import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import get_dataloaders
from model import Autoencoder


def build_registry_model_uri(resume_cfg):
    """MLflow Model Registryから読み込むモデルURIを組み立てる。"""
    explicit_uri = resume_cfg.get("model_uri")
    if explicit_uri:
        return str(explicit_uri)

    registered_model_name = resume_cfg.get("registered_model_name")
    if not registered_model_name:
        raise ValueError(
            "再学習でMLflow Model Registryを使用する場合は "
            "training.resume.registered_model_name を指定してください。"
        )

    alias = resume_cfg.get("model_alias")
    version = resume_cfg.get("model_version")

    if alias:
        return f"models:/{registered_model_name}@{alias}"
    if version is not None:
        return f"models:/{registered_model_name}/{version}"

    raise ValueError(
        "training.resume.model_alias、training.resume.model_version、"
        "training.resume.model_uri のいずれかを指定してください。"
    )


def load_initial_weights(model, cfg, device):
    """ローカルまたはMLflow Model Registryから再学習用の重みを読み込む。"""
    resume_cfg = cfg.training.get("resume", {})
    enabled = bool(resume_cfg.get("enabled", False))

    # 後方互換: 旧 resume_model_path が指定されていればローカル重みを使う
    legacy_path = cfg.training.get("resume_model_path")
    if not enabled and legacy_path:
        resume_cfg = {"enabled": True, "source": "local", "local_path": legacy_path}
        enabled = True

    if not enabled:
        print("Training mode: new model")
        return None

    source = str(resume_cfg.get("source", "registry")).lower()

    if source == "registry":
        model_uri = build_registry_model_uri(resume_cfg)
        print(f"Downloading pretrained model from MLflow Model Registry: {model_uri}")
        loaded_model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.load_state_dict(loaded_model.state_dict(), strict=True)
        del loaded_model
        print("Pretrained weights loaded from MLflow Model Registry.")
        return model_uri

    if source == "local":
        local_path = hydra.utils.to_absolute_path(str(resume_cfg.get("local_path", "")))
        if not local_path or not os.path.exists(local_path):
            raise FileNotFoundError(f"再学習用モデルが見つかりません: {local_path}")
        print(f"Loading pretrained weights from local file: {local_path}")
        state_dict = torch.load(local_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("Pretrained weights loaded from local file.")
        return local_path

    raise ValueError(f"training.resume.source は registry または local を指定してください: {source}")

@hydra.main(version_base=None, config_path="./config/", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Run Name 設定
    run_name = cfg.mlflow.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # MLflow設定
    if "tracking_uri" in cfg.mlflow:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデル保存名設定
    model_name_conf = cfg.mlflow.get("model_name", "best_model")
    model_file_path = f"{model_name_conf}.pth"
    
    # データセット取得
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    
    # モデル構築
    model = Autoencoder(
        input_channels=cfg.dataset.channels,
        input_size=cfg.dataset.input_size,
        enc_channels=cfg.model.enc_channels,
        latent_dim=cfg.model.latent_dim
    ).to(device)
    
    # 新規学習または再学習用の初期重みを読み込む
    resume_source = load_initial_weights(model, cfg, device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Early Stopping パラメータ
    patience = cfg.training.early_stopping_patience
    best_val_loss = float('inf')
    counter = 0
    
    # --- Training Loop ---
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.set_tag("training_mode", "retraining" if resume_source else "new_training")
        if resume_source:
            mlflow.set_tag("resume_source", str(resume_source))
        
        # 学習を実行するかどうか（train_loaderがあるか）
        if train_loader:
            print("Starting Training...")
            for epoch in range(cfg.training.epochs):
                model.train()
                train_loss = 0.0
                for images, _ in train_loader:
                    images = images.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)
                
                train_loss /= len(train_loader.dataset)
                
                # Validation
                val_loss = 0.0
                if val_loader:
                    model.eval()
                    with torch.no_grad():
                        for images, _ in val_loader:
                            images = images.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, images)
                            val_loss += loss.item() * images.size(0)
                    val_loss /= len(val_loader.dataset)
                else:
                    val_loss = train_loss # Valがない場合はTrainと同じにしておく
                
                # ログ出力
                print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # Early Stopping Check
                if patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                        torch.save(model.state_dict(), model_file_path)
                    else:
                        counter += 1
                        print(f"Early Stopping Counter: {counter}/{patience}")
                        if counter >= patience:
                            print("Early stopping triggered")
                            break
            
            # Save final model if not stopped early or if best model wasn't saved yet
            if not os.path.exists(model_file_path):
                torch.save(model.state_dict(), model_file_path)
                
            # Load best model for evaluation
            if os.path.exists(model_file_path):
                print(f"Loading best model ({model_file_path}) for evaluation...")
                model.load_state_dict(torch.load(model_file_path))
                # PyTorchモデルとして保存し、必要に応じてModel Registryへ自動登録
                registered_model_name = cfg.mlflow.get("registered_model_name")
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=model_name_conf,
                    registered_model_name=registered_model_name
                )
                # state_dictファイルもアーティファクトとして残したければ以下をコメントアウト解除
                # mlflow.log_artifact(model_file_path)
        
        # --- Testing Loop ---
        if test_loader:
            print("Starting Evaluation...")
            model.eval()
            
            # クラスごとのスコアを格納
            results = {name: [] for name in class_names}
            
            # 画像保存用データの蓄積
            sample_images = {name: {'orig': [], 'recon': []} for name in class_names}
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    
                    # 1枚ごとのLoss計算 (Batch, Channels, H, W) -> (Batch)
                    # MSEを計算して、各画像の平均をとる
                    loss_per_image = torch.mean((images - outputs)**2, dim=[1, 2, 3])
                    
                    for i in range(len(labels)):
                        label_idx = labels[i].item()
                        class_name = class_names[label_idx]
                        score = loss_per_image[i].item()
                        results[class_name].append(score)
                        
                        # 各クラス最大1枚保持（比較画像用）
                        if len(sample_images[class_name]['orig']) < 1:
                            sample_images[class_name]['orig'].append(images[i].cpu())
                            sample_images[class_name]['recon'].append(outputs[i].cpu())

            # 結果集計
            print("\nEvaluation Results:")
            for cls, scores in results.items():
                if scores:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    max_score = np.max(scores)
                    print(f"Class: {cls:10s} | Mean: {mean_score:.6f} | Std: {std_score:.6f} | Max: {max_score:.6f}")
                    mlflow.log_metric(f"score_mean_{cls}", mean_score)
                    mlflow.log_metric(f"score_std_{cls}", std_score)
            
            # 箱ひげ図
            valid_results = {k: v for k, v in results.items() if v}
            if valid_results:
                plt.figure(figsize=(10, 6))
                plt.boxplot(list(valid_results.values()), labels=list(valid_results.keys()))
                plt.title("Anomaly Scores by Class (Boxplot)")
                plt.ylabel("Reconstruction Error (MSE)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("boxplot.png")
                mlflow.log_artifact("boxplot.png")
                plt.close()
                
                # --- ヒストグラム作成 ---
                plt.figure(figsize=(10, 6))
                for cls, scores in valid_results.items():
                    plt.hist(scores, bins=30, alpha=0.5, label=cls, density=True)
                plt.title("Distribution of Anomaly Scores by Class")
                plt.xlabel("Anomaly Score (MSE)")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig("histogram.png")
                mlflow.log_artifact("histogram.png")
                plt.close()
                
                # --- 再構成画像の比較グリッド作成 ---
                num_classes = len(class_names)
                if num_classes > 0:
                    # 最大8列まで
                    cols = min(num_classes, 8)
                    fig, axes = plt.subplots(2, cols, figsize=(3*cols, 6))
                    
                    # 1列の場合はaxesが1次元配列になるので2次元に変換
                    if cols == 1:
                        axes = np.array([[axes[0]], [axes[1]]])
                    
                    # 表示するクラス
                    classes_to_show = class_names[:cols]
                    
                    for idx, cls in enumerate(classes_to_show):
                        # サンプルがある場合のみ表示
                        if sample_images[cls]['orig']:
                            orig = sample_images[cls]['orig'][0].permute(1, 2, 0).numpy()
                            recon = sample_images[cls]['recon'][0].permute(1, 2, 0).numpy()
                            
                            # クリップ
                            orig = np.clip(orig, 0, 1)
                            recon = np.clip(recon, 0, 1)
                            
                            # グレースケール判定
                            if orig.shape[2] == 1:
                                orig = orig.squeeze(2)
                                recon = recon.squeeze(2)
                                cmap = 'gray'
                            else:
                                cmap = None
                                
                            # 上段：オリジナル
                            if cols > 1:
                                ax_orig = axes[0, idx]
                                ax_recon = axes[1, idx]
                            else:
                                ax_orig = axes[0][0]
                                ax_recon = axes[1][0]
                                
                            ax_orig.imshow(orig, cmap=cmap)
                            ax_orig.set_title(f"{cls}\nOriginal")
                            ax_orig.axis('off')
                            
                            # 下段：再構成
                            ax_recon.imshow(recon, cmap=cmap)
                            ax_recon.set_title("Reconstructed")
                            ax_recon.axis('off')
                        else:
                            # サンプルがない場合
                            if cols > 1:
                                axes[0, idx].axis('off')
                                axes[1, idx].axis('off')
                            else:
                                axes[0][0].axis('off')
                                axes[1][0].axis('off')
                            
                    plt.tight_layout()
                    plt.savefig("reconstruction_comparison.png")
                    mlflow.log_artifact("reconstruction_comparison.png")
                    plt.close()

                print("Evaluation finished. Artifacts saved to MLflow.")

if __name__ == "__main__":
    main()
