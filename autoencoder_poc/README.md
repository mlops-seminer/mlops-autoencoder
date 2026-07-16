# モデル学習・評価ツール (autoencoder_poc)

作成したデータセットを使用してオートエンコーダーの学習を行い、異常検知の精度検証（PoC）を行うツールです。
実験管理には **MLflow** を使用しており、学習経過（Loss）、評価スコア、再構成画像の結果などをWebブラウザ上で確認・管理できます。

## 前提条件

1.  `create_traindata` ツールなどで、データセット（`train/good`, `test/good`, `test/anomaly`）が用意されていること。
2.  Python環境に `mlflow` がインストールされていること。

## 実行手順

### 1. MLflowサーバーの起動

学習ログを記録するために、まずMLflowサーバーをコンテナで起動してください。

> **注意**: すでに起動している場合はこの手順は不要です。

### 2. 学習の実行

新しいターミナルを開き（プロジェクトルートで）、以下のコマンドを実行して学習を開始します。

```bash
python autoencoder_poc/main.py
```

プログラムは自動的に以下の処理を行います：
1.  データセットの読み込み
2.  オートエンコーダーの学習（Train）
3.  検証（Validation）とEarly Stopping判定
4.  テストデータを用いた異常検知スコアの算出
5.  結果画像の生成（箱ひげ図、ヒストグラム、再構成画像の比較）

### 3. 結果の確認

ブラウザで [http://localhost:8080](http://localhost:8080) にアクセスします。
実験名 `Autoencoder_Anomaly_Detection` の中に、今回の実行結果（Run）が記録されています。

**確認できる項目:**
*   **Metrics**: `train_loss`, `val_loss`, 各クラスの異常スコア（平均・分散など）
*   **Artifacts**:
    *   `reconstruction_comparison.png`: 元画像と再構成画像の比較
    *   `boxplot.png`: クラスごとの異常スコアの分布（箱ひげ図）
    *   `histogram.png`: 異常スコアのヒストグラム

## 次のステップへ（重要：モデルの登録）

学習したモデルを次のステップ「推論 (autoencoder_inference)」で使用するためには、MLflow上でモデルを登録する必要があります。

1.  MLflow UIで、結果が良かった Run の詳細画面を開きます。
2.  **Artifacts** セクションにある `autoencoder_model` (または `best_model`) フォルダをクリックします。
3.  右側に表示される **"Register Model"** ボタンをクリックします。
4.  **Model Name** に、推論ツールの設定 (`autoencoder_inference/config/config.yaml`) と一致する名前（例: `kinoko`）を入力して **Register** します。

これで推論ツールからモデルを読み込めるようになります。

## 設定の変更

`config/config.yaml` ファイルで学習パラメータを変更できます。

*   **dataset**: データセットのパスや画像サイズ
*   **training**:
    *   `epochs`: 学習回数
    *   `learning_rate`: 学習率
    *   `early_stopping_patience`: 検証ロスが改善しなくなった時に停止するまでの回数
*   **model**:
    *   `enc_channels`: エンコーダーの層構造
    *   `latent_dim`: 潜在変数の次元数

## 補足：オートエンコーダのモデル構造
## 1. オートエンコーダの基本構造

オートエンコーダは次の2つの部分から構成されます。

### **Encoder（エンコーダ）**
- 入力画像を小さく圧縮していき、特徴量を抽出する部分  
- 畳み込み（Conv）、活性化（ReLU）、プーリング（MaxPool）を段階的に適用  
- 最終的に **潜在ベクトル（latent vector）** と呼ばれる圧縮表現へと変換します

### **Decoder（デコーダ）**
- 潜在ベクトルから元の画像サイズへ復元する部分  
- ConvTranspose を用いて空間解像度を段階的に拡大していきます  
- 最終層では `Sigmoid()` を使い、出力値を 0〜1 の範囲に収めています

学習の目的は、

> **「入力画像」と「復元画像」ができるだけ近くなるように重みを調整すること」**

であり、異常検知では正常画像のみを学習し、正常・異常で再構成誤差の差が生じることを利用します。

---

## 2. 入力画像とエンコーダの変化

設定ファイルでは以下の入力仕様が定義されています。

```yaml
input_size: [256, 256]
channels: 3
```

つまり、入力画像サイズは **256×256、RGB（3チャネル）** です。

エンコーダ側では、

```yaml
enc_channels: [32, 64, 128]
```

により、次のような3段のブロックが作られます。

各ブロックの構成  
> Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2)

MaxPool(2×2) によって **縦横サイズが1/2** になります。

| 層 | チャネル数 | サイズ変化 |
|---|---|---|
| 入力 | 3 | 256×256 |
| 1層目 | 32 | 256 → 128 |
| 2層目 | 64 | 128 → 64 |
| 3層目 | 128 | 64 → 32 |

最終的に得られる特徴マップは：

- チャネル：128  
- サイズ：32×32  

これを flatten して **128 × 32 × 32 = 131072 次元** となります。

---

## 3. 潜在空間（latent space）

設定ファイルの：

```yaml
latent_dim: 32
```

により、Encoder の最終出力（131072次元）を **32次元の潜在ベクトル**に圧縮します。

### 潜在次元を変更するとどうなるか？

- **小さくする（例: 16, 8）**  
  - 圧縮が強くなり情報が詰め込まれる  
  - 正常と異常の差が出やすくなり、異常検知性能が上がる可能性  
  - ただし、正常画像の再構成精度が落ちるリスクあり

- **大きくする（例: 64, 128）**  
  - 再構成は綺麗になりやすい  
  - しかし異常画像も少し復元できてしまい異常検知が弱くなることもある

---

## 4. デコーダによる復元

Decoder は Encoder の逆順で構築されます。

設定の `[32, 64, 128]` が逆順になり：

```
[128, 64, 32]
```

となり、次のように復元されます。

| 層 | 入力チャネル → 出力チャネル | サイズ |
|---|---|---|
| 1層目 | 128 → 64 | 32 → 64 |
| 2層目 | 64 → 32 | 64 → 128 |
| 3層目 | 32 → 3 | 128 → 256（元の画像） |

最終層では `Sigmoid()` により画素値を [0,1] に整形します。

---

## 5. 設定ファイルで調整できるポイント

### **(1) enc_channels：モデルの大きさ・深さ**
```yaml
enc_channels: [32, 64, 128]
```

- リストを長くする → 層が増え、より深いモデルに  
- 数値を大きくする → 各層の表現力アップ（ただしメモリ・計算量増加）

モデルの規模感や抽象度をここで調整できます。

---

### **(2) latent_dim：圧縮度合いの調整**
```yaml
latent_dim: 32
```

PoC では良い感じの値を探索する必要があります。

---

## 6. まとめ

本PoCで使用するオートエンコーダは次の特徴を持ちます。

- Conv + MaxPool による Encoder で画像を段階的に圧縮し特徴抽出  
- `latent_dim` 次元の潜在ベクトルに圧縮  
- ConvTranspose による Decoder で画像を元サイズに復元  
- `enc_channels` と `latent_dim` により、モデルの規模や圧縮率が調整可能  
- 異常検知では「再構成誤差」を利用

特に、  
- **enc_channels** → モデルの深さ・表現力  
- **latent_dim** → 圧縮率と異常検知性能のバランス  
を決定する重要なパラメータです。


## MLflow Model Registryのモデルを使用した再学習

学習済みモデルをModel Registryから取得し、その重みを初期値として追加学習できます。

### 1. エイリアスを指定して再学習する

`config/config.yaml` を次のように変更します。

```yaml
training:
  resume:
    enabled: true
    source: "registry"
    registered_model_name: "kinoko"
    model_alias: "champion"
    model_version: null
    model_uri: null
```

この設定では、次のMLflowモデルURIからモデルをダウンロードします。

```text
models:/kinoko@champion
```

### 2. バージョンを指定して再学習する

```yaml
training:
  resume:
    enabled: true
    source: "registry"
    registered_model_name: "kinoko"
    model_alias: null
    model_version: 3
```

この場合は `models:/kinoko/3` が読み込まれます。

### 3. モデルURIを直接指定する

```yaml
training:
  resume:
    enabled: true
    source: "registry"
    model_uri: "models:/kinoko@champion"
```

`model_uri` が指定されている場合は、モデル名・エイリアス・バージョンからのURI生成より優先されます。

### 4. 再学習の実行

```bash
python autoencoder_poc/main.py
```

実行時は、MLflow Model RegistryからPyTorchモデルがダウンロードされ、取得したモデルの `state_dict` が現在のAutoencoderへ読み込まれます。その後、設定した学習率とエポック数で再学習します。

再学習のRunには、次のタグが記録されます。

- `training_mode: retraining`
- `resume_source: models:/kinoko@champion` など

### 5. 再学習後のモデル登録

次の設定により、学習終了後のモデルがModel Registryへ自動登録されます。

```yaml
mlflow:
  model_name: "autoencoder_model"
  registered_model_name: "kinoko"
```

同じ登録モデル名を指定すると、新しいモデルバージョンとして追加されます。自動登録しない場合は `registered_model_name: null` にします。

### 注意事項

再学習元と現在の設定で、次のモデル構造を一致させてください。

- `dataset.channels`
- `dataset.input_size`
- `model.enc_channels`
- `model.latent_dim`

構造が異なる場合は、`state_dict` の読み込み時にサイズ不一致エラーになります。
