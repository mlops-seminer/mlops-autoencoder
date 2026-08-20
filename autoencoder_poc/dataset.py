import os
import glob
import hydra
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # まずはRGBで読み込む
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0 # 学習用なのでラベルはダミー

def get_transforms(cfg):
    input_size = tuple(cfg.dataset.input_size)
    
    transform_list = []
    
    # グレースケール変換 (PILレベルまたはTransformレベル)
    # ここではTransformで対応
    if cfg.dataset.channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
        
    transform_list.extend([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(transform_list)

def get_dataloaders(cfg):
    validation_ratio = float(cfg.dataset.validation_ratio)
    if not 0 <= validation_ratio < 1:
        raise ValueError("dataset.validation_ratio は0以上1未満で指定してください。")
    if int(cfg.dataset.batch_size) <= 0:
        raise ValueError("dataset.batch_size は1以上で指定してください。")
    if int(cfg.dataset.num_workers) < 0:
        raise ValueError("dataset.num_workers は0以上で指定してください。")

    transform = get_transforms(cfg)
    
    # --- Train Data Loading ---
    # Hydraを使用している場合、パスを絶対パスに変換（出力ディレクトリに移動している可能性があるため）
    root_dir = hydra.utils.to_absolute_path(cfg.dataset.root_dir)
    train_dir = os.path.join(root_dir, cfg.dataset.train_dir)
    
    # 対応する拡張子
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    train_files = []
    for ext in exts:
        # 大文字小文字両方対応のため
        train_files.extend(glob.glob(os.path.join(train_dir, ext)))
        train_files.extend(glob.glob(os.path.join(train_dir, ext.upper())))
    
    # 重複削除 (念のため)
    train_files = sorted(list(set(train_files)))
    
    if not train_files:
        print(f"Warning: No images found in {train_dir}")
    
    full_train_dataset = SimpleImageDataset(train_files, transform=transform)
    
    # Validation Split
    if len(full_train_dataset) > 0:
        val_size = int(len(full_train_dataset) * validation_ratio)
        train_size = len(full_train_dataset) - val_size
        
        # データが少なすぎてval_sizeが0になるのを防ぐ
        if val_size == 0 and len(full_train_dataset) > 1:
            val_size = 1
            train_size = len(full_train_dataset) - 1
            
        split_strategy = str(cfg.dataset.get("split_strategy", "sequential")).lower()
        if split_strategy == "sequential":
            # 時刻順ファイルの後半をvalidationにし、近接フレームの混入を抑える
            train_dataset = Subset(full_train_dataset, range(train_size))
            val_dataset = Subset(full_train_dataset, range(train_size, len(full_train_dataset)))
        elif split_strategy == "random":
            generator = torch.Generator().manual_seed(int(cfg.dataset.get("seed", 42)))
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size], generator=generator
            )
        else:
            raise ValueError(
                "dataset.split_strategy は sequential または random を指定してください。"
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.dataset.batch_size, 
            shuffle=True, 
            num_workers=cfg.dataset.num_workers
        )
        
        val_loader = None
        if val_size > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers
            )
    else:
        train_loader = None
        val_loader = None

    # --- Test Data Loading ---
    test_dir = os.path.join(root_dir, cfg.dataset.test_dir)
    
    if os.path.exists(test_dir):
        # ImageFolderを使うとサブディレクトリ名がクラス名になる
        test_dataset = ImageFolder(test_dir, transform=transform)
        class_names = test_dataset.classes
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.dataset.num_workers
        )
    else:
        print(f"Warning: Test directory {test_dir} not found.")
        test_loader = None
        class_names = []

    return train_loader, val_loader, test_loader, class_names
