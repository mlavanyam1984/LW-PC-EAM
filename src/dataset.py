"""
MVTec AD Dataset Loader
========================
Handles the MVTec Anomaly Detection dataset folder structure:

    mvtec_ad/
    ├── bottle/
    │   ├── train/
    │   │   └── good/         ← defect-free training images
    │   └── test/
    │       ├── good/         ← defect-free test images
    │       ├── broken_large/ ← anomalous test images
    │       └── ...
    └── carpet/
        └── ...

Dataset source (Kaggle):
    https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset

To download:
    pip install kaggle
    kaggle datasets download -d avdvhh/mvtec-defect-detection-dataset
    unzip mvtec-defect-detection-dataset.zip -d data/mvtec_ad/
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MVTEC AD DATASET
# ─────────────────────────────────────────────────────────────────────────────

# All 15 MVTec AD categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class MVTecDataset(Dataset):
    """
    MVTec AD Dataset loader.

    Args:
        root:      Path to MVTec AD root directory
        category:  Product category (e.g., 'bottle', 'carpet')
        split:     'train' or 'test'
        transform: Optional image transform
        image_size: Target image size (default 224 for MobileNetV2)
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        assert category in MVTEC_CATEGORIES, \
            f"Unknown category '{category}'. Choose from: {MVTEC_CATEGORIES}"
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size

        if transform is None:
            if split == "train":
                self.transform = get_train_transform(image_size)
            else:
                self.transform = get_eval_transform(image_size)
        else:
            self.transform = transform

        self.samples: List[Tuple[Path, int, str]] = []  # (img_path, label, defect_type)
        self._load_samples()

    def _load_samples(self) -> None:
        """Walk directory tree and collect (path, label, defect_type) tuples."""
        split_dir = self.root / self.category / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {split_dir}\n"
                f"Please download from:\n"
                f"  kaggle datasets download -d avdvhh/mvtec-defect-detection-dataset\n"
                f"  unzip into: {self.root}"
            )

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1

            for img_file in sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg")):
                self.samples.append((img_file, label, defect_type))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, defect_type = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, defect_type

    def get_normal_subset(self) -> "MVTecDataset":
        """Return a view with only normal (good) samples — for memory bank construction."""
        subset = MVTecDataset.__new__(MVTecDataset)
        subset.root = self.root
        subset.category = self.category
        subset.split = self.split
        subset.image_size = self.image_size
        subset.transform = self.transform
        subset.samples = [(p, l, d) for p, l, d in self.samples if l == 0]
        return subset

    @property
    def defect_types(self) -> List[str]:
        return sorted(set(d for _, _, d in self.samples))

    def __repr__(self) -> str:
        n_normal = sum(1 for _, l, _ in self.samples if l == 0)
        n_anomal = sum(1 for _, l, _ in self.samples if l == 1)
        return (f"MVTecDataset(category={self.category}, split={self.split}, "
                f"normal={n_normal}, anomalous={n_anomal})")


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    root: str,
    category: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for a single MVTec category.

    Args:
        root:       Path to MVTec AD root directory
        category:   Product category (e.g., 'bottle')
        batch_size: Training batch size (32 as per paper)
        image_size: Input image resolution (224 as per paper)
        num_workers: DataLoader worker processes

    Returns:
        train_loader: Normal images only (for memory bank)
        test_loader:  Full test set (normal + anomalous)
    """
    train_ds = MVTecDataset(root, category, split="train", image_size=image_size)
    test_ds  = MVTecDataset(root, category, split="test",  image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,           # one at a time for inference + heatmaps
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def download_mvtec_kaggle(target_dir: str = "./data") -> str:
    """
    Auto-download MVTec AD from Kaggle using the Kaggle API.

    Prerequisites:
        pip install kaggle
        Place kaggle.json in ~/.kaggle/kaggle.json  (API credentials)

    Returns:
        Path to extracted dataset root
    """
    import subprocess
    import zipfile

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "mvtec-defect-detection-dataset.zip")
    extract_path = os.path.join(target_dir, "mvtec_ad")

    if os.path.exists(extract_path):
        print(f"Dataset already exists at: {extract_path}")
        return extract_path

    print("Downloading MVTec AD dataset from Kaggle ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "avdvhh/mvtec-defect-detection-dataset",
        "-p", target_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n\n"
            "Make sure kaggle is installed: pip install kaggle\n"
            "And ~/.kaggle/kaggle.json contains your API credentials."
        )

    print(f"Extracting to {extract_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    os.remove(zip_path)
    print(f"Dataset ready at: {extract_path}")
    return extract_path
