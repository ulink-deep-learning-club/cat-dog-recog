"""
Dataset module for cat/dog image classification.
Downloads and prepares the dataset from public sources.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Tuple, Optional, Callable
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CatDogDataset(Dataset):
    """Custom dataset for cat/dog classification."""

    def __init__(
        self, root_dir: str, transform: Optional[Callable] = None, train: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing 'train' and 'test' folders
            transform: Optional transform to apply to images
            train: If True, use training data, else use test data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # Set data directory
        data_dir = os.path.join(root_dir, "train" if train else "test")

        # Get all image paths and labels
        self.image_paths = []
        self.labels = []

        # Map class names to labels
        self.class_to_idx = {"cat": 0, "dog": 1}

        # Walk through directory structure
        for class_name in ["cat", "dog"]:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

        print(f"Loaded {len(self.image_paths)} images from {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self) -> list:
        """Return list of class names."""
        return ["cat", "dog"]


class DatasetManager:
    """Manages dataset downloading and preparation."""

    KAGGLE_DATASET_URL = "https://www.kaggle.com/c/dogs-vs-cats/data"
    ALTERNATIVE_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_dataset(self, source: str = "kaggle") -> str:
        """
        Download cat/dog dataset from specified source.

        Args:
            source: 'kaggle' or 'microsoft' (alternative)

        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading dataset from {source}...")

        if source == "kaggle":
            # Note: Kaggle requires authentication
            print("Kaggle dataset requires authentication.")
            print(
                "Please download manually from: https://www.kaggle.com/c/dogs-vs-cats/data"
            )
            print("Or use the Microsoft alternative dataset.")
            return self._download_microsoft_dataset()
        else:
            return self._download_microsoft_dataset()

    def _download_microsoft_dataset(self) -> str:
        """Download Microsoft's cat/dog dataset."""
        url = self.ALTERNATIVE_URL
        zip_path = self.data_dir / "cats_and_dogs.zip"

        if zip_path.exists():
            print(f"Dataset zip already exists at {zip_path}")
        else:
            # Download with progress bar
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(zip_path, "wb") as f,
                tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))

        # Check if already extracted and organized
        train_dir = self.data_dir / "train"
        test_dir = self.data_dir / "test"
        if train_dir.exists() and test_dir.exists():
            cat_train = (train_dir / "cat").exists() and any(
                (train_dir / "cat").glob("*.jpg")
            )
            dog_train = (train_dir / "dog").exists() and any(
                (train_dir / "dog").glob("*.jpg")
            )
            if cat_train and dog_train:
                print("Dataset already organized. Skipping extraction.")
                return str(self.data_dir)

        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        # Organize into train/test structure
        self._organize_dataset()

        return str(self.data_dir)

    def _organize_dataset(self):
        """Organize extracted dataset into proper structure."""
        extracted_dir = self.data_dir / "PetImages"

        if not extracted_dir.exists():
            print(
                "Extracted directory not found. Dataset may have different structure."
            )
            return

        # Create train/test directories
        train_dir = self.data_dir / "train"
        test_dir = self.data_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # For each class
        for class_name in ["Cat", "Dog"]:
            class_dir = extracted_dir / class_name

            if not class_dir.exists():
                continue

            # Get all images and remove invalid ones
            images = self._remove_invalid_images(class_dir)

            # Split 80/20 for train/test
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            test_images = images[split_idx:]

            # Create class directories
            train_class_dir = train_dir / class_name.lower()
            test_class_dir = test_dir / class_name.lower()
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)

            # Copy images
            for img in train_images:
                shutil.copy(img, train_class_dir / img.name)

            for img in test_images:
                shutil.copy(img, test_class_dir / img.name)

        print(f"Dataset organized into {train_dir} and {test_dir}")

    def clean_invalid_images(self, directory: str = None):
        """Remove invalid images from specified directory or default data directories.

        Args:
            directory: Optional specific directory to clean. If None, cleans train/test directories.
        """
        if directory:
            dirs_to_clean = [Path(directory)]
        else:
            dirs_to_clean = [
                self.data_dir / "train" / "cat",
                self.data_dir / "train" / "dog",
                self.data_dir / "test" / "cat",
                self.data_dir / "test" / "dog",
            ]

        total_removed = 0
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                valid_images = self._remove_invalid_images(dir_path)
                # Update files in directory
                for old_file in dir_path.glob("*.jpg"):
                    if old_file not in valid_images:
                        old_file.unlink()
                        total_removed += 1

        print(f"Cleaned {total_removed} invalid images total")

    def _remove_invalid_images(self, class_dir: Path) -> list:
        """Remove invalid/corrupted images from a directory.

        Args:
            class_dir: Directory containing images

        Returns:
            List of valid image paths
        """
        valid_images = []
        invalid_count = 0

        for img_path in class_dir.glob("*.jpg"):
            if self._is_valid_image(img_path):
                valid_images.append(img_path)
            else:
                invalid_count += 1
                print(f"Removing invalid image: {img_path.name}")

        print(f"Removed {invalid_count} invalid images from {class_dir.name}")
        return valid_images

    def _is_valid_image(self, img_path: Path) -> bool:
        """Check if a file is a valid image.

        Args:
            img_path: Path to the image file

        Returns:
            True if valid image, False otherwise
        """
        try:
            # Check file size (skip 0 byte files)
            if img_path.stat().st_size == 0:
                return False

            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def get_transforms(
        self, augment: bool = True
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get train and validation transforms.

        Args:
            augment: Whether to use data augmentation for training

        Returns:
            Tuple of (train_transform, val_transform)
        """
        # Base transforms
        base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            train_transform = base_transform

        val_transform = base_transform

        return train_transform, val_transform

    def create_dataloaders(
        self, batch_size: int = 32, num_workers: int = 4, augment: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            augment: Whether to use data augmentation

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_transform, val_transform = self.get_transforms(augment)

        # Create datasets
        train_dataset = CatDogDataset(
            root_dir=str(self.data_dir), transform=train_transform, train=True
        )

        val_dataset = CatDogDataset(
            root_dir=str(self.data_dir), transform=val_transform, train=False
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader


def test_dataset():
    """Test the dataset module."""
    manager = DatasetManager()

    # Try to load existing data
    if not (manager.data_dir / "train").exists():
        print("No dataset found. Please download dataset first.")
        print("You can use: dataset_manager.download_dataset('microsoft')")
        return

    # Create dataloaders
    train_loader, val_loader = manager.create_dataloaders(batch_size=4)

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")


if __name__ == "__main__":
    test_dataset()
