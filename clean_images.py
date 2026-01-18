#!/usr/bin/env python3
"""
Script to remove invalid/corrupted images from the dataset directories.
"""

import warnings
import os
from pathlib import Path
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False


def is_valid_image(img_path: Path) -> bool:
    """Check if a file is a valid image."""
    try:
        if img_path.stat().st_size == 0:
            return False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Image.open(img_path) as img:
                img.load()

            if any("truncated" in str(warning.message).lower() for warning in w):
                return False

        return True
    except Exception:
        return False


def remove_invalid_images(data_dir: str = "data") -> int:
    """Remove all invalid images from train and test directories.

    Args:
        data_dir: Root data directory containing train/test folders

    Returns:
        Number of invalid images removed
    """
    data_path = Path(data_dir)
    total_removed = 0

    for split in ["train", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue

        for class_name in ["cat", "dog"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            print(f"Scanning {class_dir}...")
            for img_path in class_dir.glob("*.jpg"):
                if not is_valid_image(img_path):
                    print(f"  Removing: {img_path.name}")
                    img_path.unlink()
                    total_removed += 1

    print(f"\nTotal invalid images removed: {total_removed}")
    return total_removed


if __name__ == "__main__":
    remove_invalid_images()
