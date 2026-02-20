"""
Data preprocessing script for Cats vs Dogs classification.

This script:
1. Reads raw images from the dataset directory
2. Resizes them to 224x224 RGB format
3. Splits the dataset into train/validation/test sets
4. Saves processed images in structured folders

This approach ensures reproducibility and consistency
for model training in an MLOps pipeline.
"""

import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
RAW_DATA_DIR = "dvc_data/PetImages"
PROCESSED_DIR = "dvc_data/processed"
IMG_SIZE = (224, 224)

SPLIT_RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

random.seed(42)


def create_dirs():
    """Create output directory structure."""
    for split in SPLIT_RATIO.keys():
        for label in ["Cat", "Dog"]:
            Path(f"{PROCESSED_DIR}/{split}/{label}").mkdir(parents=True, exist_ok=True)


def get_image_paths():
    """Collect image file paths."""
    cat_paths = list(Path(f"{RAW_DATA_DIR}/Cat").glob("*.jpg"))
    dog_paths = list(Path(f"{RAW_DATA_DIR}/Dog").glob("*.jpg"))
    return cat_paths, dog_paths


def split_data(paths):
    """Split dataset into train/val/test."""
    random.shuffle(paths)

    n_total = len(paths)
    n_train = int(n_total * SPLIT_RATIO["train"])
    n_val = int(n_total * SPLIT_RATIO["val"])

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


def process_and_save(paths, split, label):
    """Resize and save images."""
    for img_path in tqdm(paths, desc=f"{split}-{label}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            save_path = Path(PROCESSED_DIR) / split / label / img_path.name
            img.save(save_path)

        except Exception:
            # Skip corrupted images
            continue


def main():
    print("Starting preprocessing pipeline...")

    create_dirs()

    cat_paths, dog_paths = get_image_paths()

    for label, paths in [("Cat", cat_paths), ("Dog", dog_paths)]:
        train, val, test = split_data(paths)

        process_and_save(train, "train", label)
        process_and_save(val, "val", label)
        process_and_save(test, "test", label)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()