"""
KrishiRakshak maize dataset preparation.

Validates images and creates a reproducible
train/validation/test split.
"""

from pathlib import Path
import random
import shutil

from PIL import Image


RAW_DIR = Path("data/raw/maize")
PROCESSED_DIR = Path("data/processed/maize")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "common_rust",
    "Corn_(maize)___healthy": "healthy",
    "Corn_(maize)___Northern_Leaf_Blight": "northern_leaf_blight",
}


def validate_image(image_path: Path) -> bool:
    """Check whether an image can be opened and loaded."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.load()

        return True

    except Exception:
        return False


def collect_images():
    """Collect valid images from the four maize classes."""

    all_images = {}
    invalid_images = []

    for original_class, clean_class in CLASS_NAMES.items():

        class_dir = RAW_DIR / original_class

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Dataset class folder not found: {class_dir}"
            )

        valid_images = []

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if validate_image(image_path):
                valid_images.append(image_path)
            else:
                invalid_images.append(image_path)

        all_images[clean_class] = valid_images

    return all_images, invalid_images


def split_images(images):
    """Split images into train, validation and test sets."""

    random.seed(RANDOM_SEED)

    shuffled = images.copy()
    random.shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = shuffled[:train_end]
    val_images = shuffled[train_end:val_end]
    test_images = shuffled[val_end:]

    return train_images, val_images, test_images


def copy_images(images, destination):
    """Copy images into a destination directory."""

    destination.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(
            image_path,
            destination / image_path.name
        )


def prepare_dataset():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DIR}"
        )

    print("=" * 60)
    print("KrishiRakshak Maize Dataset Preparation")
    print("=" * 60)

    all_images, invalid_images = collect_images()

    total_valid = 0

    print("\nImage validation results:")

    for class_name, images in all_images.items():

        print(f"{class_name:25s}: {len(images)} valid images")

        total_valid += len(images)

    print(f"\nTotal valid images: {total_valid}")
    print(f"Invalid images: {len(invalid_images)}")

    if invalid_images:
        print("\nFirst invalid images:")

        for image_path in invalid_images[:10]:
            print(image_path)

    print("\nCreating dataset splits...")

    for split in ["train", "val", "test"]:

        split_dir = PROCESSED_DIR / split

        if split_dir.exists():
            shutil.rmtree(split_dir)

    split_counts = {}

    for class_name, images in all_images.items():

        train_images, val_images, test_images = split_images(images)

        copy_images(
            train_images,
            PROCESSED_DIR / "train" / class_name
        )

        copy_images(
            val_images,
            PROCESSED_DIR / "val" / class_name
        )

        copy_images(
            test_images,
            PROCESSED_DIR / "test" / class_name
        )

        split_counts[class_name] = {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

    print("\nDataset split:")
    print("-" * 60)

    for class_name, counts in split_counts.items():

        print(
            f"{class_name:25s} "
            f"train={counts['train']:3d} "
            f"val={counts['val']:3d} "
            f"test={counts['test']:3d}"
        )

    print("-" * 60)
    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    prepare_dataset()