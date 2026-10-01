"""
KrishiRakshak PyTorch dataset loader.

Loads the prepared maize dataset and applies
training/validation/test transformations.
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("data/processed/maize")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# --------------------------------------------------
# Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Dataset creation
# --------------------------------------------------

def create_datasets():
    """Create training, validation and test datasets."""

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=eval_transform,
    )

    return train_dataset, val_dataset, test_dataset


# --------------------------------------------------
# DataLoader creation
# --------------------------------------------------

def create_dataloaders():
    """Create PyTorch DataLoaders."""

    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    print("=" * 60)
    print("KrishiRakshak DataLoader Test")
    print("=" * 60)

    print("Classes:", train_loader.dataset.classes)
    print("Class-to-index:", train_loader.dataset.class_to_idx)

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nFirst batch:")
    print("Image tensor shape:", images.shape)
    print("Label tensor shape:", labels.shape)
    print("Labels:", labels[:10].tolist())

    print("\nDataLoader test completed successfully.")