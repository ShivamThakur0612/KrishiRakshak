"""
KrishiRakshak MobileNetV3-Small training pipeline.

First-stage transfer learning:
- ImageNet pretrained MobileNetV3-Small
- Frozen feature extractor
- Train only the final classifier
- 4 maize disease classes
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from dataset_loader import create_dataloaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "vision"

MODEL_PATH = MODEL_DIR / "best_maize_mobilenetv3_small.pth"

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Model
# --------------------------------------------------

def create_model():
    """Create ImageNet-pretrained MobileNetV3-Small."""

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    # Freeze the feature extractor.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace ImageNet classifier with our 4-class classifier.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    return model


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation
# --------------------------------------------------

def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("KrishiRakshak MobileNetV3-Small Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    print("\nClasses:")
    print(train_loader.dataset.classes)

    # Model
    model = create_model()
    model = model.to(DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Only train parameters that require gradients.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # Save best model.
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_loader.dataset.classes,
                    "image_size": 224,
                    "model_name": "mobilenet_v3_small",
                    "val_accuracy": best_val_accuracy,
                },
                MODEL_PATH,
            )

            print(
                f"  ✓ Best model saved "
                f"({best_val_accuracy * 100:.2f}%)"
            )

    print("\n" + "=" * 60)
    print("Training completed.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()