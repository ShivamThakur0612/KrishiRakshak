"""
KrishiRakshak MobileNetV3-Small fine-tuning experiment.

Baseline:
- ImageNet pretrained MobileNetV3-Small
- Frozen feature extractor
- 90.65% test accuracy

Experiment:
- Keep early layers frozen
- Fine-tune upper MobileNetV3 layers
- Use a small learning rate
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "vision"

MODEL_PATH = (
    MODEL_DIR
    / "best_maize_mobilenetv3_small_finetuned.pth"
)

NUM_CLASSES = 4

BATCH_SIZE = 32
EPOCHS = 8

LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def create_model():

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights
    )

    # Freeze all feature layers first
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune the final feature blocks
    # MobileNetV3-Small has 12 feature blocks.
    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Replace classifier
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


def main():

    print("=" * 60)
    print("KrishiRakshak MobileNetV3-Small Fine-Tuning")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        f"\nTraining images: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )

    print("\nClasses:")
    print(train_loader.dataset.classes)

    model = create_model()

    model = model.to(DEVICE)

    # Show trainable parameter count
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Total parameters: "
        f"{total_count:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    print("\nStarting fine-tuning...\n")

    for epoch in range(EPOCHS):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: "
            f"{val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "classes":
                        train_loader.dataset.classes,

                    "image_size": 224,

                    "model_name":
                        "mobilenet_v3_small_finetuned",

                    "val_accuracy":
                        best_val_accuracy,
                },
                MODEL_PATH,
            )

            print(
                f"  ✓ Best fine-tuned model saved "
                f"({best_val_accuracy * 100:.2f}%)"
            )

    print("\n" + "=" * 60)

    print("Fine-tuning completed.")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()