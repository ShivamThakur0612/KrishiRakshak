"""
KrishiRakshak model evaluation.

Evaluates the trained MobileNetV3-Small model
on the untouched maize test set.
"""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision.models import mobilenet_v3_small

from dataset_loader import create_dataloaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_maize_mobilenetv3_small.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_model(classes):
    """Load the trained MobileNetV3-Small model."""

    model = mobilenet_v3_small(weights=None)

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(classes),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate():

    print("=" * 60)
    print("KrishiRakshak Model Evaluation")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    _, _, test_loader = create_dataloaders()

    classes = test_loader.dataset.classes

    print("\nClasses:")
    print(classes)

    model = load_model(classes)

    all_labels = []
    all_predictions = []

    print("\nRunning inference on test set...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"\nTest Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=classes,
            digits=4,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_predictions,
        )
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    evaluate()