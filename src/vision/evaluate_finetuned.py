"""
KrishiRakshak fine-tuned model evaluation.

Evaluates the fine-tuned MobileNetV3-Small model
on the same untouched maize test set used for
the original baseline.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_maize_mobilenetv3_small_finetuned.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def load_model(classes):

    model = mobilenet_v3_small(
        weights=None
    )

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(classes),
    )

    # Fine-tuned checkpoint has the same
    # architecture as MobileNetV3-Small.
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


def evaluate():

    print("=" * 60)
    print("KrishiRakshak Fine-Tuned Model Evaluation")
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

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print("\n" + "=" * 60)
    print("FINE-TUNED TEST RESULTS")
    print("=" * 60)

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
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