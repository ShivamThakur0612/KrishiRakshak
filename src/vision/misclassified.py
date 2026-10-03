"""
KrishiRakshak misclassification analysis.

Finds incorrectly classified images from the untouched
maize test set and saves them with actual/predicted labels.
"""

from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from dataset_loader import create_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_maize_mobilenetv3_small_finetuned.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "misclassified"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model(classes):
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


def analyze_misclassifications():

    print("=" * 60)
    print("KrishiRakshak Misclassification Analysis")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    _, _, test_loader = create_dataloaders()

    classes = test_loader.dataset.classes

    print("\nClasses:")
    for index, class_name in enumerate(classes):
        print(f"{index}: {class_name}")

    model = load_model(classes)

    # Remove previous analysis results
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    incorrect = 0

    confusion_counts = {}

    print("\nAnalyzing test images...\n")

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            for image_index in range(len(labels)):

                actual_index = labels[image_index].item()
                predicted_index = predictions[image_index].item()

                total += 1

                if actual_index == predicted_index:
                    continue

                incorrect += 1

                actual_class = classes[actual_index]
                predicted_class = classes[predicted_index]

                key = (
                    actual_class,
                    predicted_class,
                )

                confusion_counts[key] = (
                    confusion_counts.get(key, 0) + 1
                )

                # Get original image path
                dataset_index = (
                    batch_index * test_loader.batch_size
                    + image_index
                )

                original_path, _ = (
                    test_loader.dataset.samples[
                        dataset_index
                    ]
                )

                # Create folder:
                # actual__predicted
                pair_dir = (
                    OUTPUT_DIR
                    / f"{actual_class}__to__{predicted_class}"
                )

                pair_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                destination = (
                    pair_dir
                    / Path(original_path).name
                )

                shutil.copy2(
                    original_path,
                    destination,
                )

    print("=" * 60)
    print("MISCLASSIFICATION RESULTS")
    print("=" * 60)

    print(f"\nTotal test images: {total}")
    print(f"Incorrect predictions: {incorrect}")
    print(
        f"Correct predictions: "
        f"{total - incorrect}"
    )

    error_rate = incorrect / total

    print(
        f"Error rate: "
        f"{error_rate * 100:.2f}%"
    )

    print("\nMisclassification pairs:")

    sorted_counts = sorted(
        confusion_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (actual, predicted), count in sorted_counts:

        print(
            f"{actual:25s} -> "
            f"{predicted:25s}: "
            f"{count}"
        )

    print("\nSaved incorrectly classified images to:")

    print(OUTPUT_DIR)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    analyze_misclassifications()
    