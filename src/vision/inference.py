"""
KrishiRakshak Vision Inference

Runs the fine-tuned maize disease model on a single image
and returns disease + confidence in a structured format.
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_maize_mobilenetv3_small_finetuned.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight",
]


def load_model():
    """Load the fine-tuned MobileNetV3-Small model."""

    model = mobilenet_v3_small(weights=None)

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        len(CLASSES),
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


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

def predict(image_path: str) -> dict:
    """
    Predict the disease from a single maize image.

    Returns:
        {
            "crop": "maize",
            "disease": "...",
            "confidence": 0.00,
            "confidence_level": "...",
            "top_predictions": [...]
        }
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model = load_model()

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=2,
            dim=1,
        )

    predicted_index = top_indices[0][0].item()
    predicted_class = CLASSES[predicted_index]

    confidence = top_probabilities[0][0].item()

    top_predictions = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0],
    ):
        top_predictions.append(
            {
                "disease": CLASSES[index.item()],
                "confidence": round(
                    probability.item(),
                    4,
                ),
            }
        )

    if confidence >= 0.85:
        confidence_level = "high"
    elif confidence >= 0.60:
        confidence_level = "probable"
    else:
        confidence_level = "low"

    return {
        "crop": "maize",
        "disease": predicted_class,
        "confidence": round(
            confidence,
            4,
        ),
        "confidence_level": confidence_level,
        "top_predictions": top_predictions,
    }

if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Vision Inference")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "maize_crop.png"
    )

    print(f"\nImage: {image_path}")
    print("\nRunning prediction...\n")

    result = predict(str(image_path))

    print("Prediction Result:")
    print(result)

    print("\nInference completed successfully.")