"""
KrishiRakshak MobileNetV3-Small vision model.
4-class maize disease classifier.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


NUM_CLASSES = 4


def create_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Create a MobileNetV3-Small classifier.

    Args:
        num_classes: Number of output disease classes.
        pretrained: Whether to use ImageNet pretrained weights.
    """

    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v3_small(weights=weights)

    # Replace the original ImageNet classifier.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes,
    )

    return model


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak MobileNetV3-Small Test")
    print("=" * 60)

    model = create_model(pretrained=False)

    print("Model created successfully.")
    print("Number of classes:", NUM_CLASSES)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nModel test completed successfully.")