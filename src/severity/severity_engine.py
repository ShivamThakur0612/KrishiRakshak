"""
KrishiRakshak Severity Engine.

Estimates the proportion of suspected diseased pixels
within the detected leaf area.

IMPORTANT:
The thresholds are prototype engineering thresholds.
They are not universal agricultural treatment standards.
"""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str):
    """Load an image from disk."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


def create_leaf_mask(image):
    """
    Estimate the leaf region.

    Green vegetation is detected using HSV color ranges.
    Morphological operations remove small noise.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lower_green = np.array(
        [25, 25, 20],
        dtype=np.uint8,
    )

    upper_green = np.array(
        [100, 255, 255],
        dtype=np.uint8,
    )

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return green_mask


def calculate_affected_area(image_path: str) -> float:
    """
    Estimate the percentage of leaf area showing
    brown/yellow lesion-like discoloration.

    The calculation is performed only inside the
    detected leaf region.
    """

    image = load_image(image_path)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    leaf_mask = create_leaf_mask(image)

    leaf_pixels = np.count_nonzero(leaf_mask)

    if leaf_pixels == 0:
        raise ValueError(
            "Leaf region could not be detected."
        )

    # Brown/yellow lesion-like regions.
    lower_lesion = np.array(
        [5, 35, 20],
        dtype=np.uint8,
    )

    upper_lesion = np.array(
        [40, 255, 230],
        dtype=np.uint8,
    )

    lesion_mask = cv2.inRange(
        hsv,
        lower_lesion,
        upper_lesion,
    )

    # Only count suspected lesions inside the leaf.
    lesion_inside_leaf = cv2.bitwise_and(
        lesion_mask,
        leaf_mask,
    )

    affected_pixels = np.count_nonzero(
        lesion_inside_leaf
    )

    affected_area_percent = (
        affected_pixels / leaf_pixels
    ) * 100.0

    return float(affected_area_percent)


def estimate_severity(
    image_path: str,
    affected_area_percent: float,
) -> dict:
    """
    Convert affected leaf area into a prototype
    severity category.

    < 10%   -> mild
    10-30%  -> moderate
    > 30%   -> severe
    """

    if affected_area_percent < 10:
        severity = "mild"

    elif affected_area_percent <= 30:
        severity = "moderate"

    else:
        severity = "severe"

    return {
        "affected_area_percent": round(
            float(affected_area_percent),
            2,
        ),
        "severity": severity,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Severity Engine")
    print("=" * 60)

    image_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "maize_crop.png"
    )

    print(f"\nImage: {image_path}")
    print("\nDetecting leaf region...")

    affected_area = calculate_affected_area(
        str(image_path)
    )

    result = estimate_severity(
        str(image_path),
        affected_area,
    )

    print("\nSeverity Result:")
    print(result)

    print("\nSeverity analysis completed successfully.")