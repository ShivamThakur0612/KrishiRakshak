"""
Image quality gate for KrishiRakshak.

Rejects images that are too blurry, too dark, or too small
before they reach the crop-disease model.
"""

from pathlib import Path

import cv2


def check_image_quality(
    image_path: str,
    min_width: int = 224,
    min_height: int = 224,
    min_brightness: float = 35.0,
    max_brightness: float = 225.0,
    min_sharpness: float = 50.0,
) -> dict:
    """
    Check whether a crop image is suitable for disease detection.

    Returns a dictionary containing:
        accepted: True/False
        reason: explanation
        width: image width
        height: image height
        brightness: average brightness
        sharpness: Laplacian variance
    """

    path = Path(image_path)

    if not path.exists():
        return {
            "accepted": False,
            "reason": "Image file not found",
        }

    image = cv2.imread(str(path))

    if image is None:
        return {
            "accepted": False,
            "reason": "Unable to read image",
        }

    height, width = image.shape[:2]

    if width < min_width or height < min_height:
        return {
            "accepted": False,
            "reason": "Image resolution is too low",
            "width": width,
            "height": height,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < min_brightness:
        return {
            "accepted": False,
            "reason": "Image is too dark",
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
        }

    if brightness > max_brightness:
        return {
            "accepted": False,
            "reason": "Image is overexposed",
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
        }

    if sharpness < min_sharpness:
        return {
            "accepted": False,
            "reason": "Image is too blurry",
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
        }

    return {
        "accepted": True,
        "reason": "Image quality is acceptable",
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


if __name__ == "__main__":
    print("KrishiRakshak image-quality module loaded successfully.")