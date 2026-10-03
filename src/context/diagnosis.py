"""
KrishiRakshak Diagnosis Integration.

Combines:
    1. Image quality
    2. Vision disease prediction
    3. Confidence policy
    4. Severity estimation

Produces one structured diagnosis object.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from quality.image_quality import check_image_quality
from vision.inference import predict
from severity.severity_engine import calculate_affected_area
from severity.severity_engine import estimate_severity


def analyze_image(image_path: str) -> dict:
    """
    Run the complete KrishiRakshak diagnosis pipeline.
    """

    image_path = str(Path(image_path))

    # --------------------------------------------------
    # STEP 1: IMAGE QUALITY
    # --------------------------------------------------

    quality_result = check_image_quality(
        image_path
    )

    if not quality_result["accepted"]:
        return {
            "status": "rejected",
            "reason": quality_result["reason"],
            "quality": quality_result,
        }

    # --------------------------------------------------
    # STEP 2: DISEASE PREDICTION
    # --------------------------------------------------

    vision_result = predict(
        image_path
    )

    confidence = vision_result["confidence"]

    # --------------------------------------------------
    # STEP 3: CONFIDENCE POLICY
    # --------------------------------------------------

    if confidence < 0.60:

        return {
            "status": "uncertain",
            "reason": (
                "Model confidence is too low "
                "for a reliable diagnosis."
            ),
            "quality": quality_result,
            "vision": vision_result,
        }

    # --------------------------------------------------
    # STEP 4: SEVERITY ESTIMATION
    # --------------------------------------------------

    affected_area = calculate_affected_area(
        image_path
    )

    severity_result = estimate_severity(
        image_path,
        affected_area,
    )

    # --------------------------------------------------
    # STEP 5: FINAL DIAGNOSIS
    # --------------------------------------------------

    return {
        "status": "success",
        "crop": vision_result["crop"],
        "disease": vision_result["disease"],
        "confidence": vision_result["confidence"],
        "confidence_level": (
            vision_result["confidence_level"]
        ),
        "top_predictions": (
            vision_result["top_predictions"]
        ),
        "affected_area_percent": (
            severity_result["affected_area_percent"]
        ),
        "severity": severity_result["severity"],
    }


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Complete Diagnosis Pipeline")
    print("=" * 60)

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "maize_crop.png"
    )

    print(f"\nImage: {image_path}")

    print("\nRunning complete diagnosis...\n")

    result = analyze_image(
        str(image_path)
    )

    print("FINAL DIAGNOSIS")
    print("-" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nDiagnosis pipeline completed successfully.")