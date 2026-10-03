"""
KrishiRakshak Context Engine.

Combines disease diagnosis with agricultural context such as:
- crop
- growth stage
- region
- temperature
- humidity

This version uses manually supplied context.
Weather/API integration will be added later.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_context(
    diagnosis: dict,
    growth_stage: str,
    region: str,
    temperature_c: float,
    humidity_percent: float,
) -> dict:
    """
    Combine diagnosis with agricultural context.
    """

    context = {
        "crop": diagnosis.get(
            "crop",
            "unknown",
        ),

        "disease": diagnosis.get(
            "disease",
            "unknown",
        ),

        "confidence": diagnosis.get(
            "confidence",
            0.0,
        ),

        "confidence_level": diagnosis.get(
            "confidence_level",
            "unknown",
        ),

        "severity": diagnosis.get(
            "severity",
            "unknown",
        ),

        "affected_area_percent": diagnosis.get(
            "affected_area_percent",
            0.0,
        ),

        "growth_stage": growth_stage,

        "region": region,

        "weather": {
            "temperature_c": temperature_c,
            "humidity_percent": humidity_percent,
        },
    }

    return context


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Context Engine")
    print("=" * 60)

    # Example diagnosis from our working pipeline.
    diagnosis = {
        "status": "success",
        "crop": "maize",
        "disease": "northern_leaf_blight",
        "confidence": 0.6595,
        "confidence_level": "probable",
        "affected_area_percent": 20.55,
        "severity": "moderate",
    }

    # Example agricultural context.
    growth_stage = "vegetative"
    region = "West Bengal"
    temperature_c = 28.0
    humidity_percent = 78.0

    context = build_context(
        diagnosis=diagnosis,
        growth_stage=growth_stage,
        region=region,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
    )

    print("\nContext Result:")
    print("-" * 60)

    for key, value in context.items():
        print(f"{key}: {value}")

    print("\nContext engine completed successfully.")