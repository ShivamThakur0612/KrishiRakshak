"""
KrishiRakshak Advisory Engine.

Combines:
    1. Diagnosis
    2. Agricultural context
    3. Local agricultural knowledge

Produces a grounded advisory package.

Llama will later use this structured package to generate
farmer-friendly multilingual advice.
"""

from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from advisory.knowledge_loader import (
    get_disease_information,
)


def build_advisory(
    diagnosis: dict,
    context: dict,
) -> dict:
    """
    Build a grounded advisory package.
    """

    disease = diagnosis.get(
        "disease",
        "unknown",
    )

    knowledge_result = get_disease_information(
        disease
    )

    if not knowledge_result["found"]:
        return {
            "status": "knowledge_unavailable",
            "message": (
                "No verified agricultural knowledge "
                "was found for this diagnosis."
            ),
            "diagnosis": diagnosis,
            "context": context,
        }

    knowledge = knowledge_result["information"]

    advisory = {
        "status": "ready",
        "diagnosis": {
            "crop": diagnosis.get(
                "crop",
                "unknown",
            ),
            "disease": disease,
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
        },
        "context": {
            "growth_stage": context.get(
                "growth_stage",
                "unknown",
            ),
            "region": context.get(
                "region",
                "unknown",
            ),
            "weather": context.get(
                "weather",
                {},
            ),
        },
        "agricultural_knowledge": {
            "symptoms": knowledge.get(
                "symptoms",
                [],
            ),
            "favorable_conditions": knowledge.get(
                "favorable_conditions",
                [],
            ),
            "prevention": knowledge.get(
                "prevention",
                [],
            ),
            "management": knowledge.get(
                "management",
                [],
            ),
            "source": knowledge.get(
                "source",
                "Unknown",
            ),
        },
    }

    return advisory


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Advisory Engine")
    print("=" * 60)

    diagnosis = {
        "status": "success",
        "crop": "maize",
        "disease": "northern_leaf_blight",
        "confidence": 0.6595,
        "confidence_level": "probable",
        "severity": "moderate",
        "affected_area_percent": 20.55,
    }

    context = {
        "growth_stage": "vegetative",
        "region": "West Bengal",
        "weather": {
            "temperature_c": 28.0,
            "humidity_percent": 78.0,
        },
    }

    advisory = build_advisory(
        diagnosis=diagnosis,
        context=context,
    )

    print("\nGrounded Advisory Package:")
    print("-" * 60)

    print(
        json.dumps(
            advisory,
            indent=4,
            ensure_ascii=False,
        )
    )

    print(
        "\nAdvisory engine completed successfully."
    )