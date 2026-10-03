"""
KrishiRakshak Agricultural Knowledge Base Loader.

Loads structured disease information from the local
JSON knowledge base.

This module is designed to work offline.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_PATH = (
    PROJECT_ROOT
    / "knowledge"
    / "maize_diseases.json"
)


def load_knowledge_base() -> dict:
    """Load the local agricultural knowledge base."""

    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_PATH}"
        )

    with open(
        KNOWLEDGE_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def get_disease_information(
    disease: str,
) -> dict:
    """
    Retrieve information for a specific disease.
    """

    knowledge_base = load_knowledge_base()

    diseases = knowledge_base.get(
        "diseases",
        {},
    )

    if disease not in diseases:
        return {
            "found": False,
            "disease": disease,
            "message": (
                "No information available "
                "for this disease."
            ),
        }

    return {
        "found": True,
        "disease": disease,
        "information": diseases[disease],
    }


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Knowledge Base Loader")
    print("=" * 60)

    disease = "northern_leaf_blight"

    print(
        f"\nSearching knowledge base for: {disease}"
    )

    result = get_disease_information(
        disease
    )

    print("\nResult:")
    print("-" * 60)

    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False,
    ))

    print(
        "\nKnowledge base test completed successfully."
    )