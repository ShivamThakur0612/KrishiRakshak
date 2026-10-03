"""
KrishiRakshak End-to-End Pipeline.

Flow:
    Image
      ↓
    Image Quality + Vision + Severity
      ↓
    Agricultural Context
      ↓
    Knowledge Base
      ↓
    Llama 3.2 1B
      ↓
    Grounding Validator
      ↓
    LLM Advisory / Grounded Fallback
"""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from context.diagnosis import analyze_image
from context.context_engine import build_context
from advisory.advisory_engine import build_advisory
from advisory.llm_advisor import generate_advisory


def run_pipeline(
    image_path: str,
    growth_stage: str,
    region: str,
    temperature_c: float,
    humidity_percent: float,
    language: str = "english",
) -> dict:
    """
    Run the complete KrishiRakshak pipeline.
    """

    # --------------------------------------------------
    # STEP 1: DIAGNOSIS
    # --------------------------------------------------

    diagnosis = analyze_image(image_path)

    if diagnosis["status"] != "success":
        return {
            "status": diagnosis["status"],
            "stage": "diagnosis",
            "result": diagnosis,
        }

    # --------------------------------------------------
    # STEP 2: AGRICULTURAL CONTEXT
    # --------------------------------------------------

    context = build_context(
        diagnosis=diagnosis,
        growth_stage=growth_stage,
        region=region,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
    )

    # --------------------------------------------------
    # STEP 3: GROUNDED ADVISORY PACKAGE
    # --------------------------------------------------

    advisory_package = build_advisory(
        diagnosis=diagnosis,
        context=context,
    )

    if advisory_package["status"] != "ready":
        return {
            "status": advisory_package["status"],
            "stage": "advisory_package",
            "result": advisory_package,
        }

    # --------------------------------------------------
    # STEP 4: LLM ADVISORY
    # --------------------------------------------------

    advisory_result = generate_advisory(
        advisory_package=advisory_package,
        language=language,
    )

    # --------------------------------------------------
    # STEP 5: FINAL RESULT
    # --------------------------------------------------

    return {
        "status": advisory_result["status"],
        "diagnosis": diagnosis,
        "context": context,
        "advisory_package": advisory_package,
        "advisory": advisory_result,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("KRISHIRAKSHAK END-TO-END AI PIPELINE")
    print("=" * 70)

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "maize_crop.png"
    )

    print(f"\nImage: {image_path}")

    result = run_pipeline(
        image_path=str(image_path),
        growth_stage="vegetative",
        region="West Bengal",
        temperature_c=28.0,
        humidity_percent=78.0,
        language="english",
    )

    print("\nFINAL PIPELINE RESULT")
    print("-" * 70)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
        )
    )

    print("\nKrishiRakshak pipeline completed.")