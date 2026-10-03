"""
KrishiRakshak LLM Advisor.

Uses local Llama 3.2 1B Instruct Q4_K_M through
the local llama.cpp HTTP server.

Architecture:
    Grounded Advisory Package
            ↓
    Strict Llama Prompt
            ↓
    llama-server HTTP API
            ↓
    Farmer-facing Advisory
"""

import json
import requests


LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

def build_grounded_fallback(
    grounded_data: dict,
    language: str = "english",
) -> str:
    """
    Build a deterministic advisory using only the
    supplied grounded agricultural knowledge.

    This is the safety fallback when the LLM response
    fails the grounding validation.
    """

    crop = grounded_data["crop"]
    disease = grounded_data["disease"]
    confidence = grounded_data["confidence"]
    severity = grounded_data["severity"]

    knowledge = grounded_data["knowledge"]

    symptoms = knowledge.get("symptoms", [])
    prevention = knowledge.get("prevention", [])
    management = knowledge.get("management", [])

    lines = []

    lines.append(f"Crop: {crop}")
    lines.append(f"Probable disease: {disease}")
    lines.append(f"Confidence: {confidence}%")
    lines.append(f"Severity: {severity}")

    if symptoms:
        lines.append("")
        lines.append("Symptoms:")
        for item in symptoms:
            lines.append(f"- {item}")

    if prevention:
        lines.append("")
        lines.append("Prevention:")
        for item in prevention:
            lines.append(f"- {item}")

    if management:
        lines.append("")
        lines.append("Management:")
        for item in management:
            lines.append(f"- {item}")

    lines.append("")
    lines.append(
        "Disease identification is probabilistic. "
        "Consult the local agricultural extension service "
        "before applying chemical treatments."
    )

    return "\n".join(lines)

def validate_grounding(
        
    generated_text: str,
    grounded_data: dict,
) -> dict:
    """
    Basic deterministic grounding guard.

    Rejects generated advisory text when it contains
    agricultural terms that were not present in the
    supplied grounded knowledge.
    """

    text = generated_text.lower()

    knowledge = grounded_data["knowledge"]

    allowed_text_parts = [
        grounded_data["crop"],
        grounded_data["disease"],
        grounded_data["severity"],
        str(grounded_data["confidence"]),
        str(grounded_data["affected_area_percent"]),
        grounded_data["growth_stage"],
        grounded_data["region"],
        json.dumps(
            knowledge,
            ensure_ascii=False,
        ),
    ]

    allowed_text = " ".join(
        allowed_text_parts
    ).lower()

    suspicious_terms = [
        "yellowing",
        "bronzing",
        "insect",
        "insecticide",
        "herbicide",
        "trichogramma",
        "neem",
        "spray",
        "dosage",
        "concentration",
        "spacing",
        "planting depth",
        "temperature threshold",
        "fungicide",
    ]

    detected = []

    for term in suspicious_terms:
        if term in text and term not in allowed_text:
            detected.append(term)

    if detected:
        return {
            "valid": False,
            "reason": (
                "Generated advisory contains information "
                "not supported by the grounded knowledge."
            ),
            "unsupported_terms": detected,
        }

    return {
        "valid": True,
        "reason": "Generated advisory passed the grounding guard.",
        "unsupported_terms": [],
    }

def generate_advisory(
    advisory_package: dict,
    language: str = "english",
) -> dict:

    if advisory_package.get("status") != "ready":
        return {
            "status": "unavailable",
            "message": "A reliable advisory could not be generated.",
        }

    diagnosis = advisory_package["diagnosis"]
    context = advisory_package["context"]
    knowledge = advisory_package["agricultural_knowledge"]

    language_map = {
        "english": "English",
        "hindi": "Hindi",
        "bengali": "Bengali",
    }

    output_language = language_map.get(
        language.lower(),
        "English",
    )

    grounded_data = {
        "crop": diagnosis["crop"],
        "disease": diagnosis["disease"],
        "confidence": round(
            diagnosis["confidence"] * 100,
            1,
        ),
        "severity": diagnosis["severity"],
        "affected_area_percent": diagnosis.get(
            "affected_area_percent"
        ),
        "growth_stage": context["growth_stage"],
        "region": context["region"],
        "weather": context["weather"],
        "knowledge": knowledge,
    }

    prompt = f"""
You are KrishiRakshak, an agricultural advisory assistant.

Your job is ONLY to explain the agricultural information
provided below in simple language for a farmer.

STRICT GROUNDING RULES:

1. Use ONLY the information provided in GROUNDED DATA.
2. Do NOT invent agricultural facts.
3. Do NOT invent pesticide names.
4. Do NOT invent pesticide dosages or concentrations.
5. Do NOT invent planting distances, timings, chemicals,
   biological agents, or treatment schedules.
6. If a treatment detail is not provided, say that the farmer
   should consult the local agricultural extension service.
7. Do not introduce unrelated pest-control methods.
8. Keep the advice practical and concise.
9. Clearly state that the disease identification is probabilistic.
10. Answer only in {output_language}.
11. Do not add information from your general knowledge.

GROUNDED DATA:
{json.dumps(grounded_data, indent=2, ensure_ascii=False)}

Generate a short farmer-facing advisory containing:

- Crop and probable disease
- Confidence
- Severity
- Important symptoms or situation
- Practical prevention steps from the provided knowledge
- Management steps from the provided knowledge
- A caution to consult local agricultural experts before
  applying chemical treatments

IMPORTANT:
Do not add information that is not present in GROUNDED DATA.
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are KrishiRakshak. "
                    "Follow the user's grounding rules exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": 180,
    }

    try:

        response = requests.post(
            LLAMA_SERVER_URL,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        message = (
            result["choices"][0]["message"]["content"]
            .strip()
        )
        grounding_check = validate_grounding(
            message,
            grounded_data,
        )

        if not grounding_check["valid"]:

            fallback_message = build_grounded_fallback(
                grounded_data,
                language,
            )

            return {
                "status": "fallback",
                "message": fallback_message,
                "grounding_check": grounding_check,
                "fallback_used": True,
                "grounding": {
                    "disease": diagnosis["disease"],
                    "severity": diagnosis["severity"],
                    "region": context["region"],
                    "source": knowledge["source"],
                },
            }
        

        return {
            "status": "success",
            "language": language,
            "message": message,
            "grounding": {
                "disease": diagnosis["disease"],
                "severity": diagnosis["severity"],
                "region": context["region"],
                "source": knowledge["source"],
            },
        }

    except requests.exceptions.ConnectionError:

        return {
            "status": "error",
            "message": (
                "Could not connect to llama-server. "
                "Make sure llama-server is running on "
                "http://127.0.0.1:8080."
            ),
        }

    except requests.exceptions.Timeout:

        return {
            "status": "error",
            "message": "Llama server request timed out.",
        }

    except requests.exceptions.RequestException as exc:

        return {
            "status": "error",
            "message": f"Llama server request failed: {exc}",
        }

    except (KeyError, IndexError, TypeError, ValueError) as exc:

        return {
            "status": "error",
            "message": (
                f"Invalid response received from llama-server: {exc}"
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": f"Llama inference error: {exc}",
        }


if __name__ == "__main__":

    print("=" * 60)
    print("KrishiRakshak Llama 3.2 1B Advisor")
    print("=" * 60)

    advisory_package = {
        "status": "ready",

        "diagnosis": {
            "crop": "maize",
            "disease": "northern_leaf_blight",
            "confidence": 0.6595,
            "confidence_level": "probable",
            "severity": "moderate",
            "affected_area_percent": 20.55,
        },

        "context": {
            "growth_stage": "vegetative",
            "region": "West Bengal",
            "weather": {
                "temperature_c": 28.0,
                "humidity_percent": 78.0,
            },
        },

        "agricultural_knowledge": {
            "symptoms": [
                "Elongated gray-green to tan lesions can develop on maize leaves."
            ],
            "favorable_conditions": [
                "Disease development is favored by humid conditions."
            ],
            "prevention": [
                "Use suitable resistant or tolerant varieties where available.",
                "Manage infected crop residue according to local agricultural recommendations.",
                "Scout fields regularly for early symptoms.",
            ],
            "management": [
                "Monitor disease progression across the crop.",
                "Use integrated disease-management practices recommended for the local region.",
                "Consult current agricultural extension recommendations before applying fungicides.",
            ],
            "source": (
                "Agricultural extension and crop-disease "
                "reference material."
            ),
        },
    }

    result = generate_advisory(
        advisory_package,
        language="english",
    )

    print("\nStatus:")
    print("-" * 60)
    print(result["status"])

    if result["status"] == "success":

        print("\nGenerated Advisory:")
        print("-" * 60)
        print(result["message"])

        if "grounding" in result:

            print("\nGrounding Information:")
            print("-" * 60)

            print(
                json.dumps(
                    result["grounding"],
                    indent=4,
                    ensure_ascii=False,
                )
            )

    else:

        print("\nError:")
        print("-" * 60)
        print(result["message"])

    print("\nLlama Advisor test completed.")