import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Llama 3.2 1B model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )

    print("Model loaded successfully!")

    prompt = """You are an agricultural advisory assistant for KrishiRakshak.

Use only the information provided below.

Crop: maize
Disease: northern_leaf_blight
Confidence: 66%
Severity: moderate
Region: West Bengal
Temperature: 28 C
Humidity: 78%

Knowledge:
- Monitor disease progression across the crop.
- Manage infected crop residue according to local agricultural recommendations.
- Scout fields regularly for early symptoms.
- Use integrated disease-management practices.
- Consult current agricultural extension recommendations before applying fungicides.

Give a short, practical advisory for a farmer.
Do not invent pesticide names or dosages.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful agricultural advisory assistant. "
                "Give grounded, practical advice and do not invent "
                "chemical names or dosages."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    print("Generating advisory...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n--- Llama 3.2 1B Advisory ---")
    print(response)


if __name__ == "__main__":
    main()