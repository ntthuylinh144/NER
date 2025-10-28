"""
Annotate technical instruction sentences using a Gemini (LLM) model
to automatically label entities for NER (Named Entity Recognition).

Entity labels:
- COMPONENT: physical part or device (e.g., screw, motor, cable)
- TOOL: instrument or tool used for action (e.g., screwdriver, wrench)
- ACTION: technical action or verb (e.g., attach, tighten, connect)
- PARAMETER: numeric or physical parameter (e.g., 5V, 10mm, 30°C)
- LOCATION: position or direction in assembly (e.g., left side, base)

Input:  annotated_instructions1.json
Output: annotated_instructions2.json
"""

import google.generativeai as genai
import json
import time
import os

# Configuration
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Get here: https://aistudio.google.com/apikey
INPUT_FILE = "annotated_instructions1.json"
OUTPUT_FILE = "annotated_instructions2.json"
DELAY = 1.0 #second between API calls


# ---------- HELPER: TEST GEMINI API ----------
def test_gemini():
    """Check if Gemini API key and model are working."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say hello")
        print("Gemini API is active!")
        print(f"Response: {response.text}\n")
        return True
    except Exception as e:
        print(f"Gemini test failed: {e}")
        return False


# ---------- ENTITY ANNOTATION ----------
def annotate_sentence(model, text):
    """Annotate a sentence with entities using the Gemini model."""
    prompt = f"""
You are a technical text annotator for assembly instructions.
Label entities with:
- COMPONENT: physical part or device (e.g., screw, motor, base plate, cable)
- TOOL: instrument or tool used for action (e.g., screwdriver, wrench, multimeter)
- ACTION: technical action or verb (e.g., attach, tighten, connect, calibrate)
- PARAMETER: numeric or physical parameter (e.g., 5V, 10mm, 30°C, torque = 5 Nm)
- LOCATION: position or direction in assembly (e.g., left side, base, top, rear panel)

Return ONLY a valid JSON array of entities, for example:
[
  {{"text": "Tighten", "label": "ACTION"}},
  {{"text": "the Ethernet cable", "label": "COMPONENT"}}
]

Sentence: "{text}"
"""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()

        # Clean up possible Markdown or code formatting
        if "```json" in reply:
            reply = reply.split("```json")[1].split("```")[0].strip()
        elif "```" in reply:
            reply = reply.split("```")[1].split("```")[0].strip()

        # Parse JSON safely
        entities = json.loads(reply)

        if not isinstance(entities, list):
            print("Warning: Model output is not a list. Skipped.")
            return []

        return entities

    except json.JSONDecodeError:
        print("JSON parse error in model response.")
        return []
    except Exception as e:
        print(f"Annotation error: {e}")
        return []


# ---------- MAIN PIPELINE ----------
def main():
    # Check for API key
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print(" Please set your Gemini API key first:")
        print("Get one here: https://aistudio.google.com/apikey")
        return

    # Test API connection
    print("Testing Gemini API...\n")
    if not test_gemini():
        return

    # Initialize Gemini model
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Load input file
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sentences for annotation\n")

    annotated = []
    success_count = 0

    for i, item in enumerate(data, start=1):
        text = item.get("text", "").strip()
        if not text:
            continue

        print(f"[{i}/{len(data)}] Annotating: {text[:70]}...")

        entities = annotate_sentence(model, text)

        if entities:
            success_count += 1
            print(f"{len(entities)} entities found")

        annotated.append({
            "id": item.get("id", i),
            "text": text,
            "entities": entities
        })

        time.sleep(DELAY)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"\n Annotation completed: {success_count}/{len(data)} sentences processed.")
    print(f" Results saved to: {OUTPUT_FILE}")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()
