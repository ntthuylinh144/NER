"""
Convert annotated NER data (from LLM annotation or manual tagging)
into a spaCy-compatible training format.

Input:  annotated_instructions.json
Output: spacy_ready.json

Each entry in the output has:
{
  "id": <int>,
  "text": "<sentence>",
  "entities": [(start, end, label), ...]
}
"""

import json
import re

# CONFIGURATION
INPUT_FILE = "data\\annotated_data.json"
OUTPUT_FILE = "data\\spacy_ready.json"

def convert_to_spacy_format(data):
    """
    Convert list of annotated samples into spaCy NER training format.
    - Finds each entity text in the sentence
    - Computes (start, end, label) offsets
    - Handles case-insensitive matches safely
    """
    formatted = []
    for idx, sample in enumerate(data, start=1):
        text = sample["text"]
        print(text)
        entities = []
        for ent in sample.get("entities", []):
            pattern = re.escape(ent["text"].strip())
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                start, end = match.span()
                entities.append((start, end, ent["label"]))
        formatted.append({
            "id": idx,
            "text": text,
            "entities": entities
        })
    return formatted


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "data" in data:
    data = data["data"]

converted = convert_to_spacy_format(data)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"Converted {len(converted)} samples → spaCy format with sequential IDs (1–{len(converted)})")
