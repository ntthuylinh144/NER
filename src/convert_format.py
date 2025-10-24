import json
import re

INPUT_FILE = "annotated_instructions.json"
OUTPUT_FILE = "spacy_ready.json"

def convert_to_spacy_format(data):
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

# Đọc file JSON gốc
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Nếu JSON là {"data": [...]} thì lấy data["data"]
if isinstance(data, dict) and "data" in data:
    data = data["data"]

# Chuyển đổi
converted = convert_to_spacy_format(data)

# Ghi ra file mới
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"Converted {len(converted)} samples → spaCy format with sequential IDs (1–{len(converted)})")
