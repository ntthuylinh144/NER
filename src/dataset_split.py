import json, random

INPUT_FILE = "spacy_ready.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)
n = len(data)
train, dev, test = data[:int(0.7*n)], data[int(0.7*n):int(0.9*n)], data[int(0.9*n):]

for name, subset in zip(["train","dev","test"], [train, dev, test]):
    with open(f"data_{name}.json", "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
