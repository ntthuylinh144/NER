import json
from collections import Counter

# ---- CONFIG ----
DATA_FILE = "data_train.json"  # hoặc data_dev.json, data_test.json, hoặc spacy_ready.json
# ----------------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

label_counts = Counter()

for text, ann in data:
    for start, end, label in ann["entities"]:
        label_counts[label] += 1

print("Entity distribution:")
total = sum(label_counts.values())
for label, count in label_counts.most_common():
    print(f"  {label:12s} : {count:3d} ({count/total:.1%})")
print(f"Total entities: {total}")
