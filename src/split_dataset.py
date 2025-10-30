"""
Utility script to split a spaCy-ready JSON dataset into
training, development (validation), and test sets.

Input:  spacy_ready.json
Output: data_train.json, data_dev.json, data_test.json
"""

import json
import random
import os

# Configuration
INPUT_FILE = "data\\annotated_data.json"
TRAIN_RATIO = 0.7
DEV_RATIO = 0.2  # Remaining 0.1 goes to test automatically


def split_dataset(input_path):
    """Split dataset into train/dev/test and save as JSON files."""
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print(" Invalid or empty dataset. Expected a non-empty JSON list.")
        return

    random.shuffle(data)
    n = len(data)
    train_end = int(TRAIN_RATIO * n)
    dev_end = int((TRAIN_RATIO + DEV_RATIO) * n)

    train = data[:train_end]
    dev = data[train_end:dev_end]
    test = data[dev_end:]

    print(f"Dataset size: {n}")
    print(f"Train:{len(train)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"Dev:{len(dev)} samples ({DEV_RATIO*100:.0f}%)")
    print(f"Test:{len(test)} samples ({(1 - TRAIN_RATIO - DEV_RATIO)*100:.0f}%)\n")

    for name, subset in zip(["train", "dev", "test"], [train, dev, test]):
        output_path = f"data_{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(subset)} {name} samples → {output_path}")


if __name__ == "__main__":

    split_dataset(INPUT_FILE)

