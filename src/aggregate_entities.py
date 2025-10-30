"""
Aggregate entities from train + dev datasets
and create a unified global_entities.json file.
"""

import json
from collections import defaultdict
from manage_entities import EntityManager



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_entities(train_file: str, dev_file: str, output_file: str):
    # Load both datasets
    train_data = load_json(train_file)
    dev_data = load_json(dev_file)
    all_data = train_data + dev_data

    print(f"Loaded {len(all_data)} samples from train+dev")

    # Initialize EntityManager
    manager = EntityManager(similarity_threshold=0.85)

    total_extractions = 0
    new_entities = 0
    linked_entities = 0

    # Iterate over all texts
    for i, item in enumerate(all_data, start=1):
        text_id = f"sent_{i:04d}"
        text = item.get("text", "")
        entities = item.get("entities", [])

        for ent in entities:
            if len(ent) != 3:
                continue

            start, end, label = ent
            ent_text = text[start:end]
            total_extractions += 1

            entity, is_new = manager.add_entity(ent_text, label, source_text_id=text_id)
            if is_new:
                new_entities += 1
            else:
                linked_entities += 1

    # Compute link rate
    link_rate = f"{(linked_entities / total_extractions * 100):.1f}%" if total_extractions else "0%"

    # Build metadata
    metadata = {
        "total_entities": len(manager.entities),
        "saved_at": __import__("datetime").datetime.now().isoformat(),
    }

    statistics = {
        "total_entities": len(manager.entities),
        "total_extractions": total_extractions,
        "new_entities": new_entities,
        "linked_entities": linked_entities,
        "link_rate": link_rate,
        "entities_by_type": {
            label: len(manager.get_entities_by_label(label))
            for label in manager.label_index.keys()
        }
    }

    output_data = {
        "metadata": metadata,
        "statistics": statistics,
        "entities": [e.to_dict() for e in manager.entities]
    }

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n Global entity list saved to {output_file}")
    print(f"   Total unique entities: {len(manager.entities)}")
    print(f"   Total extractions: {total_extractions}")
    print(f"   Link rate: {link_rate}")


if __name__ == "__main__":
    aggregate_entities(
        train_file="data\\data_train.json",
        dev_file="data\\data_dev.json",
        output_file="data\\classical_global_entities.json"
    )
    aggregate_entities(
        train_file="data\\data_train.json",
        dev_file="data\\data_dev.json",
        output_file="data\\llm_global_entities.json"
    )
