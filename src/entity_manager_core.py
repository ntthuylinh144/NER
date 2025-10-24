"""
Entity Manager - Core Challenge Implementation
Maintains a growing global list and links similar entities

Save this as: src/entity_manager_core.py
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from difflib import SequenceMatcher
import os


@dataclass
class EntityOccurrence:
    """Single occurrence of an entity in text"""
    text_id: str
    matched_text: str  # Actual text extracted (may vary)
    position: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Entity:
    """Canonical entity with all its variations"""
    entity_id: str
    canonical_name: str  # Main/normalized name
    entity_type: str  # COMPONENT, PART, ACTION, etc.
    variations: List[str] = field(default_factory=list)
    occurrences: List[EntityOccurrence] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def first_seen(self) -> str:
        return self.occurrences[0].text_id if self.occurrences else "N/A"

    def add_occurrence(self, text_id: str, matched_text: str):
        """Add new occurrence and track variation"""
        self.occurrences.append(EntityOccurrence(text_id, matched_text))

        # Track variation if new
        normalized = matched_text.lower().strip()
        if normalized not in [v.lower().strip() for v in self.variations]:
            self.variations.append(matched_text)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "variations": self.variations,
            "count": self.count,
            "first_seen": self.first_seen,
            "occurrences": [
                {"text_id": occ.text_id, "matched_text": occ.matched_text}
                for occ in self.occurrences
            ]
        }


class EntityManager:
    """
    Core Entity Manager - Maintains global entity list
    Handles matching and linking of similar entities
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.entities: Dict[str, Entity] = {}  # entity_id -> Entity
        self.next_id = 1
        self.similarity_threshold = similarity_threshold

        # Statistics
        self.total_extractions = 0
        self.new_entities = 0
        self.linked_entities = 0

    def _generate_entity_id(self) -> str:
        """Generate unique entity ID"""
        entity_id = f"ENT_{self.next_id:04d}"
        self.next_id += 1
        return entity_id

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Lowercase, remove extra spaces, remove punctuation
        normalized = text.lower().strip()
        normalized = normalized.replace('-', ' ').replace('_', ' ')
        normalized = ' '.join(normalized.split())
        return normalized

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Exact match
        if norm1 == norm2:
            return 1.0

        # Sequence matching
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _find_matching_entity(self, entity_text: str, entity_type: str) -> Optional[str]:
        """
        Find if entity matches any existing entity
        Returns: entity_id if match found, None otherwise
        """
        best_match_id = None
        best_similarity = 0.0

        for ent_id, entity in self.entities.items():
            # Only compare with same type
            if entity.entity_type != entity_type:
                continue

            # Check against canonical name
            sim = self._calculate_similarity(entity_text, entity.canonical_name)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = ent_id

            # Check against all variations
            for variation in entity.variations:
                sim = self._calculate_similarity(entity_text, variation)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = ent_id

        # Return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match_id

        return None

    def process_entity(self, entity_text: str, entity_type: str,
                       text_id: str) -> Tuple[str, bool]:
        """
        Process a newly extracted entity

        Returns:
            (entity_id, is_new) - entity_id and whether it's a new entity
        """
        self.total_extractions += 1

        # Try to find matching existing entity
        matched_id = self._find_matching_entity(entity_text, entity_type)

        if matched_id:
            # LINK to existing entity
            self.entities[matched_id].add_occurrence(text_id, entity_text)
            self.linked_entities += 1
            return matched_id, False
        else:
            # CREATE new entity
            new_id = self._generate_entity_id()
            new_entity = Entity(
                entity_id=new_id,
                canonical_name=entity_text,
                entity_type=entity_type,
                variations=[entity_text],
                occurrences=[EntityOccurrence(text_id, entity_text)]
            )
            self.entities[new_id] = new_entity
            self.new_entities += 1
            return new_id, True

    def batch_process(self, extracted_entities: List[Dict], text_id: str) -> Dict:
        """
        Process multiple entities from one text

        Args:
            extracted_entities: [{"text": "control box", "type": "COMPONENT"}, ...]
            text_id: ID of the source text

        Returns:
            Processing results with statistics
        """
        results = {
            "text_id": text_id,
            "processed": [],
            "new_count": 0,
            "linked_count": 0
        }

        for entity_data in extracted_entities:
            entity_id, is_new = self.process_entity(
                entity_data["text"],
                entity_data["type"],
                text_id
            )

            results["processed"].append({
                "text": entity_data["text"],
                "type": entity_data["type"],
                "entity_id": entity_id,
                "is_new": is_new,
                "canonical": self.entities[entity_id].canonical_name
            })

            if is_new:
                results["new_count"] += 1
            else:
                results["linked_count"] += 1

        return results

    def get_entity_list(self, entity_type: Optional[str] = None) -> List[Dict]:
        """
        Get current entity list (for LLM context)

        Args:
            entity_type: Filter by type (optional)
        """
        entity_list = []

        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            entity_list.append({
                "id": entity.entity_id,
                "name": entity.canonical_name,
                "type": entity.entity_type,
                "variations": entity.variations,
                "count": entity.count
            })

        return entity_list

    def get_compact_context(self, max_entities: int = 50) -> str:
        """
        Get compact entity list for LLM context (solves context window issue)
        Returns top entities by frequency
        """
        # Sort by count (most frequent first)
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.count,
            reverse=True
        )[:max_entities]

        context_lines = []
        for entity in sorted_entities:
            # Format: canonical_name (type) [count] {variations}
            variations = ", ".join(entity.variations[:3])  # Max 3 variations
            context_lines.append(
                f"{entity.canonical_name} ({entity.entity_type}) "
                f"[seen {entity.count}x] {{{variations}}}"
            )

        return "\n".join(context_lines)

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            "total_entities": len(self.entities),
            "total_extractions": self.total_extractions,
            "new_entities": self.new_entities,
            "linked_entities": self.linked_entities,
            "link_rate": f"{(self.linked_entities / self.total_extractions * 100):.1f}%" if self.total_extractions > 0 else "0%",
            "entities_by_type": self._get_type_distribution()
        }

    def _get_type_distribution(self) -> Dict[str, int]:
        """Get entity count by type"""
        distribution = {}
        for entity in self.entities.values():
            distribution[entity.entity_type] = distribution.get(entity.entity_type, 0) + 1
        return distribution

    def save_to_file(self, filepath: str):
        """Save entity database to JSON file"""
        data = {
            "metadata": {
                "total_entities": len(self.entities),
                "saved_at": datetime.now().isoformat()
            },
            "statistics": self.get_statistics(),
            "entities": [entity.to_dict() for entity in self.entities.values()]
        }

        # Create directory if not exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Entity database saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load entity database from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.entities.clear()

        for ent_data in data["entities"]:
            entity = Entity(
                entity_id=ent_data["entity_id"],
                canonical_name=ent_data["canonical_name"],
                entity_type=ent_data["entity_type"],
                variations=ent_data["variations"],
                occurrences=[
                    EntityOccurrence(occ["text_id"], occ["matched_text"])
                    for occ in ent_data["occurrences"]
                ]
            )
            self.entities[entity.entity_id] = entity

        print(f"✅ Loaded {len(self.entities)} entities from {filepath}")


# ==================== DEMO USAGE ====================

def demo():
    """Demonstrate entity management workflow"""

    print("🚀 Entity Manager Demo - Understanding 'Link to Existing Entry'\n")
    print("=" * 70)

    manager = EntityManager(similarity_threshold=0.85)

    # Simulate processing multiple texts
    print("\n📚 SCENARIO: Processing 4 technical sentences")
    print("=" * 70)

    texts = [
        {
            "id": "sent_001",
            "text": "Connect the control box to the robot arm.",
            "entities": [
                {"text": "control box", "type": "COMPONENT"},
                {"text": "robot arm", "type": "COMPONENT"},
                {"text": "connect", "type": "ACTION"}
            ]
        },
        {
            "id": "sent_002",
            "text": "The Control Box must be grounded properly.",
            "entities": [
                {"text": "Control Box", "type": "COMPONENT"},  # Same entity, different case
                {"text": "grounded", "type": "ACTION"}
            ]
        },
        {
            "id": "sent_003",
            "text": "Mount the controller box on the wall.",
            "entities": [
                {"text": "controller box", "type": "COMPONENT"},  # Similar to "control box"
                {"text": "mount", "type": "ACTION"}
            ]
        },
        {
            "id": "sent_004",
            "text": "Install the teach pendant near operator.",
            "entities": [
                {"text": "teach pendant", "type": "COMPONENT"},  # New entity
                {"text": "install", "type": "ACTION"}
            ]
        }
    ]

    # Process each text
    for i, text_data in enumerate(texts, 1):
        print(f"\n{'=' * 70}")
        print(f"TEXT {i}: {text_data['text']}")
        print(f"{'=' * 70}")
        print(f"Extracted entities: {[e['text'] for e in text_data['entities']]}")

        result = manager.batch_process(text_data["entities"], text_data["id"])

        print(f"\n📊 Processing Results:")
        for item in result["processed"]:
            if item["is_new"]:
                print(f"   🆕 NEW: '{item['text']}' → Created {item['entity_id']}")
            else:
                print(f"   🔗 LINKED: '{item['text']}' → Linked to {item['entity_id']} ('{item['canonical']}')")
                print(f"      ↳ Reason: Similar to existing entity")

        print(f"\n   Summary: {result['new_count']} new, {result['linked_count']} linked")
        print(f"   Global entity list: {len(manager.entities)} unique entities")

    # Show what "linking" means
    print("\n" + "=" * 70)
    print("🔍 UNDERSTANDING 'LINK TO EXISTING ENTRY'")
    print("=" * 70)

    # Find the "control box" entity
    control_box_entity = None
    for entity in manager.entities.values():
        if entity.canonical_name == "control box":
            control_box_entity = entity
            break

    if control_box_entity:
        print(f"\nEntity ID: {control_box_entity.entity_id}")
        print(f"Canonical Name: {control_box_entity.canonical_name}")
        print(f"Type: {control_box_entity.entity_type}")
        print(f"Total Occurrences: {control_box_entity.count}")
        print(f"\nVariations found:")
        for var in control_box_entity.variations:
            print(f"   - '{var}'")
        print(f"\nAppeared in these texts:")
        for occ in control_box_entity.occurrences:
            print(f"   - {occ.text_id}: '{occ.matched_text}'")

        print(f"\n💡 This is what 'LINK' means:")
        print(f"   Instead of creating 3 separate entities for:")
        print(f"     - 'control box'")
        print(f"     - 'Control Box'")
        print(f"     - 'controller box'")
        print(f"   We LINKED them all to ONE entity: {control_box_entity.entity_id}")
        print(f"   This keeps the global list clean and avoids duplicates!")

    # Show final statistics
    print("\n" + "=" * 70)
    print("📈 FINAL STATISTICS")
    print("=" * 70)
    stats = manager.get_statistics()
    print(f"\nTotal unique entities: {stats['total_entities']}")
    print(f"Total extractions: {stats['total_extractions']}")
    print(f"New entities created: {stats['new_entities']}")
    print(f"Entities linked: {stats['linked_entities']}")
    print(f"Link rate: {stats['link_rate']}")

    print(f"\nEntities by type:")
    for ent_type, count in stats['entities_by_type'].items():
        print(f"   {ent_type}: {count}")

    # Show detailed entity list
    print("\n" + "=" * 70)
    print("📋 COMPLETE ENTITY DATABASE")
    print("=" * 70)
    for entity in manager.entities.values():
        print(f"\n{entity.entity_id}: {entity.canonical_name} ({entity.entity_type})")
        print(f"   Occurrences: {entity.count}x")
        print(f"   Variations: {', '.join(entity.variations)}")
        print(f"   Seen in: {', '.join([occ.text_id for occ in entity.occurrences])}")

    # Show compact context for LLM
    print("\n" + "=" * 70)
    print("🤖 COMPACT CONTEXT FOR LLM")
    print("=" * 70)
    print("(This is what you send to LLM for next text processing)\n")
    print(manager.get_compact_context(max_entities=10))

    # Save database
    print("\n" + "=" * 70)
    manager.save_to_file("results/entity_database.json")
    print("\n✅ Demo completed! Check results/entity_database.json")


if __name__ == "__main__":
    demo()