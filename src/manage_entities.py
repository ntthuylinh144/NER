"""
Entity Manager - Core Challenge Solution
Maintains a growing, global list of all extracted entities with:
- Fuzzy matching to link similar entities
- Deduplication
- Context tracking
"""

from typing import List, Dict, Set, Optional, Tuple
from difflib import SequenceMatcher
import json
from collections import defaultdict


class Entity:
    """Represents a single entity with metadata."""

    def __init__(self, text: str, label: str, entity_id: int):
        self.id = entity_id
        self.text = text
        self.label = label
        self.normalized_text = self._normalize(text)
        self.occurrences = 1
        self.aliases = set([text])  # Different forms of the same entity
        self.source_texts = []  # Which texts it appeared in

    def _normalize(self, text: str) -> str:
        """Normalize text for matching (lowercase, strip)."""
        return text.lower().strip()

    def add_occurrence(self, text: str, source_text_id: int):
        """Record another occurrence of this entity."""
        self.occurrences += 1
        self.aliases.add(text)
        if source_text_id not in self.source_texts:
            self.source_texts.append(source_text_id)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "occurrences": self.occurrences,
            "aliases": list(self.aliases),
            "source_texts": self.source_texts
        }

    def __repr__(self):
        return f"Entity(id={self.id}, text='{self.text}', label={self.label}, count={self.occurrences})"


class EntityManager:
    """
    Manages global entity list with fuzzy matching and deduplication.

    Core features:
    - Maintain growing list of unique entities
    - Match new entities to existing ones
    - Link similar entities together
    - Provide context for NER extraction
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize Entity Manager.

        Args:
            similarity_threshold: Threshold for fuzzy matching (0.0-1.0)
                                 Higher = stricter matching
        """
        self.entities: List[Entity] = []
        self.entity_map: Dict[str, Entity] = {}  # normalized_text -> Entity
        self.label_index: Dict[str, List[Entity]] = defaultdict(list)
        self.next_id = 1
        self.similarity_threshold = similarity_threshold
        self.processed_texts = 0

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _find_matching_entity(self, text: str, label: str) -> Optional[Entity]:
        """
        Find existing entity that matches the new text.

        Strategy:
        1. Exact match (normalized)
        2. Fuzzy match within same label

        Args:
            text: Entity text to match
            label: Entity label

        Returns:
            Matching Entity or None
        """
        normalized = text.lower().strip()

        # 1. Exact match
        if normalized in self.entity_map:
            existing = self.entity_map[normalized]
            if existing.label == label:
                return existing

        # 2. Fuzzy match within same label
        candidates = self.label_index.get(label, [])

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self._calculate_similarity(normalized, candidate.normalized_text)

            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_match = candidate

        return best_match

    def add_entity(self, text: str, label: str, source_text_id: int) -> Tuple[Entity, bool]:
        """
        Add entity to global list or link to existing one.

        Args:
            text: Entity text
            label: Entity label
            source_text_id: ID of source text where entity was found

        Returns:
            Tuple of (Entity, is_new)
            - Entity: The entity (new or existing)
            - is_new: True if newly created, False if matched to existing
        """
        # Try to find matching entity
        existing = self._find_matching_entity(text, label)

        if existing:
            # Link to existing entity
            existing.add_occurrence(text, source_text_id)
            return existing, False
        else:
            # Create new entity
            new_entity = Entity(text, label, self.next_id)
            new_entity.source_texts.append(source_text_id)

            self.next_id += 1
            self.entities.append(new_entity)

            # Index for fast lookup
            self.entity_map[new_entity.normalized_text] = new_entity
            self.label_index[label].append(new_entity)

            return new_entity, True

    def process_extraction_results(self, extracted_entities: List[Dict[str, str]],
                                   source_text_id: int) -> List[Dict]:
        """
        Process entities extracted from a text.

        Args:
            extracted_entities: List of {"text": str, "label": str}
            source_text_id: ID of source text

        Returns:
            List of processed entities with linking info
        """
        results = []

        for ent_dict in extracted_entities:
            text = ent_dict["text"]
            label = ent_dict["label"]

            entity, is_new = self.add_entity(text, label, source_text_id)

            results.append({
                "text": text,
                "label": label,
                "entity_id": entity.id,
                "is_new": is_new,
                "canonical_text": entity.text,  # Original form
                "occurrences": entity.occurrences
            })

        self.processed_texts += 1
        return results

    def get_entity_context(self, max_entities: Optional[int] = None) -> List[str]:
        """
        Get list of entity texts for context in NER extraction.

        Args:
            max_entities: Maximum number of entities to return
                         If None, returns all entities

        Returns:
            List of entity texts (canonical forms)
        """
        # Sort by occurrences (most common first)
        sorted_entities = sorted(
            self.entities,
            key=lambda e: e.occurrences,
            reverse=True
        )

        if max_entities:
            sorted_entities = sorted_entities[:max_entities]

        return [e.text for e in sorted_entities]

    def get_entities_by_label(self, label: str) -> List[Entity]:
        """Get all entities with specific label."""
        return self.label_index.get(label, [])

    def get_entity_by_id(self, entity_id: int) -> Optional[Entity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_statistics(self) -> Dict:
        """Get statistics about the entity collection."""
        stats = {
            "total_entities": len(self.entities),
            "processed_texts": self.processed_texts,
            "entities_by_label": {}
        }

        for label, entities in self.label_index.items():
            stats["entities_by_label"][label] = {
                "count": len(entities),
                "total_occurrences": sum(e.occurrences for e in entities)
            }

        # Top entities
        top_entities = sorted(
            self.entities,
            key=lambda e: e.occurrences,
            reverse=True
        )[:10]

        stats["top_10_entities"] = [
            {"text": e.text, "label": e.label, "count": e.occurrences}
            for e in top_entities
        ]

        return stats

    def export_to_dict(self) -> Dict:
        """Export all entities to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "statistics": self.get_statistics()
        }

    def save_to_file(self, filepath: str):
        """Save entity list to JSON file."""
        data = self.export_to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" Entity list saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load entity list from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild entities
        self.entities = []
        self.entity_map = {}
        self.label_index = defaultdict(list)
        self.next_id = 1

        for ent_data in data["entities"]:
            entity = Entity(
                ent_data["text"],
                ent_data["label"],
                ent_data["id"]
            )
            entity.occurrences = ent_data["occurrences"]
            entity.aliases = set(ent_data["aliases"])
            entity.source_texts = ent_data["source_texts"]

            self.entities.append(entity)
            self.entity_map[entity.normalized_text] = entity
            self.label_index[entity.label].append(entity)

            if entity.id >= self.next_id:
                self.next_id = entity.id + 1

        print(f" Loaded {len(self.entities)} entities from {filepath}")

    def print_summary(self):
        """Print human-readable summary."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("ENTITY MANAGER SUMMARY")
        print("=" * 60)
        print(f"Total Unique Entities: {stats['total_entities']}")
        print(f"Processed Texts: {stats['processed_texts']}")
        print(f"\nEntities by Label:")

        for label, data in stats["entities_by_label"].items():
            print(f"  {label}: {data['count']} unique ({data['total_occurrences']} total occurrences)")

        print(f"\nTop 10 Most Frequent Entities:")
        for i, ent in enumerate(stats["top_10_entities"], 1):
            print(f"  {i}. [{ent['label']}] {ent['text']} ({ent['count']} times)")

        print("=" * 60)

