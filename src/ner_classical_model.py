"""
Classical NER module using spaCy with custom trained model.
Supports both training and inference for technical instruction texts.
"""

import json
import warnings
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


class ClassicalNER:
    """
    Classical NER extractor using custom-trained spaCy model
    for technical domain entities (COMPONENT, TOOL, ACTION, PARAMETER, LOCATION).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Classical NER extractor.

        Args:
            model_path: Path to a trained model directory.
                    If None or not found, initializes a blank English model.
        """
        if model_path and Path(model_path).exists():
            print(f"Loading trained model from {model_path}")
            self.nlp = spacy.load(model_path)
        else:
            print("No trained model found. Creating blank model.")
            self.nlp = spacy.blank("en")
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner")

        self.ner = self.nlp.get_pipe("ner")

    def _load_data_to_docbin(self, file_path: str) -> DocBin:
        """
        Load training data from JSON and convert to spaCy DocBin format.

        Args:
            file_path: Path to JSON file with training data

        Returns:
            DocBin object with processed documents
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        db = DocBin()

        for sample in data:
            text = sample["text"]
            entities = sample.get("entities", [])

            # Create document
            doc = self.nlp.make_doc(text)
            spans = []

            # Convert entities to spans
            for start, end, label in entities:
                span = doc.char_span(
                    start, end, label=label,
                    alignment_mode="contract"
                )

                if span is None:
                    warnings.warn(
                        f"Skipping invalid span: '{text[start:end]}' "
                        f"at ({start}, {end})"
                    )
                    continue

                spans.append(span)

            # Handle overlapping entities (keep longest first)
            try:
                doc.ents = spans
            except ValueError:
                non_overlapping = []
                seen = set()

                for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
                    if any(i in seen for i in range(span.start, span.end)):
                        continue
                    seen.update(range(span.start, span.end))
                    non_overlapping.append(span)

                doc.ents = non_overlapping

            db.add(doc)

        return db

    def prepare_training_data(self, train_file: str, dev_file: str,
                              output_dir: str = ".") -> Tuple[str, str]:
        """
        Prepare training data by converting JSON to .spacy format.

        Args:
            train_file: Path to training JSON file
            dev_file: Path to development JSON file
            output_dir: Directory to save .spacy files

        Returns:
            Tuple of (train_path, dev_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Add all labels from both files
        for file_path in [train_file, dev_file]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sample in data:
                for _, _, label in sample.get("entities", []):
                    if label not in self.ner.labels:
                        self.ner.add_label(label)

        print(f"Added labels: {self.ner.labels}")

        # Convert to DocBin
        train_db = self._load_data_to_docbin(train_file)
        dev_db = self._load_data_to_docbin(dev_file)

        # Save to disk
        train_path = output_dir / "train.spacy"
        dev_path = output_dir / "dev.spacy"

        train_db.to_disk(train_path)
        dev_db.to_disk(dev_path)

        print(f"Training data saved to {train_path}")
        print(f"Dev data saved to {dev_path}")

        return str(train_path), str(dev_path)

    def train(self, train_file: str, dev_file: Optional[str] = None,
              n_epochs: int = 30, output_dir: str = "model_ner",
              dropout: float = 0.2) -> Dict[str, float]:
        """
        Train the NER model on provided data.

        Args:
            train_file: Path to training JSON file
            dev_file: Optional path to development JSON file
            n_epochs: Number of training epochs
            output_dir: Directory to save trained model
            dropout: Dropout rate for training

        Returns:
            Dictionary with final training losses
        """
        print(f"Starting training for {n_epochs} epochs...")

        # Load training data
        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        # Add labels if not already added
        for sample in train_data:
            for _, _, label in sample.get("entities", []):
                if label not in self.ner.labels:
                    self.ner.add_label(label)

        # Initialize optimizer
        optimizer = self.nlp.initialize()

        # Training loop
        for epoch in range(n_epochs):
            losses = {}
            examples = []

            # Create training examples
            for sample in train_data:
                doc = self.nlp.make_doc(sample["text"])
                example = Example.from_dict(doc, {"entities": sample["entities"]})
                examples.append(example)

            # Update model
            self.nlp.update(
                examples,
                sgd=optimizer,
                drop=dropout,
                losses=losses
            )

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - Losses: {losses}")

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        self.nlp.to_disk(output_path)

        print(f"Training complete! Model saved to {output_path}")

        return losses

    def extract_entities(self, text: str,
                         existing_entities: Optional[Set[str]] = None) -> List[Dict[str, str]]:
        """
        Extract entities from text using the trained model.

        Args:
            text: Input text to process
            existing_entities: Set of existing entity texts (for context)

        Returns:
            List of dictionaries with 'text' and 'label' keys
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

        return entities

    def extract_batch(self, texts: List[str],
                      existing_entities: Optional[Set[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Extract entities from multiple texts efficiently.

        Args:
            texts: List of input texts
            existing_entities: Set of existing entity texts (for context)

        Returns:
            List of entity lists for each text
        """
        results = []

        # Use spaCy's pipe for efficient batch processing
        for doc in self.nlp.pipe(texts):
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
            results.append(entities)

        return results

    def evaluate(self, test_file: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            test_file: Path to test JSON file

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        examples = []
        for sample in test_data:
            doc = self.nlp.make_doc(sample["text"])
            example = Example.from_dict(doc, {"entities": sample["entities"]})
            examples.append(example)

        # Calculate scores
        scores = self.nlp.evaluate(examples)

        print(f"\n Evaluation Results:")
        print(f"  Precision: {scores['ents_p']:.4f}")
        print(f"  Recall: {scores['ents_r']:.4f}")
        print(f"  F1: {scores['ents_f']:.4f}")

        return {
            "precision": scores['ents_p'],
            "recall": scores['ents_r'],
            "f1": scores['ents_f']
        }



if __name__ == "__main__":
    # Training mode
    print("=" * 50)
    print("Training Custom NER Model")
    print("=" * 50)

    # Initialize with blank model
    ner = ClassicalNER()

    # Train the model
    ner.train(
        train_file="data\\data_train.json",
        dev_file="data\\data_dev.json",
        n_epochs=30,
        output_dir="model_ner"
    )

    try:
        ner.evaluate("data\\data_test.json")
    except FileNotFoundError:
        print("No dev file found for evaluation")

    print("\n" + "=" * 50)
    print("Testing Trained Model")
    print("=" * 50)

