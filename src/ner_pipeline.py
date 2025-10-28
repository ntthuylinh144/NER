"""
Complete NER Pipeline
Integrates Classical NER, LLM NER, and Entity Management
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from ner_classical_model import ClassicalNER
from ner_llm_model import LLMNER
from manage_entities import EntityManager


class NERPipeline:
    """
    Complete NER Pipeline with entity management.

    Features:
    - Dual extraction methods (Classical + LLM)
    - Global entity list management
    - Entity linking and deduplication
    - Performance comparison
    """

    def __init__(self,
                 classical_model_path: str = "model_ner",
                 gemini_api_key: Optional[str] = None,
                 similarity_threshold: float = 0.85,
                 max_context_entities: int = 50):
        """
        Initialize the complete pipeline.

        Args:
            classical_model_path: Path to trained spaCy model
            gemini_api_key: Gemini API key for LLM extraction
            similarity_threshold: Threshold for entity matching
            max_context_entities: Max entities to provide as context
        """
        # Initialize NER extractors
        print(" Initializing NER Pipeline...")

        self.classical_ner = ClassicalNER(model_path=classical_model_path)
        print("  Classical NER loaded")

        self.llm_ner = LLMNER(api_key=gemini_api_key, delay=1.0) if gemini_api_key else None
        if self.llm_ner:
            print("  LLM NER loaded")
        else:
            print("    LLM NER not initialized (no API key)")

        # Initialize entity managers (separate for each method)
        self.classical_entity_manager = EntityManager(similarity_threshold)
        self.llm_entity_manager = EntityManager(similarity_threshold)

        self.max_context_entities = max_context_entities

        print(" Pipeline initialized!\n")

    def process_with_classical(self, texts: List[Dict]) -> Dict:
        """
        Process texts with classical NER and entity management.

        Args:
            texts: List of {"id": int, "text": str}

        Returns:
            Dictionary with results and statistics
        """
        print("=" * 70)
        print("CLASSICAL NER EXTRACTION (spaCy)")
        print("=" * 70)

        results = []
        start_time = time.time()

        for i, text_data in enumerate(texts, 1):
            text_id = text_data["id"]
            text = text_data["text"]

            print(f"\n[{i}/{len(texts)}] Processing text {text_id}")
            print(f"Text: {text[:80]}...")

            # Get context from entity manager
            context = self.classical_entity_manager.get_entity_context(
                max_entities=self.max_context_entities
            )

            # Extract entities
            extracted = self.classical_ner.extract_entities(
                text,
                existing_entities=set(context)
            )

            print(f"  Extracted: {len(extracted)} entities")

            # Process with entity manager
            processed = self.classical_entity_manager.process_extraction_results(
                extracted,
                text_id
            )

            # Show linking results
            new_count = sum(1 for e in processed if e["is_new"])
            linked_count = len(processed) - new_count
            print(f"  Result: {new_count} new, {linked_count} linked")

            results.append({
                "id": text_id,
                "text": text,
                "entities": processed
            })

        total_time = time.time() - start_time

        print(f"\n  Total time: {total_time:.2f}s")
        print(f" Average: {total_time / len(texts):.3f}s per text")

        return {
            "method": "classical",
            "results": results,
            "statistics": self.classical_entity_manager.get_statistics(),
            "time": total_time
        }

    def process_with_llm(self, texts: List[Dict]) -> Dict:
        """
        Process texts with LLM NER and entity management.

        Args:
            texts: List of {"id": int, "text": str}

        Returns:
            Dictionary with results and statistics
        """
        if not self.llm_ner:
            print(" LLM NER not available (API key not provided)")
            return None

        print("\n" + "=" * 70)
        print("LLM NER EXTRACTION (Gemini)")
        print("=" * 70)

        results = []
        start_time = time.time()

        for i, text_data in enumerate(texts, 1):
            text_id = text_data["id"]
            text = text_data["text"]

            print(f"\n[{i}/{len(texts)}] Processing text {text_id}")
            print(f"Text: {text[:80]}...")

            # Get context from entity manager
            context = self.llm_entity_manager.get_entity_context(
                max_entities=self.max_context_entities
            )

            # Extract entities
            extracted = self.llm_ner.extract_entities(
                text,
                existing_entities=set(context)
            )

            print(f"  Extracted: {len(extracted)} entities")

            # Process with entity manager
            processed = self.llm_entity_manager.process_extraction_results(
                extracted,
                text_id
            )

            # Show linking results
            new_count = sum(1 for e in processed if e["is_new"])
            linked_count = len(processed) - new_count
            print(f"  Result: {new_count} new, {linked_count} linked")

            results.append({
                "id": text_id,
                "text": text,
                "entities": processed
            })

            # Delay between API calls
            if i < len(texts):
                time.sleep(self.llm_ner.delay)

        total_time = time.time() - start_time

        print(f"\n  Total time: {total_time:.2f}s")
        print(f"⚡ Average: {total_time / len(texts):.3f}s per text")

        return {
            "method": "llm",
            "results": results,
            "statistics": self.llm_entity_manager.get_statistics(),
            "time": total_time
        }

    def process_both_methods(self, texts: List[Dict]) -> Dict:
        """
        Process texts with both methods and compare.

        Args:
            texts: List of {"id": int, "text": str}

        Returns:
            Dictionary with results from both methods
        """
        # Process with classical
        classical_results = self.process_with_classical(texts)

        # Process with LLM (if available)
        llm_results = self.process_with_llm(texts) if self.llm_ner else None

        # Compare results
        self._print_comparison(classical_results, llm_results)

        return {
            "classical": classical_results,
            "llm": llm_results
        }

    def _print_comparison(self, classical_results: Dict, llm_results: Optional[Dict]):
        """Print comparison between both methods."""
        print("\n" + "=" * 70)
        print("COMPARISON: CLASSICAL vs LLM")
        print("=" * 70)

        if not llm_results:
            print("  LLM results not available")
            return

        # Time comparison
        print(f"\n  PROCESSING TIME")
        print(f"  Classical: {classical_results['time']:.2f}s")
        print(f"  LLM:       {llm_results['time']:.2f}s")
        print(f"  Speed:     Classical is {llm_results['time'] / classical_results['time']:.1f}x faster")

        # Entity count comparison
        c_stats = classical_results['statistics']
        l_stats = llm_results['statistics']

        print(f"\n ENTITIES EXTRACTED")
        print(f"  Classical: {c_stats['total_entities']} unique entities")
        print(f"  LLM:       {l_stats['total_entities']} unique entities")

        print(f"\n  ENTITIES BY LABEL")
        all_labels = set(c_stats['entities_by_label'].keys()) | set(l_stats['entities_by_label'].keys())

        for label in sorted(all_labels):
            c_count = c_stats['entities_by_label'].get(label, {}).get('count', 0)
            l_count = l_stats['entities_by_label'].get(label, {}).get('count', 0)
            print(f"  {label:12s}: Classical={c_count:3d}, LLM={l_count:3d}")

        print("=" * 70)

    def load_and_process_file(self, input_file: str,
                              method: str = "both",
                              save_results: bool = True) -> Dict:
        """
        Load texts from file and process.

        Args:
            input_file: Path to JSON file with texts
            method: "classical", "llm", or "both"
            save_results: Whether to save results to files

        Returns:
            Processing results
        """
        # Load texts
        with open(input_file, "r", encoding="utf-8") as f:
            texts = json.load(f)

        print(f" Loaded {len(texts)} texts from {input_file}\n")

        # Process based on method
        if method == "classical":
            results = {"classical": self.process_with_classical(texts)}
        elif method == "llm":
            results = {"llm": self.process_with_llm(texts)}
        else:  # both
            results = self.process_both_methods(texts)

        # Save results
        if save_results:
            self._save_results(results, input_file)

        return results

    def _save_results(self, results: Dict, input_file: str):
        """Save results to JSON files inside 'results/' folder."""
        base_name = Path(input_file).stem
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)  # tạo thư mục nếu chưa có

        if "classical" in results and results["classical"]:
            # Save classical results
            output_file = results_dir / f"{base_name}_classical_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results["classical"]["results"], f, indent=2, ensure_ascii=False)
            print(f"\n Classical results saved to {output_file}")

            # Save entity list
            entity_file = results_dir / f"{base_name}_classical_entities.json"
            self.classical_entity_manager.save_to_file(entity_file)

        if "llm" in results and results["llm"]:
            # Save LLM results
            output_file = results_dir / f"{base_name}_llm_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results["llm"]["results"], f, indent=2, ensure_ascii=False)
            print(f" LLM results saved to {output_file}")

            # Save entity list
            entity_file = results_dir / f"{base_name}_llm_entities.json"
            self.llm_entity_manager.save_to_file(entity_file)

    def print_entity_summaries(self):
        """Print summaries for both entity managers."""
        print("\n" + "=" * 70)
        print("CLASSICAL NER - ENTITY SUMMARY")
        print("=" * 70)
        self.classical_entity_manager.print_summary()

        if self.llm_ner:
            print("\n" + "=" * 70)
            print("LLM NER - ENTITY SUMMARY")
            print("=" * 70)
            self.llm_entity_manager.print_summary()


# Main execution
if __name__ == "__main__":
    import sys
    import os

    # Configuration
    CLASSICAL_MODEL = "model_ner"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    INPUT_FILE = "test_file.json"

    print(" NER Pipeline - Complete System")
    print("=" * 70)

    # Check if input file exists
    if not Path(INPUT_FILE).exists():
        print(f" Input file '{INPUT_FILE}' not found!")
        print("\nPlease create a JSON file with format:")
        print('[')
        print('  {"id": 1, "text": "Your text here..."},')
        print('  {"id": 2, "text": "Another text..."}')
        print(']')
        sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = NERPipeline(
            classical_model_path=CLASSICAL_MODEL,
            gemini_api_key=GEMINI_API_KEY,
            similarity_threshold=0.85,
            max_context_entities=50
        )
    except Exception as e:
        print(f" Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Process file
    try:
        results = pipeline.load_and_process_file(
            INPUT_FILE,
            method="both",  # Change to "classical" or "llm" for single method
            save_results=True
        )

        # Print final summaries
        pipeline.print_entity_summaries()

        print("\n Pipeline execution complete!")

    except Exception as e:
        print(f"\n Error during processing: {e}")
        import traceback

        traceback.print_exc()