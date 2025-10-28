"""
LLM-based NER module using Google Gemini API.
Supports entity extraction using prompt engineering with language models.
"""

import google.generativeai as genai
import json
import time
from typing import List, Dict, Set, Optional
import os
from pathlib import Path


class LLMNER:
    """
    LLM-based NER extractor using Google Gemini API
    for technical domain entities.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash",
                 delay: float = 1.0):
        """
        Initialize the LLM NER extractor.

        Args:
            api_key: Gemini API key. If None, reads from environment variable.
            model_name: Gemini model to use (default: gemini-2.0-flash)
            delay: Delay between API calls in seconds to avoid rate limits
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.delay = delay

        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """
        Test Gemini API connection.

        Returns:
            True if connection successful
        """
        try:
            response = self.model.generate_content("Hello")
            print(f" Gemini API ({self.model_name}) connected successfully!")
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e}")

    def _create_prompt(self, text: str,
                       existing_entities: Optional[List[str]] = None) -> str:
        """
        Create prompt for entity extraction.

        Args:
            text: Input text to extract entities from
            existing_entities: List of existing entities for context

        Returns:
            Formatted prompt string
        """
        base_prompt = """You are an AI model specialized in detecting technical assembly instructions. 
        Your task is to identify and label entities in the text using the following categories:

- COMPONENT: physical part or device (e.g., screw, motor, base plate, cable, device)
- TOOL: instrument or tool used for action (e.g., screwdriver, wrench, multimeter)
- ACTION: technical action or verb (e.g., attach, tighten, connect, calibrate, hold, insert)
- PARAMETER: numeric or physical parameter (e.g., 5V, 10mm, 30°C, torque = 5 Nm)
- LOCATION: position or direction in assembly (e.g., left side, base, top, rear panel, surface)

IMPORTANT:
- Return ONLY a valid JSON array, no other text
- Each entity must have "text" and "label" fields
- Extract all relevant entities from the sentence
"""

        # Add existing entities context if provided
        if existing_entities and len(existing_entities) > 0:
            # Limit context to avoid token limits
            context_entities = existing_entities[:50]  # First 50 entities
            context_str = ", ".join(context_entities)
            base_prompt += f"\nKnown entities from previous texts: {context_str}\n"
            base_prompt += "If you find similar entities, use consistent naming.\n"

        base_prompt += f"""
Example output format:
[
  {{"text": "Tighten", "label": "ACTION"}},
  {{"text": "the Ethernet cable", "label": "COMPONENT"}},
  {{"text": "left side", "label": "LOCATION"}}
]

Now extract entities from this sentence:
"{text}"

Return ONLY the JSON array:"""

        return base_prompt

    def _parse_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        Parse LLM response and extract entities.

        Args:
            response_text: Raw response from LLM

        Returns:
            List of entity dictionaries
        """
        try:
            # Clean markdown formatting
            cleaned = response_text.strip()

            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            # Parse JSON
            entities = json.loads(cleaned)

            # Validate format
            if not isinstance(entities, list):
                print(f"  Response is not a list: {type(entities)}")
                return []

            # Validate each entity
            valid_entities = []
            for ent in entities:
                if isinstance(ent, dict) and "text" in ent and "label" in ent:
                    valid_entities.append({
                        "text": str(ent["text"]),
                        "label": str(ent["label"])
                    })

            return valid_entities

        except json.JSONDecodeError as e:
            print(f" JSON parsing error: {e}")
            print(f"Raw response: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"  Unexpected error parsing response: {e}")
            return []

    def extract_entities(self, text: str,
                         existing_entities: Optional[Set[str]] = None) -> List[Dict[str, str]]:
        """
        Extract entities from text using LLM.

        Args:
            text: Input text to process
            existing_entities: Set of existing entity texts for context

        Returns:
            List of dictionaries with 'text' and 'label' keys
        """
        try:
            # Convert set to list for prompt
            entity_list = list(existing_entities) if existing_entities else None

            # Create prompt
            prompt = self._create_prompt(text, entity_list)

            # Call API
            response = self.model.generate_content(prompt)

            # Parse response
            entities = self._parse_response(response.text)

            return entities

        except Exception as e:
            print(f" Error extracting entities: {e}")
            return []

    def extract_batch(self, texts: List[str],
                      existing_entities: Optional[Set[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Extract entities from multiple texts.
        Includes delay between API calls to avoid rate limits.

        Args:
            texts: List of input texts
            existing_entities: Set of existing entity texts for context

        Returns:
            List of entity lists for each text
        """
        results = []

        for i, text in enumerate(texts, 1):
            print(f"Processing [{i}/{len(texts)}]: {text[:60]}...")

            entities = self.extract_entities(text, existing_entities)
            results.append(entities)

            if entities:
                print(f"  Found {len(entities)} entities")
            else:
                print(f" ️  No entities found")

            # Delay between requests (except last one)
            if i < len(texts):
                time.sleep(self.delay)

        return results

    def extract_with_context_management(self, texts: List[str],
                                        max_context_entities: int = 50) -> List[List[Dict[str, str]]]:
        """
        Extract entities with growing context management.
        Maintains a global entity list that grows with each text.

        Args:
            texts: List of input texts
            max_context_entities: Maximum number of entities to include in context

        Returns:
            List of entity lists for each text
        """
        all_entities = set()  # Global entity list
        results = []

        for i, text in enumerate(texts, 1):
            print(f"Processing [{i}/{len(texts)}]: {text[:60]}...")

            # Use limited context to avoid token limits
            context_entities = list(all_entities)[-max_context_entities:]
            context_set = set(context_entities) if context_entities else None

            # Extract entities
            entities = self.extract_entities(text, context_set)
            results.append(entities)

            # Update global entity list
            for ent in entities:
                all_entities.add(ent["text"])

            if entities:
                print(f"  Found {len(entities)} entities | Total unique: {len(all_entities)}")
            else:
                print(f"   No entities found | Total unique: {len(all_entities)}")

            # Delay between requests
            if i < len(texts):
                time.sleep(self.delay)

        return results

    def save_results(self, results: List[Dict], output_file: str):
        """
        Save extraction results to JSON file.

        Args:
            results: List of results with text and entities
            output_file: Path to output JSON file
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f" Results saved to {output_file}")

    def load_and_process_file(self, input_file: str,
                              output_file: Optional[str] = None) -> List[Dict]:
        """
        Load texts from JSON file and process them.

        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (optional)

        Returns:
            List of processed results
        """
        # Load input
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f" Loaded {len(data)} texts from {input_file}\n")

        # Extract texts
        texts = [item["text"] for item in data]

        # Process with context management
        entity_results = self.extract_with_context_management(texts)

        # Format results
        results = []
        success_count = 0

        for i, (item, entities) in enumerate(zip(data, entity_results)):
            if entities:
                success_count += 1

            results.append({
                "id": item.get("id", i + 1),
                "text": item["text"],
                "entities": entities
            })

        print(f"\n Completed! {success_count}/{len(data)} texts with entities")

        # Save if output file specified
        if output_file:
            self.save_results(results, output_file)

        return results



if __name__ == "__main__":
    import sys

    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(" Error: GEMINI_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("\nOr get one at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("=" * 60)
    print("Testing LLM-based NER with Gemini")
    print("=" * 60)

    # Initialize
    try:
        llm_ner = LLMNER(api_key=api_key, delay=1.0)
    except Exception as e:
        print(f" Failed to initialize: {e}")
        sys.exit(1)
