"""
Preprocesses raw PDF text extracted from `build_dataset.py` to identify
instruction-like sentences that start with a verb or number + verb.

The script:
- Removes book codes, page references, and technical noise
- Splits long lines into valid instruction sentences
- Keeps only unique, numbered or verb-starting instructions
- Saves results as JSON
"""



import re
import json
import spacy
import os

# Configuration
INPUT_FILE = "instructions.txt"
OUTPUT_FILE = "numbered_instructions1.json"
BOOK_CODE_PATTERN = "A5W02967072002A/RS-AA/001"

# ---------- Load spaCy ----------
def load_spacy():
    """Load spaCy model for POS tagging"""
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model")
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp


# ---------- Clean punctuation ----------
def clean_punctuation(text: str) -> str:
    """Normalize quotes, punctuation, and spacing"""

    text = text.replace('\\"', '"').replace("\\'", "'").replace("\\", "")
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r'\s"(\w+)"\s', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'[.,;]+$', '', text)
    return text.strip()


def should_remove_sentence(sentence: str) -> bool:
    """Return True if sentence is not relevant or too noisy"""
    if re.search(BOOK_CODE_PATTERN, sentence):
        return True
    if re.match(r"^\s*A5W[\s\d/\-A-Z]+\s*$", sentence):
        return True
    if re.search(r"Chapter.*?(Page \d+)", sentence):
        return True
    if re.search(r"Page \d+", sentence):
        return True
    if sentence.count('"') >= 4:
        return True
    if len(sentence.strip()) < 15:
        return True
    if re.search(r"\(cid:\d+\)", sentence):
        return True

    # Common irrelevant technical fragments
    irrelevant_terms = [
        "warning notices", "terminal assignment", "location of",
        "inside faces", "terminal covers", "relevant terminals",
        "sealing wire", "wire diameter"
    ]
    lower = sentence.lower()
    return any(term in lower for term in irrelevant_terms)


# ---------- VERB DETECTION ----------
def starts_with_verb(sentence, nlp) -> bool:
    """Check if a sentence starts with a verb"""
    sentence = sentence.strip()
    if len(sentence) < 5:
        return False

    doc = nlp(sentence)
    first_token = next((t for t in doc if not t.is_punct and not t.is_space), None)
    return bool(first_token and first_token.pos_ == "VERB")


# ---------- SPLIT AND FILTER ----------
def split_into_sentences(line: str, nlp):
    """ Split a line into sentences, keeping only verb-based ones. """

    if " – " in line:
        parts = line.split(" – ")
        line = parts[0]

    doc = nlp(line)
    sentences = [sent.text.strip() for sent in doc.sents]
    valid = []

    for sent in sentences:
        if not sent or should_remove_sentence(sent):
            continue

        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", sent)
        if match:
            first_word = match.group(2)
            first_doc = nlp(first_word)
            if first_doc and first_doc[0].pos_ == "VERB":
                valid.append(sent)
        elif starts_with_verb(sent, nlp):
            valid.append(sent)

    return valid


# ---------- Extract Instruction ----------
def extract_numbered_instructions(text: str, nlp):
    """Extract numbered or verb-starting instructions from text."""
    lines = text.split("\n")
    instructions = []
    line_buffer = []
    current_number = None

    for line in lines:
        line = line.strip()
        if not line or re.search(BOOK_CODE_PATTERN, line):
            continue

        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", line)
        if match:
            # Process previous buffer
            if line_buffer:
                full_text = " ".join(line_buffer)
                for sent in split_into_sentences(full_text, nlp):
                    cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sent)
                    cleaned = clean_punctuation(cleaned)
                    if cleaned and not should_remove_sentence(cleaned):
                        instructions.append({
                            "number": current_number or len(instructions) + 1,
                            "text": cleaned
                        })

            # Reset
            line_buffer = []
            current_number = int(match.group(1))
            first_word = match.group(2)

            doc = nlp(first_word)
            if doc and doc[0].pos_ == "VERB":
                line_buffer = [line]
        elif line_buffer:
            line_buffer.append(line)

    # Final buffer
    if line_buffer:
        full_text = " ".join(line_buffer)
        for sent in split_into_sentences(full_text, nlp):
            cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sent)
            cleaned = clean_punctuation(cleaned)
            if cleaned and not should_remove_sentence(cleaned):
                instructions.append({
                    "number": current_number or len(instructions) + 1,
                    "text": cleaned
                })

    return instructions


def process_file():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(f" Loaded {len(text)} characters\n")

    nlp = load_spacy()

    print("Extracting numbered/verb-based instructions...")
    instructions = extract_numbered_instructions(text, nlp)
    print(f"Found {len(instructions)} raw instructions")

    # Remove duplicates
    unique = []
    seen = set()
    for inst in instructions:
        norm = inst["text"].lower().strip()
        if norm not in seen:
            seen.add(norm)
            unique.append({
                "id": len(unique) + 1,
                "text": inst["text"]
            })

    print(f"Removed {len(instructions) - len(unique)} duplicates")
    print(f"Saving {len(unique)} clean instructions to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    # Show preview
    print("\nSample instructions:")
    for i in unique[:10]:
        print(f" {i['id']}. {i['text']}")


if __name__ == "__main__":
    process_file()


