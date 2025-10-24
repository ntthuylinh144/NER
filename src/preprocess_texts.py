import re
import json
import spacy

# ---------- Configuration ----------
INPUT_FILE = "instructions.txt"
OUTPUT_FILE = "numbered_instructions1.json"

# Regex pattern để phát hiện mã sách (có thể có dấu cách)
# BOOK_CODE_PATTERN = r"A5W\s*\d+\s*/\s*RS-AA\s*/\s*\d+"
BOOK_CODE_PATTERN = "A5W02967072002A/RS-AA/001"

# ---------- Load spaCy ----------
def load_spacy():
    """Load spaCy model để kiểm tra động từ"""
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model")
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp


# ---------- Clean punctuation ----------
def clean_punctuation(text):
    """Delete punctuation and redundant characters"""
    """Xóa dấu câu và ký tự dư thừa"""

    # 1. Xóa backslash trước quotes: the \"Products\" -> the "Products"
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")

    # 2. Xóa backslash đơn lẻ
    text = text.replace("\\", "")

    # 3. Chuẩn hóa quotes kép thừa: ""text"" -> "text"
    text = re.sub(r'"{2,}', '"', text)

    # 4. Xóa quotes nếu không cần thiết (ví dụ: the "Products" field -> the Products field)
    # Chỉ xóa nếu quotes bao quanh 1 từ đơn
    text = re.sub(r'\s"(\w+)"\s', r' \1 ', text)

    # 5. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)

    # 6. Xóa khoảng trắng trước dấu câu
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # 7. Xóa dấu câu thừa ở cuối (nhiều dấu chấm, dấu phẩy...)
    text = re.sub(r'[.,;]+$', '', text)


def should_remove_sentence(sentence):
    """Kiểm tra xem câu có nên bị loại bỏ không"""

    # 1. Loại bỏ câu chứa mã sách (với hoặc không có dấu cách)
    if re.search(BOOK_CODE_PATTERN, sentence):
        return True

    # 2. Loại bỏ câu chỉ chứa mã sách và không có gì khác
    if re.match(r"^\s*A5W[\s\d/\-A-Z]+\s*$", sentence):
        return True

    # 3. Loại bỏ câu có Chapter/Page references
    if re.search(r"Chapter.*?(Page \d+)", sentence):
        return True

    if re.search(r"Page \d+", sentence):
        return True

    # 4. Loại bỏ câu có nhiều quotes liên tiếp (tài liệu tham khảo)
    if sentence.count('"') >= 4:
        return True

    # 5. Loại bỏ câu quá ngắn
    if len(sentence.strip()) < 15:
        return True


    # 6. Loại bỏ câu có ký tự đặc biệt lạ
    if re.search(r"\(cid:\d+\)", sentence):
        return True

    technical_terms = [
        "warning notices", "terminal assignment", "location of",
        "inside faces", "terminal covers", "relevant terminals",
        "sealing wire", "wire diameter"
    ]
    lower_sentence = sentence.lower()
    for term in technical_terms:
        if term in lower_sentence:
            return True

    return False


# ---------- Check if starts with verb ----------
def starts_with_verb(sentence, nlp):
    """Kiểm tra câu có bắt đầu bằng động từ không"""
    sentence = sentence.strip()

    # Bỏ qua câu quá ngắn
    if len(sentence) < 5:
        return False

    # Parse câu
    doc = nlp(sentence)

    # Tìm token đầu tiên không phải dấu câu
    first_token = next((t for t in doc if not t.is_punct and not t.is_space), None)

    if not first_token:
        return False

    # Kiểm tra xem có phải động từ không
    return first_token.pos_ == "VERB"


# ---------- Split sentences in line ----------
def split_into_sentences(line, nlp):
    """
    Tách một dòng thành nhiều câu nếu có nhiều dấu chấm
    Sau đó kiểm tra từng câu xem có hợp lệ không
    """
    # Xóa dấu – (dash) vì nó thường dùng để ngăn cách phần reference
    # Chỉ lấy phần trước dấu –
    if " – " in line:
        parts = line.split(" – ")
        line = parts[0]  # Chỉ lấy phần đầu

    # Tách câu bằng spaCy (chính xác hơn split by ".")
    doc = nlp(line)
    sentences = [sent.text.strip() for sent in doc.sents]

    valid_sentences = []

    for sent in sentences:
        # Bỏ qua câu rỗng
        if not sent:
            continue

        # QUAN TRỌNG: Kiểm tra câu có chứa mã sách không - loại bỏ ngay
        if re.search(BOOK_CODE_PATTERN, sent):
            print(f"  Removed (contains book code): {sent[:80]}...")
            continue

        # Kiểm tra có nên loại bỏ không
        if should_remove_sentence(sent):
            continue

        # Kiểm tra format: số + động từ hoặc chỉ động từ
        # Pattern: số + dấu + động từ
        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", sent)

        if match:
            first_word = match.group(2)
            first_doc = nlp(first_word)
            if first_doc and first_doc[0].pos_ == "VERB":
                valid_sentences.append(sent)
        elif starts_with_verb(sent, nlp):
            # Câu bắt đầu bằng động từ (không có số)
            valid_sentences.append(sent)

    return valid_sentences


# ---------- Extract numbered instructions ----------
def extract_numbered_instructions(text, nlp):
    """
    Trích xuất các câu có dạng: số + dấu + động từ
    Xử lý trường hợp nhiều câu trong 1 dòng
    """

    lines = text.split("\n")
    instructions = []
    current_instruction = ""
    current_number = None
    line_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # KIỂM TRA NGAY: Nếu dòng chứa mã sách, bỏ qua toàn bộ dòng
        if re.search(BOOK_CODE_PATTERN, line):
            print(f"  Skipped line (contains book code): {line[:80]}...")
            continue

        # Pattern: số + dấu + từ
        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", line)

        if match:
            # Xử lý câu trước đó (nếu có)
            if line_buffer:
                full_text = " ".join(line_buffer)
                sub_sentences = split_into_sentences(full_text, nlp)

                for sub_sent in sub_sentences:
                    # Xóa số thứ tự ở đầu nếu có
                    cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sub_sent)
                    # Làm sạch dấu câu
                    cleaned = clean_punctuation(cleaned)
                    if cleaned and not should_remove_sentence(cleaned):
                        instructions.append({
                            "number": current_number if current_number else len(instructions) + 1,
                            "text": cleaned.strip()
                        })

            # Reset buffer
            line_buffer = []

            # Kiểm tra từ đầu tiên có phải động từ không
            number = int(match.group(1))
            first_word = match.group(2)

            doc = nlp(first_word)
            if doc and doc[0].pos_ == "VERB":
                current_number = number
                line_buffer = [line]
            else:
                current_number = None
                line_buffer = []
        else:
            # Dòng tiếp theo của câu hiện tại
            if line_buffer:
                line_buffer.append(line)

    # Xử lý buffer cuối cùng
    if line_buffer:
        full_text = " ".join(line_buffer)
        sub_sentences = split_into_sentences(full_text, nlp)

        for sub_sent in sub_sentences:
            cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sub_sent)
            # Làm sạch dấu câu
            cleaned = clean_punctuation(cleaned)
            if cleaned and not should_remove_sentence(cleaned):
                instructions.append({
                    "number": current_number if current_number else len(instructions) + 1,
                    "text": cleaned.strip()
                })

    return instructions


# ---------- Process file ----------
def process_file():
    """Đọc file txt và trích xuất instructions"""

    print(f" Reading file: {INPUT_FILE}")

    # Đọc file
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    print(f" Loaded {len(text)} characters\n")

    # Load spaCy
    nlp = load_spacy()

    # Extract instructions
    print(" Extracting numbered instructions...")
    instructions = extract_numbered_instructions(text, nlp)

    print(f" Found {len(instructions)} instructions\n")

    # Remove duplicates
    unique_instructions = []
    seen_texts = set()

    for inst in instructions:
        normalized = inst["text"].lower().strip()
        if normalized not in seen_texts and normalized:
            seen_texts.add(normalized)
            unique_instructions.append({
                "id": inst["number"],
                "text": inst["text"]
            })

    duplicates_removed = len(instructions) - len(unique_instructions)
    if duplicates_removed > 0:
        print(f" Removed {duplicates_removed} duplicate instructions\n")

    # Re-index IDs
    for i, inst in enumerate(unique_instructions, 1):
        inst["id"] = i

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_instructions, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(unique_instructions)} instructions to {OUTPUT_FILE}\n")

    # Show samples
    print(" Sample instructions:")
    for inst in unique_instructions[:10]:
        print(f"  {inst['id']}. {inst['text']}\n")


# ---------- Main ----------
# if __name__ == "__main__":
#     process_file()
# , '', text)
#
# return text.strip()


# ---------- Check if should remove ----------
def should_remove_sentence(sentence):
    """Kiểm tra xem câu có nên bị loại bỏ không"""

    # 1. Loại bỏ câu chứa mã sách
    if re.search(BOOK_CODE_PATTERN, sentence):
        return True

    # 2. Loại bỏ câu có Chapter/Page references
    if re.search(r"Chapter.*?(Page \d+)", sentence):
        return True

    if re.search(r"Page \d+", sentence):
        return True

    # 3. Loại bỏ câu có nhiều quotes liên tiếp (tài liệu tham khảo)
    if sentence.count('"') >= 4:
        return True

    # 4. Loại bỏ câu quá ngắn
    if len(sentence.strip()) < 15:
        return True

    # 5. Loại bỏ câu có ký tự đặc biệt lạ
    if re.search(r"\(cid:\d+\)", sentence):
        return True

    technical_terms = [
        "warning notices", "terminal assignment", "location of",
        "inside faces", "terminal covers", "relevant terminals",
        "sealing wire", "wire diameter",  "for example", "terminal area"
    ]
    lower_sentence = sentence.lower()
    for term in technical_terms:
        if term in lower_sentence:
            return True

    return False


# ---------- Check if starts with verb ----------
def starts_with_verb(sentence, nlp):
    """Kiểm tra câu có bắt đầu bằng động từ không"""
    sentence = sentence.strip()

    # Bỏ qua câu quá ngắn
    if len(sentence) < 5:
        return False

    # Parse câu
    doc = nlp(sentence)

    # Tìm token đầu tiên không phải dấu câu
    first_token = next((t for t in doc if not t.is_punct and not t.is_space), None)

    if not first_token:
        return False

    # Kiểm tra xem có phải động từ không
    return first_token.pos_ == "VERB"


# ---------- Split sentences in line ----------
def split_into_sentences(line, nlp):
    """
    Tách một dòng thành nhiều câu nếu có nhiều dấu chấm
    Sau đó kiểm tra từng câu xem có hợp lệ không
    """
    # Xóa dấu – (dash) vì nó thường dùng để ngăn cách phần reference
    # Chỉ lấy phần trước dấu –
    if " – " in line:
        parts = line.split(" – ")
        line = parts[0]  # Chỉ lấy phần đầu

    # Tách câu bằng spaCy (chính xác hơn split by ".")
    doc = nlp(line)
    sentences = [sent.text.strip() for sent in doc.sents]

    valid_sentences = []

    for sent in sentences:
        # Bỏ qua câu rỗng
        if not sent:
            continue

        # Kiểm tra có nên loại bỏ không
        if should_remove_sentence(sent):
            continue

        # Kiểm tra format: số + động từ hoặc chỉ động từ
        # Pattern: số + dấu + động từ
        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", sent)

        if match:
            first_word = match.group(2)
            first_doc = nlp(first_word)
            if first_doc and first_doc[0].pos_ == "VERB":
                valid_sentences.append(sent)
        elif starts_with_verb(sent, nlp):
            # Câu bắt đầu bằng động từ (không có số)
            valid_sentences.append(sent)

    return valid_sentences


# ---------- Extract numbered instructions ----------
def extract_numbered_instructions(text, nlp):
    """
    Trích xuất các câu có dạng: số + dấu + động từ
    Xử lý trường hợp nhiều câu trong 1 dòng
    """

    lines = text.split("\n")
    instructions = []
    current_instruction = ""
    current_number = None
    line_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Kiểm tra xem dòng có chứa mã sách không
        if re.search(BOOK_CODE_PATTERN, line):
            continue

        # Pattern: số + dấu + từ
        match = re.match(r"^(\d+)[\.\)\-\:]\s+(\w+)", line)

        if match:
            # Xử lý câu trước đó (nếu có)
            if line_buffer:
                full_text = " ".join(line_buffer)
                sub_sentences = split_into_sentences(full_text, nlp)

                for sub_sent in sub_sentences:
                    # Xóa số thứ tự ở đầu nếu có
                    cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sub_sent)
                    if cleaned and not should_remove_sentence(cleaned):
                        instructions.append({
                            "number": current_number if current_number else len(instructions) + 1,
                            "text": cleaned.strip()
                        })

            # Reset buffer
            line_buffer = []

            # Kiểm tra từ đầu tiên có phải động từ không
            number = int(match.group(1))
            first_word = match.group(2)

            doc = nlp(first_word)
            if doc and doc[0].pos_ == "VERB":
                current_number = number
                line_buffer = [line]
            else:
                current_number = None
                line_buffer = []
        else:
            # Dòng tiếp theo của câu hiện tại
            if line_buffer:
                line_buffer.append(line)

    # Xử lý buffer cuối cùng
    if line_buffer:
        full_text = " ".join(line_buffer)
        sub_sentences = split_into_sentences(full_text, nlp)

        for sub_sent in sub_sentences:
            cleaned = re.sub(r"^\d+[\.\)\-\:]\s+", "", sub_sent)
            if cleaned and not should_remove_sentence(cleaned):
                instructions.append({
                    "number": current_number if current_number else len(instructions) + 1,
                    "text": cleaned.strip()
                })

    return instructions


# ---------- Process file ----------
def process_file():
    """Đọc file txt và trích xuất instructions"""

    print(f" Reading file: {INPUT_FILE}")

    # Đọc file
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f" File not found: {INPUT_FILE}")
        return

    print(f" Loaded {len(text)} characters\n")

    # Load spaCy
    nlp = load_spacy()

    # Extract instructions
    print(" Extracting numbered instructions...")
    instructions = extract_numbered_instructions(text, nlp)

    print(f" Found {len(instructions)} instructions\n")

    # Remove duplicates
    unique_instructions = []
    seen_texts = set()

    for inst in instructions:
        normalized = inst["text"].lower().strip()
        if normalized not in seen_texts and normalized:
            seen_texts.add(normalized)
            unique_instructions.append({
                "id": inst["number"],
                "text": inst["text"]
            })

    duplicates_removed = len(instructions) - len(unique_instructions)
    if duplicates_removed > 0:
        print(f" Removed {duplicates_removed} duplicate instructions\n")

    # Re-index IDs
    for i, inst in enumerate(unique_instructions, 1):
        inst["id"] = i

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_instructions, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(unique_instructions)} instructions to {OUTPUT_FILE}\n")

    # Show samples
    print(" Sample instructions:")
    for inst in unique_instructions[:10]:
        print(f"  {inst['id']}. {inst['text']}\n")


# ---------- Main ----------
if __name__ == "__main__":
    process_file()

