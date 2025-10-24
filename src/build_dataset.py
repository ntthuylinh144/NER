import pdfplumber

# Đường dẫn file PDF và file TXT đầu ra
PDF_FILE = "coupling_relay_3RQ1_en-US.pdf"
OUTPUT_FILE = "instructions.txt"

# Danh sách chứa các dòng trích xuất
instructions = []

# Đọc từng trang PDF
with pdfplumber.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            instructions.extend(lines)

# Xóa trùng lặp và sắp xếp theo thứ tự xuất hiện ban đầu
unique_instructions = list(dict.fromkeys(instructions))

# Ghi ra file TXT
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in unique_instructions:
        f.write(line + "\n")

print(f"Đã lưu {len(unique_instructions)} dòng vào file '{OUTPUT_FILE}'")