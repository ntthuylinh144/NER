"""
build_dataset.py
----------------
Extracts raw unique text lines from a PDF manual
"""

import pdfplumber

# Configuration
PDF_FILE = "coupling_relay_3RQ1_en-US.pdf"
OUTPUT_FILE = "instructions.txt"


instructions = []

with pdfplumber.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            instructions.extend(lines)

unique_instructions = list(dict.fromkeys(instructions)) # remove duplicates

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in unique_instructions:
        f.write(line + "\n")

print(f"Saved {len(unique_instructions)} lines to '{OUTPUT_FILE}'")