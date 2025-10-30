"""
build_dataset.py
----------------
Extracts raw unique text lines from a PDF manual
"""

import pdfplumber

# Configuration
PDF_FILE = "coupling_relay_3RQ1_en-US.pdf"
OUTPUT_FILE = "data\\raw_data.txt"


raw_data = []

with pdfplumber.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            raw_data.extend(lines)

unique_raw_data = list(dict.fromkeys(raw_data)) # remove duplicates

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in unique_raw_data:
        f.write(line + "\n")

print(f"Saved {len(unique_raw_data)} lines to '{OUTPUT_FILE}'")