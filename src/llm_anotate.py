
import google.generativeai as genai
import json
import time

# -------- CONFIG --------
GEMINI_API_KEY = "AIzaSyDk-ducsOxHRf4Cbhumr9l8kj1A0gneoc8"  # Lấy tại: https://aistudio.google.com/apikey
INPUT_FILE = "annotated_instructions1.json"
OUTPUT_FILE = "annotated_instructions2.json"
DELAY = 1.0


# -------------------------

def test_gemini():
    """Test Gemini API key"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say hello")
        print("Gemini API hoạt động!")
        print(f"Response: {response.text}\n")
        return True
    except Exception as e:
        print(f"Gemini lỗi: {e}")
        return False


def annotate_sentence(model, text):
    """Gán nhãn entity cho 1 câu"""
    prompt = f"""You are a technical text annotator for assembly instructions.
Label entities with:
# - COMPONENT: physical part or device (e.g., screw, motor, base plate, cable)
# - TOOL: instrument or tool used for action (e.g., screwdriver, wrench, multimeter)
# - ACTION: technical action or verb (e.g., attach, tighten, connect, calibrate)
# - PARAMETER: numeric or physical parameter (e.g., 5V, 10mm, 30°C, torque = 5 Nm)
# - LOCATION: position or direction in assembly (e.g., left side, base, top, rear panel)
#
# Return ONLY a JSON array, no other text (example):
# [
#   {{"text": "Tighten", "label": "ACTION"}},
#   {{"text": "the Ethernet cable", "label": "COMPONENT"}}
# ]

Return ONLY valid JSON array:
[{{"text": "word", "label": "CATEGORY"}}]

Sentence: "{text}"
"""

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()

        # Làm sạch markdown
        if "```json" in reply:
            reply = reply.split("```json")[1].split("```")[0].strip()
        elif "```" in reply:
            reply = reply.split("```")[1].split("```")[0].strip()

        # Parse JSON
        entities = json.loads(reply)

        if not isinstance(entities, list):
            print(f"Không phải list")
            return []

        return entities

    except Exception as e:
        print(f" Lỗi: {e}")
        return []


def main():
    # Kiểm tra API key
    if GEMINI_API_KEY == "PASTE_YOUR_GEMINI_KEY_HERE":
        print(" Vui lòng lấy Gemini API key tại:")
        print(" https://aistudio.google.com/apikey")
        return

    # Test API
    print("🔍 Đang kiểm tra Gemini API...\n")
    if not test_gemini():
        return

    # Khởi tạo model
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Đọc input
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f" Đã load {len(data)} câu\n")
    except FileNotFoundError:
        print(f" Không tìm thấy {INPUT_FILE}")
        return

    annotated = []
    success_count = 0

    for i, item in enumerate(data, start=1):
        text = item["text"]
        print(f" [{i}/{len(data)}] {text[:60]}...")

        entities = annotate_sentence(model, text)

        if entities:
            success_count += 1
            print(f"=> {len(entities)} entities")

        annotated.append({
            "id": item["id"],
            "text": text,
            "entities": entities
        })

        time.sleep(DELAY)

    # Lưu kết quả
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"\nHoàn thành! {success_count}/{len(data)} câu")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

