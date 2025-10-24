import spacy
import json
from spacy.training import Example
from spacy.scorer import Scorer
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------- CONFIG ----------
MODEL_PATH = "model_ner"          # thư mục chứa model bạn đã train
TEST_FILE = "data_test.json"      # file test có format [(text, {"entities": [...]})]
# -----------------------------

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_offsets(test_data):
    """Kiểm tra offset không khớp với text"""
    invalid = 0
    for i, (text, ann) in enumerate(test_data):
        for s, e, label in ann.get("entities", []):
            span = text[s:e]
            if not span.strip():
                print(f" Invalid offset in sample {i}: {label} [{s}:{e}] -> '{span}'")
                invalid += 1
    if invalid == 0:
        print(" All offsets are valid!")
    else:
        print(f"⚠ Found {invalid} invalid entity spans!")

def evaluate_by_text(nlp, test_data):
    y_true, y_pred = [], []
    results = []  # lưu kết quả từng câu

    for idx, sample in enumerate(test_data, start=1):
        text = sample["text"]
        entities = sample.get("entities", [])
        true_ents = [(text[s:e], label) for s, e, label in entities]

        doc = nlp(text)
        pred_ents = [(ent.text, ent.label_) for ent in doc.ents]

        # lưu chi tiết từng sample
        results.append({
            "id": idx,
            "text": text,
            "true_entities": true_ents,
            "pred_entities": pred_ents
        })

        # dùng cho tính toán thống kê
        for te, tl in true_ents:
            y_true.append(tl)
            if any(pe.lower() == te.lower() and pl == tl for pe, pl in pred_ents):
                y_pred.append(tl)
            else:
                y_pred.append("NONE")

    # --- In ra từng mẫu ---
    print("\n================= DETAILED RESULTS =================")
    for r in results:
        print(f"\n🟩 ID {r['id']}")
        print(f"Text: {r['text']}")
        print(f"True : {r['true_entities']}")
        print(f" Pred : {r['pred_entities']}")
        # Đánh dấu so sánh
        missed = [t for t in r["true_entities"] if t not in r["pred_entities"]]
        extra = [p for p in r["pred_entities"] if p not in r["true_entities"]]
        if missed:
            print(f" Missed: {missed}")
        if extra:
            print(f"️ Extra: {extra}")

    # --- Thống kê tổng hợp ---
    labels = list(set(label for sample in test_data for _, _, label in sample.get("entities", [])))
    print("\n================= LENIENT TEXT-BASED METRICS =================")
    print(f"Precision: {precision_score(y_true, y_pred, average='micro', zero_division=0):.2f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='micro', zero_division=0):.2f}")
    print(f"F1-score : {f1_score(y_true, y_pred, average='micro', zero_division=0):.2f}")

    # --- Lưu ra file để xem lại nếu cần ---
    # with open("detailed_results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print("\n Saved detailed predictions to detailed_results.json")




def evaluate_model(model_path, test_file):
    print(f"Loading model from {model_path}")
    nlp = spacy.load(model_path)
    test_data = load_data(test_file)
    evaluate_by_text(nlp, test_data)


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, TEST_FILE)



