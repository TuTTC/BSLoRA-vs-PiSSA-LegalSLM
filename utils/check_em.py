import json
# Import sẵn hàm bạn vừa hỏi từ file metrics.py của dự án
from evaluation.metrics import compute_exact_match, _extract_answer_after_think

# 1. Đường dẫn tới file JSON của bạn
file_path = r"outputs-final\outputs\results\lora_task3_detailed_responses.json"

# Đọc file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

predictions = []
references = []

for item in data:
    # Lấy model_answer và reference ra
    # Hàm _extract_answer_after_think giúp tự động loại bỏ phần suy luận <think>...</think> (nếu có)
    # để chỉ tính Exact Match cho phần kết luận cuối cùng thôi
    pred = _extract_answer_after_think(item.get("model_answer", ""))
    ref = _extract_answer_after_think(item.get("reference", ""))
    
    predictions.append(pred)
    references.append(ref)

# 2. Tính điểm Exact Match (EM)
em_score = compute_exact_match(predictions, references)

print(f"Đã đọc {len(data)} câu từ file JSON.")
print(f"Điểm Exact Match (EM): {em_score:.4f} (tức {(em_score * 100):.2f}%)")
