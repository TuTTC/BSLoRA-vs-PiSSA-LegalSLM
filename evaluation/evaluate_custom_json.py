import json
import argparse
import re
import os
import sys

# Thêm thư mục gốc vào sys.path để import từ evaluation.metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_rouge_l, compute_hierarchical_match

def remove_think_tags(text):
    """Xóa phần suy nghĩ <think>...</think> để chỉ đánh giá kết quả trả lời cuối cùng."""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Xử lý các dấu cách hoặc khoảng trắng thừa
    return cleaned_text.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Custom JSON using Bloom/VLegal-Bench Framework")
    parser.add_argument("--input_file", type=str, required=True, help="Đường dẫn đến file JSON chứa kết quả sinh của mô hình.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"[LỖI] Không tìm thấy file: {args.input_file}")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = []
    references = []

    print(f"[DATA] Đang tải {len(data)} mẫu từ file {args.input_file}...")

    for item in data:
        ref = item.get("reference", "")
        raw_pred = item.get("model_response", "")
        
        # Bóc tách phần <think> ra khỏi response để đánh giá khách quan
        pred = remove_think_tags(raw_pred)
        
        predictions.append(pred)
        references.append(ref)

    print("\n============================================================")
    print("  ĐÁNH GIÁ THANG ĐO NHẬN THỨC BLOOM (VLEGAL-BENCH STYLE)   ")
    print("============================================================")
    
    # Trục 1: Hierarchical Statutory Accuracy (Độ chính xác Điều-Khoản-Điểm)
    print("\n[1] Đang tính toán cấu trúc viện dẫn pháp luật (Hierarchical Match)...")
    hierarchy_metrics = compute_hierarchical_match(predictions, references)
    
    print("  => Khớp Điều (Article Match):  {:.2f}%".format(hierarchy_metrics.get("article_match", 0) * 100))
    print("  => Khớp Khoản (Clause Match):  {:.2f}%".format(hierarchy_metrics.get("clause_match", 0) * 100))
    print("  => Khớp Điểm (Point Match):    {:.2f}%".format(hierarchy_metrics.get("point_match", 0) * 100))
    print("  => Khớp Toàn Bộ (Full Exact):  {:.2f}%".format(hierarchy_metrics.get("full_match", 0) * 100))

    # Trục 2: ROUGE-L (Sự tương đồng về mặt văn bản và lập luận Syllogism)
    print("\n[2] Đang tính toán độ tương đồng lập luận (ROUGE-L)...")
    try:
        rouge_metrics = compute_rouge_l(predictions, references)
        print("  => ROUGE-L: {:.4f}".format(rouge_metrics))
    except Exception as e:
        print(f"  [CẢNH BÁO] Không thể tính ROUGE-L. Lỗi: {e}")

    print("\n============================================================")
    print("Hoàn tất đánh giá!")

if __name__ == "__main__":
    main()
