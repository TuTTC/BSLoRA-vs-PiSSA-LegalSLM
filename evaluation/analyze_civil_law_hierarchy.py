"""
Civil Law Hierarchy Analysis
==============================
Công cụ phân tích lỗi (Error Analysis) theo đặc thù Civil Law của Việt Nam.
Tương tự như VLegal-Bench đã nhấn mạnh, cấu trúc "Điều -> Khoản -> Điểm" 
là một rào cản lớn với LLMs. Sub-script này tính toán số lượng "near-misses".

Usage:
    python evaluation/analyze_civil_law_hierarchy.py --results_file outputs/results/vlegal_raw.json
"""

import sys
import json
import argparse
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import extract_legal_citation

def analyze_near_misses(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {}

    correct_article_wrong_clause = 0
    correct_clause_wrong_point = 0
    correct_all = 0
    wrong_article = 0

    for pred, ref in zip(predictions, references):
        pred_cites = extract_legal_citation(pred)
        ref_cites = extract_legal_citation(ref)
        
        if not ref_cites:
            continue
            
        ref_articles = {c["article"] for c in ref_cites if c["article"]}
        ref_clauses = {(c["article"], c["clause"]) for c in ref_cites if c["article"] and c["clause"]}
        ref_points = {(c["article"], c["clause"], c["point"]) for c in ref_cites if c["article"] and c["clause"] and c["point"]}
        
        best_match = 0 # 0: wrong, 1: art, 2: clause, 3: point
        
        for p in pred_cites:
            art = p["article"]
            cls = p["clause"]
            pnt = p["point"]
            
            if art in ref_articles:
                if (art, cls) in ref_clauses:
                    if (art, cls, pnt) in ref_points:
                        best_match = max(best_match, 3)
                    else:
                        best_match = max(best_match, 2)
                else:
                    best_match = max(best_match, 1)

        if best_match == 3:
            correct_all += 1
        elif best_match == 2:
            correct_clause_wrong_point += 1
        elif best_match == 1:
            correct_article_wrong_clause += 1
        else:
            wrong_article += 1

    return {
        "total_samples": total,
        "correct_all_hierarchy": correct_all,
        "near_miss_correct_clause_wrong_point": correct_clause_wrong_point,
        "near_miss_correct_article_wrong_clause": correct_article_wrong_clause,
        "wrong_article": wrong_article,
        "analysis_quote": (
            f"Trong {total} mẫu, {correct_article_wrong_clause} mẫu đoán đúng Điều nhưng sai Khoản, "
            f"và {correct_clause_wrong_point} mẫu sai ở bước Điểm. "
            "Đây là bằng chứng rõ ràng cho thử thách 'hierarchical statutory interpretation' của Civil Law Việt Nam."
        )
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="JSON file with 'predictions' and 'references'")
    args = parser.parse_args()

    with open(args.results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    refs = data.get("references", [])

    if not preds or not refs:
        print("[ERROR] File JSON phải chứa danh sách 'predictions' và 'references' để phân tích.")
        return

    print(f"\n[ANALYSIS] Chạy phân tích Civil Law Hierarchy trên {len(preds)} mẫu...")
    analysis = analyze_near_misses(preds, refs)
    
    print("\n--- KẾT QUẢ PHÂN TÍCH (NEAR-MISSES) ---")
    for k, v in analysis.items():
        if k != "analysis_quote":
            print(f"  {k:>40}: {v}")
    
    print(f"\n--- LUẬN ĐIỂM (Dùng cho bài báo) ---")
    print(f"  {analysis['analysis_quote']}\n")

if __name__ == "__main__":
    main()
