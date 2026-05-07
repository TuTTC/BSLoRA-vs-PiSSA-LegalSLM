import json
import os
import argparse
from typing import List, Dict

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=str, default="8,9,12", help="Indices of samples to compare")
    args = parser.parse_args()
    
    indices = [int(i) for i in args.indices.split(",")]
    
    # Define models to compare
    models = [
        {
            "name": "BASELINE",
            "resp_path": "outputs-final/outputs/results/none_task3_detailed_responses.json",
            "judge_path": "evaluation/results/judge_baseline.json",
            "score_key": "reasoning"
        },
        {
            "name": "LoRA",
            "resp_path": "outputs-final/outputs/results/lora_nonthinking_task3_detailed_responses.json",
            "judge_path": "evaluation/results/judge_groq_lora.json",
            "score_key": "reasoning"
        },
        {
            "name": "PiSSA",
            "resp_path": "outputs-final/outputs/results/pissa_task3_detailed_non_thinking.json",
            "judge_path": "evaluation/results/judge_pissa_task3_detailed_non_thinking.json",
            "score_key": "logical_structure"
        },
        {
            "name": "PiSSA + THINKING",
            "resp_path": "outputs-final/outputs/results/pissa_task3_detailed_responses_thinking.json",
            "judge_path": "evaluation/results/judge_pissa_task3_detailed_thinking.json",
            "score_key": "reasoning"
        },
        {
            "name": "PiSSA + RAG",
            "resp_path": "outputs-final/outputs/results/detailed_responses_rag.json",
            "judge_path": "evaluation/results/judge_pissa_rag_task3_detailed_responses.json",
            "score_key": "reasoning"
        }
    ]
    
    # Pre-load all data
    for m in models:
        m["data"] = load_json(m["resp_path"])
        judge_data = load_json(m["judge_path"])
        m["judge"] = judge_data.get("per_sample_results", []) if judge_data else []
        m["aggregate"] = judge_data.get("aggregate_scores", {}) if judge_data else {}

    print("\n" + "="*100)
    print(" BÁO CÁO TỔNG HỢP SO SÁNH ĐA MÔ HÌNH (TASK 3)")
    print("="*100 + "\n")
    
    for idx in indices:
        print(f"### VÍ DỤ #{idx}")
        # Get question from any model that has data
        question = "N/A"
        for m in models:
            if m["data"] and idx < len(m["data"]):
                d = m["data"][idx]
                question = d.get("user_input") or d.get("input") or d.get("user") or "N/A"
                break
        
        print(f"**Câu hỏi:** {question[:300]}...")
        print("-" * 60)
        
        for m in models:
            if not m["data"] or idx >= len(m["data"]):
                continue
                
            resp_item = m["data"][idx]
            response = resp_item.get("model_response") or resp_item.get("model_answer") or "N/A"
            
            # Find score
            score = "N/A"
            if m["judge"] and idx < len(m["judge"]):
                # Try specific score key, fallback to avg_total
                score = m["judge"][idx].get(m["score_key"]) or m["judge"][idx].get("overall_quality") or "N/A"
                
            print(f"[{m['name']}] (Score: {score}):")
            print(f"   > {response[:400].replace('\\n', ' ')}...")
            print("-" * 30)
            
        print("\n" + "="*100 + "\n")

    # Aggregate summary table
    print("### BẢNG ĐIỂM TỔNG HỢP (AGGREGATE SCORES)")
    header = f"{'Model':<20} | {'Avg Score':<10} | {'Eval Samples':<12}"
    print(header)
    print("-" * len(header))
    
    for m in models:
        if m["aggregate"]:
            avg = m["aggregate"].get("avg_total") or m["aggregate"].get("avg_overall_quality") or 0.0
            num = m["aggregate"].get("num_evaluated", 0)
            print(f"{m['name']:<20} | {avg:<10.2f} | {num:<12}")
        else:
            print(f"{m['name']:<20} | {'N/A':<10} | {'0':<12}")

if __name__ == "__main__":
    main()
