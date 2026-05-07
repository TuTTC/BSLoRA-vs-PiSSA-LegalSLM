"""
RAG Evaluation Script
======================
So sánh hiệu suất Model-only vs RAG+Model trên Public Test.

Modes:
  - direct: Inference trực tiếp (không RAG)
  - rag: Inference có Hierarchical RAG context

Metrics:
  - Task 1: Citation Accuracy (Có/Không)
  - Task 2: MCQ Accuracy
  - Task 3: LLM-Judge Score (nếu có Gemini key)
  - Retrieval Precision@K (RAG mode only)

Usage:
    # So sánh cả 2 mode
    python evaluation/evaluate_rag.py --config configs/rag_config.yaml

    # Chỉ chạy RAG mode
    python evaluation/evaluate_rag.py --config configs/rag_config.yaml --mode rag

    # Test nhanh với ít samples
    python evaluation/evaluate_rag.py --config configs/rag_config.yaml --max_samples 10
"""

import os
import sys
import json
import yaml
import argparse
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_test_data(test_path: str = "data/processed/test.json") -> List[Dict]:
    """Load test dataset."""
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Tính metrics cho predictions."""
    task_results = {"task1": [], "task2": [], "task3": []}

    for pred, gold in zip(predictions, test_data):
        task_type = gold.get("task_type", "task3")
        pred_answer = pred.get("answer", "").strip()
        gold_answer = gold.get("output", gold.get("assistant", "")).strip()

        if task_type == "task1":
            correct = (
                pred_answer.lower() in gold_answer.lower()
                or gold_answer.lower() in pred_answer.lower()
            )
            task_results["task1"].append(1 if correct else 0)

        elif task_type == "task2":
            correct = pred_answer == gold_answer
            task_results["task2"].append(1 if correct else 0)

        elif task_type == "task3":
            # Simple word overlap score for task3 (nếu không có LLM judge)
            pred_words = set(pred_answer.lower().split())
            gold_words = set(gold_answer.lower().split())
            if gold_words:
                overlap = len(pred_words & gold_words) / len(gold_words)
            else:
                overlap = 0.0
            task_results["task3"].append(overlap)

    metrics = {}
    for task, scores in task_results.items():
        if scores:
            metrics[f"{task}_accuracy"] = sum(scores) / len(scores)
            metrics[f"{task}_count"] = len(scores)

    metrics["overall_count"] = len(predictions)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG vs Direct Inference")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["rag", "direct", "both"],
        help="Evaluation mode"
    )
    parser.add_argument("--test_data", type=str, default="data/processed/test.json")
    parser.add_argument("--task_type", type=str, default=None, choices=["task1", "task2", "task3"], help="Chỉ định task cụ thể (VD: task3)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/results/rag_comparison.json")
    args = parser.parse_args()

    # Load test data
    print("[EVAL] Loading test data...")
    test_data = load_test_data(args.test_data)
    
    # Filter by task_type if specified
    if args.task_type:
        test_data = [d for d in test_data if d.get("task_type") == args.task_type]
        print(f"[EVAL] Filtered for task_type: {args.task_type}")

    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"[EVAL] Test samples: {len(test_data)}")

    # Load pipeline
    from rag.pipeline import HiRAGPipeline
    pipeline = HiRAGPipeline.from_config(args.config)

    results = {}

    # Prepare queries
    queries = [
        {"query": d.get("user", d.get("instruction", "")), "task_type": d.get("task_type", "task3")}
        for d in test_data
    ]

    # Evaluate modes
    modes = ["direct", "rag"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {mode.upper()} mode")
        print(f"{'=' * 60}")

        use_rag = (mode == "rag")
        start_time = time.time()

        predictions = pipeline.answer_batch(queries, use_rag=use_rag)

        elapsed = time.time() - start_time

        metrics = evaluate_predictions(predictions, test_data)
        metrics["elapsed_seconds"] = elapsed
        metrics["mode"] = mode

        results[mode] = metrics

        print(f"\n[EVAL] {mode.upper()} Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Save detailed predictions for LLM Judge
        detailed_list = []
        for pred, gold in zip(predictions, test_data):
            detailed_list.append({
                "id": gold.get("id", "N/A"),
                "task_type": gold.get("task_type", "task3"),
                "question": gold.get("user", gold.get("instruction", "")),
                "reference": gold.get("output", gold.get("assistant", "")),
                "model_answer": pred.get("answer", ""),
                "context_used": pred.get("context", "")
            })
            
        detailed_output_path = os.path.join(
            os.path.dirname(args.output), 
            f"detailed_responses_{mode}.json"
        )
        os.makedirs(os.path.dirname(detailed_output_path), exist_ok=True)
        with open(detailed_output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_list, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved detailed responses to: {detailed_output_path}")

    # Comparison
    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print("  Comparison: Direct vs RAG")
        print(f"{'=' * 60}")
        for task in ["task1", "task2", "task3"]:
            key = f"{task}_accuracy"
            if key in results.get("direct", {}) and key in results.get("rag", {}):
                direct_score = results["direct"][key]
                rag_score = results["rag"][key]
                delta = rag_score - direct_score
                emoji = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                print(f"  {task}: Direct={direct_score:.4f} → RAG={rag_score:.4f} ({emoji} {delta:+.4f})")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
