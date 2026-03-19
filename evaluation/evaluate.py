"""
Evaluation Script (Multi-task)
===============================
Chạy inference và đánh giá mô hình trên tập test cho từng task.

Usage:
    python evaluation/evaluate.py \\
        --peft_config configs/lora_config.yaml \\
        --task_type task1 \\
        --num_samples 100
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from training.trainer_utils import load_config, load_model, apply_peft
from evaluation.metrics import compute_all_metrics
from utils.helpers import set_seed, log_vram_usage


# ---------------------------------------------------------------------------
# System prompts (đồng bộ với prepare_data.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "task1": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Nhiệm vụ của bạn là xác định xem một điều luật "
        "có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không."
    ),
    "task2": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi trắc nghiệm sau dựa trên "
        "văn bản pháp luật được cung cấp."
    ),
    "task3": (
        "Bạn là một chuyên gia pháp luật Việt Nam. "
        "Hãy trả lời câu hỏi mở sau theo cấu trúc "
        "lập luận pháp lý chuyên sâu."
    ),
}


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Sinh câu trả lời từ mô hình dùng ChatML format.

    Args:
        model: Language model với adapter
        tokenizer: Tokenizer
        system_prompt: System prompt cho task
        user_input: User input đã format
        max_new_tokens: Số token tối đa sinh ra
        temperature: Temperature sampling
        top_p: Top-p sampling

    Returns:
        Câu trả lời (string)
    """
    # FastLanguageModel.for_inference(model) --> Moved to main()

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
    )

    # Decode chỉ phần response (bỏ prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model (Multi-task)")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to adapter checkpoint (default: from config)")
    parser.add_argument(
        "--task_type", type=str, default=None,
        choices=["task1", "task2", "task3"],
        help="Đánh giá cho task cụ thể (mặc định: auto-detect từ dữ liệu)",
    )
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Số lượng mẫu đánh giá (-1 = tất cả)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_ppl", action="store_true",
                        help="Bỏ qua tính Perplexity (tốn thời gian)")
    args = parser.parse_args()

    # =========================================================================
    # 1. Load config
    # =========================================================================
    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    task_type = args.task_type or config.get("data", {}).get("task_type")

    print(f"\n{'='*60}")
    print(f"  Evaluating: {peft_method.upper()}")
    if task_type:
        print(f"  Task:       {task_type}")
    else:
        print(f"  Task:       ALL (auto-detect per sample)")
    print(f"{'='*60}\n")

    set_seed(config["training"]["seed"])

    # =========================================================================
    # 2. Load model + adapter
    # =========================================================================
    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]
    actual_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        # Check if it has weights
        if any(os.path.exists(os.path.join(checkpoint_dir, f)) 
               for f in ["adapter_model.safetensors", "adapter_model.bin"]):
            actual_checkpoint = checkpoint_dir
        else:
            # Search for subdirectories
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
            if checkpoints:
                actual_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                print(f"[EVAL] Found latest checkpoint: {actual_checkpoint}")

    if actual_checkpoint:
        print(f"[EVAL] Loading model with adapter from: {actual_checkpoint}")
        # Use force_transformers=True for stable evaluation (bypasses Unsloth inference bugs)
        model, tokenizer = load_model(config, adapter_path=actual_checkpoint, force_transformers=True)
    else:
        print("[WARNING] No trained adapter found. Using base model + random initialization.")
        model, tokenizer = load_model(config, force_transformers=True)
        model = apply_peft(model, config, force_transformers=True)

    # Optimization for inference (Standard Transformers handles this correctly)
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    log_vram_usage("After loading model")

    # =========================================================================
    # 3. Load test data
    # =========================================================================
    test_data = load_dataset("json", data_files=config["data"]["test_file"], split="train")
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    print(f"[EVAL] Test samples: {len(test_data)}")

    # =========================================================================
    # 4. Group samples by task if needed
    # =========================================================================
    # Nếu task_type được chỉ định, lọc samples
    if task_type and "task_type" in test_data.column_names:
        test_data = test_data.filter(lambda x: x["task_type"] == task_type)
        print(f"[EVAL] Filtered to {len(test_data)} samples for {task_type}")

    # =========================================================================
    # 5. Generate predictions
    # =========================================================================
    predictions = []
    references = []

    print("[EVAL] Generating responses...")
    for sample in tqdm(test_data, desc="Inference"):
        # Xác định system prompt
        sample_task = sample.get("task_type", task_type or "task3")
        system_prompt = sample.get("system", SYSTEM_PROMPTS.get(sample_task, SYSTEM_PROMPTS["task3"]))
        user_input = sample.get("user", sample.get("instruction", ""))

        response = generate_response(
            model, tokenizer,
            system_prompt=system_prompt,
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(response)
        references.append(sample.get("assistant", sample.get("output", "")))

    # =========================================================================
    # 6. Compute metrics
    # =========================================================================
    print("\n[EVAL] Computing metrics...")

    # Nếu có mixed tasks, đánh giá theo từng task
    if task_type is None and "task_type" in test_data.column_names:
        # Đánh giá tổng thể + per-task
        results = {}

        # Group by task
        task_groups = {}
        for i, sample in enumerate(test_data):
            t = sample.get("task_type", "unknown")
            if t not in task_groups:
                task_groups[t] = {"preds": [], "refs": []}
            task_groups[t]["preds"].append(predictions[i])
            task_groups[t]["refs"].append(references[i])

        for t, group in sorted(task_groups.items()):
            print(f"\n--- Metrics for {t} ({len(group['preds'])} samples) ---")
            task_metrics = compute_all_metrics(
                model=model if not args.skip_ppl else None,
                tokenizer=tokenizer if not args.skip_ppl else None,
                predictions=group["preds"],
                references=group["refs"],
                compute_ppl=not args.skip_ppl,
                task_type=t,
            )
            for k, v in task_metrics.items():
                results[f"{t}/{k}"] = v
    else:
        # Đánh giá cho task cụ thể
        results = compute_all_metrics(
            model=model if not args.skip_ppl else None,
            tokenizer=tokenizer if not args.skip_ppl else None,
            predictions=predictions,
            references=references,
            compute_ppl=not args.skip_ppl,
            task_type=task_type,
        )

    results["method"] = peft_method
    results["num_samples"] = len(test_data)
    if task_type:
        results["task_type"] = task_type

    # =========================================================================
    # 7. Save results
    # =========================================================================
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    suffix = f"_{task_type}" if task_type else ""
    results_path = os.path.join(results_dir, f"{peft_method}{suffix}_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to: {results_path}")

    # In kết quả tóm tắt
    print(f"\n{'='*60}")
    print(f"  Results Summary: {peft_method.upper()}" + (f" ({task_type})" if task_type else ""))
    print(f"{'='*60}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:>30}: {value:.4f}")
        else:
            print(f"  {key:>30}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
