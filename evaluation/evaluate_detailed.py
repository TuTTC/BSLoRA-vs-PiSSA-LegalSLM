"""
Detailed Evaluation Script (Multi-task)
=======================================
Chạy inference và đánh giá mô hình, đồng thời lưu lại toàn bộ câu trả lời 
chi tiết của mô hình để phân tích định tính (đặc biệt cho Task 3).

Usage:
    python evaluation/evaluate_detailed.py \
        --peft_config configs/lora_config.yaml \
        --task_type task3
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
    """Sinh câu trả lời từ mô hình dùng ChatML format."""
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
    parser = argparse.ArgumentParser(description="Evaluate with detailed logging")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None, choices=["task1", "task2", "task3"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_ppl", action="store_true")
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    task_type = args.task_type or config.get("data", {}).get("task_type")

    print(f"\n[DETAILED EVAL] Method: {peft_method.upper()}, Task: {task_type or 'ALL'}")
    set_seed(config["training"]["seed"])

    # 2. Load model + adapter
    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]
    actual_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        if any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in ["adapter_model.safetensors", "adapter_model.bin"]):
            actual_checkpoint = checkpoint_dir
        else:
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
            if checkpoints:
                actual_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    
    if actual_checkpoint:
        print(f"[EVAL] Loading adapter from: {actual_checkpoint}")
        model, tokenizer = load_model(config, adapter_path=actual_checkpoint, force_transformers=True)
    else:
        print("[WARNING] No adapter found. Using base model.")
        model, tokenizer = load_model(config, force_transformers=True)
        model = apply_peft(model, config, force_transformers=True)

    model.config.use_cache = True
    tokenizer.padding_side = "left"
    log_vram_usage("After loading model")

    # 3. Load data
    test_data = load_dataset("json", data_files=config["data"]["test_file"], split="train")
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    if task_type and "task_type" in test_data.column_names:
        test_data = test_data.filter(lambda x: x["task_type"] == task_type)

    # 4. Generate & Log
    predictions = []
    references = []
    detailed_results = []

    print(f"[EVAL] Generating {len(test_data)} responses...")
    for sample in tqdm(test_data, desc="Inference"):
        sample_task = sample.get("task_type", task_type or "task3")
        system_prompt = sample.get("system", SYSTEM_PROMPTS.get(sample_task, SYSTEM_PROMPTS["task3"]))
        user_input = sample.get("user", sample.get("instruction", ""))
        reference = sample.get("assistant", sample.get("output", ""))

        response = generate_response(
            model, tokenizer,
            system_prompt=system_prompt,
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
        )
        
        predictions.append(response)
        references.append(reference)
        
        detailed_results.append({
            "id": sample.get("id", "N/A"),
            "task_type": sample_task,
            "system_prompt": system_prompt,
            "user_input": user_input,
            "reference": reference,
            "model_response": response
        })

    # 5. Save detailed results
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    suffix = f"_{task_type}" if task_type else ""
    
    detailed_path = os.path.join(results_dir, f"{peft_method}{suffix}_detailed_responses.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Detailed responses saved to: {detailed_path}")

    # 6. Compute metrics
    print("[EVAL] Computing metrics summary...")
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
    
    results_path = os.path.join(results_dir, f"{peft_method}{suffix}_summary_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Summary results saved to: {results_path}")

if __name__ == "__main__":
    main()
