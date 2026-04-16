"""
VLegal-Bench Evaluation Script
================================
Đánh giá mô hình trên các task của VLegal-Bench theo framework nhận thức pháp lý.
Đặc biệt tập trung vào:
- Task 1.4 (Level 1): Article Recall
- Task 3.3 (Level 3): Multi-Article Reasoning
- Task 5.3 (Level 5): Ethical Consistency

Usage:
    python evaluation/evaluate_vlegal_bench.py \
        --peft_config configs/pissa_config.yaml \
        --task 3.3 \
        --num_samples 100
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from training.trainer_utils import load_config, load_model, apply_peft
from evaluation.metrics import compute_rouge_l, compute_accuracy, _extract_answer_after_think, compute_hierarchical_match
from utils.helpers import set_seed, log_vram_usage

# Mapping task ID to HuggingFace dataset subset names
TASK_CONFIG_MAP = {
    "1.4": "task_1.4",
    "3.3": "task_3.3",
    "5.3": "task_5.3"
}

SYSTEM_PROMPTS = {
    "1.4": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy xác định chính xác số hiệu điều luật tương ứng.",
    "3.3": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy thực hiện lập luận đa bước, kết nối nhiều điều luật để đưa ra kết luận chính xác.",
    "5.3": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy đánh giá tính nhất quán và đạo đức trong tình huống pháp lý này một cách khách quan."
}

def generate_response(model, tokenizer, system_prompt: str, user_input: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
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
        do_sample=True if temperature > 0 else False,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate on VLegal-Bench")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--task", type=str, choices=["1.4", "3.3", "5.3"], required=True, help="VLegal-Bench Task ID")
    parser.add_argument("--num_samples", type=int, default=-1, help="Số lượng mẫu đánh giá (-1 = tất cả)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    set_seed(config["training"]["seed"])

    print(f"\n{'='*60}")
    print(f"  Evaluating VLegal-Bench Task: {args.task}")
    print(f"  Method: {peft_method.upper()}")
    print(f"{'='*60}\n")

    # Load Model
    checkpoint_dir = (args.checkpoint_dir or config["output"]["output_dir"]).replace("\\", "/")
    actual_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        # Check if current dir is already a checkpoint
        if any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in ["adapter_model.safetensors", "adapter_model.bin"]):
            actual_checkpoint = checkpoint_dir
        else:
            # Look for checkpoint-* subdirectories
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                actual_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            else:
                # One last try: if there's a 'pissa' or 'lora' subfolder, look in there
                for sub in ["pissa", "lora", "scenario1_pissa", "scenario1_lora"]:
                    sub_path = os.path.join(checkpoint_dir, sub)
                    if os.path.exists(sub_path):
                        sub_checkpoints = [os.path.join(sub_path, d) for d in os.listdir(sub_path) if d.startswith("checkpoint-")]
                        if sub_checkpoints:
                            actual_checkpoint = max(sub_checkpoints, key=lambda x: int(x.split("-")[-1]))
                            break

    if actual_checkpoint:
        print(f"[EVAL] Loading model with adapter from: {actual_checkpoint}")
        model, tokenizer = load_model(config, adapter_path=actual_checkpoint, force_transformers=True)
    else:
        print("[WARNING] No trained adapter found. Using base model.")
        model, tokenizer = load_model(config, force_transformers=True)
        model = apply_peft(model, config, force_transformers=True)

    model.config.use_cache = True
    tokenizer.padding_side = "left"

    # Load Dataset
    local_task_path = os.path.join("data", "vlegal_bench", args.task, f"{args.task.replace('.', '_')}.jsonl")
    # Some tasks might not have dots in filenames
    if not os.path.exists(local_task_path):
        local_task_path = os.path.join("data", "vlegal_bench", args.task, f"{args.task}.jsonl")

    if os.path.exists(local_task_path):
        print(f"[EVAL] Loading LOCAL dataset from: {local_task_path}")
        dataset = load_dataset("json", data_files=local_task_path, split="train")
    else:
        print(f"[WARNING] Local file not found at {local_task_path}. Trying Hugging Face...")
        subset_name = TASK_CONFIG_MAP.get(args.task, f"task_{args.task}")
        try:
            print(f"[EVAL] Downloading CMC-OPENAI/VLegal-Bench subset: {subset_name}")
            dataset = load_dataset("CMC-OPENAI/VLegal-Bench", subset_name, split="test")
        except Exception as e:
            print(f"[ERROR] Could not load dataset from Hugging Face. Error: {e}")
            sys.exit(1)

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"[EVAL] Test samples: {len(dataset)}")

    predictions = []
    references = []

    print("[EVAL] Generating responses...")
    system_prompt = SYSTEM_PROMPTS.get(args.task, "Bạn là chuyên gia pháp lý.")
    
    for sample in tqdm(dataset, desc="Inference"):
        # VLegal-Bench structure typically has 'question' and 'answer' or 'instruction'/'input'/'output'
        user_input = sample.get('question', sample.get('instruction', ''))
        reference = sample.get('answer', sample.get('output', ''))
        
        response = generate_response(
            model, tokenizer,
            system_prompt=system_prompt,
            user_input=user_input,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(response)
        references.append(reference)

    # Clean outputs (remove think tags if present)
    preds_clean = [_extract_answer_after_think(p) for p in predictions]
    refs_clean = [_extract_answer_after_think(r) for r in references]

    print("\n[EVAL] Computing VLegal metrics...")
    results = {}
    
    if args.task == "1.4": # Recall - usually Accuracy / Hierarchical Matching
        h_metrics = compute_hierarchical_match(preds_clean, refs_clean)
        results.update(h_metrics)
        # F1-Score proxy for matching
        acc = compute_accuracy(preds_clean, refs_clean)
        results["accuracy"] = acc

    elif args.task == "3.3": # Reasoning - ROUGE-L & Hierarchical 
        rouge_l = compute_rouge_l(preds_clean, refs_clean)
        results["rouge_l"] = rouge_l
        h_metrics = compute_hierarchical_match(preds_clean, refs_clean)
        results.update(h_metrics)

    elif args.task == "5.3": # Ethics - Accuracy/F1/ROUGE-L depending on format
        rouge_l = compute_rouge_l(preds_clean, refs_clean)
        results["rouge_l"] = rouge_l
        acc = compute_accuracy(preds_clean, refs_clean)
        results["accuracy"] = acc

    results["task"] = args.task
    results["peft_method"] = peft_method
    
    # Save results
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"vlegal_task_{args.task}_{peft_method}_eval.json")
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  VLegal-Bench Leaderboard Format (Task {args.task})")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:>25}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f"  {k:>25}: {v}")
    print(f"{'='*60}\n")
    print(f"[SAVE] Results saved to {results_path}")

if __name__ == "__main__":
    main()
