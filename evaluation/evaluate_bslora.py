import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Environment Hack: Prepend BSLoRA directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSLORA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "BSLoRA")
sys.path.insert(0, BSLORA_DIR)
sys.path.insert(0, PROJECT_ROOT)

import peft
from peft import PeftModel
from training.trainer_utils import load_config
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
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model (BSLoRA)")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, default="configs/bslora_config.yaml")
    parser.add_argument("--share_mode", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--skip_ppl", action="store_true")
    args = parser.parse_args()

    # Load config
    config = load_config(args.base_config, args.peft_config)
    if args.share_mode:
        config["peft"]["share_mode"] = args.share_mode
        if not args.checkpoint_dir:
            args.checkpoint_dir = f"outputs/checkpoints/bslora_{args.share_mode}"

    peft_method = config["peft"]["method"]
    share_mode = config["peft"].get("share_mode", "unknown")
    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {peft_method.upper()} (Mode: {share_mode})")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}\n")

    set_seed(config["training"]["seed"])

    # Load base model & tokenizer (Standard transformers)
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
    )

    # Load BSLoRA adapter
    if os.path.exists(checkpoint_dir):
        print(f"[EVAL] Loading BSLoRA adapter from: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_dir}")

    model.eval()
    log_vram_usage("After loading model")

    # Load test data
    test_data = load_dataset("json", data_files=config["data"]["test_file"], split="train")
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    print(f"[EVAL] Test samples: {len(test_data)}")

    # Group samples by task
    predictions = []
    references = []

    print("[EVAL] Generating responses...")
    for sample in tqdm(test_data, desc="Inference"):
        sample_task = sample.get("task_type", "task3")
        system_prompt = sample.get("system", SYSTEM_PROMPTS.get(sample_task, SYSTEM_PROMPTS["task3"]))
        user_input = sample.get("user", sample.get("instruction", ""))

        response = generate_response(
            model, tokenizer,
            system_prompt=system_prompt,
            user_input=user_input,
        )
        predictions.append(response)
        references.append(sample.get("assistant", sample.get("output", "")))

    # Compute metrics
    print("\n[EVAL] Computing metrics...")
    results = {}
    task_groups = {}
    for i, sample in enumerate(test_data):
        t = sample.get("task_type", "task3")
        if t not in task_groups:
            task_groups[t] = {"preds": [], "refs": []}
        task_groups[t]["preds"].append(predictions[i])
        task_groups[t]["refs"].append(references[i])

    for t, group in sorted(task_groups.items()):
        print(f"\n--- Metrics for {t} ({len(group['preds'])} samples) ---")
        task_metrics = compute_all_metrics(
            model=None, # Skip PPL here to save time/memory in this custom script
            tokenizer=tokenizer,
            predictions=group["preds"],
            references=group["refs"],
            compute_ppl=False,
            task_type=t,
        )
        for k, v in task_metrics.items():
            results[f"{t}/{k}"] = v

    results["method"] = f"bslora_{share_mode}"
    results["num_samples"] = len(test_data)

    # Save results
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"bslora_{share_mode}_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to: {results_path}")

if __name__ == "__main__":
    main()
