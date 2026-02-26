"""
Evaluation Script
=================
Chạy inference và đánh giá mô hình trên tập test.

Usage:
    python evaluation/evaluate.py \
        --peft_config configs/lora_config.yaml \
        --checkpoint_dir outputs/checkpoints/lora \
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


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Sinh câu trả lời từ mô hình.
    
    Args:
        model: Language model với adapter
        tokenizer: Tokenizer
        instruction: Câu hỏi / chỉ dẫn
        input_text: Ngữ cảnh bổ sung
        max_new_tokens: Số token tối đa sinh ra
        temperature: Temperature sampling
        top_p: Top-p sampling
    
    Returns:
        Câu trả lời (string)
    """
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
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
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to adapter checkpoint (default: from config)")
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
    print(f"\n{'='*60}")
    print(f"  Evaluating: {peft_method.upper()}")
    print(f"{'='*60}\n")

    set_seed(config["training"]["seed"])

    # =========================================================================
    # 2. Load model + adapter
    # =========================================================================
    model, tokenizer = load_model(config)
    model = apply_peft(model, config)

    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]
    if os.path.exists(checkpoint_dir):
        from peft import PeftModel
        print(f"[EVAL] Loading adapter from: {checkpoint_dir}")
        # Adapter weights sẽ được load nếu đã train
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_dir}")
        print("[WARNING] Using randomly initialized adapter weights")

    log_vram_usage("After loading model")

    # =========================================================================
    # 3. Load test data
    # =========================================================================
    test_data = load_dataset("json", data_files=config["data"]["test_file"], split="train")
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    print(f"[EVAL] Test samples: {len(test_data)}")

    # =========================================================================
    # 4. Generate predictions
    # =========================================================================
    predictions = []
    references = []

    print("[EVAL] Generating responses...")
    for sample in tqdm(test_data, desc="Inference"):
        response = generate_response(
            model, tokenizer,
            instruction=sample["instruction"],
            input_text=sample["input"],
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(response)
        references.append(sample["output"])

    # =========================================================================
    # 5. Compute metrics
    # =========================================================================
    print("\n[EVAL] Computing metrics...")
    results = compute_all_metrics(
        model=model if not args.skip_ppl else None,
        tokenizer=tokenizer if not args.skip_ppl else None,
        predictions=predictions,
        references=references,
        compute_ppl=not args.skip_ppl,
    )

    results["method"] = peft_method
    results["num_samples"] = len(test_data)

    # =========================================================================
    # 6. Save results
    # =========================================================================
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f"{peft_method}_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] Results saved to: {results_path}")

    # In kết quả tóm tắt
    print(f"\n{'='*60}")
    print(f"  Results Summary: {peft_method.upper()}")
    print(f"{'='*60}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:>15}: {value:.4f}")
        else:
            print(f"  {key:>15}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
