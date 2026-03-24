"""
Evaluation Script (Multi-task)
===============================
Sử dụng Transformers thuần túy để Inference, né hoàn toàn lỗi shape của Unsloth.
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from training.trainer_utils import load_config
from evaluation.metrics import compute_all_metrics
from utils.helpers import set_seed

SYSTEM_PROMPTS = {
    "task1": "Bạn là một chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là xác định xem một điều luật có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không.",
    "task2": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi trắc nghiệm sau dựa trên văn bản pháp luật được cung cấp.",
    "task3": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi mở sau theo cấu trúc lập luận pháp lý chuyên sâu.",
}

def generate_response(model, tokenizer, system_prompt: str, user_input: str, max_new_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.9) -> str:
    # KHÔNG GỌI UNSLOTH Ở ĐÂY NỮA. CHẠY THUẦN TRANSFORMERS.
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Đảm bảo dùng torch.inference_mode để tắt lưu gradient (tiết kiệm VRAM)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_p=top_p, 
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id # Thêm dòng này để tránh warning
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None, choices=["task1", "task2", "task3"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--test_file", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    task_type = args.task_type or config.get("data", {}).get("task_type")

    print(f"\n{'='*60}\n  Evaluating: {peft_method.upper()}\n  Task:       {task_type or 'ALL'}\n{'='*60}\n")
    set_seed(config["training"]["seed"])

    # =========================================================================
    # LÁ CHẮN THÉP: CHỈ DÙNG TRANSFORMERS & PEFT GỐC ĐỂ LOAD MODEL
    # =========================================================================
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_name = config["model"]["name"]
    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]

    print("\n[EVAL] Đang khởi tạo Transformers thuần túy (Không dùng Unsloth)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model gốc bằng Transformers
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if os.path.exists(checkpoint_dir):
        print(f"[EVAL] Đang nạp trọng số PiSSA/LoRA từ: {checkpoint_dir}")
        model = PeftModel.from_pretrained(model, checkpoint_dir)
    else:
        print(f"[WARNING] KHÔNG TÌM THẤY TRỌNG SỐ TẠI: {checkpoint_dir}")
        print("[WARNING] Sẽ chạy đánh giá trên Mô hình gốc (Base Model)!")
    
    # Ép model vào chế độ eval
    model.eval()
    print("[EVAL] Đã nạp xong mô hình thành công!\n")
    # =========================================================================

    test_file_path = args.test_file if args.test_file else config["data"]["test_file"]
    test_data = load_dataset("json", data_files=test_file_path, split="train")
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    print(f"[EVAL] Đã load {len(test_data)} mẫu từ: {test_file_path}")

    if task_type and "task_type" in test_data.column_names:
        test_data = test_data.filter(lambda x: x["task_type"] == task_type)

    predictions = []
    references = []

    print("\n" + "="*50 + "\n🔎 XEM THỬ 3 MẪU DỰ ĐOÁN ĐẦU TIÊN\n" + "="*50)
    
    for i, sample in enumerate(tqdm(test_data, desc="Inference")):
        sample_task = task_type if task_type else sample.get("task_type", "task3")
        system_prompt = sample.get("system", SYSTEM_PROMPTS.get(sample_task, SYSTEM_PROMPTS["task3"]))
        user_input = sample.get("user", sample.get("instruction", ""))

        response = generate_response(model, tokenizer, system_prompt, user_input, args.max_new_tokens)
        predictions.append(response)
        
        ref = sample.get("assistant", sample.get("output", ""))
        references.append(ref)

        if i < 3:
            print(f"\n🔹 Câu {i+1} (Dự đoán theo chuẩn {sample_task}):")
            print(f"   - Mô hình đáp : {response}")
            print(f"   - Đáp án chuẩn: {ref}")
            
    print("\n" + "="*50)

    print("\n[EVAL] Computing metrics...")
    if task_type is None and "task_type" in test_data.column_names:
        results = {}
        task_groups = {}
        for i, sample in enumerate(test_data):
            t = sample.get("task_type", "unknown")
            if t not in task_groups: task_groups[t] = {"preds": [], "refs": []}
            task_groups[t]["preds"].append(predictions[i])
            task_groups[t]["refs"].append(references[i])

        for t, group in sorted(task_groups.items()):
            print(f"\n--- Metrics for {t} ({len(group['preds'])} samples) ---")
            task_metrics = compute_all_metrics(model=model if not args.skip_ppl else None, tokenizer=tokenizer if not args.skip_ppl else None, predictions=group["preds"], references=group["refs"], compute_ppl=not args.skip_ppl, task_type=t)
            for k, v in task_metrics.items(): results[f"{t}/{k}"] = v
    else:
        eval_task = task_type or "task3"
        results = compute_all_metrics(model=model if not args.skip_ppl else None, tokenizer=tokenizer if not args.skip_ppl else None, predictions=predictions, references=references, compute_ppl=not args.skip_ppl, task_type=eval_task)

    results["method"] = peft_method
    results["num_samples"] = len(test_data)
    if task_type: results["task_type"] = task_type

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    suffix = f"_{task_type}" if task_type else ""
    results_path = os.path.join(results_dir, f"{peft_method}{suffix}_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\n  Results Summary: {peft_method.upper()}" + (f" ({task_type})" if task_type else "") + f"\n{'='*60}")
    for key, value in results.items():
        if isinstance(value, float): print(f"  {key:>30}: {value:.4f}")
        else: print(f"  {key:>30}: {value}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
