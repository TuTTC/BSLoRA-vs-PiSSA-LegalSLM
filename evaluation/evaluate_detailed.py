"""
Detailed Evaluation Script (Multi-task) - STABLE VERSION
=======================================================
Vá lỗi RuntimeError của Unsloth và TypeError của trainer_utils.
Sử dụng Standard Transformers để đảm bảo tính ổn định tuyệt đối.
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from training.trainer_utils import load_config
from evaluation.metrics import compute_all_metrics, _extract_answer_after_think
from utils.helpers import set_seed, log_vram_usage

SYSTEM_PROMPTS = {
    "task1": "Bạn là một chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là xác định xem một điều luật có thể được sử dụng để trả lời câu hỏi pháp lý cụ thể hay không.",
    "task2": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi trắc nghiệm sau dựa trên văn bản pháp luật được cung cấp.",
    "task3": "Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi mở sau theo cấu trúc lập luận pháp lý chuyên sâu.",
}

def generate_response(model, tokenizer, system_prompt: str, user_input: str, max_new_tokens: int = 1024):
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    full_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return full_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--skip_ppl", action="store_true")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    set_seed(config["training"]["seed"])

    # 2. Load Model & Adapter (Dùng Transformers thuần để né lỗi Unsloth)
    model_name = config["model"]["name"]
    checkpoint_dir = args.checkpoint_dir or config["output"]["output_dir"]
    
    print(f"\n[STABLE EVAL] Loading Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Xác định checkpoint thực tế
    actual_checkpoint = None
    if os.path.exists(checkpoint_dir):
        if any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in ["adapter_model.safetensors", "adapter_model.bin"]):
            actual_checkpoint = checkpoint_dir
        else:
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                actual_checkpoint = max([os.path.join(checkpoint_dir, d) for d in checkpoints], key=lambda x: int(x.split("-")[-1]))

    if actual_checkpoint:
        print(f"[EVAL] Loading PiSSA Adapter from: {actual_checkpoint}")
        model = PeftModel.from_pretrained(model, actual_checkpoint)
    
    model.eval()
    tokenizer.padding_side = "left"
    log_vram_usage("Model & Adapter loaded")

    # 3. Load & Filter Data
    test_file = config["data"]["test_file"]
    ds = load_dataset("json", data_files=test_file, split="train")
    
    target_task = args.task_type.lower() if args.task_type else None
    if target_task:
        short_name = target_task.replace("task", "").strip()
        task_data = ds.filter(lambda x: str(x.get("task_type", "")).lower() in [target_task, short_name, f"task {short_name}"])
    else:
        task_data = ds

    if args.num_samples > 0:
        task_data = task_data.select(range(min(args.num_samples, len(task_data))))

    print(f"[EVAL] Processing {len(task_data)} samples...")
    if len(task_data) == 0:
        print("[ERROR] Không tìm thấy dữ liệu khớp với task_type!")
        return

    # 4. Inference
    predictions, references, detailed_results = [], [], []

    for sample in tqdm(task_data, desc="Inference"):
        t_type = str(sample.get("task_type", "task3")).lower()
        clean_t_type = "task" + t_type.replace("task", "").strip()
        s_prompt = SYSTEM_PROMPTS.get(clean_t_type, SYSTEM_PROMPTS["task3"])
        
        user_in = sample.get("user", sample.get("instruction", ""))
        ref = sample.get("assistant", sample.get("output", ""))

        full_output = generate_response(model, tokenizer, s_prompt, user_in, args.max_new_tokens)
        
        final_ans = _extract_answer_after_think(full_output)
        think_text = ""
        if "</think>" in full_output:
            think_text = full_output.split("</think>")[0].replace("<think>", "").strip()

        predictions.append(final_ans)
        references.append(ref)
        
        detailed_results.append({
            "id": sample.get("id", "N/A"),
            "task": t_type,
            "input": user_in,
            "reference": ref,
            "thinking": think_text,
            "model_answer": final_ans
        })

    # 5. Lưu kết quả
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    suffix = f"_{target_task}" if target_task else ""
    
    d_path = os.path.join(results_dir, f"{peft_method}{suffix}_detailed.json")
    with open(d_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    # 6. Tính Metrics
    print("\n[EVAL] Computing metrics summary...")
    results = compute_all_metrics(
        model=model if not args.skip_ppl else None,
        tokenizer=tokenizer if not args.skip_ppl else None,
        predictions=predictions,
        references=references,
        task_type=target_task,
        compute_ppl=not args.skip_ppl
    )
    
    results.update({"method": peft_method, "num_samples": len(task_data)})
    
    s_path = os.path.join(results_dir, f"{peft_method}{suffix}_summary.json")
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*40}")
    print(f"✅ ĐÁNH GIÁ HOÀN TẤT ({peft_method.upper()})")
    print(f"   Chi tiết: {d_path}")
    print(f"   Tổng kết: {s_path}")
    for k, v in results.items():
        if isinstance(v, (float, int)): print(f"   {k:>20}: {v:.4f}" if isinstance(v, float) else f"   {k:>20}: {v}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()
