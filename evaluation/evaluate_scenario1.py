"""
Scenario 1: Generalization - Evaluation Script
==============================================
Loads the trained PiSSA adapter and evaluates accuracy on uitnlp/ViANLI test set.
"""

import os
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel
from training.preprocess_vianli import map_label
import yaml
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_scenario1(config_path, checkpoint_path=None):
    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    
    checkpoint_dir = checkpoint_path or output_cfg["output_dir"]
    
    print(f"\n[INFO] Loading model and adapter from: {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_dir,
        max_seq_length = model_cfg["max_seq_length"],
        load_in_4bit = model_cfg["load_in_4bit"],
        dtype = None,
    )
    FastLanguageModel.for_inference(model)

    # 2. Load test dataset
    print(f"[INFO] Loading dataset: {data_cfg['dataset_name']}")
    dataset = load_dataset(data_cfg["dataset_name"])
    test_ds = dataset["test"]
    
    # 3. Inference loop
    correct = 0
    total = len(test_ds)
    
    print(f"[INFO] Evaluating {total} samples...")
    
    results = []
    
    for i in tqdm(range(total)):
        example = test_ds[i]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label_id = example["label"]
        true_label = map_label(label_id)
        
        # Format prompt (User part only)
        prompt = f"User: \"Câu 1: {premise}. Câu 2: {hypothesis}. Mối quan hệ giữa hai câu là gì?\"\nAssistant: \""
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean response (extract first word/label)
        pred_label = response.strip().split('"')[0].strip().lower()
        
        if pred_label == true_label:
            correct += 1
            
        results.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "true": true_label,
            "pred": pred_label
        })

    accuracy = correct / total
    print(f"\n[RESULTS] Scenario 1 Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save results
    eval_results_path = os.path.join(output_cfg["results_dir"], "scenario1_eval_results.json")
    os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
    import json
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Detailed results saved to: {eval_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/scenario1_pissa.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to adapter checkpoint")
    args = parser.parse_args()
    
    evaluate_scenario1(args.config, args.checkpoint)
