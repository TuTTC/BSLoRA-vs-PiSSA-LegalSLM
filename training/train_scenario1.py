"""
Scenario 1: Generalization - Training Script
============================================
Model: Qwen/Qwen2.5-3B
Dataset: uitnlp/ViANLI
Technique: PiSSA (r=32, alpha=64)
Optimization: Unsloth
"""

import os
import sys
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from training.preprocess_vianli import get_vianli_formatter
import yaml
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_scenario1(config_path):
    # 1. Load config
    config = load_yaml_config(config_path)
    model_cfg = config["model"]
    peft_cfg = config["peft"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    
    print(f"\n[INFO] Loading model: {model_cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_cfg["model_name"],
        max_seq_length = model_cfg["max_seq_length"],
        load_in_4bit = model_cfg["load_in_4bit"],
        dtype = None, # Auto detect
    )

    # 2. Apply PiSSA
    print(f"[INFO] Applying PiSSA (r={peft_cfg['r']}, alpha={peft_cfg['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r = peft_cfg["r"],
        target_modules = peft_cfg["target_modules"],
        lora_alpha = peft_cfg["lora_alpha"],
        lora_dropout = peft_cfg["lora_dropout"],
        bias = peft_cfg["bias"],
        use_gradient_checkpointing = "unsloth",
        random_state = train_cfg["seed"],
        init_lora_weights = peft_cfg["init_lora_weights"], # "pissa"
    )

    # 3. Load and Preprocess Dataset
    print(f"[INFO] Loading dataset: {data_cfg['dataset_name']}")
    dataset = load_dataset(data_cfg["dataset_name"])
    
    formatter = get_vianli_formatter(tokenizer, data_cfg["prompt_template"])
    
    train_ds = dataset["train"].map(formatter, batched=True)
    val_ds = dataset["validation"].map(formatter, batched=True)
    test_ds = dataset["test"].map(formatter, batched=True)

    # 4. Setup Training Arguments
    training_args = TrainingArguments(
        learning_rate = float(train_cfg["learning_rate"]),
        lr_scheduler_type = "linear",
        per_device_train_batch_size = train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps = train_cfg["gradient_accumulation_steps"],
        num_train_epochs = train_cfg["num_train_epochs"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = train_cfg["logging_steps"],
        optim = train_cfg["optim"],
        weight_decay = train_cfg["weight_decay"],
        warmup_steps = train_cfg["warmup_steps"],
        output_dir = output_cfg["output_dir"],
        seed = train_cfg["seed"],
        evaluation_strategy = train_cfg["eval_strategy"],
        eval_steps = train_cfg["eval_steps"],
        report_to = "wandb" if config.get("wandb") else "none",
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        dataset_text_field = "text",
        max_seq_length = model_cfg["max_seq_length"],
        dataset_num_proc = 2,
        packing = False, # Can be True for faster training with short sequences
        args = training_args,
    )

    # 6. Train
    print("[INFO] Starting training...")
    trainer.train()

    # 7. Evaluate on Test Set
    print("[INFO] Evaluating on Test Set...")
    metrics = trainer.evaluate(test_ds)
    print(f"\n[RESULTS] Test Metrics: {metrics}")
    
    # Save accuracy specifically if possible
    accuracy = metrics.get("eval_accuracy", "N/A")
    print(f"[RESULTS] Test Accuracy: {accuracy}")

    # 8. Save Model
    print(f"[INFO] Saving model to {output_cfg['output_dir']}")
    model.save_pretrained(output_cfg["output_dir"])
    tokenizer.save_pretrained(output_cfg["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/scenario1_pissa.yaml")
    args = parser.parse_args()
    
    train_scenario1(args.config)
