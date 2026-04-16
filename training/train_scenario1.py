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
from peft import LoraConfig, TaskType
from training.preprocess_vianli import get_vianli_formatter
import yaml
import argparse

# Add project root to path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    )

    # 2. Apply PEFT (Bypassing Unsloth validation while keeping optimizations)
    is_pissa = peft_cfg.get("method") == "pissa" or peft_cfg.get("init_lora_weights") == "pissa"
    
    print(f"[INFO] Applying PEFT (method={'PiSSA' if is_pissa else 'LoRA'}, r={peft_cfg['r']}, alpha={peft_cfg['lora_alpha']})")

    # Create LoraConfig directly to support PiSSA properly
    lora_config = LoraConfig(
        r = peft_cfg["r"],
        target_modules = peft_cfg["target_modules"],
        lora_alpha = peft_cfg["lora_alpha"],
        lora_dropout = peft_cfg["lora_dropout"],
        bias = peft_cfg["bias"],
        task_type = TaskType.CAUSAL_LM,
        init_lora_weights = peft_cfg["init_lora_weights"], # Now "pissa" is safe
    )
    
    # 1. Use PEFT to get the model structure
    from peft import get_peft_model as peft_get_peft_model
    model = peft_get_peft_model(model, lora_config)
    
    # 2. Apply Unsloth's performance patches to the PEFT model
    model = FastLanguageModel.patch_peft_model(model, use_gradient_checkpointing = "unsloth")

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
        eval_strategy = train_cfg["eval_strategy"],
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
    # 6. Train or Resume
    print("[INFO] Checking for checkpoints to resume...")
    resume_from_checkpoint = None
    if os.path.exists(output_cfg["output_dir"]):
        checkpoints = [d for d in os.listdir(output_cfg["output_dir"]) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(output_cfg["output_dir"], latest_checkpoint)
            print(f"[INFO] Found checkpoint: {resume_from_checkpoint}. Resuming/Loading...")

    print("[INFO] Starting training...")
    trainer.train(resume_from_checkpoint = resume_from_checkpoint)

    # 7. Save Model FIRST (Safe approach)
    print(f"[INFO] Saving model to {output_cfg['output_dir']}")
    model.save_pretrained(output_cfg["output_dir"])
    tokenizer.save_pretrained(output_cfg["output_dir"])

    # 8. Evaluate on Test Set
    print("[INFO] Preparing Test Set for evaluation...")
    # SFTTrainer evaluation needs input_ids and labels. Let's tokenize manually to be safe.
    def tokenize_test(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_cfg["max_seq_length"],
            padding=False,
        )
        # For evaluation, labels are usually the same as input_ids
        outputs["labels"] = [ids.copy() for ids in outputs["input_ids"]]
        return outputs

    tokenized_test_ds = test_ds.map(
        tokenize_test, 
        batched=True, 
        remove_columns=test_ds.column_names,
        desc="Tokenizing test set"
    )

    print("[INFO] Evaluating on Test Set...")
    metrics = trainer.evaluate(eval_dataset=tokenized_test_ds)
    print(f"\n[RESULTS] Test Metrics: {metrics}")
    
    # Save accuracy specifically if possible
    accuracy = metrics.get("eval_accuracy", metrics.get("eval_loss", "Check metrics above"))
    print(f"[RESULTS] Primary Metric (Loss/Acc): {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/scenario1_pissa.yaml")
    args = parser.parse_args()
    
    train_scenario1(args.config)
