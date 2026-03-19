import os
import sys
import json
import argparse
import torch
import transformers
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Environment Hack: Prepend BSLoRA directory to sys.path
# This ensures "import peft" loads the local version with ShareLora support.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BSLORA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "BSLoRA")
sys.path.insert(0, BSLORA_DIR)
sys.path.insert(0, PROJECT_ROOT)

import peft
from peft import get_peft_model, ShareLoraConfig, prepare_model_for_kbit_training
from training.trainer_utils import load_config, load_model, format_prompts
from utils.logger import setup_wandb, log_config, log_vram_to_wandb
from utils.helpers import set_seed, get_device_info, log_vram_usage, VRAMTracker

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with BSLoRA")
    parser.add_argument("--base_config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--peft_config", type=str, default="configs/bslora_config.yaml")
    parser.add_argument("--share_mode", type=str, default=None, help="Override share_mode (slice, gate, kron)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.base_config, args.peft_config)
    if args.share_mode:
        config["peft"]["share_mode"] = args.share_mode
        # Update output dir based on share_mode if overridden
        config["output"]["output_dir"] = f"outputs/checkpoints/bslora_{args.share_mode}"
        config["output"]["logging_dir"] = f"outputs/logs/bslora_{args.share_mode}"
    
    peft_method = config["peft"]["method"]
    share_mode = config["peft"].get("share_mode", "unknown")
    
    print(f"\n{'='*60}")
    print(f"  PEFT Method: {peft_method.upper()} (Mode: {share_mode})")
    print(f"{'='*60}\n")

    # Setup
    set_seed(config["training"]["seed"])
    get_device_info()
    wandb_run = setup_wandb(config)
    log_config(config)

    vram_tracker = VRAMTracker(
        method=f"bslora_{share_mode}",
        output_dir=os.path.join(config["output"].get("results_dir", "outputs/results"), "vram_logs"),
    )

    # 3. Load model (Standard 4-bit loading via bitsandbytes)
    # Note: We use the helper from trainer_utils which uses Unsloth for loading
    # but we will wrap it with standard PEFT instead of Unsloth's PEFT.
    with vram_tracker.track("model_loading"):
        # We still use load_model from trainer_utils because it handles quantization and Flash Attention well
        model, tokenizer = load_model(config)

    # 4. Apply BSLoRA PEFT (NOT Unsloth's apply_peft)
    with vram_tracker.track("peft_apply"):
        peft_cfg = config["peft"]
        
        # Prepare for k-bit training (Gradient Checkpointing happens here or in SFTTrainer)
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=peft_cfg.get("use_gradient_checkpointing", True)
        )
        
        share_config = ShareLoraConfig(
            r_local=peft_cfg["r_local"],
            r_intra=peft_cfg["r_intra"],
            r_inter=peft_cfg["r_inter"],
            share_mode=peft_cfg["share_mode"],
            kron_share_size=peft_cfg.get("kron_share_size", 256),
            gate_share_size=peft_cfg.get("gate_share_size", 1024),
            lora_alpha=peft_cfg["lora_alpha"],
            target_modules=peft_cfg["target_modules"],
            lora_dropout=peft_cfg["lora_dropout"],
            bias=peft_cfg["bias"],
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, share_config)
        
        if peft_cfg.get("use_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            
        print(f"[PEFT] Applied BSLoRA ({share_mode})")
        model.print_trainable_parameters()

    # 5. Load dataset
    data_cfg = config["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["val_file"],
        },
    )
    
    print("[DATA] Formatting datasets...")
    prompt_template = config["data"]["prompt_template"]
    train_texts = format_prompts(dataset["train"][:], tokenizer, prompt_template)
    val_texts = format_prompts(dataset["validation"][:], tokenizer, prompt_template)
    
    dataset["train"] = dataset["train"].remove_columns([col for col in dataset["train"].column_names if col != "text"])
    dataset["validation"] = dataset["validation"].remove_columns([col for col in dataset["validation"].column_names if col != "text"])
    
    dataset["train"] = dataset["train"].add_column("text", train_texts)
    dataset["validation"] = dataset["validation"].add_column("text", val_texts)

    # 6. Training Arguments
    train_cfg = config["training"]
    output_cfg = config["output"]
    
    training_args = SFTConfig(
        output_dir=output_cfg["output_dir"],
        logging_dir=output_cfg["logging_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        max_steps=train_cfg["max_steps"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=output_cfg.get("save_strategy", "steps"),
        save_steps=output_cfg.get("save_steps", 100),
        save_total_limit=output_cfg.get("save_total_limit", 3),
        eval_strategy=output_cfg.get("eval_strategy", "steps"),
        eval_steps=output_cfg.get("eval_steps", 100),
        report_to="wandb",
        max_seq_length=config["model"].get("max_seq_length", 1024),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # 7. Train
    print(f"\n[TRAIN] Starting training with BSLoRA ({share_mode})...")
    with vram_tracker.track("training"):
        trainer_stats = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"[TRAIN] Training completed!")
    
    # Save adapter
    model.save_pretrained(output_cfg["output_dir"])
    tokenizer.save_pretrained(output_cfg["output_dir"])
    
    # Save metrics
    metrics_path = os.path.join(output_cfg["results_dir"], f"bslora_{share_mode}_train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
        
    vram_tracker.save()
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()
