"""
Training Script (Multi-task + VRAM Tracking + WandB Tags)
==========================================================
Script huấn luyện chính sử dụng Unsloth + TRL SFTTrainer.
Tích hợp VRAMTracker và WandB tag-based logging.

Usage:
    python training/train.py --peft_config configs/lora_config.yaml
    python training/train.py --peft_config configs/dora_config.yaml
    python training/train.py --peft_config configs/pissa_config.yaml
"""

import os
import sys
import json
import argparse

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
# from trl import SFTTrainer
from trl import SFTTrainer, SFTConfig

from training.trainer_utils import (
    load_config,
    load_model,
    apply_peft,
    get_training_args,
    format_prompts,
)
from utils.logger import setup_wandb, log_config, log_vram_to_wandb
from utils.helpers import set_seed, get_device_info, log_vram_usage, VRAMTracker


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA/DoRA/PiSSA")
    parser.add_argument(
        "--base_config", type=str, default="configs/base_config.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--peft_config", type=str, required=True,
        help="Path to PEFT config YAML (lora/dora/pissa)",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. Load config
    # =========================================================================
    config = load_config(args.base_config, args.peft_config)
    peft_method = config["peft"]["method"]
    print(f"\n{'='*60}")
    print(f"  PEFT Method: {peft_method.upper()}")
    print(f"{'='*60}\n")

    # =========================================================================
    # 2. Setup
    # =========================================================================
    set_seed(config["training"]["seed"])
    get_device_info()
    wandb_run = setup_wandb(config)
    log_config(config)

    # Khởi tạo VRAM Tracker
    vram_tracker = VRAMTracker(
        method=peft_method,
        output_dir=os.path.join(config["output"].get("results_dir", "outputs/results"), "vram_logs"),
    )

    # =========================================================================
    # 3. Load model & apply PEFT (tracked)
    # =========================================================================
    with vram_tracker.track("model_loading"):
        model, tokenizer = load_model(config)

    with vram_tracker.track("peft_apply"):
        model = apply_peft(model, config)

    log_vram_usage("After loading model + PEFT")

    # =========================================================================
    # 4. Load dataset
    # =========================================================================
    data_cfg = config["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "validation": data_cfg["val_file"],
        },
    )
    print(f"[DATA] Train samples: {len(dataset['train'])}")
    print(f"[DATA] Val samples:   {len(dataset['validation'])}")

    # =========================================================================
    # 5. Setup trainer
    # =========================================================================
    training_args = get_training_args(config)
    prompt_template = config["data"]["prompt_template"]

    def _format_batch(examples):
        return {"text": format_prompts(examples, tokenizer, prompt_template)}

    print("[DATA] Formatting datasets...")
    dataset["train"] = dataset["train"].map(_format_batch, batched=True)
    dataset["validation"] = dataset["validation"].map(_format_batch, batched=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    # =========================================================================
    # 6. Train (tracked)
    # =========================================================================
    print(f"\n[TRAIN] Starting training with {peft_method.upper()}...")

    with vram_tracker.track("training"):
        trainer_stats = trainer.train()

    log_vram_usage("After training")
    print(f"[TRAIN] Training completed!")
    print(f"[TRAIN] Total training time: {trainer_stats.metrics['train_runtime']:.2f}s")

    # =========================================================================
    # 7. Save adapter
    # =========================================================================
    save_path = config["output"]["output_dir"]
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[SAVE] Adapter saved to: {save_path}")

    # Lưu training metrics
    metrics_path = os.path.join(
        config["output"]["results_dir"], f"{peft_method}_train_metrics.json"
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    print(f"[SAVE] Training metrics saved to: {metrics_path}")

    # =========================================================================
    # 8. Save VRAM tracking + log to WandB
    # =========================================================================
    vram_csv_path = vram_tracker.save()
    vram_summary = vram_tracker.summary()

    # Log VRAM peak to WandB
    if wandb_run is not None:
        log_vram_to_wandb(wandb_run, vram_summary)
        wandb_run.finish()
        print("[WANDB] Run finished.")


if __name__ == "__main__":
    main()
