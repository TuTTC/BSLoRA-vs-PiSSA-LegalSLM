"""
Trainer Utilities
=================
Hàm hỗ trợ cho quá trình huấn luyện: load model, apply PEFT, format prompts.
"""
"""
Trainer Utilities (Bản chuẩn lách Unsloth - Hỗ trợ Eval)
=================
"""

import yaml
import copy
import torch
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(base_path: str, peft_path: str) -> Dict[str, Any]:
    with open(base_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    with open(peft_path, "r", encoding="utf-8") as f:
        peft_config = yaml.safe_load(f)

    config = copy.deepcopy(base_config)
    for key, value in peft_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    return config

def load_model(config: Dict[str, Any], adapter_path=None, force_transformers=False, **kwargs):
    # NẾU LÀ ĐÁNH GIÁ (EVAL): Dùng Transformers thuần để né lỗi ép xung của Unsloth
    if force_transformers:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        print("\n[EVAL] Đang bỏ qua Unsloth, dùng Transformers thuần để an toàn tuyệt đối...\n")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        return model, tokenizer

    # NẾU LÀ TRAIN: Vẫn dùng Unsloth để load nhanh
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported

    dtype = config["model"].get("dtype")
    if dtype is None or dtype == "null":
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16

    model_name_or_path = adapter_path if adapter_path else config["model"]["name"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=config.get("model", {}).get("max_seq_length", 2048),
        dtype=dtype,
        load_in_4bit=config.get("model", {}).get("load_in_4bit", False),
    )

    print(f"[MODEL] Loaded: {model_name_or_path}")
    print(f"[MODEL] 4-bit quantization: {config.get('model', {}).get('load_in_4bit', False)}")

    return model, tokenizer

def apply_peft(model, config: Dict[str, Any]):
    # SỬ DỤNG PEFT NGUYÊN BẢN ĐỂ VƯỢT MẶT UNSLOTH
    from peft import get_peft_model, LoraConfig

    peft_cfg = config["peft"]
    
    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg["bias"],
        task_type="CAUSAL_LM",
        init_lora_weights=peft_cfg.get("init_lora_weights", "pissa")
    )
    
    model = get_peft_model(model, lora_config)
    
    print(f"\n[PEFT] ĐÃ LÁCH QUA UNSLOTH! Khởi tạo thành công với init_lora_weights='{peft_cfg.get('init_lora_weights')}'")
    
    # In số lượng trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PEFT] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)\n")

    return model

def get_training_args(config: Dict[str, Any]):
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported

    train_cfg = config["training"]
    output_cfg = config["output"]
    model_cfg = config.get("model", {})

    fp16 = train_cfg.get("fp16", False)
    bf16 = train_cfg.get("bf16", False)
    if not fp16 and not bf16:
        bf16 = is_bfloat16_supported()
        fp16 = not bf16

    args = SFTConfig(
        output_dir=output_cfg["output_dir"],
        logging_dir=output_cfg["logging_dir"],
        num_train_epochs=train_cfg.get("num_epochs", 1),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_steps=train_cfg.get("max_steps", -1),
        fp16=fp16,
        bf16=bf16,
        optim=train_cfg.get("optim", "adamw_8bit"),
        seed=train_cfg.get("seed", 42),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=output_cfg["save_strategy"],
        save_total_limit=output_cfg["save_total_limit"],
        report_to="wandb",
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        packing=False,
        dataset_text_field="text",
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    return args

def format_prompts(examples, tokenizer, template: str):
    texts = []

    has_chatml = "system" in examples and "user" in examples

    if has_chatml:
        for system, user, assistant in zip(
            examples["system"], examples["user"], examples["assistant"]
        ):
            text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant}<|im_end|>"
            )
            text = text + tokenizer.eos_token
            texts.append(text)
    else:
        for instruction, input_text, output_text in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            text = template.format(
                instruction=instruction,
                input=input_text,
                output=output_text,
            )
            text = text + tokenizer.eos_token
            texts.append(text)

    return texts