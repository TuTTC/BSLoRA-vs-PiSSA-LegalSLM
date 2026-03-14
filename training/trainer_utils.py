"""
Trainer Utilities
=================
Hàm hỗ trợ cho quá trình huấn luyện: load model, apply PEFT, format prompts.
"""

import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(base_path: str, peft_path: str) -> Dict[str, Any]:
    """
    Load và merge config từ base_config + peft_config.
    
    Args:
        base_path: Đường dẫn đến base_config.yaml
        peft_path: Đường dẫn đến lora/dora/pissa_config.yaml
    
    Returns:
        Dict chứa config đã merge
    """
    with open(base_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    with open(peft_path, "r", encoding="utf-8") as f:
        peft_config = yaml.safe_load(f)

    # Deep merge: peft_config overrides base_config
    config = copy.deepcopy(base_config)
    for key, value in peft_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    return config


def load_model(config: Dict[str, Any]):
    """
    Load model qua Unsloth với quantization 4-bit.
    
    Returns:
        (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=config["model"]["dtype"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    print(f"[MODEL] Loaded: {config['model']['name']}")
    print(f"[MODEL] Max seq length: {config['model']['max_seq_length']}")
    print(f"[MODEL] 4-bit quantization: {config['model']['load_in_4bit']}")

    return model, tokenizer


def apply_peft(model, config: Dict[str, Any]):
    """
    Apply PEFT adapter (LoRA / DoRA / PiSSA) dựa trên config.
    
    - LoRA: use_dora=False, init_lora_weights=True
    - DoRA: use_dora=True, init_lora_weights=True  
    - PiSSA: use_dora=False, init_lora_weights="pissa"
    
    Returns:
        model with PEFT adapter applied
    """
    from unsloth import FastLanguageModel

    peft_cfg = config["peft"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg["bias"],
        use_gradient_checkpointing=peft_cfg["use_gradient_checkpointing"],
        use_rslora=peft_cfg.get("use_rslora", False),
        use_dora=peft_cfg.get("use_dora", False),
        init_lora_weights=peft_cfg.get("init_lora_weights", True),
    )

    method = peft_cfg["method"]
    print(f"[PEFT] Applied method: {method.upper()}")
    print(f"[PEFT] Rank: {peft_cfg['r']}, Alpha: {peft_cfg['lora_alpha']}")
    print(f"[PEFT] Target modules: {peft_cfg['target_modules']}")

    # In số lượng trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PEFT] Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model


def get_training_args(config: Dict[str, Any]):
    """
    Tạo TrainingArguments từ config dict.
    
    Returns:
        transformers.TrainingArguments
    """
    from transformers import TrainingArguments

    train_cfg = config["training"]
    output_cfg = config["output"]

    args = TrainingArguments(
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
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=output_cfg["save_strategy"],
        save_total_limit=output_cfg["save_total_limit"],
        report_to="wandb",
    )

    return args


def format_prompts(examples, tokenizer, template: str):
    """
    Format dữ liệu sang prompt template cho SFTTrainer.

    Hỗ trợ cả ChatML format (system/user/assistant) và 
    Alpaca format (instruction/input/output) cho backward compatibility.

    Args:
        examples: Dataset batch
        tokenizer: Tokenizer
        template: Prompt template string

    Returns:
        List[str] formatted texts
    """
    texts = []

    # Kiểm tra xem dữ liệu có trường ChatML không
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
        # Fallback: Alpaca format
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
