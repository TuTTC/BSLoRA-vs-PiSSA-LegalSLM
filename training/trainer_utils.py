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


def load_model(config: Dict[str, Any], adapter_path: Optional[str] = None, force_transformers: bool = False):
    """
    Load model qua Unsloth hoặc standard Transformers.
    
    Args:
        config: Config dict
        adapter_path: Đường dẫn đến adapter (nếu muốn load model kèm adapter)
        force_transformers: Bắt buộc dùng standard transformers (dùng cho eval ổn định)
    
    Returns:
        (model, tokenizer)
    """
    from unsloth import is_bfloat16_supported
    import torch

    dtype = config["model"].get("dtype")
    if dtype is None or dtype == "null":
        dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16

    model_name = adapter_path if adapter_path else config["model"]["name"]

    if force_transformers:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print(f"[MODEL] Loading via standard Transformers (Stable Mode): {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if adapter_path else config["model"]["name"])
        
        bnb_config = None
        if config["model"]["load_in_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
        )

        if adapter_path:
            print(f"[PEFT] Loading adapter weights from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
    else:
        from unsloth import FastLanguageModel
        
        # Kiểm tra Flash Attention 2
        attn_impl = "eager"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[MODEL] Flash Attention 2 detected → enabling FA2 for faster training!")
        except ImportError:
            print("[MODEL] flash-attn not installed → using default attention (slower).")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=config["model"]["max_seq_length"],
            dtype=dtype,
            load_in_4bit=config["model"]["load_in_4bit"],
            attn_implementation=attn_impl,
        )

    print(f"[MODEL] Final model: {model_name}")
    print(f"[MODEL] Max seq length: {config['model'].get('max_seq_length', 'N/A')}")
    print(f"[MODEL] 4-bit quantization: {config['model']['load_in_4bit']}")

    return model, tokenizer


def apply_peft(model, config: Dict[str, Any], force_transformers: bool = False):
    """
    Apply PEFT adapter (LoRA / DoRA / PiSSA) dựa trên config.
    
    - LoRA: use_dora=False, init_lora_weights=True
    - DoRA: use_dora=True, init_lora_weights=True  
    - PiSSA: use_dora=False, init_lora_weights="pissa"
    
    Args:
        model: Base model
        config: Config dict
        force_transformers: Dùng standard peft instead of Unsloth
        
    Returns:
        model with PEFT adapter applied
    """
    peft_cfg = config["peft"]
    method = peft_cfg.get("method", "lora").lower()
    init_lora_weights = peft_cfg.get("init_lora_weights", True)

    if method == "fft":
        print("[PEFT] Bypassing PEFT -> Full Fine-Tuning (FFT) Mode Enabled")
        # Ensure all parameters are trainable for FFT
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[FFT] Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        return model

    if method == "bslora":
        print("[PEFT] 🚀 Applying Bi-Share LoRA (BSLoRA) Mode...")
        # TODO: Chèn logic can thiệp trọng số Bi-Share của bạn vào phía dưới.
        # Tạm thời map "bslora" về "lora" để build khung wrapper chuản cho Unsloth
        method = "lora"
        peft_cfg["method"] = "lora"

    if method == "pissa":
        init_lora_weights = "pissa"

    if force_transformers:
        from peft import get_peft_model, LoraConfig
        
        print(f"[PEFT] Applying standard PEFT (Stable Mode): {method.upper()}")
        
        lora_config = LoraConfig(
            r=peft_cfg["r"],
            lora_alpha=peft_cfg["lora_alpha"],
            target_modules=peft_cfg["target_modules"],
            lora_dropout=peft_cfg["lora_dropout"],
            bias=peft_cfg["bias"],
            task_type="CAUSAL_LM",
            use_dora=peft_cfg.get("use_dora", False),
            init_lora_weights=init_lora_weights,
            use_rslora=peft_cfg.get("use_rslora", False),
        )
        model = get_peft_model(model, lora_config)
    else:
        from unsloth import FastLanguageModel

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
            init_lora_weights=init_lora_weights,
        )

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
    Tạo SFTConfig từ config dict.
    
    Returns:
        trl.SFTConfig
    """
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
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        max_steps=train_cfg["max_steps"],
        fp16=fp16,
        bf16=bf16,
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=output_cfg.get("save_strategy", "epoch"),
        save_steps=output_cfg.get("save_steps", 500),
        save_total_limit=output_cfg.get("save_total_limit", 3),
        eval_strategy=output_cfg.get("eval_strategy", "no"),
        eval_steps=output_cfg.get("eval_steps", None),
        report_to="wandb",
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        packing=False,
        dataset_text_field="text",
        dataset_num_proc=1,        # Ép xử lý dữ liệu trên 1 core duy nhất
        dataloader_num_workers=0,  # Tắt đa luồng khi load data vào GPU
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
