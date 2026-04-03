"""
Trainer Utilities
=================
Hàm hỗ trợ cho quá trình huấn luyện: load model, apply PEFT, format prompts.
"""

import os
import sys
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

        import math
        import torch
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        r_local = peft_cfg["r_local"]
        r_intra = peft_cfg["r_intra"]
        r_inter = peft_cfg["r_inter"]
        total_r = r_local + r_intra + r_inter
        share_mode = peft_cfg.get("share_mode", "slice")

        # --- Step 1: Apply standard LoRA with r = r_local ---
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=peft_cfg.get("use_gradient_checkpointing", True),
        )

        lora_config = LoraConfig(
            r=r_local,
            lora_alpha=peft_cfg["lora_alpha"],
            target_modules=peft_cfg["target_modules"],
            lora_dropout=peft_cfg["lora_dropout"],
            bias=peft_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # --- Step 2: Create shared modules (intra per-layer + inter global) ---
        base_model = model.base_model
        inner = getattr(base_model, "model", base_model)

        # Find decoder layers
        decoder = None
        for attr in ["model", "transformer", "gpt_neox"]:
            cand = getattr(inner, attr, None)
            if cand and hasattr(cand, "layers"):
                decoder = cand
                break
        if decoder is None:
            for _, module in inner.named_modules():
                if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                    decoder = module
                    break

        if decoder is None:
            print("[BSLoRA] ⚠️ Cannot find decoder layers, using standard LoRA only")
            model.print_trainable_parameters()
            return model

        num_layers = len(decoder.layers)
        print(f"[BSLoRA] Found {num_layers} decoder layers")

        # Find max in/out features
        max_in, max_out = 0, 0
        for _, module in model.named_modules():
            if hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
                max_in = max(max_in, module.base_layer.in_features)
                max_out = max(max_out, module.base_layer.out_features)

        share_in = max_in if share_mode == "slice" else peft_cfg.get("kron_share_size", 256)
        share_out = max_out if share_mode == "slice" else share_in
        print(f"[BSLoRA] Share dims: in={share_in}, out={share_out}, mode={share_mode}")

        # Intra: 1 shared (A, B) per layer
        intra_A = nn.ModuleList()
        intra_B = nn.ModuleList()
        for _ in range(num_layers):
            a = nn.Linear(share_in, r_intra, bias=False)
            b = nn.Linear(r_intra, share_out, bias=False)
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
            intra_A.append(a)
            intra_B.append(b)

        # Inter: 1 shared (A, B) global
        inter_A_mod = nn.Linear(share_in, r_inter, bias=False)
        inter_B_mod = nn.Linear(r_inter, share_out, bias=False)
        nn.init.kaiming_uniform_(inter_A_mod.weight, a=math.sqrt(5))
        nn.init.zeros_(inter_B_mod.weight)

        # Move to device
        device = next(model.parameters()).device
        intra_A = intra_A.to(device)
        intra_B = intra_B.to(device)
        inter_A_mod = inter_A_mod.to(device)
        inter_B_mod = inter_B_mod.to(device)

        # Attach to model so optimizer can find them
        model.bslora_intra_A = intra_A
        model.bslora_intra_B = intra_B
        model.bslora_inter_A = inter_A_mod
        model.bslora_inter_B = inter_B_mod

        # --- Step 3: Register forward hooks ---
        # Scaling theo chuẩn LoRA: alpha / rank (cho phần intra và inter)
        bslora_scaling = peft_cfg["lora_alpha"] / (r_intra + r_inter)

        def _make_hook(layer_idx, in_f, out_f):
            def hook(module, inp, out):
                x = inp[0]
                a_in = intra_A[layer_idx]
                b_in = intra_B[layer_idx]
                x_c = x.to(a_in.weight.dtype)

                if share_mode == "slice":
                    wA = a_in.weight.T
                    wB = b_in.weight.T
                    bA = (wA.shape[0] - in_f) // 2
                    bB = (wB.shape[1] - out_f) // 2
                    intra_o = x_c @ wA[bA:bA+in_f, :] @ wB[:, bB:bB+out_f]

                    wAg = inter_A_mod.weight.T
                    wBg = inter_B_mod.weight.T
                    bAg = (wAg.shape[0] - in_f) // 2
                    bBg = (wBg.shape[1] - out_f) // 2
                    inter_o = x_c @ wAg[bAg:bAg+in_f, :] @ wBg[:, bBg:bBg+out_f]
                else:
                    intra_o = b_in(a_in(x_c))
                    inter_o = inter_B_mod(inter_A_mod(x_c))

                return out + (intra_o + inter_o).to(out.dtype) * bslora_scaling
            return hook

        hook_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
                layer_idx = None
                for p in name.split("."):
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                if layer_idx is not None and layer_idx < num_layers:
                    in_f = module.base_layer.in_features
                    out_f = module.base_layer.out_features
                    module.register_forward_hook(_make_hook(layer_idx, in_f, out_f))
                    hook_count += 1

        print(f"[BSLoRA] ✅ Registered {hook_count} forward hooks")

        # Count params (BSLoRA modules đã nằm trong model.parameters() rồi, không cần cộng thêm)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[PEFT] Applied: BSLoRA (mode={share_mode})")
        print(f"[PEFT] Rank: local={r_local}, intra={r_intra}, inter={r_inter} (total={total_r})")
        print(f"[PEFT] Alpha: {peft_cfg['lora_alpha']}, Scaling (intra+inter): {bslora_scaling:.4f}")
        print(f"[PEFT] Trainable: {trainable:,} / {total_params:,} "
              f"({100*trainable/total_params:.2f}%)")

        return model

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
