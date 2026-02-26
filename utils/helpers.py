"""
Helper Utilities
================
Seed, device info, VRAM tracking, config loading.
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any


def set_seed(seed: int = 3407) -> None:
    """Đặt seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Set random seed: {seed}")


def get_device_info() -> Dict[str, Any]:
    """Lấy thông tin thiết bị (GPU/CPU)."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
            info[f"gpu_{i}"] = {
                "name": gpu_name,
                "total_memory_gb": round(gpu_mem, 2),
            }
            print(f"[DEVICE] GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("[DEVICE] No GPU available. Using CPU.")

    return info


def log_vram_usage(stage: str = "") -> None:
    """Log VRAM usage hiện tại."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

    prefix = f"[VRAM] {stage}" if stage else "[VRAM]"
    print(f"{prefix} - Allocated: {allocated:.2f} GB | "
          f"Reserved: {reserved:.2f} GB | "
          f"Peak: {max_allocated:.2f} GB")


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load config từ file YAML."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model) -> Dict[str, int]:
    """Đếm tham số của model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / total, 4) if total > 0 else 0,
    }


def ensure_dirs(*dirs: str) -> None:
    """Tạo các thư mục nếu chưa tồn tại."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
