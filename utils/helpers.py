"""
Helper Utilities
================
Seed, device info, VRAM tracking, memory efficiency, config loading.
"""

import os
import csv
import random
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, List

import yaml
import torch
import numpy as np


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


# ============================================================================
# VRAM Tracking cho Paper (DoRA vs PiSSA comparison)
# ============================================================================
class VRAMTracker:
    """
    Context manager để theo dõi VRAM peak trong từng giai đoạn.

    Ghi nhận VRAM allocated, reserved, và peak cho mỗi stage,
    lưu kết quả ra CSV để so sánh giữa các phương pháp PEFT.

    Usage:
        tracker = VRAMTracker(method="dora", output_dir="outputs/vram_logs")
        with tracker.track("model_loading"):
            model, tokenizer = load_model(config)
        with tracker.track("training"):
            trainer.train()
        tracker.save()
        tracker.summary()
    """

    def __init__(self, method: str, output_dir: str = "outputs/vram_logs"):
        self.method = method
        self.output_dir = output_dir
        self.records: List[Dict[str, Any]] = []

    @contextmanager
    def track(self, stage: str):
        """Track VRAM usage trong một giai đoạn cụ thể."""
        if not torch.cuda.is_available():
            yield
            return

        # Reset peak stats trước khi bắt đầu
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        start_time = time.time()

        yield

        torch.cuda.synchronize()
        end_time = time.time()

        end_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        record = {
            "method": self.method,
            "stage": stage,
            "start_allocated_gb": round(start_allocated, 4),
            "end_allocated_gb": round(end_allocated, 4),
            "peak_allocated_gb": round(peak_allocated, 4),
            "reserved_gb": round(reserved, 4),
            "duration_s": round(end_time - start_time, 2),
            "timestamp": datetime.now().isoformat(),
        }
        self.records.append(record)

        print(
            f"[VRAM] {stage} | "
            f"Peak: {peak_allocated:.2f} GB | "
            f"Current: {end_allocated:.2f} GB | "
            f"Duration: {record['duration_s']:.1f}s"
        )

    def save(self) -> str:
        """Lưu kết quả VRAM tracking ra file CSV."""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, f"{self.method}_vram.csv")

        fieldnames = [
            "method", "stage", "start_allocated_gb", "end_allocated_gb",
            "peak_allocated_gb", "reserved_gb", "duration_s", "timestamp",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[VRAM] Saved tracking data to: {csv_path}")
        return csv_path

    def summary(self) -> Dict[str, Any]:
        """Trả về tóm tắt VRAM usage."""
        if not self.records:
            return {}

        overall_peak = max(r["peak_allocated_gb"] for r in self.records)
        total_duration = sum(r["duration_s"] for r in self.records)

        summary = {
            "method": self.method,
            "overall_peak_vram_gb": overall_peak,
            "total_duration_s": total_duration,
            "stages": {r["stage"]: r["peak_allocated_gb"] for r in self.records},
        }

        print(f"\n[VRAM] ===== {self.method.upper()} Summary =====")
        print(f"  Overall Peak VRAM: {overall_peak:.2f} GB")
        print(f"  Total Duration:    {total_duration:.1f} s")
        for stage, peak in summary["stages"].items():
            print(f"  {stage:>20}: {peak:.2f} GB peak")

        return summary


def check_memory_efficiency(
    vram_log_dir: str = "outputs/vram_logs",
) -> None:
    """
    So sánh VRAM efficiency giữa các phương pháp PEFT.
    Đọc tất cả CSV trong vram_log_dir và tạo bảng so sánh.

    Dùng cho Paper: chứng minh tính hiệu quả PEFT so với Full-parameter.
    """
    import glob

    csv_files = glob.glob(os.path.join(vram_log_dir, "*_vram.csv"))
    if not csv_files:
        print("[VRAM] No VRAM log files found.")
        return

    print(f"\n{'='*70}")
    print("  VRAM Efficiency Comparison (PEFT Methods)")
    print(f"{'='*70}")
    print(f"  {'Method':<12} {'Stage':<20} {'Peak VRAM (GB)':>15} {'Duration (s)':>13}")
    print(f"  {'-'*12} {'-'*20} {'-'*15} {'-'*13}")

    all_data = []
    for csv_path in sorted(csv_files):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_data.append(row)
                print(
                    f"  {row['method']:<12} {row['stage']:<20} "
                    f"{float(row['peak_allocated_gb']):>15.2f} "
                    f"{float(row['duration_s']):>13.1f}"
                )

    print(f"{'='*70}\n")


def plot_vram_comparison(
    vram_log_dir: str = "outputs/vram_logs",
    output_path: str = "outputs/results/vram_comparison.png",
) -> None:
    """
    Tạo biểu đồ so sánh VRAM peak giữa các phương pháp PEFT.
    Biểu đồ bar chart cho từng stage, nhóm theo method.
    """
    import glob

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[PLOT] matplotlib not installed. Skipping plot.")
        return

    csv_files = glob.glob(os.path.join(vram_log_dir, "*_vram.csv"))
    if not csv_files:
        print("[PLOT] No VRAM log files found.")
        return

    # Đọc data
    data = {}  # {method: {stage: peak_gb}}
    for csv_path in sorted(csv_files):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row["method"]
                if method not in data:
                    data[method] = {}
                data[method][row["stage"]] = float(row["peak_allocated_gb"])

    if not data:
        return

    methods = list(data.keys())
    all_stages = sorted(set(s for m in data.values() for s in m.keys()))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_stages))
    width = 0.8 / len(methods)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for i, method in enumerate(methods):
        values = [data[method].get(stage, 0) for stage in all_stages]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method.upper(), color=colors[i % len(colors)])
        # Thêm label trên bar
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("VRAM Usage Comparison: LoRA vs DoRA vs PiSSA")
    ax.set_xticks(x)
    ax.set_xticklabels(all_stages, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # RTX 3090 reference line
    ax.axhline(y=24, color="red", linestyle="--", alpha=0.5, label="RTX 3090 (24 GB)")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[PLOT] VRAM comparison chart saved to: {output_path}")


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
