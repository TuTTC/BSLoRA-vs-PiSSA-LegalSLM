"""
Logger Utilities (Enhanced)
============================
Setup Python logging, WandB integration với tag-based logging,
và VRAM metric tracking.
"""

import os
import logging
from typing import Dict, Any, Optional


def get_logger(name: str = "CS431", level: int = logging.INFO) -> logging.Logger:
    """Tạo logger với format chuẩn."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_wandb(config: Dict[str, Any]) -> Optional[Any]:
    """
    Khởi tạo WandB run từ config với tag-based logging.

    Tags tự động bao gồm phương pháp PEFT (LoRA/DoRA/PiSSA)
    để dễ so sánh convergence curves trên WandB dashboard.

    Args:
        config: Dict chứa wandb settings + peft settings

    Returns:
        wandb.Run object hoặc None nếu wandb không available
    """
    try:
        import wandb

        wandb_cfg = config.get("wandb", {})
        peft_method = config.get("peft", {}).get("method", "unknown")
        peft_rank = config.get("peft", {}).get("r", "?")

        # Tags cho phép lọc và so sánh trên WandB
        tags = [
            peft_method,                    # LoRA / DoRA / PiSSA
            f"rank-{peft_rank}",            # rank-16, rank-32, ...
            config.get("model", {}).get("name", "unknown").split("/")[-1],  # model name
        ]

        # Thêm task_type nếu có
        task_type = config.get("data", {}).get("task_type")
        if task_type:
            tags.append(task_type)

        run_name = wandb_cfg.get("run_name") or f"{peft_method}-r{peft_rank}"

        run = wandb.init(
            project=wandb_cfg.get("project", "CS431-DoRA-vs-PiSSA-LegalSLM"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config=config,
            tags=tags,
            reinit=True,
        )
        print(f"[WANDB] Initialized: {run.name} ({run.url})")
        print(f"[WANDB] Tags: {tags}")
        return run

    except ImportError:
        print("[WANDB] wandb not installed. Skipping WandB logging.")
        return None
    except Exception as e:
        print(f"[WANDB] Failed to initialize: {e}")
        return None


def log_vram_to_wandb(wandb_run, vram_summary: Dict[str, Any]) -> None:
    """
    Log VRAM metrics vào WandB summary.

    Args:
        wandb_run: Active wandb.Run object
        vram_summary: Dict từ VRAMTracker.summary()
    """
    if wandb_run is None or not vram_summary:
        return

    try:
        import wandb

        # Log overall peak
        wandb_run.summary["vram_peak_gb"] = vram_summary.get("overall_peak_vram_gb", 0)
        wandb_run.summary["total_duration_s"] = vram_summary.get("total_duration_s", 0)

        # Log per-stage peaks
        stages = vram_summary.get("stages", {})
        for stage, peak in stages.items():
            wandb_run.summary[f"vram_{stage}_peak_gb"] = peak

        # Log as a WandB table for comparison
        table = wandb.Table(
            columns=["method", "stage", "peak_vram_gb"],
        )
        method = vram_summary.get("method", "unknown")
        for stage, peak in stages.items():
            table.add_data(method, stage, peak)

        wandb_run.log({"vram_by_stage": table})
        print(f"[WANDB] Logged VRAM metrics (peak: {vram_summary.get('overall_peak_vram_gb', 0):.2f} GB)")

    except Exception as e:
        print(f"[WANDB] Failed to log VRAM metrics: {e}")


def log_config(config: Dict[str, Any]) -> None:
    """Log config ra console."""
    logger = get_logger()
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info("-" * 50)

    def _log_dict(d: Dict, prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"  {prefix}{key}:")
                _log_dict(value, prefix + "  ")
            else:
                logger.info(f"  {prefix}{key}: {value}")

    _log_dict(config)
    logger.info("=" * 50)
