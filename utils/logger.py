"""
Logger Utilities
================
Setup Python logging và WandB integration.
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
    Khởi tạo WandB run từ config.
    
    Args:
        config: Dict chứa wandb settings
    
    Returns:
        wandb.Run object hoặc None nếu wandb không available
    """
    try:
        import wandb

        wandb_cfg = config.get("wandb", {})
        peft_method = config.get("peft", {}).get("method", "unknown")

        run = wandb.init(
            project=wandb_cfg.get("project", "CS431-DoRA-vs-PiSSA-LegalSLM"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", f"{peft_method}-run"),
            config=config,
            reinit=True,
        )
        print(f"[WANDB] Initialized: {run.name} ({run.url})")
        return run

    except ImportError:
        print("[WANDB] wandb not installed. Skipping WandB logging.")
        return None
    except Exception as e:
        print(f"[WANDB] Failed to initialize: {e}")
        return None


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
