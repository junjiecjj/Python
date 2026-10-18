"""Checkpoint save/load helpers."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, best_val_iou: float = 0.0,
                    config: Optional[Dict[str, Any]] = None) -> None:
    """Save a full training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_val_iou": best_val_iou,
        "config": config,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(path, map_location="cpu") -> Dict[str, Any]:
    """Load a checkpoint dict (weights_only=False: it stores a config dict too)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch without the weights_only kwarg.
        return torch.load(path, map_location=map_location)
