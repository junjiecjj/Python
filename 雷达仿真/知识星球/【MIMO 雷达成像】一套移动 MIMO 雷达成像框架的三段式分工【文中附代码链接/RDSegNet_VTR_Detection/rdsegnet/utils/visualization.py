"""Visualization helpers (headless-safe: uses the Agg backend)."""

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")  # must come before the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_prediction_grid(log_amp, probs, targets, path,
                         max_samples: int = 2, threshold: float = 0.5) -> None:
    """Save a grid of rows = samples, cols = [log-amp ch0 | predicted mask | GT]."""
    log_amp = _to_numpy(log_amp)
    probs = _to_numpy(probs)
    targets = _to_numpy(targets)

    n = min(max_samples, log_amp.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    for i in range(n):
        axes[i][0].imshow(log_amp[i, 0], cmap="viridis")
        axes[i][0].set_title("log-amplitude (ch 0)")
        axes[i][1].imshow((probs[i, 0] >= threshold).astype(np.float32),
                          cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title("predicted mask")
        axes[i][2].imshow(targets[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("GT mask")
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_mask_png(mask: Union[torch.Tensor, np.ndarray], path) -> None:
    """Save a single binary mask [H, W] (or [1, H, W]) as a PNG."""
    mask = _to_numpy(mask)
    if mask.ndim == 3:
        mask = mask[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
