"""Binary segmentation metrics for RDM target detection.

SegmentationMetrics accumulates TP/FP/FN/TN over batches, then derives IoU,
Dice, Precision, Recall, F1, PoD (= TP/(TP+FN), recall) and FAR (= FP/(FP+TN)).
per_sample_metrics computes the same per sample in a batch.
"""

from typing import Dict, List

import torch

EPS = 1e-8


def _binarize(pred: torch.Tensor, threshold: float, from_logits: bool) -> torch.Tensor:
    if from_logits:
        pred = torch.sigmoid(pred)
    return (pred >= threshold).float()


def _confusion(pred_bin: torch.Tensor, target: torch.Tensor):
    """Element-wise confusion counts; both inputs are {0,1} float tensors."""
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1.0 - target)).sum()
    fn = ((1.0 - pred_bin) * target).sum()
    tn = ((1.0 - pred_bin) * (1.0 - target)).sum()
    return tp, fp, fn, tn


def _derive(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    iou = tp / (tp + fp + fn + EPS)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    pod = recall  # probability of detection == recall
    far = fp / (fp + tn + EPS)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pod": pod,
        "far": far,
    }


class SegmentationMetrics:
    """Accumulates confusion counts across batches and derives metrics."""

    def __init__(self, threshold: float = 0.5, from_logits: bool = True):
        self.threshold = threshold
        self.from_logits = from_logits
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred [B, 1, H, W] logits/probs; target [B, 1, H, W] binary
        pred_bin = _binarize(pred, self.threshold, self.from_logits)
        tp, fp, fn, tn = _confusion(pred_bin, target.float())
        self.tp += float(tp)
        self.fp += float(fp)
        self.fn += float(fn)
        self.tn += float(tn)

    def compute(self) -> Dict[str, float]:
        return _derive(self.tp, self.fp, self.fn, self.tn)


@torch.no_grad()
def per_sample_metrics(pred: torch.Tensor, target: torch.Tensor,
                       threshold: float = 0.5,
                       from_logits: bool = True) -> List[Dict[str, float]]:
    """Per-sample metrics for one batch; returns a list of dicts."""
    pred_bin = _binarize(pred, threshold, from_logits)
    target = target.float()

    rows = []
    for i in range(pred_bin.shape[0]):
        tp, fp, fn, tn = _confusion(pred_bin[i], target[i])
        rows.append(_derive(float(tp), float(fp), float(fn), float(tn)))
    return rows
