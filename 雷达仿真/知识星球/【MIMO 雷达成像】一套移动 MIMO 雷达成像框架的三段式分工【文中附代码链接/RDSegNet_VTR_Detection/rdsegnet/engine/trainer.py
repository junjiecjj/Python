"""Training engine: scheduler, AMP helpers, and the Trainer class."""

import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import CombinedLoss
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.metrics import SegmentationMetrics
# Per-iteration warmup+cosine scheduler.
from ..utils.scheduler import build_warmup_cosine_scheduler
from ..utils.visualization import save_prediction_grid


def _make_grad_scaler(enabled: bool):
    """GradScaler with API compatibility across torch versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Trains RDSegNet with epoch-level validation, checkpointing & logging."""

    def __init__(self, model: torch.nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict[str, Any],
                 device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.device = device

        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.0),
            focal_weight=loss_cfg.get("focal_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            tversky_weight=loss_cfg.get("tversky_weight", 0.0),
            lovasz_weight=loss_cfg.get("lovasz_weight", 0.0),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.3),
            tversky_beta=loss_cfg.get("tversky_beta", 0.7),
            tversky_gamma=loss_cfg.get("tversky_gamma", 1.0),
        )
        # Lovasz term is ramped 0->1 over the first N epochs (it is noisy early).
        self.lovasz_ramp_epochs = int(loss_cfg.get("lovasz_ramp_epochs", 15))

        self.epochs = int(config.get("epochs", 50))
        base_lr = float(config.get("lr", 1e-3))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
        min_lr = float(config.get("min_lr", 1e-6))
        # Stepped once per optimizer update (per iteration), not per epoch.
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            epochs=self.epochs,
            steps_per_epoch=max(len(train_loader), 1),
            warmup_epochs=int(config.get("warmup_epochs", 5)),
            min_lr_ratio=(min_lr / base_lr) if base_lr > 0 else 0.0,
        )

        # AMP only makes sense on CUDA.
        self.amp = bool(config.get("amp", True)) and device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp)

        self.threshold = float(config.get("threshold", 0.5))
        self.grad_clip = float(config.get("grad_clip", 0.0))

        self.output_dir = Path(config.get("output_dir", "outputs/rdsegnet"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.output_dir / "vis"
        self.vis_every = int(config.get("vis_every", 1))
        self.vis_samples = int(config.get("vis_samples", 2))

        self.start_epoch = 0
        # -1 sentinel so the first validated epoch always writes best.pth,
        # even when early IoU is exactly 0.0.
        self.best_val_iou = -1.0

        # Optional TensorBoard. Falls back silently if unavailable.
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        except Exception:
            self.tb = None

        self.log_csv = self.output_dir / "train_log.csv"
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "lr", "train_loss", "train_bce", "train_focal",
                    "train_dice", "val_loss", "val_iou", "val_dice",
                    "val_precision", "val_recall", "val_f1", "val_pod",
                    "val_far",
                ])

    # ------------------------------------------------------------------ #
    def load_resume(self, ckpt_path) -> None:
        """Resume model/optimizer/scheduler/scaler state from a checkpoint."""
        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and self.amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_val_iou = float(ckpt.get("best_val_iou", 0.0))
        print(f"[resume] from {ckpt_path}: start_epoch={self.start_epoch}, "
              f"best_val_iou={self.best_val_iou:.4f}")

    def init_from_weights(self, ckpt_path) -> None:
        """Warm-start: load ONLY model weights, keep a fresh optimizer/scheduler.

        Leaves start_epoch=0 and best_val_iou=-1 so a new LR schedule and loss
        take over from a good initialization.
        """
        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        print(f"[init] warm-started model weights from {ckpt_path} "
              f"(fresh optimizer/scheduler, start_epoch=0)")

    # ------------------------------------------------------------------ #
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {"total": 0.0, "bce": 0.0, "focal": 0.0, "dice": 0.0,
                   "tversky": 0.0, "lovasz": 0.0}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"train {epoch + 1}/{self.epochs}",
                    leave=False)
        for batch in pbar:
            complex_input = batch["complex_input"].to(self.device)  # [B,6,H,W]
            log_amp = batch["log_amp"].to(self.device)              # [B,3,H,W]
            mask = batch["mask"].to(self.device)                    # [B,1,H,W]

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp:
                with torch.autocast(device_type=self.device.type):
                    logits = self.model(complex_input, log_amp)
                    loss = self.criterion(logits, mask)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(complex_input, log_amp)
                loss = self.criterion(logits, mask)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()   # per-iteration LR update

            for k in running:
                running[k] += self.criterion.components[k]
            n_batches += 1
            pbar.set_postfix(loss=f"{running['total'] / n_batches:.4f}")

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        metrics = SegmentationMetrics(threshold=self.threshold,
                                      from_logits=True)
        total_loss = 0.0
        n_batches = 0
        vis_saved = False

        for batch in tqdm(self.val_loader, desc="val", leave=False):
            complex_input = batch["complex_input"].to(self.device)
            log_amp = batch["log_amp"].to(self.device)
            mask = batch["mask"].to(self.device)

            logits = self.model(complex_input, log_amp)
            loss = self.criterion(logits, mask)
            total_loss += float(loss.detach())
            n_batches += 1
            metrics.update(logits, mask)

            should_vis = (self.vis_every > 0
                          and (epoch + 1) % self.vis_every == 0)
            if should_vis and not vis_saved:
                probs = torch.sigmoid(logits)
                save_prediction_grid(
                    log_amp, probs, mask,
                    self.vis_dir / f"epoch_{epoch + 1:03d}.png",
                    max_samples=self.vis_samples,
                    threshold=self.threshold,
                )
                vis_saved = True

        result = metrics.compute()
        result["loss"] = total_loss / max(n_batches, 1)
        return result

    # ------------------------------------------------------------------ #
    def fit(self) -> float:
        """Run the full training loop; returns the best validation IoU."""
        for epoch in range(self.start_epoch, self.epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            ramp = (1.0 if self.lovasz_ramp_epochs <= 0
                    else min(epoch / self.lovasz_ramp_epochs, 1.0))
            self.criterion.set_lovasz_ramp(ramp)
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.validate(epoch)
            # scheduler is stepped per-iteration inside train_one_epoch

            print(f"[epoch {epoch + 1:3d}/{self.epochs}] lr={lr:.2e} "
                  f"train_loss={train_stats['total']:.4f} "
                  f"val_loss={val_stats['loss']:.4f} "
                  f"val_iou={val_stats['iou']:.4f} "
                  f"val_dice={val_stats['dice']:.4f} "
                  f"val_pod={val_stats['pod']:.4f} "
                  f"val_far={val_stats['far']:.6f}")

            with open(self.log_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f"{lr:.6e}",
                    f"{train_stats['total']:.6f}", f"{train_stats['bce']:.6f}",
                    f"{train_stats['focal']:.6f}", f"{train_stats['dice']:.6f}",
                    f"{val_stats['loss']:.6f}", f"{val_stats['iou']:.6f}",
                    f"{val_stats['dice']:.6f}", f"{val_stats['precision']:.6f}",
                    f"{val_stats['recall']:.6f}", f"{val_stats['f1']:.6f}",
                    f"{val_stats['pod']:.6f}", f"{val_stats['far']:.6f}",
                ])

            if self.tb is not None:
                self.tb.add_scalar("lr", lr, epoch + 1)
                self.tb.add_scalar("loss/train", train_stats["total"], epoch + 1)
                self.tb.add_scalar("loss/val", val_stats["loss"], epoch + 1)
                for key in ("iou", "dice", "pod", "far"):
                    self.tb.add_scalar(f"val/{key}", val_stats[key], epoch + 1)

            is_best = val_stats["iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_stats["iou"]

            common = dict(model=self.model, optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          scaler=self.scaler if self.amp else None,
                          epoch=epoch, best_val_iou=self.best_val_iou,
                          config=self.cfg)
            save_checkpoint(self.output_dir / "last.pth", **common)
            if is_best:
                save_checkpoint(self.output_dir / "best.pth", **common)
                print(f"  -> new best val IoU {self.best_val_iou:.4f}; "
                      f"saved best.pth")

        if self.tb is not None:
            self.tb.close()
        return self.best_val_iou
