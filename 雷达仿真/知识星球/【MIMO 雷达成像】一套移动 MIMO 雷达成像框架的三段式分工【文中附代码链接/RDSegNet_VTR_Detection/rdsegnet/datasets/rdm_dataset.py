"""Dataset for range-Doppler map (RDM) segmentation.

Two directory layouts:
1. split_dirs (default): data_root/{train,val,test}/{rdm,mask}/
2. flat: one rdm dir + one mask dir, split by an index list (see
   make_subset_splits / make_holdout_splits).

Each RDM file holds a complex array [H, W, C] or [C, H, W] (channel order
auto-detected). Masks may be .npy/.npz or an image. Adapt ensure_complex() for
custom real/imag storage. Use scripts/create_dummy_data.py for code testing.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# RDM input + mask/image extensions handled by the loaders.
SUPPORTED_EXTENSIONS = {".npy", ".npz", ".mat"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTENSIONS = {".npy", ".npz"} | IMAGE_EXTENSIONS

# Preferred keys for container formats (.npz / .mat).
RDM_KEYS = ("radar_RDM256", "rdm", "RDM", "input", "data", "arr_0")


# --- File loading helpers (also used by scripts/predict.py) ---
def _load_array(path: Path, rdm_key: Optional[str] = None) -> np.ndarray:
    """Load a raw RDM array from .npy / .npz / .mat."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as z:
            keys = ([rdm_key] if rdm_key else []) + list(RDM_KEYS)
            key = next((k for k in keys if k in z.files), z.files[0])
            return z[key]
    if suffix == ".mat":
        import scipy.io as scio  # local import
        mat = scio.loadmat(path)
        names = [k for k in mat.keys() if not k.startswith("__")]
        if not names:
            raise ValueError(f"No arrays found in MATLAB file {path}")
        # Prefer the explicit key, then the first complex array, then the first.
        for k in ([rdm_key] if rdm_key else []) + list(RDM_KEYS):
            if k in mat:
                return mat[k]
        for k in names:
            if np.iscomplexobj(mat[k]):
                return mat[k]
        return mat[names[0]]
    raise ValueError(f"Unsupported RDM file extension: {path.suffix} ({path})")


def _load_mask_array(path: Path) -> np.ndarray:
    """Load a mask from .npy / .npz or an image file."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as z:
            key = "mask" if "mask" in z.files else z.files[0]
            return z[key]
    if suffix in IMAGE_EXTENSIONS:
        return np.asarray(Image.open(path).convert("L"))
    raise ValueError(f"Unsupported mask file extension: {path.suffix} ({path})")


def ensure_complex(arr: np.ndarray) -> np.ndarray:
    """Convert a stored array to a complex ndarray.

    MODIFY THIS for custom real/imag storage. Handles native complex dtypes and
    real arrays with a trailing axis of size 2 holding (real, imag).
    """
    if np.iscomplexobj(arr):
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    raise ValueError(
        "RDM array is real-valued with an unrecognized layout "
        f"(shape={arr.shape}, dtype={arr.dtype}). Adapt "
        "`rdsegnet.datasets.rdm_dataset.ensure_complex` to your storage convention."
    )


def to_channel_first(arr: np.ndarray) -> np.ndarray:
    """[H, W, C] or [C, H, W] -> [C, H, W] (channel axis = smallest dim)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D RDM array, got shape {arr.shape}.")
    channel_axis = int(np.argmin(arr.shape))
    return np.moveaxis(arr, channel_axis, 0)


def load_rdm_file(path: str | Path, rdm_key: Optional[str] = None) -> np.ndarray:
    """Load an RDM file and return a contiguous complex64 array [C, H, W]."""
    arr = ensure_complex(_load_array(Path(path), rdm_key=rdm_key))
    return np.ascontiguousarray(to_channel_first(arr)).astype(np.complex64)


# --- Preprocessing (phase-preserving): configurable complex_norm / log_amp_norm ---
def _normalize_complex(sel: np.ndarray, mag: np.ndarray, mode: str,
                       percentile: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """(real, imag) float32 [K,H,W] normalized by mode.

    'p99_global'/'p99'/'max' divide real and imag by the SAME scale (phase
    preserved); 'zscore' normalizes them independently (phase NOT preserved).
    """
    mode = (mode or "none").lower()
    re = sel.real.astype(np.float32)
    im = sel.imag.astype(np.float32)
    if mode in ("none",):
        return re, im
    if mode in ("p99_global", "global", "p99g"):
        scale = float(np.percentile(mag, percentile)) + eps
        return re / scale, im / scale
    if mode in ("p99", "p99_per_channel", "percentile"):
        scale = np.percentile(mag.reshape(mag.shape[0], -1), percentile,
                              axis=1).reshape(-1, 1, 1) + eps
        return re / scale, im / scale
    if mode == "max":
        scale = mag.reshape(mag.shape[0], -1).max(axis=1).reshape(-1, 1, 1) + eps
        return re / scale, im / scale
    if mode in ("zscore", "zscore_per_channel"):
        mr = re.mean(axis=(1, 2), keepdims=True)
        sr = re.std(axis=(1, 2), keepdims=True)
        mi = im.mean(axis=(1, 2), keepdims=True)
        si = im.std(axis=(1, 2), keepdims=True)
        return (re - mr) / (sr + eps), (im - mi) / (si + eps)
    raise ValueError(f"Unknown complex_norm: {mode!r}")


def _normalize_log_amp(log_amp: np.ndarray, mode: str,
                       mean: Optional[float], std: Optional[float],
                       eps: float) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode in ("none",):
        return log_amp
    if mode == "per_frame":
        return (log_amp - log_amp.mean()) / (log_amp.std() + eps)
    if mode in ("zscore_per_channel", "per_channel"):
        m = log_amp.mean(axis=(1, 2), keepdims=True)
        s = log_amp.std(axis=(1, 2), keepdims=True)
        return (log_amp - m) / (s + eps)
    if mode == "dataset":
        return (log_amp - float(mean or 0.0)) / (float(1.0 if std is None else std) + eps)
    raise ValueError(f"Unknown log_amp_norm: {mode!r}")


def preprocess_rdm(
    rdm: np.ndarray,
    selected_channels: Sequence[int],
    percentile: float = 99.0,
    complex_norm: str = "p99_global",
    complex_layout: str = "interleaved",
    log_amp_norm: str = "per_frame",
    log_amp_mean: Optional[float] = None,
    log_amp_std: Optional[float] = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build network inputs from a channel-first complex RDM [C, H, W].

    Returns (complex_input float32 [2K, H, W], log_amp float32 [K, H, W]).
    'interleaved' layout (required by the model) is [re_0, im_0, re_1, im_1, ...].
    """
    selected_channels = list(selected_channels)
    if max(selected_channels) >= rdm.shape[0] or min(selected_channels) < 0:
        raise IndexError(
            f"selected_channels={selected_channels} out of range for an RDM with "
            f"{rdm.shape[0]} channels."
        )

    sel = rdm[selected_channels]                       # [K, H, W] complex
    mag = np.abs(sel)
    re, im = _normalize_complex(sel, mag, complex_norm, percentile, eps)

    k, h, w = sel.shape
    if complex_layout == "interleaved":
        complex_input = np.empty((2 * k, h, w), dtype=np.float32)
        complex_input[0::2] = re
        complex_input[1::2] = im
    elif complex_layout == "stacked":
        complex_input = np.concatenate([re, im], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unknown complex_layout: {complex_layout!r}")

    log_amp = np.log1p(mag).astype(np.float32)
    log_amp = _normalize_log_amp(log_amp, log_amp_norm, log_amp_mean, log_amp_std, eps)

    return complex_input.astype(np.float32), log_amp.astype(np.float32)


# --- Subset / split helpers for the flat layout ---
def _numeric_key(stem: str):
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def make_subset_splits(
    input_dir: str | Path,
    input_ext: str = ".mat",
    limit: int = 2000,
    test_size: int = 100,
    val_size: int = 0,
    val_equals_test: bool = True,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministically split a flat directory into train/val/test.

    Takes the first `limit` files (numeric order), shuffles with `seed`, peels
    off `test_size` for test. If val_equals_test, val reuses the test stems and
    the rest is train; otherwise val_size distinct stems are used.
    """
    input_dir = Path(input_dir)
    ext = input_ext.lower()
    stems = sorted((p.stem for p in input_dir.iterdir() if p.suffix.lower() == ext),
                   key=_numeric_key)
    stems = stems[:limit]
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)

    test = shuffled[:test_size]
    if val_equals_test:
        val = list(test)
        train = shuffled[test_size:]
    else:
        val = shuffled[test_size:test_size + val_size]
        train = shuffled[test_size + val_size:]
    return {"train": train, "val": val, "test": test}


def make_holdout_splits(
    input_dir: str | Path,
    input_ext: str = ".mat",
    train_limit: int = 3800,
    test_size: int = 100,
    val_size: int = 100,
    exclude_first: int = 0,
    split_seed: int = 1234,
) -> Dict[str, List[str]]:
    """Fixed-holdout split: test/val stay constant as the training pool grows.

    The full candidate list is shuffled before any truncation, then test/val are
    carved off the top and train is sliced from the rest, so growing train_limit
    leaves test/val identical. exclude_first drops the first N numeric stems
    (e.g. an earlier run's samples) from the pool.
    """
    input_dir = Path(input_dir)
    ext = input_ext.lower()
    all_stems = sorted((p.stem for p in input_dir.iterdir() if p.suffix.lower() == ext),
                       key=_numeric_key)
    if exclude_first > 0:
        excluded = set(all_stems[:exclude_first])
        candidates = [s for s in all_stems if s not in excluded]
    else:
        candidates = list(all_stems)
    shuffled = list(candidates)
    random.Random(split_seed).shuffle(shuffled)

    test = shuffled[:test_size]
    val = shuffled[test_size:test_size + val_size]
    pool = shuffled[test_size + val_size:]
    train = pool[:train_limit]
    return {"train": train, "val": val, "test": test}


def _apply_augment(arrays: list, aug: dict) -> list:
    """Apply the SAME random flip/roll to a list of [C, H, W] arrays.

    Phase-safe: spatial flips/rolls move each complex pixel as a unit. No
    rotation / intensity / phase ops.
    """
    if aug.get("hflip") and random.random() < 0.5:
        arrays = [a[:, :, ::-1] for a in arrays]
    if aug.get("vflip") and random.random() < 0.5:
        arrays = [a[:, ::-1, :] for a in arrays]
    roll = int(aug.get("roll", 0) or 0)
    if roll > 0:
        dh = random.randint(-roll, roll)
        dw = random.randint(-roll, roll)
        if dh or dw:
            arrays = [np.roll(a, (dh, dw), axis=(1, 2)) for a in arrays]
    return [np.ascontiguousarray(a) for a in arrays]


# --- Dataset ---
class RDMDataset(Dataset):
    """RDM segmentation dataset.

    Construct with a split dir containing rdm/ and mask/:
        RDMDataset(data_root / "train", selected_channels=[0, 1, 2])
    or with explicit flat dirs + a stem list:
        RDMDataset(rdm_dir=..., mask_dir=..., stems=[...], input_ext=".mat")

    Each item: complex_input [2K, H, W], log_amp [K, H, W], mask [1, H, W] (if
    available), all_rdm [C, H, W] complex (if return_all_rdm), filename.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        rdm_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        stems: Optional[Sequence[str]] = None,
        selected_channels: Sequence[int] = (0, 1, 2),   # choose per your MIMO array
        rdm_key: Optional[str] = None,
        input_ext: Optional[str] = None,
        mask_suffix: Optional[str] = None,
        percentile: float = 99.0,
        complex_norm: str = "p99_global",
        complex_layout: str = "interleaved",
        log_amp_norm: str = "per_frame",
        log_amp_mean: Optional[float] = None,
        log_amp_std: Optional[float] = None,
        eps: float = 1e-8,
        augment: Optional[dict] = None,
        return_all_rdm: bool = False,
        require_mask: bool = True,
    ):
        super().__init__()
        # Resolve directories: explicit flat dirs, or <root>/{rdm,mask}.
        if rdm_dir is not None:
            self.rdm_dir = Path(rdm_dir)
            self.mask_dir = Path(mask_dir) if mask_dir is not None else self.rdm_dir.parent / "mask"
        else:
            if root is None:
                raise ValueError("Provide either `root` or `rdm_dir`.")
            self.rdm_dir = Path(root) / "rdm"
            self.mask_dir = Path(root) / "mask"

        self.selected_channels = list(selected_channels)
        self.rdm_key = rdm_key
        self.input_ext = input_ext.lower() if input_ext else None
        self.mask_suffix = mask_suffix
        self.percentile = float(percentile)
        self.complex_norm = complex_norm
        self.complex_layout = complex_layout
        self.log_amp_norm = log_amp_norm
        self.log_amp_mean = log_amp_mean
        self.log_amp_std = log_amp_std
        self.eps = float(eps)
        self.augment = augment or None
        self.return_all_rdm = return_all_rdm
        self.require_mask = require_mask

        if not self.rdm_dir.is_dir():
            raise FileNotFoundError(
                f"RDM directory not found: {self.rdm_dir}\n"
                "For a quick sanity check, create dummy data first:\n"
                "  python scripts/create_dummy_data.py --output data/dummy "
                "--num-train 8 --num-val 4 --num-test 4"
            )

        # Build the file list, either from an explicit stem list or by scanning.
        if stems is not None:
            ext = self.input_ext or ".mat"
            self.files = [self.rdm_dir / f"{s}{ext}" for s in stems]
            missing_rdm = [f.name for f in self.files if not f.exists()]
            if missing_rdm:
                raise FileNotFoundError(
                    f"{len(missing_rdm)} RDM files from the split list are missing in "
                    f"{self.rdm_dir}, e.g. {missing_rdm[:3]}")
        else:
            self.files = sorted(
                p for p in self.rdm_dir.iterdir()
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        if not self.files:
            raise FileNotFoundError(f"No RDM files found in {self.rdm_dir}")

        if self.require_mask:
            missing = [f.name for f in self.files if self._mask_path(f) is None]
            if missing:
                raise FileNotFoundError(
                    f"Missing masks for {len(missing)} samples in {self.mask_dir}, "
                    f"e.g. {missing[:3]}"
                )

    def _mask_path(self, rdm_path: Path) -> Optional[Path]:
        """Resolve the mask file for an RDM file, or None if not found."""
        stem = rdm_path.stem
        if self.mask_suffix:
            cand = self.mask_dir / f"{stem}{self.mask_suffix}"
            return cand if cand.exists() else None
        # Auto-detect: prefer .npy, then common image formats.
        for ext in (".npy", ".npz", *sorted(IMAGE_EXTENSIONS)):
            cand = self.mask_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        rdm_path = self.files[idx]
        rdm = load_rdm_file(rdm_path, rdm_key=self.rdm_key)   # complex64 [C, H, W]

        complex_input, log_amp = preprocess_rdm(
            rdm,
            self.selected_channels,
            percentile=self.percentile,
            complex_norm=self.complex_norm,
            complex_layout=self.complex_layout,
            log_amp_norm=self.log_amp_norm,
            log_amp_mean=self.log_amp_mean,
            log_amp_std=self.log_amp_std,
            eps=self.eps,
        )

        mask_arr = None
        mask_path = self._mask_path(rdm_path)
        if mask_path is None and self.require_mask:
            raise FileNotFoundError(f"Mask not found for {rdm_path.name} in {self.mask_dir}")
        if mask_path is not None:
            mask = np.squeeze(np.asarray(_load_mask_array(mask_path)))
            if mask.ndim != 2:
                raise ValueError(f"Mask {mask_path} must be 2-D, got shape {mask.shape}.")
            mask_arr = (mask.astype(np.float32) > 0.5).astype(np.float32)[None]   # [1, H, W]

        # Joint spatial augmentation (train split only); all_rdm is left untouched.
        if self.augment:
            if mask_arr is not None:
                complex_input, log_amp, mask_arr = _apply_augment(
                    [complex_input, log_amp, mask_arr], self.augment)
            else:
                complex_input, log_amp = _apply_augment([complex_input, log_amp], self.augment)

        sample = {
            "complex_input": torch.from_numpy(complex_input),   # [2K, H, W]
            "log_amp": torch.from_numpy(log_amp),               # [K, H, W]
            "filename": rdm_path.name,
        }
        if mask_arr is not None:
            sample["mask"] = torch.from_numpy(mask_arr)          # [1, H, W]

        if self.return_all_rdm:
            sample["all_rdm"] = torch.from_numpy(rdm)            # complex64 [C, H, W]

        return sample


# --- Config-driven dataset factory (used by train.py / evaluate.py) ---
def _preprocess_kwargs(cfg: dict) -> dict:
    return dict(
        selected_channels=cfg.get("selected_channels", [0, 1, 2]),
        percentile=float(cfg.get("percentile", 99.0)),
        complex_norm=cfg.get("complex_norm", "p99_global"),
        complex_layout=cfg.get("complex_layout", "interleaved"),
        log_amp_norm=cfg.get("log_amp_norm", "per_frame"),
        log_amp_mean=cfg.get("log_amp_mean"),
        log_amp_std=cfg.get("log_amp_std"),
        eps=float(cfg.get("eps", 1e-8)),
    )


# Cache split stem lists per (input_dir, params) so train/val/test agree.
_SPLIT_CACHE: Dict[tuple, Dict[str, List[str]]] = {}


def flat_splits_from_cfg(cfg: dict) -> Dict[str, List[str]]:
    scheme = cfg.get("split_scheme", "subset")
    if scheme == "holdout":
        key = (
            "holdout", str(cfg["input_dir"]), cfg.get("input_ext", ".mat"),
            int(cfg.get("train_limit", 3800)), int(cfg.get("test_size", 100)),
            int(cfg.get("val_size", 100)), int(cfg.get("exclude_first", 0)),
            int(cfg.get("split_seed", 1234)),
        )
        if key not in _SPLIT_CACHE:
            _SPLIT_CACHE[key] = make_holdout_splits(
                cfg["input_dir"],
                input_ext=cfg.get("input_ext", ".mat"),
                train_limit=int(cfg.get("train_limit", 3800)),
                test_size=int(cfg.get("test_size", 100)),
                val_size=int(cfg.get("val_size", 100)),
                exclude_first=int(cfg.get("exclude_first", 0)),
                split_seed=int(cfg.get("split_seed", 1234)),
            )
        return _SPLIT_CACHE[key]

    key = (
        str(cfg["input_dir"]), cfg.get("input_ext", ".mat"),
        int(cfg.get("subset_limit", 2000)), int(cfg.get("test_size", 100)),
        int(cfg.get("val_size", 0)), bool(cfg.get("val_equals_test", True)),
        int(cfg.get("split_seed", cfg.get("seed", 42))),
    )
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = make_subset_splits(
            cfg["input_dir"],
            input_ext=cfg.get("input_ext", ".mat"),
            limit=int(cfg.get("subset_limit", 2000)),
            test_size=int(cfg.get("test_size", 100)),
            val_size=int(cfg.get("val_size", 0)),
            val_equals_test=bool(cfg.get("val_equals_test", True)),
            seed=int(cfg.get("split_seed", cfg.get("seed", 42))),
        )
    return _SPLIT_CACHE[key]


def build_dataset_for_split(cfg: dict, split: str, *,
                            require_mask: bool = True,
                            return_all_rdm: bool = False) -> RDMDataset:
    """Build an RDMDataset for split ('train'|'val'|'test') from config.

    dataset_mode 'flat' uses input_dir/mask_dir with a deterministic split;
    otherwise the split_dirs layout data_root/<split> is used.
    """
    pp = _preprocess_kwargs(cfg)
    aug = cfg.get("augment") if split == "train" else None
    mode = cfg.get("dataset_mode", "split_dirs")
    if mode == "flat":
        splits = flat_splits_from_cfg(cfg)
        return RDMDataset(
            rdm_dir=cfg["input_dir"], mask_dir=cfg["mask_dir"],
            stems=splits[split], rdm_key=cfg.get("rdm_key"),
            input_ext=cfg.get("input_ext", ".mat"),
            mask_suffix=cfg.get("mask_suffix"), augment=aug,
            require_mask=require_mask, return_all_rdm=return_all_rdm, **pp,
        )
    data_root = Path(cfg.get("data_root", "data/dummy"))
    return RDMDataset(data_root / split, augment=aug, require_mask=require_mask,
                      return_all_rdm=return_all_rdm, **pp)
