"""Tests for RDMDataset: layouts, shapes, dtypes."""

import numpy as np
import pytest
import torch

from rdsegnet.datasets import RDMDataset


def _write_sample(root, name, rdm, mask):
    rdm_dir = root / "rdm"
    mask_dir = root / "mask"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.save(rdm_dir / f"{name}.npy", rdm)
    np.save(mask_dir / f"{name}.npy", mask)


@pytest.mark.parametrize("channels_first", [False, True])
def test_dataset_loading_both_orders(tmp_path, channels_first):
    size, channels = 32, 3
    rng = np.random.default_rng(0)

    rdm = (rng.normal(size=(size, size, channels))
           + 1j * rng.normal(size=(size, size, channels))).astype(np.complex64)
    if channels_first:
        rdm = np.transpose(rdm, (2, 0, 1))  # [3, 32, 32]
    mask = (rng.random((size, size)) > 0.8).astype(np.uint8)

    split = tmp_path / "train"
    _write_sample(split, "sample_000000", rdm, mask)

    ds = RDMDataset(split, selected_channels=[0, 1, 2])
    assert len(ds) == 1

    item = ds[0]
    assert item["complex_input"].shape == (6, size, size)
    assert item["log_amp"].shape == (3, size, size)
    assert item["mask"].shape == (1, size, size)
    assert item["complex_input"].dtype == torch.float32
    assert item["mask"].dtype == torch.float32
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
    assert item["filename"] == "sample_000000.npy"


def test_dataset_return_all_rdm(tmp_path):
    size, channels = 32, 3
    rng = np.random.default_rng(1)
    rdm = (rng.normal(size=(size, size, channels))
           + 1j * rng.normal(size=(size, size, channels))).astype(np.complex64)
    mask = np.zeros((size, size), dtype=np.uint8)

    split = tmp_path / "test"
    _write_sample(split, "sample_000000", rdm, mask)

    ds = RDMDataset(split, selected_channels=[0, 1, 2], return_all_rdm=True)
    item = ds[0]
    all_rdm = item["all_rdm"]
    assert all_rdm.shape == (channels, size, size)
    assert all_rdm.dtype == torch.complex64
    # all_rdm must keep ORIGINAL (unnormalized) values, channel-first.
    expected = np.transpose(rdm, (2, 0, 1))
    assert np.allclose(all_rdm.numpy(), expected)


def test_dataset_missing_mask_raises(tmp_path):
    size, channels = 32, 3
    rdm = np.zeros((size, size, channels), dtype=np.complex64)
    split = tmp_path / "train"
    rdm_dir = split / "rdm"
    rdm_dir.mkdir(parents=True)
    np.save(rdm_dir / "sample_000000.npy", rdm)

    with pytest.raises(FileNotFoundError):
        _ = RDMDataset(split)[0]
