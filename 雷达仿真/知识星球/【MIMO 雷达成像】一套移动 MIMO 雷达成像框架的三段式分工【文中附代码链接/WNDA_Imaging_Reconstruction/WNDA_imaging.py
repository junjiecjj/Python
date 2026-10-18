from __future__ import annotations

import numpy as np


def _as_3d(coords: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N, 2) or (N, 3).")
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(arr.shape[0])])
    return arr


def wnda_kspace_samples(
    s_nf_tx_rx: np.ndarray,
    tx_coords: np.ndarray,
    rx_coords: np.ndarray,
    f0: float,
    chirp_slope: float,
    adc_sample_rate: float,
    c: float = 3e8,
    doi_origin: np.ndarray | None = None,
    range_bin_indices: np.ndarray | None = None,
    phase_sign: int = -1,
    amplitude_compensation: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate WNDA nonuniform k-space samples from frequency-domain Tx-Rx data."""

    s = np.asarray(s_nf_tx_rx, dtype=np.complex128)
    if s.ndim != 3:
        raise ValueError("s_nf_tx_rx must have shape (num_freq_bins, num_tx, num_rx).")

    num_freq, num_tx, num_rx = s.shape
    tx = _as_3d(tx_coords, "tx_coords")
    rx = _as_3d(rx_coords, "rx_coords")

    if tx.shape[0] != num_tx:
        raise ValueError("tx_coords length does not match the Tx dimension of s_nf_tx_rx.")
    if rx.shape[0] != num_rx:
        raise ValueError("rx_coords length does not match the Rx dimension of s_nf_tx_rx.")

    if doi_origin is None:
        origin = np.zeros(3, dtype=np.float64)
    else:
        origin = np.asarray(doi_origin, dtype=np.float64).reshape(-1)
        if origin.size == 2:
            origin = np.array([origin[0], origin[1], 0.0], dtype=np.float64)
        if origin.size != 3:
            raise ValueError("doi_origin must have length 2 or 3.")

    if range_bin_indices is None:
        freq_idx = np.arange(num_freq, dtype=np.float64)
    else:
        freq_idx = np.asarray(range_bin_indices, dtype=np.float64).reshape(-1)
        if freq_idx.size != num_freq:
            raise ValueError("range_bin_indices length must match num_freq_bins.")

    tx_rel = tx - origin[None, :]
    rx_rel = rx - origin[None, :]

    rt = np.linalg.norm(tx_rel, axis=1)
    rr = np.linalg.norm(rx_rel, axis=1)

    if np.any(rt <= 0.0) or np.any(rr <= 0.0):
        raise ValueError("Tx/Rx distance to doi_origin must be nonzero.")

    tx_unit = tx_rel / rt[:, None]
    rx_unit = rx_rel / rr[:, None]

    kx_dir = tx_unit[:, 0, None] + rx_unit[None, :, 0]
    ky_dir = tx_unit[:, 1, None] + rx_unit[None, :, 1]

    path_len = rt[:, None] + rr[None, :]
    amp = (4.0 * np.pi) ** 2 * rt[:, None] * rr[None, :]

    if not amplitude_compensation:
        amp = np.ones((num_tx, num_rx), dtype=np.float64)

    k_inst = 2.0 * np.pi * (f0 + chirp_slope * freq_idx / adc_sample_rate) / c

    kx_all = []
    ky_all = []
    data_all = []

    for i, k in enumerate(k_inst):
        compensated = s[i] * amp * np.exp(1j * phase_sign * k * path_len)
        kx_all.append((k * kx_dir).ravel())
        ky_all.append((k * ky_dir).ravel())
        data_all.append(compensated.ravel())

    return (
        np.concatenate(kx_all),
        np.concatenate(ky_all),
        np.concatenate(data_all),
        k_inst,
    )


def _adjoint_nudft2d(
    data: np.ndarray,
    omega: np.ndarray,
    image_shape: tuple[int, int],
    block_size: int = 512,
) -> np.ndarray:
    """Compute a dependency-free adjoint NUDFT for small validation cases."""

    nx, ny = image_shape
    x_grid = np.arange(nx, dtype=np.float64) - nx / 2.0
    y_grid = np.arange(ny, dtype=np.float64) - ny / 2.0

    img = np.zeros((nx, ny), dtype=np.complex128)

    for start in range(0, data.size, block_size):
        stop = min(start + block_size, data.size)
        wx = omega[start:stop, 0]
        wy = omega[start:stop, 1]
        db = data[start:stop]

        ex = np.exp(1j * x_grid[:, None] * wx[None, :])
        ey = np.exp(1j * wy[:, None] * y_grid[None, :])
        img += (ex * db[None, :]) @ ey

    return img


def adjoint_nufft2d(
    data: np.ndarray,
    omega: np.ndarray,
    image_shape: tuple[int, int] = (800, 800),
    method: str = "auto",
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply 2D adjoint NUFFT from nonuniform k-space samples to an image grid."""

    data = np.asarray(data, dtype=np.complex128).reshape(-1)
    omega = np.asarray(omega, dtype=np.float64)

    if omega.ndim != 2 or omega.shape[1] != 2:
        raise ValueError("omega must have shape (num_samples, 2).")
    if omega.shape[0] != data.size:
        raise ValueError("omega and data must contain the same number of samples.")

    if method not in {"auto", "finufft", "direct"}:
        raise ValueError("method must be 'auto', 'finufft', or 'direct'.")

    if method in {"auto", "finufft"}:
        try:
            import finufft

            return finufft.nufft2d1(
                omega[:, 0],
                omega[:, 1],
                data,
                image_shape,
                eps=eps,
                isign=1,
            )
        except ImportError:
            if method == "finufft":
                raise ImportError("Install finufft or use method='direct' for small tests.")

    nx, ny = image_shape
    if data.size * nx * ny > 2.0e8:
        raise RuntimeError(
            "Direct adjoint NUDFT is too slow for this size. Install finufft or reduce image_shape."
        )

    return _adjoint_nudft2d(data, omega, image_shape)


def wnda_reconstruct(
    s_nf_tx_rx: np.ndarray,
    tx_coords: np.ndarray,
    rx_coords: np.ndarray,
    f0: float,
    chirp_slope: float,
    adc_sample_rate: float,
    image_shape: tuple[int, int] = (800, 800),
    c: float = 3e8,
    doi_origin: np.ndarray | None = None,
    range_bin_indices: np.ndarray | None = None,
    phase_sign: int = -1,
    amplitude_compensation: bool = True,
    omega_scale: float = 0.5,
    method: str = "auto",
    eps: float = 1e-6,
    return_kspace: bool = False,
):
    """Reconstruct a DOI sub-image using WNDA and adjoint NUFFT."""

    kx, ky, data, k_inst = wnda_kspace_samples(
        s_nf_tx_rx=s_nf_tx_rx,
        tx_coords=tx_coords,
        rx_coords=rx_coords,
        f0=f0,
        chirp_slope=chirp_slope,
        adc_sample_rate=adc_sample_rate,
        c=c,
        doi_origin=doi_origin,
        range_bin_indices=range_bin_indices,
        phase_sign=phase_sign,
        amplitude_compensation=amplitude_compensation,
    )

    k_ref = float(k_inst[-1])
    omega_x = kx * (np.pi / (omega_scale * k_ref))
    omega_y = ky * (np.pi / (omega_scale * k_ref))
    omega = np.column_stack([omega_x, omega_y])

    image = adjoint_nufft2d(
        data=data,
        omega=omega,
        image_shape=image_shape,
        method=method,
        eps=eps,
    )

    if return_kspace:
        return image, {
            "kx": kx,
            "ky": ky,
            "omega": omega,
            "data": data,
            "k_inst": k_inst,
        }

    return image


def normalize_wnda_image(
    image: np.ndarray,
    threshold: float = 0.05,
    gamma: float = 1.8,
    rotate: bool = True,
) -> np.ndarray:
    
    img = np.abs(image)
    max_val = np.max(img)

    if max_val > 0.0:
        img = img / max_val

    img[img < threshold] = 0.0
    img = img ** gamma

    if rotate:
        img = np.rot90(img)

    return img
