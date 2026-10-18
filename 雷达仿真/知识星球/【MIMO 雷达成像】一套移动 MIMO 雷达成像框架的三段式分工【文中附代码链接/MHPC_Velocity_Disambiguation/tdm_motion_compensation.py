import numpy as np


def tdm_motion_compensation(
    s_if_total_2dfft: np.ndarray,
    doppler_fft_size: int | None = None,
    num_tx: int | None = None,
    ambiguity_index: int = 0,
) -> np.ndarray:
    """Apply TDM-MIMO Doppler phase compensation with a candidate ambiguity index."""

    x = np.asarray(s_if_total_2dfft)

    if x.ndim != 4:
        raise ValueError(
            "s_if_total_2dfft must have shape "
            "(num_range_bins, num_doppler_bins, num_tx, num_rx)."
        )

    _, n_doppler, n_tx, _ = x.shape

    if doppler_fft_size is None:
        doppler_fft_size = n_doppler
    if num_tx is None:
        num_tx = n_tx

    if doppler_fft_size != n_doppler:
        raise ValueError("doppler_fft_size does not match input Doppler dimension.")
    if num_tx != n_tx:
        raise ValueError("num_tx does not match input Tx dimension.")

    doppler_idx = np.arange(doppler_fft_size, dtype=np.float64)
    tx_idx = np.arange(num_tx, dtype=np.float64)

    delta_phi = (
        2.0
        * np.pi
        * (doppler_idx - doppler_fft_size / 2.0)
        / (num_tx * doppler_fft_size)
    )

    phase_tdm = delta_phi[:, None] * tx_idx[None, :]
    phase_ambiguity = 2.0 * np.pi * ambiguity_index * tx_idx / num_tx

    compensation = np.exp(-1j * (phase_tdm + phase_ambiguity[None, :]))

    return x * compensation[None, :, :, None]


TDM_MotionCompensation = tdm_motion_compensation
