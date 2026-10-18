"""
MHPC velocity disambiguation for TDM-MIMO radar.
Need:
    1. range-Doppler FFT data  [range, doppler, tx, rx];
    2. detected RDM cells; 
    3. a virtual-array mapping from physical Tx/Rx channels to virtual array
       coordinates.


from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class RadarConfig:
    """TI-CAS-EVM radar parameters required by MHPC."""

    c: float = 3.0e8
    f0: float = 77.0e9
    t_chirp: float = 4.505083e-5
    num_tx: int = 12
    num_rx: int = 16
    num_chirp_per_tx: int = 32
    range_fft_size: int = 256
    doppler_fft_size: int = 32
    azi_fft_size: int = 1024
    ele_fft_size: int = 256

    @property
    def wavelength(self) -> float:
        return self.c / self.f0

    @property
    def vmax(self) -> float:

        return self.wavelength / (4.0 * self.t_chirp * self.num_chirp_per_tx) / self.num_tx

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2.0 * self.num_chirp_per_tx * self.t_chirp * self.num_tx)


@dataclass(frozen=True)
class MHPCWeights:
    """Weights for J(zeta) = mu1*max(P) + mu2*SNR(P) - mu3*H(P). CAN change and adjust"""

    mu_peak: float = 0.01
    mu_snr: float = 2.0
    mu_entropy: float = 1.0


@dataclass
class MHPCResult:
    """Result for one detected RDM cell."""

    range_index: int
    doppler_index: int
    zeta_hat: int
    observed_velocity: float
    recovered_velocity: float
    score_table: np.ndarray
    # score_table columns: [zeta, peak, snr, entropy, score]
    best_spectrum: np.ndarray


def doppler_index_to_velocity(doppler_index: int, cfg: RadarConfig, one_based: bool = False) -> float:
    """
    Convert a Doppler FFT bin index to the observed radial velocity.
    """
    idx = doppler_index - 1 if one_based else doppler_index
    centered_bin = idx - cfg.doppler_fft_size // 2
    return centered_bin * cfg.velocity_resolution


def recover_true_velocity(v_obs: float, zeta: int, cfg: RadarConfig) -> float:
    """Recover the true velocity from V_obs = V_true - 2*zeta*Vmax."""
    return v_obs + 2.0 * zeta * cfg.vmax


def compensate_tdm_phase(
    rd_cube: np.ndarray,
    v_hyp: float,
    cfg: RadarConfig,
    tx_axis: int = 2,
) -> np.ndarray:
    """
    Compensate TDM-induced phase for a hypothesized true velocity.

    Parameters
    ----------
    rd_cube:
        Complex range-Doppler data with Tx dimension at `tx_axis`.
    v_hyp:
        Hypothesized true radial velocity in m/s.
    cfg:
        Radar configuration.
    tx_axis:
        Axis corresponding to Tx index.


    out = np.array(rd_cube, dtype=np.complex128, copy=True)
    tx_idx = np.arange(cfg.num_tx, dtype=float)
    phase = np.exp(1j * 2.0 * np.pi * cfg.f0 * v_hyp * (tx_idx * cfg.t_chirp) / cfg.c)
    shape = [1] * out.ndim
    shape[tx_axis] = cfg.num_tx
    return out * phase.reshape(shape)


def build_virtual_array(
    rd_cube: np.ndarray,
    virtual_mapping: np.ndarray,
    out_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:

    mapping = np.asarray(virtual_mapping, dtype=int).copy()
    if mapping.ndim != 2 or mapping.shape[1] != 4:
        raise ValueError("virtual_mapping must have shape [N, 4]: [virt_az, virt_el, tx, rx].")


    if mapping.min() >= 1:
        mapping -= 1

    if out_shape is None:
        n_az = int(mapping[:, 0].max()) + 1
        n_el = int(mapping[:, 1].max()) + 1
    else:
        n_az, n_el = out_shape

    n_r, n_d = rd_cube.shape[:2]
    vir = np.zeros((n_r, n_d, n_az, n_el), dtype=np.complex128)

    for virt_az, virt_el, tx, rx in mapping:
        if not np.any(vir[:, :, virt_az, virt_el]):
            vir[:, :, virt_az, virt_el] = rd_cube[:, :, tx, rx]
    return vir


def angular_spectrum_azimuth(virtual_snapshot: np.ndarray, azi_fft_size: int) -> np.ndarray:
    """
    Form a 1-D azimuth angular spectrum from one virtual-array snapshot.

    """
    if virtual_snapshot.ndim != 2:
        raise ValueError("virtual_snapshot must have shape [virt_az, virt_el].")
    return np.fft.fftshift(np.fft.fft(virtual_snapshot[:, 0], n=azi_fft_size))


def spectrum_metrics(spectrum: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Compute peak power, SNR-like sharpness, and Shannon entropy.
    """
    power = np.abs(spectrum).astype(float)
    peak = float(np.max(power))

    prob = power / (np.sum(power) + eps)
    snr = float(np.max(prob) / (np.mean(prob) + eps))
    entropy = float(-np.sum(prob * np.log2(prob + eps)))
    return peak, snr, entropy


def mhpc_for_cell(
    rd_cube: np.ndarray,
    r_idx: int,
    d_idx: int,
    cfg: RadarConfig,
    virtual_mapping: np.ndarray,
    zeta_candidates: Sequence[int] = tuple(range(-5, 6)),
    weights: MHPCWeights = MHPCWeights(),
    one_based_indices: bool = False,
    phase_sign: int = +1,
) -> MHPCResult:

    rr = r_idx - 1 if one_based_indices else r_idx
    dd = d_idx - 1 if one_based_indices else d_idx

    v_obs = doppler_index_to_velocity(d_idx, cfg, one_based=one_based_indices)

    rows = []
    spectra = []

    for zeta in zeta_candidates:
        v_hyp = recover_true_velocity(v_obs, int(zeta), cfg)

        # Apply the hypothesized TDM phase correction. The sign is exposed
        # because different IF/FFT implementations may use opposite signs.
        compensated = compensate_tdm_phase(rd_cube, phase_sign * v_hyp, cfg)
        vir = build_virtual_array(compensated, virtual_mapping)
        snapshot = vir[rr, dd, :, :]
        spec = angular_spectrum_azimuth(snapshot, cfg.azi_fft_size)

        peak, snr, entropy = spectrum_metrics(spec)
        score = weights.mu_peak * peak + weights.mu_snr * snr - weights.mu_entropy * entropy
        rows.append([int(zeta), peak, snr, entropy, score])
        spectra.append(spec)

    score_table = np.asarray(rows, dtype=float)
    best_idx = int(np.argmax(score_table[:, 4]))
    zeta_hat = int(score_table[best_idx, 0])
    v_true = recover_true_velocity(v_obs, zeta_hat, cfg)

    return MHPCResult(
        range_index=r_idx,
        doppler_index=d_idx,
        zeta_hat=zeta_hat,
        observed_velocity=v_obs,
        recovered_velocity=v_true,
        score_table=score_table,
        best_spectrum=spectra[best_idx],
    )


def mhpc_batch(
    rd_cube: np.ndarray,
    detected_cells: Iterable[Tuple[int, int]],
    cfg: RadarConfig,
    virtual_mapping: np.ndarray,
    zeta_candidates: Sequence[int] = tuple(range(-5, 6)),
    weights: MHPCWeights = MHPCWeights(),
    one_based_indices: bool = False,
    phase_sign: int = +1,
) -> list[MHPCResult]:
    """Run MHPC for all detected RDM cells."""
    return [
        mhpc_for_cell(
            rd_cube=rd_cube,
            r_idx=r,
            d_idx=d,
            cfg=cfg,
            virtual_mapping=virtual_mapping,
            zeta_candidates=zeta_candidates,
            weights=weights,
            one_based_indices=one_based_indices,
            phase_sign=phase_sign,
        )
        for r, d in detected_cells
    ]


def example_virtual_mapping_ula(num_tx: int = 12, num_rx: int = 16) -> np.ndarray:
    """
    Simple placeholder mapping for a 1-D virtual ULA.
    """
    mapping = []
    for tx in range(num_tx):
        for rx in range(num_rx):
            mapping.append([tx * num_rx + rx, 0, tx, rx])
    return np.asarray(mapping, dtype=int)


if __name__ == "__main__":
    # Minimal smoke test with random complex data. Replace with real RDM data.
    cfg = RadarConfig()
    rng = np.random.default_rng(0)
    rd = rng.normal(size=(cfg.range_fft_size, cfg.doppler_fft_size, cfg.num_tx, cfg.num_rx)) \
        + 1j * rng.normal(size=(cfg.range_fft_size, cfg.doppler_fft_size, cfg.num_tx, cfg.num_rx))

    cells = [(120, 20)]  # Python 0-based [range, doppler] indices
    mapping = example_virtual_mapping_ula(cfg.num_tx, cfg.num_rx)
    results = mhpc_batch(rd, cells, cfg, mapping)

    for res in results:
        print(f"cell=({res.range_index}, {res.doppler_index}), "
              f"zeta_hat={res.zeta_hat}, "
              f"v_obs={res.observed_velocity:.4f} m/s, "
              f"v_true={res.recovered_velocity:.4f} m/s")
        print("score table columns: zeta, peak, snr, entropy, score")
        print(res.score_table)
