"""Shared parameters and numerical operations for the five radar examples.

Complex samples use a power-normalized envelope: E[abs(x)**2] is power.
Positive radial velocity means closing motion and positive Doppler here.
"""

from pathlib import Path
import argparse

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import gammaln

# Monostatic radar and nonfluctuating (Swerling 0) target parameters.
c = 3e8                        # Speed of light (m/s).
Pt = 1.5e6                     # Transmit peak power (W).
fc = 5.6e9                     # Carrier frequency (Hz), C band.
G_dB = 45.0                    # Antenna gain, used for both Tx and Rx.
G = 10 ** (G_dB / 10)
B = 5e6                        # LFM sweep bandwidth (Hz).
F_dB = 1.5                     # Receiver noise figure (dB).
F = 10 ** (F_dB / 10)
L_dB = 3.0                     # System losses (dB).
L = 10 ** (L_dB / 10)
PRF = 2000.0                   # Pulse repetition frequency (Hz).
PRI = 1.0 / PRF                # Pulse repetition interval (s).
Pfa = 1e-6                     # Nominal per-cell false-alarm probability.
PD = 0.80                      # Desired detection probability.
R0 = 35e3                      # Target range (m).
vr = 20.0                      # Closing radial velocity (m/s).
sigma_dB = -10.0               # Radar cross section (dBsm).
sigma = 10 ** (sigma_dB / 10)   # Radar cross section (m^2).
tau_p = 20e-6                  # Pulse duration (s).
k_r = B / tau_p                # Chirp rate (Hz/s).
lambda_c = c / fc              # Wavelength (m).
tau_delay = 2 * R0 / c          # Round-trip delay (s).
fd = 2 * vr / lambda_c          # Doppler frequency (Hz).
k_B = 1.380649e-23             # Boltzmann constant (J/K).
T0 = 290.0                     # Reference noise temperature (K).

# Four samples per nominal range-resolution cell. A complex baseband chirp
# sweeping from -B/2 to +B/2 requires fs >= B, rather than the real-signal 2B.
fs = 4 * B
dt = 1.0 / fs
N_PRI = int(round(PRI * fs))
N_pulse = int(round(tau_p * fs))
t_PRI = np.arange(N_PRI) * dt
t_pulse = np.arange(N_pulse) * dt - tau_p / 2
s_tx_pulse = np.exp(1j * np.pi * k_r * t_pulse**2)
range_axis = np.arange(N_PRI) * c / (2 * fs)
dR = c / (2 * B)                # Nominal range resolution, not sample spacing.

# IID complex noise occupies the whole sampled bandwidth fs. Using k*T*B*F
# as its sample variance at fs=4B would overestimate matched-filter SNR by 6 dB.
Pn_band = k_B * T0 * B * F      # Noise power in bandwidth B (W).
Pn = k_B * T0 * fs * F          # Total discrete-time noise variance (W).
Pr = Pt * G**2 * lambda_c**2 * sigma / ((4 * np.pi)**3 * R0**4 * L)
A_target = np.sqrt(Pr)          # Same power convention as complex noise.
c_w = 2.0                      # Weibull shape; 2 gives a Rayleigh envelope.
CNR_dB = 30.0
CNR = 10 ** (CNR_dB / 10)
Pc = CNR * Pn                  # Clutter-to-noise ratio at the sampled input.


def make_echo(vr_target=vr, R_target=R0, sigma_target=sigma):
    """Build a causal, fractionally delayed echo with within-pulse Doppler.

    Both Tx and echo use the same local chirp time, starting at -tau_p/2.
    Range is held fixed during a CPI (the stop-and-hop approximation).
    The complete echo must fit inside the sampled PRI.
    """
    if R_target <= 0 or sigma_target < 0:
        raise ValueError("Range must be positive and RCS must be nonnegative.")
    tau_delay = 2 * R_target / c
    if tau_delay + tau_p > PRI:
        raise ValueError("The full echo must fit inside one PRI.")
    Pr = Pt * G**2 * lambda_c**2 * sigma_target / ((4 * np.pi)**3 * R_target**4 * L)
    A_target = np.sqrt(Pr)
    fd = 2 * vr_target / lambda_c
    t_echo = t_PRI - tau_delay
    in_echo = (t_echo >= 0) & (t_echo < tau_p)
    s_echo = np.zeros(N_PRI, dtype=complex)
    s_echo[in_echo] = (
        A_target * np.exp(1j * np.pi * k_r * (t_echo[in_echo] - tau_p / 2)**2)
        * np.exp(1j * 2 * np.pi * fd * t_PRI[in_echo])
    )
    return s_echo


def make_noise(rng, size):
    """Circular complex Gaussian noise with total sample variance Pn."""
    sigma_n = np.sqrt(Pn / 2)
    return sigma_n * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def make_clutter(rng, size, c_w=c_w, Pc=Pc):
    """IID Weibull amplitudes and uniform phases, with E[abs(x)**2] = Pc.

    Reuse the same realization across pulses to model stationary clutter.
    This marginal amplitude model is not a terrain-scattering simulation.
    """
    if c_w <= 0 or Pc < 0:
        raise ValueError("Weibull shape must be positive and power nonnegative.")
    lambda_w = np.sqrt(Pc / np.exp(gammaln(1 + 2 / c_w)))
    U = rng.random(size)
    clutter_amp = lambda_w * (-np.log1p(-U)) ** (1 / c_w)
    return clutter_amp * np.exp(1j * 2 * np.pi * rng.random(size))


def matched_filter(data, s_ref):
    """Linear cross-correlation on the last axis; return nonnegative lags.

    Zero padding prevents circular wraparound. Only full-overlap lags are
    returned so partial echoes do not bias the far-range noise estimate.
    Supports a single pulse or an array of pulses.
    """
    data = np.asarray(data)
    s_ref = np.asarray(s_ref)
    if s_ref.ndim != 1 or not s_ref.size or data.shape[-1] < s_ref.size:
        raise ValueError("Reference must be a nonempty 1-D pulse no longer than data.")
    N_fft = next_fast_len(data.shape[-1] + s_ref.size - 1)
    y_mf = ifft(fft(data, N_fft, axis=-1) * np.conj(fft(s_ref, N_fft)), axis=-1)
    return y_mf[..., :data.shape[-1] - s_ref.size + 1]


def ca_cfar(power, N_ref=16, N_guard=8, Pfa=Pfa):
    """Return thresholds and a detection mask for 1-D cell-averaging CFAR.

    N_ref and N_guard are counts PER SIDE. The analytic alpha assumes IID
    exponential power samples. Filtered/oversampled clutter violates that
    assumption: Pfa is nominal until the complete chain is calibrated.
    Cells without a complete training window receive a NaN threshold.
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 1 or not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("Power must be a finite, nonnegative 1-D array.")
    if not isinstance(N_ref, (int, np.integer)) or N_ref < 1:
        raise ValueError("N_ref must be a positive integer.")
    if not isinstance(N_guard, (int, np.integer)) or N_guard < 0:
        raise ValueError("N_guard must be a nonnegative integer.")
    if not 0 < Pfa < 1:
        raise ValueError("Pfa must be between zero and one.")
    N_ref_total = 2 * N_ref
    alpha = N_ref_total * np.expm1(-np.log(Pfa) / N_ref_total)
    threshold = np.full(power.size, np.nan)
    valid_start = N_guard + N_ref
    valid_end = power.size - valid_start
    i_arr = np.arange(valid_start, max(valid_start, valid_end))
    cs = np.concatenate(([0.0], np.cumsum(power)))
    left_sum = cs[i_arr - N_guard] - cs[i_arr - N_guard - N_ref]
    right_sum = cs[i_arr + N_guard + N_ref + 1] - cs[i_arr + N_guard + 1]
    threshold[i_arr] = alpha * (left_sum + right_sum) / N_ref_total
    return threshold, power > threshold


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Figure output directory.")
    return parser.parse_args()


def save_figure(fig, filename, output_dir, show):
    """Save every figure to one explicit output directory."""
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir / filename}")
    if show:
        plt.show()
    plt.close(fig)
