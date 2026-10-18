"""Part 1: time-domain LFM modeling and the thermal-noise SNR budget.

A long chirp combines pulse energy with bandwidth-dependent range resolution:
delta_R = c/(2B). The RF carrier is represented by a complex baseband envelope.
The PD curve describes a single nonfluctuating target in Gaussian noise;
it does not predict detection probability in the clutter-plus-CFAR example.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2

from radar_utils import (
    B, CNR_dB, F, G_dB, L_dB, N_PRI, N_pulse, PD, PRF, PRI, Pfa,
    Pn, Pn_band, Pr, Pt, R0, T0, c, dR, fc, fd, fs, k_B, k_r,
    lambda_c, make_clutter, make_echo, make_noise, matched_filter,
    parse_args, s_tx_pulse, save_figure, sigma, t_PRI, tau_delay, tau_p, vr,
)


def marcum_q1(a, b):
    """First-order Marcum Q via the noncentral chi-squared survival function."""
    return ncx2.sf(b**2, df=2, nc=a**2)


def main(output_dir="figures", show=True):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)

    # The transmitted waveform has unit amplitude; physical Tx power is Pt.
    s_tx_full = np.zeros(N_PRI, dtype=complex)
    s_tx_full[:N_pulse] = s_tx_pulse
    noise = make_noise(rng, N_PRI)
    clutter = make_clutter(rng, N_PRI)
    s_echo = make_echo()
    s_rx = s_echo + noise + clutter

    # Ideal transmit/receive blanking limit, ambiguity limit, and full-echo limit.
    R_min = c * tau_p / 2
    R_max = c * PRI / 2
    R_full_echo = c * (PRI - tau_p) / 2
    b_thresh = np.sqrt(-2 * np.log(Pfa))
    SNR_req_dB = brentq(lambda x: marcum_q1(np.sqrt(2 * 10**(x / 10)), b_thresh) - PD, -30, 60)
    SNR_actual = Pr * tau_p / (k_B * T0 * F)
    SNR_actual_dB = 10 * np.log10(SNR_actual)
    margin_dB = SNR_actual_dB - SNR_req_dB

    # Check the discrete-time filter against the continuous-time energy budget.
    # Fractional delay and within-pulse Doppler introduce a small peak loss.
    y_echo = matched_filter(s_echo, s_tx_pulse)
    SNR_sampled_dB = 10 * np.log10(np.max(np.abs(y_echo)**2) / (Pn * np.sum(np.abs(s_tx_pulse)**2)))
    print("PART 1 - LFM SIGNAL MODELING")
    print(f"Carrier: {fc/1e9:.2f} GHz | Bandwidth: {B/1e6:.1f} MHz | Pulse: {tau_p*1e6:.0f} us")
    print(f"PRF: {PRF:.0f} Hz | Duty cycle: {100*tau_p/PRI:.1f}% | TB: {B*tau_p:.0f}")
    print(f"Target: {R0/1e3:.3f} km, {vr:.1f} m/s closing, RCS {sigma:.2f} m^2")
    print(f"Delay: {tau_delay*1e6:.3f} us | Doppler: {fd:.2f} Hz")
    print(f"Received target power: {Pr:.4e} W | Tx/Rx power ratio: {10*np.log10(Pt/Pr):.2f} dB")
    print(f"Noise in B: {Pn_band:.4e} W | IID sample variance over fs: {Pn:.4e} W")
    print(f"Minimum range: {R_min/1e3:.1f} km | Unambiguous range: {R_max/1e3:.1f} km")
    print(f"Full echo before next pulse: <= {R_full_echo/1e3:.1f} km | Resolution: {dR:.1f} m")
    print(f"Required SNR: {SNR_req_dB:.2f} dB | Ideal MF SNR: {SNR_actual_dB:.2f} dB")
    print(f"Sampled MF SNR: {SNR_sampled_dB:.2f} dB | Ideal thermal-noise margin: {margin_dB:.2f} dB")
    print("The SNR margin excludes clutter, window loss, and adaptive-threshold loss.")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("LFM pulse radar: time-domain signal model", fontweight="bold")
    axes[0].plot(t_PRI[:N_pulse] * 1e6, s_tx_pulse.real, lw=0.9)
    axes[0].set(title="Transmitted baseband chirp (-B/2 to +B/2)", xlabel="Time (us)", ylabel="Normalized amplitude")
    rx_env_dB = 10 * np.log10(np.abs(s_rx)**2 + 1e-30)
    axes[1].plot(t_PRI * 1e3, rx_env_dB, color="slategray", lw=0.5, label="Target + noise + clutter")
    axes[1].axvline(tau_delay * 1e3, color="green", ls="--", label="Echo start")
    axes[1].set(title=f"Received envelope power | Sampled-input CNR = {CNR_dB:.0f} dB", xlabel="Time (ms)", ylabel="Power (dBW)")
    axes[1].legend()
    # Both traces now share the same physical Tx-amplitude normalization.
    tx_env = np.abs(s_tx_full)
    rx_env_lin = np.abs(s_rx) / np.sqrt(Pt)
    axes[2].plot(t_PRI * 1e3, np.maximum(tx_env, 1e-12), label="Tx envelope / sqrt(Pt)")
    axes[2].plot(t_PRI * 1e3, np.maximum(rx_env_lin, 1e-12), lw=0.6, label="Rx envelope / sqrt(Pt)")
    axes[2].set(yscale="log", ylim=(1e-12, 2), xlabel="Time (ms)", ylabel="Relative amplitude", title="Tx/Rx amplitude comparison (logarithmic scale)")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    save_figure(fig, "part1_signal_modeling.png", output_dir, show)

    snr_plot = np.linspace(0, max(20, SNR_actual_dB + 3), 1500)
    pd_plot = marcum_q1(np.sqrt(2 * 10**(snr_plot / 10)), b_thresh)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(snr_plot, 100 * pd_plot, lw=2)
    ax.axhline(100 * PD, color="red", ls="--", label=f"PD = {100*PD:.0f}%")
    ax.axvline(SNR_req_dB, color="green", ls="--", label=f"Required: {SNR_req_dB:.2f} dB")
    ax.axvline(SNR_actual_dB, color="orange", ls="-.", label=f"Ideal MF: {SNR_actual_dB:.2f} dB")
    ax.set(xlabel="SNR (dB)", ylabel="Detection probability (%)", ylim=(0, 105), title=f"Swerling 0 in Gaussian noise | Pfa = {Pfa:.0e}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "part1_snr_detection_probability.png", output_dir, show)
    return {"SNR_req_dB": SNR_req_dB, "SNR_actual_dB": SNR_actual_dB, "SNR_sampled_dB": SNR_sampled_dB}


if __name__ == "__main__":
    args = parse_args(__doc__)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    main(args.output_dir, show=not args.no_show)
