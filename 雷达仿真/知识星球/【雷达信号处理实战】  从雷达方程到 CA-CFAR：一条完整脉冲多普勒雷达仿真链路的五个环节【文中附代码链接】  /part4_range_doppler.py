"""Part 4: fast-time pulse compression and slow-time Doppler processing.

Stationary clutter has the same amplitude and phase in every pulse. A target
produces a pulse-to-pulse phase rotation. Slow-time Hann weighting reduces
leakage while broadening the Doppler main lobe; it does not eliminate leakage.
"""

import numpy as np

from radar_utils import (
    N_PRI, PRF, PRI, R0, c, fs, lambda_c, make_clutter, make_echo,
    make_noise, matched_filter, parse_args, s_tx_pulse, save_figure, vr,
)

N_pulses = 64
v_max = lambda_c * PRF / 4
delta_v = lambda_c * PRF / (2 * N_pulses)  # Doppler-bin spacing, not an MDV guarantee.


def build_rd_map(vr_target, apply_window=True, seed_offset=0):
    """Return (normalized amplitude in dB, velocity in m/s, range in m).

    The same seed produces identical target/noise/clutter data for a fair
    window comparison. Range migration is neglected during the 32 ms CPI.
    """
    rng = np.random.default_rng(42 + seed_offset)
    fd_t = 2 * vr_target / lambda_c
    s_echo = make_echo(vr_target=vr_target)
    clutter_fix = make_clutter(rng, N_PRI)
    n_arr = np.arange(N_pulses)
    doppler_phase = np.exp(1j * 2 * np.pi * fd_t * n_arr * PRI)
    noise_mat = make_noise(rng, (N_pulses, N_PRI))
    echo_mat = s_echo[np.newaxis, :] * doppler_phase[:, np.newaxis]
    data = echo_mat + noise_mat + clutter_fix[np.newaxis, :]

    # Linear matched filtering along fast time, then a slow-time FFT.
    data_mf = matched_filter(data, s_tx_pulse)
    if apply_window:
        win_slow = np.hanning(N_pulses)
        data_mf = data_mf * win_slow[:, np.newaxis]
    RD = np.fft.fftshift(np.fft.fft(data_mf, axis=0), axes=0)
    RD_dB = 20 * np.log10(np.abs(RD) + 1e-30)
    RD_dB_norm = RD_dB - np.max(RD_dB)
    range_ax = np.arange(data_mf.shape[1]) * c / (2 * fs)
    dopp_freqs = np.fft.fftshift(np.fft.fftfreq(N_pulses, d=PRI))
    vel_ax = dopp_freqs * lambda_c / 2
    return RD_dB_norm, vel_ax, range_ax


def draw_map(ax, RD, vel_ax, rng_ax, v_apparent, title):
    """Plot bin centers at the correct coordinates using half-bin edges."""
    dv = vel_ax[1] - vel_ax[0]
    dr = (rng_ax[1] - rng_ax[0]) / 1e3
    ext_l = [vel_ax[0] - dv/2, vel_ax[-1] + dv/2,
             rng_ax[0]/1e3 - dr/2, rng_ax[-1]/1e3 + dr/2]
    im = ax.imshow(RD.T, extent=ext_l, origin="lower", aspect="auto",
                   cmap="viridis", vmin=-60, vmax=0, interpolation="nearest")
    ax.axvline(0, color="cyan", ls="--", lw=0.8, label="Stationary clutter")
    ax.axhline(R0/1e3, color="white", ls=":", lw=0.8)
    ax.plot(v_apparent, R0/1e3, "+", color="red", ms=10, mew=1.8, label="Expected target")
    ax.set(xlabel="Velocity (m/s, positive = closing)", ylabel="Range (km)", title=title)
    ax.legend(fontsize=8, loc="upper left")
    return im


def main(output_dir="figures", show=True):
    import matplotlib.pyplot as plt
    RD_win, vel_ax, rng_ax = build_rd_map(vr, apply_window=True)
    RD_nowin, _, _ = build_rd_map(vr, apply_window=False)
    print("PART 4 - RANGE-DOPPLER PROCESSING")
    print(f"Pulses: {N_pulses} | CPI: {N_pulses*PRI*1e3:.1f} ms")
    print(f"Unambiguous velocity: [-{v_max:.3f}, +{v_max:.3f}) m/s")
    print(f"Doppler-bin spacing: {delta_v:.4f} m/s")
    print("Minimum detectable velocity also depends on clutter, windowing, and the detector.")
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle(f"Range-Doppler maps | Target {R0/1e3:.0f} km, {vr:.0f} m/s", fontweight="bold")
    for ax, RD, title in [(ax_l, RD_nowin, "Rectangular slow-time window"), (ax_r, RD_win, "Hann slow-time window")]:
        im = draw_map(ax, RD, vel_ax, rng_ax, vr, title)
        fig2.colorbar(im, ax=ax, label="Amplitude relative to each map peak (dB)", fraction=0.045)
    save_figure(fig2, "part4_range_doppler.png", output_dir, show)

    # Modulo wrapping handles any signed Doppler, not just one negative alias.
    vr_neg30 = -30.0
    fd_neg30 = 2 * vr_neg30 / lambda_c
    fd_alias = (fd_neg30 + PRF / 2) % PRF - PRF / 2
    v_alias = fd_alias * lambda_c / 2
    RD_neg30, _, _ = build_rd_map(vr_neg30, seed_offset=10)
    print(f"True velocity: {vr_neg30:.1f} m/s (receding) | True Doppler: {fd_neg30:.1f} Hz")
    print(f"Aliased Doppler: {fd_alias:.1f} Hz | Apparent velocity: {v_alias:+.3f} m/s")
    print("A higher PRF widens the velocity interval but reduces unambiguous range.")
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    im3 = draw_map(ax3, RD_neg30, vel_ax, rng_ax, v_alias,
                   f"Velocity ambiguity: {vr_neg30:.0f} m/s appears at {v_alias:+.2f} m/s")
    fig3.colorbar(im3, ax=ax3, label="Normalized amplitude (dB)")
    save_figure(fig3, "part4_velocity_ambiguity.png", output_dir, show)
    # Estimate velocity at the expected target-range bin, away from clutter.
    r_idx = np.argmin(np.abs(rng_ax - R0))
    v_est = vel_ax[np.argmax(RD_win[:, r_idx])]
    v_alias_est = vel_ax[np.argmax(RD_neg30[:, r_idx])]
    print(f"Measured target Doppler bins: {v_est:.3f} and {v_alias_est:.3f} m/s")
    return {"v_est": v_est, "v_alias": v_alias, "v_alias_est": v_alias_est, "delta_v": delta_v}


if __name__ == "__main__":
    args = parse_args(__doc__)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    main(args.output_dir, show=not args.no_show)
