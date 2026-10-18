"""Part 3: matched filtering and the Hann sidelobe/main-lobe trade-off.

Rectangular weighting gives the narrowest main lobe in this comparison.
Hann weighting reduces sidelobes at the cost of a wider main lobe and SNR loss.
The often-quoted -13.3 and -31.5 dB values are large-time-bandwidth reference
values, not exact predictions for a sampled finite chirp in noise.
"""

import numpy as np
from scipy.signal import find_peaks

from radar_utils import (
    B, N_PRI, N_pulse, R0, c, dR, fs, make_echo, make_noise,
    matched_filter, parse_args, s_tx_pulse, save_figure, tau_p,
)


def main_lobe_width_m(y_norm, r_ax):
    """Interpolated half-power width in meters; r_ax is supplied in km."""
    peak_idx = np.argmax(y_norm)
    half_pow = y_norm[peak_idx] / np.sqrt(2)
    left = peak_idx
    while left > 0 and y_norm[left] > half_pow:
        left -= 1
    right = peak_idx
    while right < len(y_norm) - 1 and y_norm[right] > half_pow:
        right += 1
    if left == peak_idx or right == peak_idx or y_norm[left] > half_pow or y_norm[right] > half_pow:
        return np.nan
    left_r = np.interp(half_pow, y_norm[left:left + 2], r_ax[left:left + 2])
    right_r = np.interp(half_pow, y_norm[right - 1:right + 1][::-1], r_ax[right - 1:right + 1][::-1])
    return 1000 * (right_r - left_r)


def peak_sidelobe_db(y_norm):
    """Peak sidelobe outside the nearest local minima around the main peak.

    A fixed exclusion of two resolution cells incorrectly removes the first
    rectangular-window sidelobe. Locate each waveform's own main-lobe bounds.
    """
    peak_idx = int(np.argmax(y_norm))
    minima, _ = find_peaks(-np.asarray(y_norm))
    left = minima[minima < peak_idx]
    right = minima[minima > peak_idx]
    if not len(left) or not len(right):
        return np.nan
    sidelobes = np.concatenate((y_norm[:left[-1]], y_norm[right[0] + 1:]))
    if not sidelobes.size:
        return np.nan
    return 20 * np.log10(np.max(sidelobes) / y_norm[peak_idx] + 1e-30)


def main(output_dir="figures", show=True):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    s_echo = make_echo()
    noise = make_noise(rng, N_PRI)
    s_ref_rect = s_tx_pulse
    win_hann = np.hanning(N_pulse)
    s_ref_hann = s_tx_pulse * win_hann
    y_rect = matched_filter(s_echo + noise, s_ref_rect)
    y_hann = matched_filter(s_echo + noise, s_ref_hann)
    range_axis = np.arange(y_rect.size) * c / (2 * fs)
    y_rect_norm = np.abs(y_rect) / np.max(np.abs(y_rect))
    y_hann_norm = np.abs(y_hann) / np.max(np.abs(y_hann))
    y_rect_dB = 20 * np.log10(y_rect_norm + 1e-30)
    y_hann_dB = 20 * np.log10(y_hann_norm + 1e-30)
    mlw_rect = main_lobe_width_m(y_rect_norm, range_axis / 1e3)
    mlw_hann = main_lobe_width_m(y_hann_norm, range_axis / 1e3)
    psl_rect_dB = peak_sidelobe_db(y_rect_norm)
    psl_hann_dB = peak_sidelobe_db(y_hann_norm)
    mf_peak_range = range_axis[np.argmax(y_rect_norm)]
    window_loss_dB = -10 * np.log10(np.sum(win_hann)**2 / (N_pulse * np.sum(win_hann**2)))
    print("PART 3 - PULSE COMPRESSION AND WINDOWING")
    print(f"Nominal range resolution: {dR:.1f} m | Sample spacing: {c/(2*fs):.2f} m")
    print(f"Time-bandwidth product: {B*tau_p:.0f} ({10*np.log10(B*tau_p):.1f} dB)")
    print(f"Target: {R0/1e3:.3f} km | Rectangular MF peak: {mf_peak_range/1e3:.4f} km")
    print(f"Half-power width: rectangular {mlw_rect:.2f} m, Hann {mlw_hann:.2f} m")
    print(f"Peak sidelobe: rectangular {psl_rect_dB:.2f} dB, Hann {psl_hann_dB:.2f} dB")
    print(f"Ideal aligned-pulse Hann SNR loss: {window_loss_dB:.2f} dB")
    print("Range resolution c/(2B) is not the half-power width or the estimation error.")

    # A few hundred meters show the compressed pulse structure clearly.
    zoom_range = 180.0
    mask = np.abs(range_axis - R0) <= zoom_range
    r_zoom = (range_axis[mask] - R0)
    fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8))
    axes1[0].plot(range_axis / 1e3, y_rect_norm, lw=0.8)
    axes1[0].axvline(R0 / 1e3, color="red", ls="--", label="True target range")
    axes1[0].set(xlabel="Range (km)", ylabel="Normalized amplitude", title="Rectangular matched-filter output")
    axes1[1].plot(r_zoom, y_rect_norm[mask], marker=".", label="Sampled response")
    axes1[1].axhline(1 / np.sqrt(2), color="green", ls=":", label=f"Half-power width: {mlw_rect:.1f} m")
    axes1[1].axvline(0, color="red", ls="--", label="True target range")
    axes1[1].set(xlabel="Range offset from target (m)", ylabel="Normalized amplitude", title="Compressed pulse: main lobe and sidelobes")
    for ax in axes1:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    save_figure(fig1, "part3_matched_filter.png", output_dir, show)

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, values, name, width, psl, guide in zip(axes2, [y_rect_dB, y_hann_dB], ["Rectangular", "Hann"], [mlw_rect, mlw_hann], [psl_rect_dB, psl_hann_dB], [-13.3, -31.5]):
        ax.plot(r_zoom, values[mask], marker=".", lw=1.2)
        ax.axhline(guide, color="orange", ls="--", label=f"Large-TB reference: {guide:.1f} dB")
        ax.axhline(-3.0103, color="green", ls=":", label="Half power")
        ax.axvline(0, color="gray", ls=":")
        ax.set(ylim=(-65, 3), ylabel="Normalized amplitude (dB)", title=f"{name}: width {width:.1f} m | Measured peak sidelobe {psl:.1f} dB")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes2[1].set_xlabel("Range offset from target (m)")
    save_figure(fig2, "part3_window_comparison.png", output_dir, show)
    return {"mf_peak_range": mf_peak_range, "mlw_rect": mlw_rect, "mlw_hann": mlw_hann, "psl_rect_dB": psl_rect_dB, "psl_hann_dB": psl_hann_dB}


if __name__ == "__main__":
    args = parse_args(__doc__)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    main(args.output_dir, show=not args.no_show)
