"""Part 5: adaptive 1-D cell-averaging CFAR on the range profile.

At each cell under test (CUT), estimate background power from reference cells
on both sides. Guard cells exclude the target main lobe from this estimate.
The nominal multiplier is alpha = N*(Pfa**(-1/N)-1), where N is the TOTAL
reference count. Exact Pfa control requires IID exponential background power;
the oversampled matched-filter output here is correlated.
"""

import numpy as np

from radar_utils import (
    CNR_dB, N_PRI, N_pulse, Pfa, R0, c, ca_cfar, dR, fs,
    make_clutter, make_echo, make_noise, matched_filter,
    parse_args, s_tx_pulse, save_figure, sigma, vr,
)

N_ref = 16                     # Reference cells PER SIDE.
N_guard = 8                    # Guard cells PER SIDE, covering the Hann main lobe.
N_ref_total = 2 * N_ref


def simulate(seed=42, sigma_target=sigma):
    """Return the range profile and detections; a target-free scene is valid."""
    rng = np.random.default_rng(seed)
    s_echo = make_echo(sigma_target=sigma_target)
    noise = make_noise(rng, N_PRI)
    clutter = make_clutter(rng, N_PRI)
    s_rx = s_echo + noise + clutter
    s_ref = s_tx_pulse * np.hanning(N_pulse)
    y_mf = matched_filter(s_rx, s_ref)
    power = np.abs(y_mf)**2
    range_axis = np.arange(power.size) * c / (2 * fs)
    threshold, detections = ca_cfar(power, N_ref=N_ref, N_guard=N_guard, Pfa=Pfa)
    det_indices = np.flatnonzero(detections)
    best_det = int(det_indices[np.argmax(power[det_indices])]) if det_indices.size else None
    # A target association gate uses resolution, not an arbitrary +/-1.5 km.
    near_target = det_indices[np.abs(range_axis[det_indices] - R0) <= 2 * dR]
    best_target = int(near_target[np.argmax(power[near_target])]) if near_target.size else None
    margin_dB_at_target = (
        10 * np.log10(power[best_target] / threshold[best_target])
        if best_target is not None else None
    )
    return dict(power=power, threshold=threshold, range_axis=range_axis,
                det_indices=det_indices, best_det=best_det, best_target=best_target,
                near_target=near_target, margin_dB_at_target=margin_dB_at_target)


def main(output_dir="figures", show=True, sigma_target=sigma):
    import matplotlib.pyplot as plt
    result = simulate(sigma_target=sigma_target)
    power = result["power"]
    threshold = result["threshold"]
    range_axis = result["range_axis"]
    det_indices = result["det_indices"]
    best_det = result["best_det"]
    best_target = result["best_target"]
    margin_dB_at_target = result["margin_dB_at_target"]
    mf_peak_range = range_axis[np.argmax(power)] / 1e3
    alpha = N_ref_total * np.expm1(-np.log(Pfa) / N_ref_total)
    power_dB = 10 * np.log10(power + 1e-30)
    threshold_dB = 10 * np.log10(threshold + 1e-30)
    print("PART 5 - CA-CFAR DETECTION")
    print(f"Reference cells: {N_ref} per side | Guard cells: {N_guard} per side")
    print(f"Nominal Pfa: {Pfa:.0e} | Alpha: {alpha:.4f} ({10*np.log10(alpha):.2f} dB)")
    print(f"MF peak: {mf_peak_range:.4f} km | Detected cells: {len(det_indices)}")
    if best_det is not None:
        print(f"Strongest detection: {range_axis[best_det]/1e3:.4f} km")
    else:
        print("No detections.")
    if best_target is not None:
        print(f"Detection within +/-{2*dR:.0f} m of target: {range_axis[best_target]/1e3:.4f} km")
        print(f"Range error: {abs(range_axis[best_target]-R0):.2f} m | Margin: {margin_dB_at_target:.2f} dB")
    else:
        print("No detection within the target association gate.")
    print("Detected cells are not distinct targets; neighboring cells may belong to one echo.")
    print("One realization does not validate Pfa=1e-6; correlated cells require calibration.")

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(range_axis/1e3, power_dB, lw=0.6, label="Hann-weighted MF power")
    ax1.plot(range_axis/1e3, threshold_dB, color="red", lw=1, label="CA-CFAR threshold")
    ax1.axvline(R0/1e3, color="green", ls="--", label="Configured target range")
    if det_indices.size:
        ax1.plot(range_axis[det_indices]/1e3, power_dB[det_indices], "g*", label="Detected cells")
    ax1.set(xlabel="Range (km)", ylabel="Output power (dB re 1 W; unnormalized filter)",
            title=f"CA-CFAR | Nominal Pfa {Pfa:.0e} | CNR {CNR_dB:.0f} dB | Target {vr:.0f} m/s")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    save_figure(fig1, "part5_cfar_full_range.png", output_dir, show)

    # Show the target main lobe, guard region scale, and nearby reference cells.
    zoom_range = 300.0
    mask = np.abs(range_axis - R0) <= zoom_range
    r_z = range_axis[mask] - R0
    p_z = power_dB[mask]
    t_z = threshold_dB[mask]
    mg_z = p_z - t_z
    fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax_top.plot(r_z, p_z, label="MF output")
    ax_top.plot(r_z, t_z, color="red", label="Adaptive threshold")
    ax_top.fill_between(r_z, p_z, t_z, where=p_z > t_z, alpha=0.25, color="green")
    ax_top.set(ylabel="Output power (dB re 1 W)", title="Target neighborhood and adaptive threshold")
    if best_target is not None:
        ax_top.set_title(f"Target neighborhood | Detection margin {margin_dB_at_target:.2f} dB")
    ax_bot.plot(r_z, mg_z, color="darkorange", label="Signal minus threshold")
    ax_bot.axhline(0, color="red", ls="--", label="Detection boundary")
    ax_bot.fill_between(r_z, mg_z, 0, where=mg_z > 0, alpha=0.25, color="green")
    ax_bot.set(xlabel="Range offset from configured target (m)", ylabel="Detection margin (dB)", title="Positive margin indicates a detected cell")
    for ax in (ax_top, ax_bot):
        ax.axvline(0, color="green", ls=":")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    save_figure(fig2, "part5_cfar_target_zoom.png", output_dir, show)
    return result


if __name__ == "__main__":
    args = parse_args(__doc__)
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    main(args.output_dir, show=not args.no_show)
