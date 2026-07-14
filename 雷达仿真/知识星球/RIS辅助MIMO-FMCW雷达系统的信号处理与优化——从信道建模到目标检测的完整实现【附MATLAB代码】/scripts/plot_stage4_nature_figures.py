"""Create Nature-style Stage 4 two-dimensional and three-dimensional RD figures.

The script reads the MATLAB source data produced by main_stage4_rd_detection.m.
It only handles plotting/export; MATLAB remains responsible for radar simulation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "outputs" / "data" / "stage4_rd_four_targets_latest.mat"
CFAR_DATA_PATH = PROJECT_ROOT / "outputs" / "data" / "stage4_rd_four_targets_cfar_latest.mat"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
HEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "rd_height_layers",
    ["#EDF3F8", "#C6DCEB", "#80B7D2", "#3E86B5", "#1E567F"],
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def field(obj, name: str) -> np.ndarray:
    return np.asarray(getattr(obj, name)).astype(float).reshape(-1)


def export_figure(fig: mpl.figure.Figure, stem: str, *, tight_bbox: bool = True) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / stem
    save_kwargs = {"bbox_inches": "tight"} if tight_bbox else {}
    fig.savefig(base.with_suffix(".svg"), **save_kwargs)
    #fig.savefig(base.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(base.with_suffix(".png"), dpi=600, **save_kwargs)
    #fig.savefig(base.with_suffix(".tiff"), dpi=600, **save_kwargs)


def load_source(path: Path = DATA_PATH) -> dict[str, object]:
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def adapt_cfar_source(data: dict[str, object]) -> dict[str, object]:
    """Map CFAR-specific MATLAB source fields onto the common plotting contract."""
    adapted = dict(data)
    adapted["noRisDetection"] = data["noRisCfarDetection"]
    adapted["randomDetection"] = data["randomCfarDetection"]
    adapted["optimizedDetection"] = data["optimizedCfarDetection"]
    adapted["rdPeakImprovementVsNoRisDb"] = data["cfarPeakImprovementVsNoRisDb"]
    adapted["rdPeakImprovementDb"] = data["cfarPeakImprovementDb"]
    return adapted


def crop_rd(
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    range_lim: tuple[float, float],
    velocity_lim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    range_mask = (range_axis >= range_lim[0]) & (range_axis <= range_lim[1])
    velocity_mask = (velocity_axis >= velocity_lim[0]) & (velocity_axis <= velocity_lim[1])
    return rd_db[np.ix_(range_mask, velocity_mask)], range_axis[range_mask], velocity_axis[velocity_mask]


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_heatmap(
    ax: mpl.axes.Axes,
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    title: str,
    clim: tuple[float, float],
    targets,
    detection,
    show_ylabel: bool,
) -> mpl.image.AxesImage:
    image = ax.imshow(
        rd_db,
        origin="lower",
        aspect="auto",
        extent=[velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]],
        cmap="viridis",
        vmin=clim[0],
        vmax=clim[1],
        interpolation="nearest",
    )
    ax.scatter(field(targets, "velocity_mps"), field(targets, "range_m"), marker="x", s=28, c="#D62728", linewidths=1.1)
    ax.scatter(
        field(detection, "peakVelocity_mps"),
        field(detection, "peakRange_m"),
        marker="o",
        s=22,
        facecolors="none",
        edgecolors="white",
        linewidths=0.9,
    )
    for idx, (vel, rng) in enumerate(zip(field(targets, "velocity_mps"), field(targets, "range_m")), start=1):
        ax.text(vel + 0.12, rng + 0.45, f"T{idx}", color="white", fontsize=6.4, weight="bold")
    ax.set_title(title, pad=3)
    ax.set_xlabel("Velocity (m s$^{-1}$)")
    if show_ylabel:
        ax.set_ylabel("Range (m)")
    else:
        ax.set_yticklabels([])
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 28)
    ax.tick_params(length=2.5, width=0.65)
    return image


def make_2d_figure(data: dict[str, object], stem: str = "stage4_rd_four_targets_nature_2d") -> None:
    range_axis = np.asarray(data["rangeAxis"], dtype=float).reshape(-1)
    velocity_axis = np.asarray(data["velocityAxis"], dtype=float).reshape(-1)
    rd_no_ris = np.asarray(data["RDnoRisDb"], dtype=float)
    rd_random = np.asarray(data["RDrandomDb"], dtype=float)
    rd_opt = np.asarray(data["RDoptimizedDb"], dtype=float)
    targets = data["targets"]
    det_no_ris = data["noRisDetection"]
    det_random = data["randomDetection"]
    det_opt = data["optimizedDetection"]
    improvement_vs_random = np.asarray(data["rdPeakImprovementDb"], dtype=float).reshape(-1)
    improvement_vs_no_ris = np.asarray(data["rdPeakImprovementVsNoRisDb"], dtype=float).reshape(-1)

    rd_no_ris_crop, range_crop, velocity_crop = crop_rd(rd_no_ris, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    rd_random_crop, range_crop, velocity_crop = crop_rd(rd_random, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    rd_opt_crop, _, _ = crop_rd(rd_opt, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    vmax = float(np.nanmax([rd_no_ris_crop.max(), rd_random_crop.max(), rd_opt_crop.max()]))
    clim = (vmax - 48, vmax)

    fig = plt.figure(figsize=(8.6, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.92], height_ratios=[1, 0.88])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[1, 3])

    image = draw_heatmap(ax_a, rd_no_ris_crop, range_crop, velocity_crop, "No RIS", clim, targets, det_no_ris, True)
    draw_heatmap(ax_b, rd_random_crop, range_crop, velocity_crop, "Random RIS", clim, targets, det_random, False)
    draw_heatmap(ax_c, rd_opt_crop, range_crop, velocity_crop, "Fixed-grid ZF-SNR RIS", clim, targets, det_opt, False)
    cbar = fig.colorbar(image, ax=[ax_a, ax_b, ax_c], fraction=0.025, pad=0.018)
    cbar.set_label("Magnitude (dB)")
    cbar.ax.tick_params(length=2.5, width=0.65)

    target_ids = np.arange(1, improvement_vs_random.size + 1)
    no_ris_peaks = field(det_no_ris, "peakDb")
    random_peaks = field(det_random, "peakDb")
    opt_peaks = field(det_opt, "peakDb")
    ax_d.plot(target_ids, no_ris_peaks, "o-", color="#8D8D8D", lw=0.85, ms=3.0, label="no RIS")
    ax_d.plot(target_ids, random_peaks, "o-", color="#555555", lw=0.9, ms=3.2, label="random")
    ax_d.plot(target_ids, opt_peaks, "o-", color="#0072B2", lw=0.9, ms=3.2, label="optimized")
    for tid, y0, y1 in zip(target_ids, no_ris_peaks, opt_peaks):
        ax_d.plot([tid, tid], [y0, y1], color="#D5E5F1", lw=0.7, zorder=0)
    ax_d.set_xticks(target_ids)
    ax_d.set_xticklabels([f"T{i}" for i in target_ids])
    ax_d.set_ylabel("Local peak (dB)")
    ax_d.set_title("Peak recovery")
    ax_d.legend(loc="lower right", handlelength=1.4, fontsize=5.6)
    ax_d.grid(axis="y", color="#D9D9D9", lw=0.45)

    width = 0.34
    ax_e.bar(target_ids - width / 2, improvement_vs_no_ris, width, color="#A9D4E8", edgecolor="#356B8C", linewidth=0.4, label="vs no RIS")
    ax_e.bar(target_ids + width / 2, improvement_vs_random, width, color="#56B4E9", edgecolor="#2F5D7C", linewidth=0.4, label="vs random")
    ax_e.axhline(0, color="black", lw=0.6)
    ax_e.set_xticks(target_ids)
    ax_e.set_xticklabels([f"T{i}" for i in target_ids])
    ax_e.set_ylabel("Gain (dB)")
    ax_e.set_title(f"Gain, mean {improvement_vs_random.mean():.2f} dB vs random")
    finite_vs_no_ris = improvement_vs_no_ris[np.isfinite(improvement_vs_no_ris)]
    gain_ceiling = finite_vs_no_ris.max() + 1 if finite_vs_no_ris.size else improvement_vs_random.max() + 1
    ax_e.set_ylim(0, max(12, gain_ceiling))
    ax_e.legend(loc="upper right", fontsize=5.6)
    ax_e.grid(axis="y", color="#D9D9D9", lw=0.45)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], ["a", "b", "c", "d", "e"]):
        add_panel_label(ax, label)

    export_figure(fig, stem)
    plt.close(fig)


def prepare_3d_rd(
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    floor_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rd_crop, r_crop, v_crop = crop_rd(rd_db, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    stride_r = max(1, rd_crop.shape[0] // 34)
    stride_v = max(1, rd_crop.shape[1] // 28)
    rd_plot = rd_crop[::stride_r, ::stride_v]
    # Keep the noise floor continuous while compressing weak bins into a narrow
    # lower visual band. Hard clipping makes the 3D surface look artificially
    # empty below the target peaks.
    weak_bins = rd_plot < floor_db
    rd_plot[weak_bins] = floor_db + 0.15 * (rd_plot[weak_bins] - floor_db)
    return rd_plot, r_crop[::stride_r], v_crop[::stride_v]


def normalize_3d_heights(values: np.ndarray, zlim: tuple[float, float]) -> np.ndarray:
    span = max(zlim[1] - zlim[0], np.finfo(float).eps)
    return np.clip((values - zlim[0]) / span, 0, 1)


def layered_facecolors(rd_plot: np.ndarray, zlim: tuple[float, float], alpha: float) -> np.ndarray:
    facecolors = HEIGHT_CMAP(normalize_3d_heights(rd_plot, zlim))
    facecolors[..., -1] = alpha
    return facecolors


def style_3d_axis(
    ax,
    title: str,
    zlim: tuple[float, float],
) -> None:
    ax.set_title(title, pad=4)
    ax.set_xlabel("Velocity (m s$^{-1}$)", labelpad=-1)
    ax.set_ylabel("Range (m)", labelpad=-1)
    ax.set_zlabel("Magnitude (dB)", labelpad=-1)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 28)
    ax.set_zlim(zlim)
    ax.view_init(elev=29, azim=-54)
    ax.set_box_aspect((1.15, 1.15, 0.55))
    ax.tick_params(axis="both", which="major", pad=-2, labelsize=6)
    ax.tick_params(axis="z", which="major", pad=0, labelsize=6)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.xaxis.pane.set_edgecolor("#D9D9D9")
    ax.yaxis.pane.set_edgecolor("#D9D9D9")
    ax.zaxis.pane.set_edgecolor("#D9D9D9")


def mark_3d_peaks(ax, detection) -> None:
    if detection is None:
        return
    ax.scatter(
        field(detection, "peakVelocity_mps"),
        field(detection, "peakRange_m"),
        field(detection, "peakDb") + 0.8,
        c="#D62728",
        s=26,
        marker="x",
        linewidths=1.1,
        depthshade=False,
        zorder=10,
    )


def clean_surface_panel(
    ax,
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    title: str,
    zlim: tuple[float, float],
    floor_db: float,
    detection,
) -> None:
    rd_plot, r_plot, v_plot = prepare_3d_rd(rd_db, range_axis, velocity_axis, floor_db)
    V, R = np.meshgrid(v_plot, r_plot)
    ax.plot_surface(
        V,
        R,
        rd_plot,
        facecolors=layered_facecolors(rd_plot, zlim, 0.92),
        edgecolor="#C3D8EB",
        linewidth=0.16,
        antialiased=True,
        shade=False,
        alpha=0.90,
    )
    mark_3d_peaks(ax, detection)
    style_3d_axis(ax, title, zlim)


def wireframe_panel(
    ax,
    rd_db: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    title: str,
    zlim: tuple[float, float],
    floor_db: float,
    detection,
) -> None:
    rd_plot, r_plot, v_plot = prepare_3d_rd(rd_db, range_axis, velocity_axis, floor_db)
    V, R = np.meshgrid(v_plot, r_plot)
    ax.plot_wireframe(
        V,
        R,
        rd_plot,
        rstride=1,
        cstride=1,
        color="#7EA6C8",
        linewidth=0.40,
        alpha=0.95,
    )
    ax.plot_surface(
        V,
        R,
        rd_plot,
        facecolors=layered_facecolors(rd_plot, zlim, 0.22),
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    mark_3d_peaks(ax, detection)
    style_3d_axis(ax, title, zlim)


def shared_3d_limits(
    rd_no_ris: np.ndarray,
    rd_random: np.ndarray,
    rd_opt: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
) -> tuple[float, tuple[float, float]]:
    """Return the shared Stage 4 3D display floor and z-axis limits."""
    no_ris_crop, _, _ = crop_rd(rd_no_ris, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    random_crop, _, _ = crop_rd(rd_random, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    opt_crop, _, _ = crop_rd(rd_opt, range_axis, velocity_axis, (0, 28), (-3.2, 3.2))
    vmax = float(np.nanmax([no_ris_crop.max(), random_crop.max(), opt_crop.max()]))
    floor_db = vmax - 40
    vmin = float(np.nanmin([no_ris_crop.min(), random_crop.min(), opt_crop.min()]))
    compressed_floor = floor_db + 0.15 * (vmin - floor_db)
    zlim = (max(floor_db - 6, compressed_floor), vmax + 2)
    return floor_db, zlim


def make_clean_surface_panels(
    data: dict[str, object],
    stem_prefix: str = "stage4_rd_four_targets_nature_3d_clean_surface",
) -> None:
    range_axis = np.asarray(data["rangeAxis"], dtype=float).reshape(-1)
    velocity_axis = np.asarray(data["velocityAxis"], dtype=float).reshape(-1)
    rd_no_ris = np.asarray(data["RDnoRisDb"], dtype=float)
    rd_random = np.asarray(data["RDrandomDb"], dtype=float)
    rd_opt = np.asarray(data["RDoptimizedDb"], dtype=float)
    floor_db, zlim = shared_3d_limits(rd_no_ris, rd_random, rd_opt, range_axis, velocity_axis)
    scalar_map = mpl.cm.ScalarMappable(norm=Normalize(vmin=zlim[0], vmax=zlim[1]), cmap=HEIGHT_CMAP)
    panel_specs = [
        ("No RIS", "no_ris", rd_no_ris, None),
        ("Random RIS", "random_ris", rd_random, data["randomDetection"]),
        ("Optimized RIS", "optimized_ris", rd_opt, data["optimizedDetection"]),
    ]

    for title, stem_suffix, rd_db, detection in panel_specs:
        fig = plt.figure(figsize=(3.55, 3.25))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        clean_surface_panel(ax, rd_db, range_axis, velocity_axis, title, zlim, floor_db, detection)
        ax.set_zlabel("")
        cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.045, pad=0.08, shrink=0.56)
        cbar.set_label("Magnitude (dB)")
        cbar.ax.tick_params(length=2.2, width=0.6, labelsize=5.8)
        fig.subplots_adjust(left=0.00, right=0.92, bottom=0.03, top=0.92)
        export_figure(fig, f"{stem_prefix}_{stem_suffix}")
        plt.close(fig)


def make_clean_surface_comparison_figure(
    data: dict[str, object],
    stem: str = "stage4_rd_four_targets_nature_3d_clean_surface_comparison",
) -> None:
    """Create one 1x3 clean-surface comparison figure with one compact shared colorbar.

    This figure keeps the same z-axis limits, colormap and display floor for
    No RIS, Random RIS and Optimized RIS. The colorbar is deliberately short
    and placed in a reserved right margin, so it will not cover the z-axis
    ticks of panel c.
    """
    range_axis = np.asarray(data["rangeAxis"], dtype=float).reshape(-1)
    velocity_axis = np.asarray(data["velocityAxis"], dtype=float).reshape(-1)
    rd_no_ris = np.asarray(data["RDnoRisDb"], dtype=float)
    rd_random = np.asarray(data["RDrandomDb"], dtype=float)
    rd_opt = np.asarray(data["RDoptimizedDb"], dtype=float)

    floor_db, zlim = shared_3d_limits(rd_no_ris, rd_random, rd_opt, range_axis, velocity_axis)
    scalar_map = mpl.cm.ScalarMappable(norm=Normalize(vmin=zlim[0], vmax=zlim[1]), cmap=HEIGHT_CMAP)
    scalar_map.set_array([])

    # Reserve a visible right margin for the compact shared colorbar.
    # Do not put the colorbar inside the GridSpec, otherwise it can overlap
    # the right-side z-axis ticks/labels of panel c after 3D projection.
    fig = plt.figure(figsize=(9.7, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.00)
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")

    clean_surface_panel(ax_a, rd_no_ris, range_axis, velocity_axis, "No RIS", zlim, floor_db, None)
    clean_surface_panel(ax_b, rd_random, range_axis, velocity_axis, "Random RIS", zlim, floor_db, data["randomDetection"])
    clean_surface_panel(ax_c, rd_opt, range_axis, velocity_axis, "Optimized RIS", zlim, floor_db, data["optimizedDetection"])

    # Keep only coordinate tick values on z-axis; the single semantic magnitude
    # label is provided by the shared colorbar to avoid duplicate "Magnitude".
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_zlabel("")
    ax_b.set_ylabel("")
    ax_c.set_ylabel("")

    for ax, label in zip([ax_a, ax_b, ax_c], ["a", "b", "c"]):
        ax.text2D(-0.02, 0.98, label, transform=ax.transAxes, fontsize=9, fontweight="bold")

    # A shorter colorbar is easier to read and avoids visually dominating the figure.
    cax = fig.add_axes([0.935, 0.31, 0.012, 0.38])
    cbar = fig.colorbar(scalar_map, cax=cax)
    cbar.set_label("Magnitude (dB)", labelpad=4)
    cbar.ax.tick_params(length=2.2, width=0.6, labelsize=5.8, pad=1.5)
    cbar.outline.set_linewidth(0.55)

    fig.subplots_adjust(left=0.00, right=0.905, bottom=0.03, top=0.93)
    export_figure(fig, stem)
    plt.close(fig)


def make_3d_figure(data: dict[str, object], variant: str, stem: str | None = None) -> None:
    range_axis = np.asarray(data["rangeAxis"], dtype=float).reshape(-1)
    velocity_axis = np.asarray(data["velocityAxis"], dtype=float).reshape(-1)
    rd_no_ris = np.asarray(data["RDnoRisDb"], dtype=float)
    rd_random = np.asarray(data["RDrandomDb"], dtype=float)
    rd_opt = np.asarray(data["RDoptimizedDb"], dtype=float)
    det_no_ris = data["noRisDetection"]
    det_random = data["randomDetection"]
    det_opt = data["optimizedDetection"]

    floor_db, zlim = shared_3d_limits(rd_no_ris, rd_random, rd_opt, range_axis, velocity_axis)

    fig = plt.figure(figsize=(9.0, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.01)
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    if variant == "wireframe":
        wireframe_panel(ax_a, rd_no_ris, range_axis, velocity_axis, "No RIS", zlim, floor_db, None)
        wireframe_panel(ax_b, rd_random, range_axis, velocity_axis, "Random RIS", zlim, floor_db, det_random)
        wireframe_panel(ax_c, rd_opt, range_axis, velocity_axis, "Optimized RIS", zlim, floor_db, det_opt)
    else:
        raise ValueError(f"Unsupported 3D variant: {variant}")
    ax_a.text2D(-0.02, 0.98, "a", transform=ax_a.transAxes, fontsize=9, fontweight="bold")
    ax_b.text2D(-0.02, 0.98, "b", transform=ax_b.transAxes, fontsize=9, fontweight="bold")
    ax_c.text2D(-0.02, 0.98, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.00, right=0.99, bottom=0.03, top=0.93)
    if stem is None:
        stem = f"stage4_rd_four_targets_nature_3d_{variant}"
    export_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    data = load_source()
    make_2d_figure(data)
    make_clean_surface_panels(data)
    make_clean_surface_comparison_figure(data)
    make_3d_figure(data, "wireframe")
    if CFAR_DATA_PATH.exists():
        cfar_data = adapt_cfar_source(load_source(CFAR_DATA_PATH))
        make_2d_figure(cfar_data, "stage4_rd_four_targets_cfar_nature_2d")
        make_clean_surface_panels(cfar_data, "stage4_rd_four_targets_cfar_nature_3d_clean_surface")
        make_clean_surface_comparison_figure(
            cfar_data,
            "stage4_rd_four_targets_cfar_nature_3d_clean_surface_comparison",
        )
        make_3d_figure(cfar_data, "wireframe", "stage4_rd_four_targets_cfar_nature_3d_wireframe")
    print(f"Saved Nature-style figures under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
