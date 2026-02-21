import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

from configs.data import (
    categories,
    output_dir,
    root_dir,
    transform_funcs,
)
from configs.ml import model
from data.dataloader import PRBSDataloader
from matplotlib import cm


def plot_distributions_3d(
    overal_distributions,
    aggregation="mean",
    figsize=(20, 18),
    split_figures=False,
    output_dir="3d_plots",
):
    """
    Plot 3D surface plots for each distribution in overal_distributions.

    Parameters:
    -----------
    overal_distributions : dict
        Dictionary with keys as distribution names and values as numpy arrays
        with shape [K, C, H, W] where K is number of trials, C is channels,
        H is height (255), W is width (256)
    aggregation : str
        Method to aggregate over dimensions 0 and 1. Options: 'mean', 'sum', 'median'
    figsize : tuple
        Figure size for the plot (or individual figures if split_figures=True)
    split_figures : bool
        If True, create individual figures for each distribution and save to folder.
        If False, create a single figure with all distributions as subplots.
    output_dir : str
        Directory to save individual figures when split_figures=True
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    if split_figures:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot each distribution in a separate figure
        for key, value in overal_distributions.items():
            # Aggregate over dimensions 0 and 1
            if aggregation == "mean":
                aggregated = np.mean(value, axis=(0, 1))
            elif aggregation == "sum":
                aggregated = np.sum(value, axis=(0, 1))
            elif aggregation == "median":
                aggregated = np.median(value, axis=(0, 1))
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            # aggregated shape should be (255, 256)
            H, W = aggregated.shape

            # Create meshgrid for x and y coordinates
            X = np.arange(W)  # 0 to 255 (width/x-axis)
            Y = np.arange(H)  # 0 to 254 (height/y-axis)
            X, Y = np.meshgrid(X, Y)

            # Create individual figure
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            # Plot surface
            surf = ax.plot_surface(
                X,
                Y,
                abs(aggregated),
                cmap=cm.viridis,
                linewidth=0,
                antialiased=True,
                alpha=0.8,
            )

            # Labels
            clean_key = key.strip("$")
            ax.set_xlabel("X (Width: 256)", fontsize=10, labelpad=8)
            ax.set_ylabel("Y (Height: 255)", fontsize=10, labelpad=8)
            ax.set_zlabel("Value", fontsize=10, labelpad=8)

            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

            # Adjust viewing angle
            ax.view_init(elev=30, azim=45)

            # Set title
            ax.set_title(f"${clean_key}$ ({aggregation})", fontsize=12, pad=20)

            # Save individual figure
            safe_filename = (
                clean_key.replace("\\", "").replace("/", "_").replace(" ", "_")
            )
            save_path = os.path.join(
                output_dir, f"{safe_filename}_3d_{aggregation}.png"
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
            print(f"Saved 3D plot to {save_path}")
            plt.close()

        return None

    else:
        # Original behavior: plot all in one figure
        n_distributions = len(overal_distributions)
        n_cols = 3
        n_rows = (n_distributions + n_cols - 1) // n_cols

        fig = plt.figure(figsize=figsize)

        for idx, (key, value) in enumerate(overal_distributions.items(), 1):
            # Aggregate over dimensions 0 and 1
            if aggregation == "mean":
                aggregated = np.mean(value, axis=(0, 1))
            elif aggregation == "sum":
                aggregated = np.sum(value, axis=(0, 1))
            elif aggregation == "median":
                aggregated = np.median(value, axis=(0, 1))
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            H, W = aggregated.shape
            X = np.arange(W)
            Y = np.arange(H)
            X, Y = np.meshgrid(X, Y)

            ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")

            surf = ax.plot_surface(
                X,
                Y,
                abs(aggregated),
                cmap=cm.viridis,
                linewidth=0,
                antialiased=True,
                alpha=0.8,
            )

            clean_key = key.strip("$")
            ax.set_xlabel("X (Width: 256)", fontsize=8, labelpad=5)
            ax.set_ylabel("Y (Height: 255)", fontsize=8, labelpad=5)
            ax.set_zlabel("Value", fontsize=8, labelpad=5)

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            ax.view_init(elev=30, azim=45)
            ax.tick_params(labelsize=6)

            ax.text2D(
                0.5,
                -0.15,
                f"${clean_key}$ ({aggregation})",
                transform=ax.transAxes,
                fontsize=10,
                ha="center",
                va="top",
            )

        plt.tight_layout(pad=3.0, h_pad=5.0, w_pad=3.0)
        save_path = f"distributions_3d_{aggregation}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.5)
        print(f"Saved 3D plot to {save_path}")
        plt.close()

        return fig


def plot_distributions_2d_heatmap(
    overal_distributions,
    aggregation="mean",
    bins=100,
    figsize=(20, 18),
    split_figures=False,
    output_dir="2d_plots",
    comparison_groups=None,
    use_log_scale=False,
    normalize_densities=True,
    use_dual_axis=False,
    use_sqrt_scale=False,
    xlim_percentile=None,
    xlim_density_threshold=0.001,
):
    """
    Plot 2D histograms (PDF) for each distribution in overal_distributions.

    Parameters:
    -----------
    overal_distributions : dict
        Dictionary with keys as distribution names and values as numpy arrays
    aggregation : str
        Method to aggregate over dimensions 0 and 1. Options: 'mean', 'sum', 'median'
    bins : int
        Number of bins for histogram
    figsize : tuple
        Figure size for the plot (or individual figures if split_figures=True)
    split_figures : bool
        If True, create individual figures for each distribution and save to folder.
        If False, create a single figure with all distributions as subplots.
    output_dir : str
        Directory to save individual figures when split_figures=True
    comparison_groups : list of lists or tuples, optional
        Groups of distribution keys to plot together for comparison.
        Example: [['dist1', 'dist2'], ['dist3', 'dist4', 'dist5']]
        If provided, these groups will be plotted as overlaid distributions.
    use_log_scale : bool, optional
        If True, use log scale for y-axis to better visualize distributions with different scales.
    normalize_densities : bool, optional
        If True, normalize each density to have max value of 1 for better comparison.
    use_dual_axis : bool, optional
        If True, use dual y-axes with different scales for each distribution.
    use_sqrt_scale : bool, optional
        If True, use square root scale for y-axis (gentler than log scale).
    xlim_percentile : tuple or float, optional
        Set x-axis limits based on data percentiles.
        - If tuple (e.g., (1, 99)): use 1st and 99th percentiles
        - If single value (e.g., 99): use symmetric range around median
        - If None: automatically determine based on density threshold
    xlim_density_threshold : float, optional
        When xlim_percentile is None, set x-axis limits to regions where
        density > threshold * max_density. Default: 0.001 (0.1% of max)
    """
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    # Define colors for comparison plots
    comparison_colors = [
        "steelblue",
        "coral",
        "mediumseagreen",
        "mediumpurple",
        "gold",
        "crimson",
        "teal",
        "orange",
    ]

    if split_figures:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Handle comparison groups
        if comparison_groups:
            plotted_keys = set()

            # Plot comparison groups
            for group_idx, group in enumerate(comparison_groups):
                fig, ax = plt.subplots(figsize=(8, 6))

                # Store all densities for scaling
                all_densities = []
                plot_data = []

                for idx, key in enumerate(group):
                    if key not in overal_distributions:
                        print(f"Warning: Key '{key}' not found in distributions")
                        continue

                    value = overal_distributions[key]
                    plotted_keys.add(key)

                    # Aggregate over dimensions 0 and 1
                    if aggregation == "mean":
                        aggregated = np.mean(value, axis=(0, 1))
                    elif aggregation == "sum":
                        aggregated = np.sum(value, axis=(0, 1))
                    elif aggregation == "median":
                        aggregated = np.median(value, axis=(0, 1))
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregation}")

                    # Flatten data
                    flattened_data = aggregated.ravel()
                    color = comparison_colors[idx % len(comparison_colors)]
                    clean_key = key.strip("$")

                    # Use KDE for smooth density estimation
                    kde = stats.gaussian_kde(flattened_data)
                    x_range = np.linspace(
                        flattened_data.min(), flattened_data.max(), 500
                    )
                    density = kde(x_range)

                    # Store for later processing
                    plot_data.append(
                        {
                            "x_range": x_range,
                            "density": density,
                            "color": color,
                            "label": f"${clean_key}$",
                            "max_density": np.max(density),
                        }
                    )
                    all_densities.append(density)

                # Use dual axis if requested and there are exactly 2 distributions
                if use_dual_axis and len(plot_data) == 2:
                    # First distribution on left axis
                    ax.fill_between(
                        plot_data[0]["x_range"],
                        plot_data[0]["density"],
                        alpha=0.5,
                        color=plot_data[0]["color"],
                        label=plot_data[0]["label"],
                    )
                    ax.plot(
                        plot_data[0]["x_range"],
                        plot_data[0]["density"],
                        color=plot_data[0]["color"],
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax.set_ylabel(
                        f"Density - {plot_data[0]['label']}",
                        fontsize=10,
                        color=plot_data[0]["color"],
                    )
                    ax.tick_params(axis="y", labelcolor=plot_data[0]["color"])

                    # Second distribution on right axis
                    ax2 = ax.twinx()
                    ax2.fill_between(
                        plot_data[1]["x_range"],
                        plot_data[1]["density"],
                        alpha=0.5,
                        color=plot_data[1]["color"],
                        label=plot_data[1]["label"],
                    )
                    ax2.plot(
                        plot_data[1]["x_range"],
                        plot_data[1]["density"],
                        color=plot_data[1]["color"],
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax2.set_ylabel(
                        f"Density - {plot_data[1]['label']}",
                        fontsize=10,
                        color=plot_data[1]["color"],
                    )
                    ax2.tick_params(axis="y", labelcolor=plot_data[1]["color"])

                    # Combine legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(
                        lines1 + lines2, labels1 + labels2, loc="best", fontsize=9
                    )

                    ylabel = "Density (dual axes)"
                else:
                    # Plot all densities on same axis with transformations
                    for data in plot_data:
                        density = data["density"]

                        # Apply transformations
                        if normalize_densities:
                            density = density / np.max(density)

                        if use_sqrt_scale:
                            density = np.sqrt(density)

                        # Plot filled density curve
                        ax.fill_between(
                            data["x_range"],
                            density,
                            alpha=0.5,
                            color=data["color"],
                            label=data["label"],
                        )
                        ax.plot(
                            data["x_range"],
                            density,
                            color=data["color"],
                            linewidth=1.5,
                            alpha=0.8,
                        )

                    # Set y-axis label based on transformations
                    if use_log_scale:
                        ax.set_yscale("log")
                        ylabel = "Density (log scale)"
                        ax.set_ylim(bottom=1e-6)
                    elif use_sqrt_scale and normalize_densities:
                        ylabel = "Normalized Density"
                    elif use_sqrt_scale:
                        ylabel = "Density"
                    elif normalize_densities:
                        ylabel = "Normalized Density"
                    else:
                        ylabel = "Density"

                    ax.set_ylabel(ylabel, fontsize=10)
                    ax.legend(loc="best", fontsize=9)

                # Set x-axis limits based on where density is significant
                if xlim_percentile is not None:
                    # Use percentile-based limits
                    all_data = []
                    for data in plot_data:
                        # Reconstruct data points from KDE (approximate)
                        # Better: use original flattened data if available
                        all_data.extend(data["x_range"])

                    if (
                        isinstance(xlim_percentile, (list, tuple))
                        and len(xlim_percentile) == 2
                    ):
                        x_min = np.percentile(all_data, xlim_percentile[0])
                        x_max = np.percentile(all_data, xlim_percentile[1])
                    else:
                        # Symmetric range
                        median = np.median(all_data)
                        p_low = (100 - xlim_percentile) / 2
                        p_high = 100 - p_low
                        x_min = np.percentile(all_data, p_low)
                        x_max = np.percentile(all_data, p_high)

                    ax.set_xlim(x_min, x_max)
                else:
                    # Automatically determine limits based on density threshold
                    x_min_all = float("inf")
                    x_max_all = float("-inf")

                    for data in plot_data:
                        max_dens = data["max_density"]
                        threshold = xlim_density_threshold * max_dens

                        # Find where density exceeds threshold
                        significant_mask = data["density"] > threshold
                        if np.any(significant_mask):
                            significant_x = data["x_range"][significant_mask]
                            x_min_all = min(x_min_all, significant_x.min())
                            x_max_all = max(x_max_all, significant_x.max())

                    if x_min_all != float("inf") and x_max_all != float("-inf"):
                        # Add some padding (5% on each side)
                        x_range = x_max_all - x_min_all
                        padding = 0.05 * x_range
                        ax.set_xlim(x_min_all - padding, x_max_all + padding)

                ax.set_xlabel(f"Value ({aggregation})", fontsize=10)
                ax.grid(True, alpha=0.3)

                # Create group title
                group_names = "_".join(
                    [
                        k.strip("$")
                        .replace("\\", "")
                        .replace("/", "_")
                        .replace(" ", "_")
                        for k in group
                    ]
                )
                ax.set_title(f"Comparison: Group {group_idx + 1}", fontsize=12, pad=15)

                # Save comparison figure
                scale_suffix = ""
                if use_dual_axis:
                    scale_suffix = "_dual"
                elif use_log_scale:
                    scale_suffix = "_log"
                elif use_sqrt_scale:
                    scale_suffix = "_sqrt"
                elif normalize_densities:
                    scale_suffix = "_norm"

                save_path = os.path.join(
                    output_dir,
                    f"comparison_group_{group_idx + 1}_{aggregation}{scale_suffix}.png",
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved comparison plot to {save_path}")
                plt.close()

            # Plot remaining individual distributions not in any group
            for key, value in overal_distributions.items():
                if key in plotted_keys:
                    continue

                # Aggregate over dimensions 0 and 1
                if aggregation == "mean":
                    aggregated = np.mean(value, axis=(0, 1))
                elif aggregation == "sum":
                    aggregated = np.sum(value, axis=(0, 1))
                elif aggregation == "median":
                    aggregated = np.median(value, axis=(0, 1))
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")

                fig, ax = plt.subplots(figsize=(8, 6))
                flattened_data = aggregated.ravel()
                ax.hist(
                    flattened_data,
                    bins=bins,
                    density=True,
                    alpha=0.7,
                    edgecolor="black",
                    color="steelblue",
                )

                clean_key = key.strip("$")
                ax.set_xlabel(f"Value ({aggregation})", fontsize=10)
                ax.set_ylabel("PDF", fontsize=10)
                ax.set_title(f"${clean_key}$", fontsize=12, pad=15)
                ax.grid(True, alpha=0.3)

                safe_filename = (
                    clean_key.replace("\\", "").replace("/", "_").replace(" ", "_")
                )
                save_path = os.path.join(
                    output_dir, f"{safe_filename}_2d_{aggregation}.png"
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved 2D histogram to {save_path}")
                plt.close()

        else:
            # Plot each distribution individually (original behavior)
            for key, value in overal_distributions.items():
                # Aggregate over dimensions 0 and 1
                if aggregation == "mean":
                    aggregated = np.mean(value, axis=(0, 1))
                elif aggregation == "sum":
                    aggregated = np.sum(value, axis=(0, 1))
                elif aggregation == "median":
                    aggregated = np.median(value, axis=(0, 1))
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")

                # Create individual figure
                fig, ax = plt.subplots(figsize=(8, 6))

                # Flatten the spatial map and plot histogram
                flattened_data = aggregated.ravel()
                ax.hist(
                    flattened_data,
                    bins=bins,
                    density=True,
                    alpha=0.7,
                    edgecolor="black",
                    color="steelblue",
                )

                clean_key = key.strip("$")
                ax.set_xlabel(f"Value ({aggregation})", fontsize=10)
                ax.set_ylabel("PDF", fontsize=10)
                ax.set_title(f"${clean_key}$", fontsize=12, pad=15)
                ax.grid(True, alpha=0.3)

                # Save individual figure
                safe_filename = (
                    clean_key.replace("\\", "").replace("/", "_").replace(" ", "_")
                )
                save_path = os.path.join(
                    output_dir, f"{safe_filename}_2d_{aggregation}.png"
                )
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Saved 2D histogram to {save_path}")
                plt.close()

        return None

    else:
        # Original behavior: plot all in one figure
        n_distributions = len(overal_distributions)
        n_cols = 4
        n_rows = (n_distributions + n_cols - 1) // n_cols

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        for idx, (key, value) in enumerate(overal_distributions.items()):
            # Aggregate over dimensions 0 and 1
            if aggregation == "mean":
                aggregated = np.mean(value, axis=(0, 1))
            elif aggregation == "sum":
                aggregated = np.sum(value, axis=(0, 1))
            elif aggregation == "median":
                aggregated = np.median(value, axis=(0, 1))
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            row = idx // n_cols
            col = idx % n_cols

            ax = fig.add_subplot(gs[row, col])

            flattened_data = aggregated.ravel()
            ax.hist(
                flattened_data,
                bins=bins,
                density=True,
                alpha=0.7,
                edgecolor="black",
                color="steelblue",
            )

            clean_key = key.strip("$")
            ax.set_xlabel(f"Value ({aggregation})", fontsize=8)
            ax.set_ylabel("PDF", fontsize=8)
            ax.grid(True, alpha=0.3)

            ax.text(
                0.5,
                -0.25,
                f"${clean_key}$",
                transform=ax.transAxes,
                fontsize=10,
                ha="center",
                va="top",
            )

        # Hide unused subplots
        for idx in range(n_distributions, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

        save_path = f"distributions_2d_{aggregation}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 2D histogram to {save_path}")
        plt.close()

        return fig


if __name__ == "__main__":
    # Create a single dataloader for all categories
    dataloader = PRBSDataloader(
        root_dir=root_dir,
        categories=categories,
        snrs=['20'],
        transform_funcs=transform_funcs,
        batch_size=1,
        pin_memory=True,
        num_worker=1,
        train_test_split=0.01,
        shuffle=True,
    )

    train_loader, test_loader = dataloader.get_dataloaders()

    # Initialize models
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint
    checkpoint_path = f"{output_dir}/checkpoints/best_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        exit(1)

    # Set models to evaluation mode
    model.eval()

    # Initialize lists to collect data
    complex_Z_PRBS_waveforms_no_noises = []
    complex_Z_PRBS_waveforms = []
    complex_Z_PRBS_waveforms_pred = []
    complex_Y_PRBS_waveforms_no_noises = []
    complex_Y_PRBS_waveforms = []

    nmses = 0
    overal_distributions = {}

    with torch.no_grad():
        for batch_idx, batch_images in enumerate(tqdm(test_loader)):
            (
                PRBS_waveform,
                Z_PRBS_waveform_no_noise,
                Z_PRBS_waveform,
                Y_PRBS_waveform_no_noise,
                Y_PRBS_waveform,
            ) = batch_images
            PRBS_waveform = PRBS_waveform.to(device)
            Z_PRBS_waveform_no_noise = Z_PRBS_waveform_no_noise.to(device)
            Z_PRBS_waveform = Z_PRBS_waveform.to(device)
            Y_PRBS_waveform_no_noise = Y_PRBS_waveform_no_noise.to(device)
            Y_PRBS_waveform = Y_PRBS_waveform.to(device)

            distributions = {
                r"$p(\mathbf{P}_{\rm prbs})$": PRBS_waveform,
                r"$p(\mathbf{Y}_{\rm sen, fast}^{\rm prbs})$": Y_PRBS_waveform,
                r"$p(\mathbf{\tilde{Z}}_{\rm sen, fast}^{\rm prbs})$": Z_PRBS_waveform_no_noise,
                r"$p(\mathbf{Z}_{\rm sen, fast}^{\rm prbs})$": Z_PRBS_waveform,
            }

            distributions = model.distribution(
                Y_PRBS_waveform, PRBS_waveform, Z_PRBS_waveform_no_noise, distributions
            )
            # distributions = model.distribution(
            #     Y_PRBS_waveform, None, Z_PRBS_waveform_no_noise, distributions
            # )

            # Initialize overal_distributions on first batch
            if batch_idx == 0:
                overal_distributions = {key: [] for key in distributions.keys()}

            # Append each distribution
            for key, value in distributions.items():
                overal_distributions[key].append(value.cpu())

    # Stack and convert to numpy
    for key in overal_distributions.keys():
        overal_distributions[key] = (
            torch.stack(overal_distributions[key], dim=0).squeeze(1).numpy()
        )

    print("Overall distributions shapes:")
    for key, value in overal_distributions.items():
        print(f"{key}: {value.shape}")

    # Plot distributions
    print("\nGenerating 3D plots...")
    plot_distributions_3d(
        overal_distributions,
        aggregation="sum",
        split_figures=True,
        output_dir="results/3d_plots",
    )

    print("\nGenerating 2D heatmaps...")
    comparation_groups = [
        [
            r"$p(\mathbf{\tilde{Z}}_{\rm sen, fast}^{\rm prbs})$",
            r"$p(\mathbf{Z}_{\rm sen, fast}^{\rm prbs})$",
            r"$p(\mathbf{Y}_{\rm sen, fast}^{\rm prbs})$",
        ],
        [
            r"$p(\mathbf{\tilde{Z}}_{\rm sen, rd}^{\rm prbs})$",
            r"$p(\mathbf{Y}_{\rm sen, rd}^{\rm prbs})$",
        ],
        [r"$q(\vec{z}_{1}\mid \vec{z}_{0})$", r"$p(\vec{z}_{1}\mid \vec{z}_{2})$"],
        [r"$q(\vec{z}_{2}\mid \vec{z}_{1})$", r"$p(\vec{z}_{2}\mid \vec{z}_{3})$"],
        [r"$q(\vec{z}_{3}\mid \vec{z}_{2})$", r"$p(\vec{z}_{3})$"],
        [
            r"$p(\mathbf{\tilde{Z}}_{\rm sen, rd}^{\rm prbs})$",
            r"$p(\mathbf{\hat{Z}}_{\rm sen, rd}^{\rm prbs})$",
        ],
        [
            r"$p(\mathbf{\tilde{Z}}_{\rm sen, fast}^{\rm prbs})$",
            r"$p(\mathbf{\hat{Z}}_{\rm sen, fast}^{\rm prbs})$",
        ],
    ]
    plot_distributions_2d_heatmap(
        overal_distributions,
        aggregation="mean",
        split_figures=True,
        output_dir="results/2d_plots",
        comparison_groups=comparation_groups,
        use_log_scale=False,
        use_dual_axis=False,
        use_sqrt_scale=True,
        normalize_densities=False,
        xlim_density_threshold=0.0005,
    )
