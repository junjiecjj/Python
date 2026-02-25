import matplotlib.pyplot as plt
import os
import torch
import torchvision.utils as vutils
import numpy as np
from utils.utils import preprocess, convert_to_complex
from configs.isac import speed_axis, range_axis, r_max, v_max
from utils.rd import range_doppler_response


def plot_and_save_mask_grid(
    mask: torch.Tensor, filename: str = "mask_grid.png", cmap: str = "viridis"
):
    """
    Plots a batch of masks and saves them to an image file.
    Args:
        mask (torch.Tensor): Tensor of shape [B, H, W], batch of masks.
        filename (str): Output image filename (e.g., "mask_grid.png").
        cmap (str): Colormap for visualization ('gray', 'viridis', 'hot', etc.).
    """

    # Move to CPU and normalize
    mask_cpu = mask.detach().cpu()
    mask_norm = (mask_cpu - mask_cpu.min()) / (mask_cpu.max() - mask_cpu.min() + 1e-8)

    # Add channel dimension for make_grid: [B, H, W] -> [B, 1, H, W]
    mask_norm = mask_norm.unsqueeze(1)

    # Create grid image from batch
    grid_img = vutils.make_grid(mask_norm, nrow=4, padding=2, normalize=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap=cmap)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved mask grid to '{filename}'")


def plot_loss_curve(
    epochs, train_losses, val_losses, title, ylabel, filename, output_dir
):
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label=f"Train {ylabel}")
    plt.plot(epochs, val_losses, label=f"Validation {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} over Epochs")
    plt.legend()
    plt.grid(True)

    # Save figure
    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path)
    plt.close()

    # Save raw data as .npy
    base_filename = os.path.splitext(filename)[0]
    np.save(os.path.join(output_dir, f"{base_filename}_epochs.npy"), np.array(epochs))
    np.save(
        os.path.join(output_dir, f"{base_filename}_train.npy"), np.array(train_losses)
    )
    np.save(os.path.join(output_dir, f"{base_filename}_val.npy"), np.array(val_losses))


def plot_dncnn_loss(epochs, train_dncnn_losses, val_dncnn_losses, output_dir):
    plot_loss_curve(
        epochs,
        train_dncnn_losses,
        val_dncnn_losses,
        title="DnCNN Loss",
        ylabel="DnCNN Loss",
        filename="dncnn_loss_plot.png",
        output_dir=output_dir,
    )


def plot_nmse(epochs, train_nmse, val_nmse, output_dir):
    plot_loss_curve(
        epochs,
        train_nmse,
        val_nmse,
        title="NMSE",
        ylabel="NMSE",
        filename="nmse_plot.png",
        output_dir=output_dir,
    )


def plot_afm_and_nmse(
    epochs, train_afm_losses, val_afm_losses, train_nmse, val_nmse, output_dir
):
    plot_loss_curve(
        epochs,
        train_afm_losses,
        val_afm_losses,
        title="AFM Loss",
        ylabel="AFM Loss",
        filename="afm_loss_plot.png",
        output_dir=output_dir,
    )

    plot_loss_curve(
        epochs,
        train_nmse,
        val_nmse,
        title="NMSE",
        ylabel="NMSE",
        filename="nmse_plot.png",
        output_dir=output_dir,
    )


def plot_matrix(
    matrices: dict, cmap="viridis", save_path: str = None, domain: str = "freq"
):
    n = len(matrices)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axs = [axs]  # ensure it's iterable

    for ax, (name, mat) in zip(axs, matrices.items()):
        assert mat.shape[0] == 2, f"Matrix '{name}' must have shape (2, H, W)"

        # Processing
        complex_tensor = convert_to_complex(mat)
        fft_tensor = preprocess(complex_tensor, domain=domain)
        log_magnitude = torch.log10(torch.abs(fft_tensor) + 1e-10)
        img = log_magnitude.detach().cpu().numpy()

        # Plotting
        ax.imshow(
            img,
            cmap=cmap,
            extent=[speed_axis[0], speed_axis[-1], range_axis[0], range_axis[-1]],
            aspect="auto",
            origin="lower",
        )
        ax.set_title(name)
        ax.axis("off")
        ax.set_xlim([-v_max, v_max])
        ax.set_ylim([0, r_max])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_rd(Z_PRBS_waveform, domain):
    rd_map, ranges, speeds = range_doppler_response(Z_PRBS_waveform, domain)
    # Plot response
    plt.figure()
    plt.imshow(
        rd_map,
        extent=[speeds[0], speeds[-1], ranges[0], ranges[-1]],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )  # Use 'jet' colormap for MATLAB-like visualization
    plt.colorbar(label="Amplitude (dB)")
    plt.xlim([-v_max, v_max])
    plt.ylim([0, r_max])
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Range-Doppler Response")
    plt.show()
