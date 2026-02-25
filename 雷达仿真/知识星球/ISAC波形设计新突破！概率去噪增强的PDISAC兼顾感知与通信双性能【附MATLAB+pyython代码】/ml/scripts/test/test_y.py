import os

import torch
from scipy.io import savemat
from tqdm import tqdm

from configs.data import (
    categories,
    eval_dir,
    output_dir,
    root_dir,
    snrs,
    transform_funcs,
)
from configs.ml import domain, model
from data.dataloader import PRBSDataloader
from utils.plots import plot_matrix
from utils.utils import (
    complex_mul_conj,
    compute_nmse,
    convert_to_complex,
    freq_to_time,
    reverse_from_complex,
)

if __name__ == "__main__":
    for snr in snrs:
        # Create a single dataloader for all categories
        dataloader = PRBSDataloader(
            root_dir=root_dir,
            categories=categories,
            snrs=[snr],
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
        complex_Z_PRBS_waveforms_no_noise = []
        complex_Z_PRBS_waveforms = []
        complex_Z_PRBS_waveforms_pred = []

        nmses = 0

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

                Z_PRBS_waveform_pred = model.predict(Y_PRBS_waveform)

                # Convert to complex matrices
                real_part = Z_PRBS_waveform[:, 0, :, :]
                imag_part = Z_PRBS_waveform[:, 1, :, :]
                complex_Z_PRBS_waveform = torch.complex(real_part, imag_part)

                real_part_pred = Z_PRBS_waveform_pred[:, 0, :, :]
                imag_part_pred = Z_PRBS_waveform_pred[:, 1, :, :]
                complex_Z_PRBS_waveform_pred = torch.complex(
                    real_part_pred, imag_part_pred
                )

                real_part_no_noise = Z_PRBS_waveform_no_noise[:, 0, :, :]
                imag_part_no_noise = Z_PRBS_waveform_no_noise[:, 1, :, :]
                complex_Z_PRBS_waveform_no_noise = torch.complex(
                    real_part_no_noise, imag_part_no_noise
                )

                # Compute NMSE
                nmse = compute_nmse(
                    Z_PRBS_waveform_pred.flatten(start_dim=1),
                    Z_PRBS_waveform_no_noise.flatten(start_dim=1),
                    dim=1,
                )
                nmses += nmse.item()

                # Collect data
                complex_Z_PRBS_waveforms.append(complex_Z_PRBS_waveform.cpu())
                complex_Z_PRBS_waveforms_pred.append(complex_Z_PRBS_waveform_pred.cpu())
                complex_Z_PRBS_waveforms_no_noise.append(
                    complex_Z_PRBS_waveform_no_noise.cpu()
                )

        nmses = nmses / len(test_loader)
        print(f"At {snr}, the NMSE score: {nmses}")

        # Stack the collected complex tensors and remove the batch dimension (since batch_size=1)
        complex_Z_PRBS_waveforms = torch.stack(complex_Z_PRBS_waveforms, dim=0).squeeze(
            1
        )  # Shape: (num_samples, 255, 256)
        complex_Z_PRBS_waveforms_pred = torch.stack(
            complex_Z_PRBS_waveforms_pred, dim=0
        ).squeeze(1)  # Shape: (num_samples, 255, 256)

        complex_Z_PRBS_waveforms_no_noise = torch.stack(
            complex_Z_PRBS_waveforms_no_noise, dim=0
        ).squeeze(1)  # Shape: (num_samples, 255, 256)

        # Convert to NumPy arrays
        complex_Z_PRBS_waveforms_np = complex_Z_PRBS_waveforms.numpy()
        complex_Z_PRBS_waveforms_pred_np = complex_Z_PRBS_waveforms_pred.numpy()
        complex_Z_PRBS_waveforms_no_noise_np = complex_Z_PRBS_waveforms_no_noise.numpy()

        # Save to .mat file
        output_path = f"{eval_dir}/data/{snr}/data.mat"  # Adjust path as needed

        # create directory if it does not exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plot_matrix(
            {
                "Y_PRBS_waveform": Y_PRBS_waveform[0],
                "Z_PRBS_waveform": Z_PRBS_waveform[0],
                "Z_PRBS_waveform_pred": Z_PRBS_waveform_pred[0],
                "Z_PRBS_waveform_no_noise": Z_PRBS_waveform_no_noise[0],
            },
            save_path=f"{eval_dir}/data/{snr}/visualize_matrix.png",
            domain=domain,
        )

        savemat(
            output_path,
            {
                "complex_Z_PRBS_waveforms_no_noise": complex_Z_PRBS_waveforms_no_noise_np,
                "complex_Z_PRBS_waveforms": complex_Z_PRBS_waveforms_np,
                "complex_Z_PRBS_waveforms_pred": complex_Z_PRBS_waveforms_pred_np,
            },
        )

        print(f"Data saved to {output_path}")
