import torch
import os
from data.dataloader import PRBSDataloader
from utils.utils import (
    compute_nmse,
    preprocess,
    reverse_from_complex,
    convert_to_complex,
)
from utils.plots import plot_dncnn_loss, plot_afm_and_nmse, plot_matrix
from configs.data import root_dir, output_dir, categories, snrs, transform_funcs
from configs.ml import (
    epochs,
    device,
    model,
    afm_model,
    batch_size,
    train_test_split,
    kls_thesh,
    lr,
    betas,
    domain,
)
from losses.losses import PSNRLoss
from tqdm import tqdm

if __name__ == "__main__":
    # Create a single dataloader for all categories
    dataloader = PRBSDataloader(
        root_dir=root_dir,
        categories=categories,
        snrs=snrs,
        transform_funcs=transform_funcs,
        batch_size=batch_size,
        pin_memory=True,
        num_worker=1,
        train_test_split=train_test_split,
        shuffle=True,
    )

    train_loader, test_loader = dataloader.get_dataloaders()

    # Optimizers
    afm_model_opt = torch.optim.Adam(afm_model.parameters(), lr=lr, betas=betas)
    model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    # Loss function
    psnr_loss = PSNRLoss().to(device)

    # Create output directory for images and checkpoints
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training parameters
    best_val_nmse = float("inf")  # Initialize with infinity for minimization

    # Lists to store metrics
    train_dncnn_losses = []
    train_afm_losses = []
    train_nmse = []
    val_dncnn_losses = []
    val_afm_losses = []
    val_nmse = []

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        afm_model.train()
        model.train()

        total_loss_dncnn = 0.0
        total_loss_afm = 0.0
        total_nmse = 0.0
        num_batches = 0

        for batch_images in tqdm(train_loader):
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
            RD_no_noises = reverse_from_complex(
                preprocess(
                    convert_to_complex(Z_PRBS_waveform_no_noise, dim=1),
                    dims=[1, 2],
                    domain=domain,
                ),
                dim=1,
            ).to(device)

            # Zero gradients
            afm_model_opt.zero_grad()
            model_opt.zero_grad()

            # Train DnCNN
            Z_PRBS_waveform_pred, RD_pred, kls = model(
                Y=Y_PRBS_waveform, Z=Z_PRBS_waveform_no_noise
            )
            loss_dncnn = psnr_loss(Z_PRBS_waveform_pred, Z_PRBS_waveform_no_noise)

            # Compute NMSE
            nmse = compute_nmse(
                Z_PRBS_waveform_pred.flatten(start_dim=1),
                Z_PRBS_waveform_no_noise.flatten(start_dim=1),
                dim=1,
            )

            # AFM forward pass
            Z_hat_PRBS_waveform, Y_hat_PRBS_waveform, PRBS_waveform_hard_mask = (
                afm_model(Z_PRBS_waveform_pred, Y_PRBS_waveform, PRBS_waveform)
            )
            Z_hat_PRBS_waveform_pred, _, _ = model(
                Y=Y_hat_PRBS_waveform, Z=Z_PRBS_waveform_no_noise
            )
            loss_afm = psnr_loss(Z_hat_PRBS_waveform_pred, Z_PRBS_waveform_no_noise)

            # Total DnCNN loss
            l_total_dncnn = (
                loss_dncnn
                + loss_afm
                + psnr_loss(RD_pred, RD_no_noises)
                + torch.mean(torch.stack(kls)) * kls_thesh
            )
            l_total_dncnn.backward()
            model_opt.step()

            # Train AFM
            Z_PRBS_waveform_pred, _, _ = model(
                Y=Y_PRBS_waveform, Z=Z_PRBS_waveform_no_noise
            )
            Z_hat_PRBS_waveform, Y_hat_PRBS_waveform, PRBS_waveform_hard_mask = (
                afm_model(Z_PRBS_waveform_pred, Y_PRBS_waveform, PRBS_waveform)
            )
            loss_afm_z = psnr_loss(Z_hat_PRBS_waveform, Z_PRBS_waveform_no_noise)
            loss_afm_y = psnr_loss(Y_hat_PRBS_waveform, Y_PRBS_waveform_no_noise)

            # Total AFM loss
            l_total_afm = loss_afm_z + loss_afm_y
            l_total_afm.backward()
            afm_model_opt.step()

            # Accumulate metrics
            total_loss_dncnn += l_total_dncnn.item()
            total_loss_afm += l_total_afm.item()
            total_nmse += nmse.item()
            num_batches += 1

        # Average metrics for the epoch
        avg_loss_dncnn = total_loss_dncnn / num_batches
        avg_loss_afm = total_loss_afm / num_batches
        avg_nmse = total_nmse / num_batches
        train_dncnn_losses.append(avg_loss_dncnn)
        train_afm_losses.append(avg_loss_afm)
        train_nmse.append(avg_nmse)
        print(
            f"Train - Avg DnCNN Loss: {avg_loss_dncnn:.4f}, Avg AFM Loss: {avg_loss_afm:.4f}, Avg NMSE: {avg_nmse:.4f}"
        )

        # Validation loop
        afm_model.eval()
        model.eval()
        total_test_loss_dncnn = 0.0
        total_test_loss_afm = 0.0
        total_test_nmse = 0.0
        num_test_batches = 0

        with torch.no_grad():
            for batch_idx, batch_images in enumerate(test_loader):
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

                # DnCNN evaluation
                Z_PRBS_waveform_pred = model.predict(Y=Y_PRBS_waveform)
                loss_dncnn = psnr_loss(Z_PRBS_waveform_pred, Z_PRBS_waveform_no_noise)

                # Compute NMSE
                nmse = compute_nmse(
                    Z_PRBS_waveform_pred.flatten(start_dim=1),
                    Z_PRBS_waveform_no_noise.flatten(start_dim=1),
                    dim=1,
                )

                # AFM evaluation
                Z_hat_PRBS_waveform, Y_hat_PRBS_waveform, PRBS_waveform_hard_mask = (
                    afm_model(Z_PRBS_waveform_pred, Y_PRBS_waveform, PRBS_waveform)
                )
                Z_hat_PRBS_waveform_pred = model.predict(Y=Y_hat_PRBS_waveform)
                loss_afm_z = psnr_loss(
                    Z_hat_PRBS_waveform_pred, Z_PRBS_waveform_no_noise
                )
                loss_afm_y = psnr_loss(Y_hat_PRBS_waveform, Y_PRBS_waveform_no_noise)

                # Total AFM loss for evaluation
                l_total_eval_afm = loss_afm_z + loss_afm_y

                # Accumulate test metrics
                total_test_loss_dncnn += loss_dncnn.item()
                total_test_loss_afm += l_total_eval_afm.item()
                total_test_nmse += nmse.item()
                num_test_batches += 1

        # Average test metrics
        avg_test_loss_dncnn = total_test_loss_dncnn / num_test_batches
        avg_test_loss_afm = total_test_loss_afm / num_test_batches
        avg_test_nmse = total_test_nmse / num_test_batches
        val_dncnn_losses.append(avg_test_loss_dncnn)
        val_afm_losses.append(avg_test_loss_afm)
        val_nmse.append(avg_test_nmse)
        print(
            f"Eval - Avg DnCNN Loss: {avg_test_loss_dncnn:.4f}, Avg AFM Loss: {avg_test_loss_afm:.4f}, Avg NMSE: {avg_test_nmse:.4f}"
        )

        # Save checkpoint if validation NMSE is improved
        if avg_test_nmse < best_val_nmse:
            best_val_nmse = avg_test_nmse
            checkpoint = {
                "epoch": epoch + 1,
                "afm_model_state_dict": afm_model.state_dict(),
                "model_state_dict": model.state_dict(),
                "afm_optimizer_state_dict": afm_model_opt.state_dict(),
                "model_optimizer_state_dict": model_opt.state_dict(),
                "val_nmse": best_val_nmse,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_checkpoint.pth"))
            print(f"Saved checkpoint with best validation NMSE: {best_val_nmse:.4f}")

            plot_matrix(
                {
                    "Y_PRBS_waveform": Y_PRBS_waveform[0],
                    "Z_PRBS_waveform": Z_PRBS_waveform[0],
                    "Z_PRBS_waveform_pred": Z_PRBS_waveform_pred[0],
                    "Z_PRBS_waveform_no_noise": Z_PRBS_waveform_no_noise[0],
                },
                save_path=f"{output_dir}/visualize_matrix.png",
                domain=domain,
            )

    # Plotting and saving the metrics
    epochs = range(1, epochs + 1)

    plot_dncnn_loss(epochs, train_dncnn_losses, val_dncnn_losses, output_dir)
    plot_afm_and_nmse(
        epochs, train_afm_losses, val_afm_losses, train_nmse, val_nmse, output_dir
    )
