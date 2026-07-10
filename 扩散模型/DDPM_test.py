#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 23:30:50 2026

@author: jack
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 时间步正弦编码
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / max(half_dim - 1, 1))
        angles = t.float().unsqueeze(1) * frequency.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


# ============================================================
# 2. 一维残差卷积模块
# ============================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        time_condition = self.time_projection(F.silu(time_embedding))
        h = h + time_condition[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.shortcut(x)


# ============================================================
# 3. 一维简化 U-Net
# ============================================================
class TinyUNet1D(nn.Module):
    def __init__(self, signal_channels=1, base_channels=32, time_dim=64):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_conv = nn.Conv1d(signal_channels, base_channels, kernel_size=3, padding=1)

        self.down_block1 = ResidualBlock1D(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.down_block2 = ResidualBlock1D(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.middle_block1 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim)
        self.middle_block2 = ResidualBlock1D(base_channels * 4, base_channels * 4, time_dim)

        self.upsample2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResidualBlock1D(base_channels * 4, base_channels * 2, time_dim)

        self.upsample1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResidualBlock1D(base_channels * 2, base_channels, time_dim)

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv1d(base_channels, signal_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        time_embedding = self.time_embedding(t)

        x = self.input_conv(x)

        skip1 = self.down_block1(x, time_embedding)
        x = self.downsample1(skip1)

        skip2 = self.down_block2(x, time_embedding)
        x = self.downsample2(skip2)

        x = self.middle_block1(x, time_embedding)
        x = self.middle_block2(x, time_embedding)

        x = self.upsample2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_block2(x, time_embedding)

        x = self.upsample1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_block1(x, time_embedding)

        return self.output_conv(F.silu(self.output_norm(x)))


# ============================================================
# 4. DDPM 扩散模型
# ============================================================
class DDPM1D(nn.Module):
    def __init__(self, model, num_steps=300, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.model = model
        self.num_steps = num_steps

        beta = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    @staticmethod
    def extract(values, t, x_shape):
        batch_size = t.shape[0]
        output = values.gather(0, t)
        return output.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def forward_diffusion(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt, noise

    def training_loss(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=x0.device, dtype=torch.long)
        xt, true_noise = self.forward_diffusion(x0, t)
        predicted_noise = self.model(xt, t)

        return F.mse_loss(predicted_noise, true_noise)

    def sigma_to_timestep(self, sigma):
        target_alpha_bar = 1.0 / (1.0 + sigma ** 2)
        distance = torch.abs(self.alpha_bar - target_alpha_bar)
        timestep = int(torch.argmin(distance).item())

        return timestep

    @torch.no_grad()
    def ddim_reverse_step(self, xt, step):
        batch_size = xt.shape[0]
        t = torch.full((batch_size,), step, device=xt.device, dtype=torch.long)

        predicted_noise = self.model(xt, t)

        alpha_bar_t = self.alpha_bar[step]
        predicted_x0 = (xt - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        predicted_x0 = predicted_x0.clamp(-1.2, 1.2)

        if step == 0:
            return predicted_x0

        alpha_bar_previous = self.alpha_bar[step - 1]
        xt_previous = torch.sqrt(alpha_bar_previous) * predicted_x0 + torch.sqrt(1.0 - alpha_bar_previous) * predicted_noise

        return xt_previous

    @torch.no_grad()
    def denoise_observation(self, observation, sigma):
        timestep = self.sigma_to_timestep(sigma)
        alpha_bar_t = self.alpha_bar[timestep]

        xt = torch.sqrt(alpha_bar_t) * observation

        for step in reversed(range(timestep + 1)):
            xt = self.ddim_reverse_step(xt, step)

        return xt.clamp(-1.0, 1.0), timestep


# ============================================================
# 5. 生成合成干净信号
# ============================================================
def generate_clean_signal(signal_length):
    time_axis = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)
    signal = np.zeros(signal_length, dtype=np.float32)

    number_of_sinusoids = np.random.randint(1, 4)

    for _ in range(number_of_sinusoids):
        amplitude = np.random.uniform(0.2, 1.0)
        frequency = np.random.uniform(0.5, 5.0)
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        signal += amplitude * np.sin(2.0 * np.pi * frequency * time_axis + phase)

    number_of_pulses = np.random.randint(0, 3)

    for _ in range(number_of_pulses):
        amplitude = np.random.uniform(-0.8, 0.8)
        center = np.random.uniform(0.1, 0.9)
        width = np.random.uniform(0.02, 0.12)
        signal += amplitude * np.exp(-(time_axis - center) ** 2 / (2.0 * width ** 2))

    trend_amplitude = np.random.uniform(-0.3, 0.3)
    signal += trend_amplitude * (time_axis - 0.5)

    maximum_value = np.max(np.abs(signal))

    if maximum_value > 1e-8:
        signal = 0.9 * signal / maximum_value

    return signal.astype(np.float32)


def create_training_dataset(number_of_samples=12000, signal_length=128):
    signals = np.zeros((number_of_samples, 1, signal_length), dtype=np.float32)

    for index in range(number_of_samples):
        signals[index, 0, :] = generate_clean_signal(signal_length)

    signals = torch.from_numpy(signals)

    return TensorDataset(signals)


# ============================================================
# 6. 训练扩散模型
# ============================================================
def train_model(ddpm, dataloader, optimizer, device, num_epochs, checkpoint_path):
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_history = []

    for epoch in range(num_epochs):
        ddpm.train()
        running_loss = 0.0

        for batch_index, (clean_signals,) in enumerate(dataloader):
            clean_signals = clean_signals.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = ddpm.training_loss(clean_signals)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_index % 50 == 0:
                print(f"Epoch: {epoch + 1:02d}/{num_epochs:02d}, Batch: {batch_index:03d}/{len(dataloader):03d}, Loss: {loss.item():.6f}")

        average_loss = running_loss / len(dataloader)
        loss_history.append(average_loss)

        print(f"Epoch: {epoch + 1:02d}, Average loss: {average_loss:.6f}")

        torch.save({"epoch": epoch + 1, "model_state_dict": ddpm.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss_history": loss_history}, checkpoint_path)

    return loss_history


# ============================================================
# 7. 未知噪声水平下的交替估计
# ============================================================
@torch.no_grad()
def blind_diffusion_denoising(ddpm, observation, initial_sigma, sigma_min=0.01, sigma_max=0.80, num_iterations=8, relaxation=0.5):
    estimated_sigma = float(initial_sigma)
    sigma_history = [estimated_sigma]
    reconstructed_signal = observation.clone()

    for iteration in range(num_iterations):
        reconstructed_signal, timestep = ddpm.denoise_observation(observation, estimated_sigma)

        residual = observation - reconstructed_signal
        residual_sigma = torch.sqrt(torch.mean(residual ** 2)).item()

        updated_sigma = (1.0 - relaxation) * estimated_sigma + relaxation * residual_sigma
        updated_sigma = float(np.clip(updated_sigma, sigma_min, sigma_max))

        print(f"Iteration: {iteration + 1:02d}, timestep: {timestep:03d}, sigma before: {estimated_sigma:.6f}, residual sigma: {residual_sigma:.6f}, sigma after: {updated_sigma:.6f}")

        estimated_sigma = updated_sigma
        sigma_history.append(estimated_sigma)

    reconstructed_signal, timestep = ddpm.denoise_observation(observation, estimated_sigma)

    return reconstructed_signal, estimated_sigma, sigma_history, timestep


# ============================================================
# 8. 绘图
# ============================================================
def plot_training_loss(loss_history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
    plt.show()


def plot_signal_results(clean_signal, noisy_signal, reconstructed_signal, output_dir):
    clean_signal = clean_signal.squeeze().cpu().numpy()
    noisy_signal = noisy_signal.squeeze().cpu().numpy()
    reconstructed_signal = reconstructed_signal.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(clean_signal, linewidth=2.0, label="Clean signal")
    plt.plot(noisy_signal, linewidth=1.0, alpha=0.65, label="Noisy observation")
    plt.plot(reconstructed_signal, linewidth=2.0, label="Diffusion reconstruction")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_denoising_result.png"), dpi=300)
    plt.show()


def plot_sigma_history(sigma_history, true_sigma, output_dir):
    iterations = np.arange(len(sigma_history))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, sigma_history, marker="o", label="Estimated noise standard deviation")
    plt.axhline(true_sigma, linestyle="--", label="True noise standard deviation")
    plt.xlabel("Alternating iteration")
    plt.ylabel("Noise standard deviation")
    plt.xticks(iterations)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sigma_estimation.png"), dpi=300)
    plt.show()


def plot_residual(clean_signal, noisy_signal, reconstructed_signal, output_dir):
    true_noise = noisy_signal - clean_signal
    residual = noisy_signal - reconstructed_signal

    true_noise = true_noise.squeeze().cpu().numpy()
    residual = residual.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(true_noise, linewidth=1.0, label="True noise")
    plt.plot(residual, linewidth=1.0, alpha=0.8, label="Estimated residual")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_comparison.png"), dpi=300)
    plt.show()


# ============================================================
# 9. 主程序
# ============================================================
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signal_length = 128
    number_of_training_samples = 12000
    batch_size = 128
    base_channels = 32
    time_dim = 64
    num_steps = 300
    num_epochs = 20
    learning_rate = 2e-4

    true_sigma = 0.20
    initial_sigma = 0.40
    alternating_iterations = 8
    relaxation = 0.5

    output_dir = "./blind_diffusion_outputs"
    checkpoint_path = os.path.join(output_dir, "blind_diffusion_model.pth")
    train_new_model = True

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    model = TinyUNet1D(signal_channels=1, base_channels=base_channels, time_dim=time_dim)
    ddpm = DDPM1D(model=model, num_steps=num_steps, beta_start=1e-4, beta_end=0.02).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=learning_rate, weight_decay=1e-4)

    total_parameters = sum(parameter.numel() for parameter in ddpm.parameters())
    print(f"Number of parameters: {total_parameters:,}")

    if train_new_model:
        training_dataset = create_training_dataset(number_of_samples=number_of_training_samples, signal_length=signal_length)
        dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
        loss_history = train_model(ddpm, dataloader, optimizer, device, num_epochs, checkpoint_path)
        plot_training_loss(loss_history, output_dir)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ddpm.load_state_dict(checkpoint["model_state_dict"])
        loss_history = checkpoint.get("loss_history", [])
        print(f"Model loaded from: {checkpoint_path}")

    ddpm.eval()

    clean_signal_np = generate_clean_signal(signal_length)
    clean_signal = torch.from_numpy(clean_signal_np).reshape(1, 1, signal_length).to(device)

    true_noise = true_sigma * torch.randn_like(clean_signal)
    noisy_observation = clean_signal + true_noise

    reconstructed_signal, estimated_sigma, sigma_history, final_timestep = blind_diffusion_denoising(
        ddpm=ddpm,
        observation=noisy_observation,
        initial_sigma=initial_sigma,
        sigma_min=0.01,
        sigma_max=0.80,
        num_iterations=alternating_iterations,
        relaxation=relaxation
    )

    noisy_mse = torch.mean((noisy_observation - clean_signal) ** 2).item()
    reconstructed_mse = torch.mean((reconstructed_signal - clean_signal) ** 2).item()
    input_snr = 10.0 * math.log10(torch.mean(clean_signal ** 2).item() / noisy_mse)
    output_snr = 10.0 * math.log10(torch.mean(clean_signal ** 2).item() / reconstructed_mse)

    print("\n================ Final result ================")
    print(f"True sigma:              {true_sigma:.6f}")
    print(f"Initial sigma:           {initial_sigma:.6f}")
    print(f"Estimated sigma:         {estimated_sigma:.6f}")
    print(f"Final diffusion step:    {final_timestep}")
    print(f"Noisy-signal MSE:        {noisy_mse:.6e}")
    print(f"Reconstructed MSE:       {reconstructed_mse:.6e}")
    print(f"Input SNR:               {input_snr:.3f} dB")
    print(f"Output SNR:              {output_snr:.3f} dB")
    print(f"SNR improvement:         {output_snr - input_snr:.3f} dB")

    plot_signal_results(clean_signal, noisy_observation, reconstructed_signal, output_dir)
    plot_sigma_history(sigma_history, true_sigma, output_dir)
    plot_residual(clean_signal, noisy_observation, reconstructed_signal, output_dir)


if __name__ == "__main__":
    main()
