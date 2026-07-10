import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


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
# 2. 带时间条件的残差模块
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        time_condition = self.time_projection(F.silu(time_embedding))
        h = h + time_condition[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.shortcut(x)


# ============================================================
# 3. 简化 U-Net 噪声预测网络
# ============================================================
class TinyUNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=64, time_dim=128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.down_block1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.down_block2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.middle_block1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.middle_block2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)

        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(base_channels * 2, base_channels, time_dim)

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

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
class DDPM(nn.Module):
    def __init__(self, model, num_steps=300, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.model = model
        self.num_steps = num_steps

        beta = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_previous = F.pad(alpha_bar[:-1], pad=(1, 0), value=1.0)
        posterior_variance = beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
        posterior_variance[0] = 0.0

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def extract(values, t, x_shape):
        batch_size = t.shape[0]
        output = values.gather(dim=0, index=t)
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

    @torch.no_grad()
    def reverse_step(self, xt, t, stochastic=True):
        predicted_noise = self.model(xt, t)

        alpha_t = self.extract(self.alpha, t, xt.shape)
        beta_t = self.extract(self.beta, t, xt.shape)
        alpha_bar_t = self.extract(self.alpha_bar, t, xt.shape)
        posterior_variance_t = self.extract(self.posterior_variance, t, xt.shape)

        model_mean = (xt - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)

        if not stochastic:
            return model_mean

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
        xt_previous = model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20)) * noise

        return xt_previous

    @torch.no_grad()
    def denoise_from_xt(self, xt, start_step, deterministic=True):
        x = xt.clone()

        for step in reversed(range(start_step + 1)):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x = self.reverse_step(x, t, stochastic=not deterministic)

        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, batch_size, image_size=28, image_channels=1, device="cpu"):
        x = torch.randn(batch_size, image_channels, image_size, image_size, device=device)

        for step in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.reverse_step(x, t, stochastic=True)

        return x.clamp(-1.0, 1.0)

# ============================================================
# 5. 创建 MNIST 数据集
# ============================================================
def create_datasets(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)

    return train_loader, test_dataset

# ============================================================
# 6. 从测试集中选取固定的 0 到 9
# ============================================================
def get_fixed_digit_samples(dataset, device):
    fixed_images = []
    fixed_labels = []
    found_digits = set()

    for image, label in dataset:
        if label not in found_digits:
            fixed_images.append(image)
            fixed_labels.append(label)
            found_digits.add(label)

        if len(found_digits) == 10:
            break

    sorted_samples = sorted(zip(fixed_labels, fixed_images), key=lambda item: item[0])
    fixed_labels = torch.tensor([item[0] for item in sorted_samples], dtype=torch.long, device=device)
    fixed_images = torch.stack([item[1] for item in sorted_samples], dim=0).to(device)

    return fixed_images, fixed_labels

# ============================================================
# 7. 保存固定原始数字
# ============================================================
@torch.no_grad()
def save_fixed_reference(fixed_images, output_path):
    images = (fixed_images.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(images, output_path, nrow=fixed_images.shape[0], padding=2)
    print(f"Fixed reference image saved to: {output_path}")


# ============================================================
# 8. 保存固定样本的加噪和恢复结果
# ============================================================
@torch.no_grad()
def save_fixed_reconstruction(ddpm, fixed_images, device, output_path, t_show=150):
    ddpm.eval()

    batch_size = fixed_images.shape[0]
    t = torch.full((batch_size,), t_show, device=device, dtype=torch.long)

    xt, _ = ddpm.forward_diffusion(fixed_images, t)
    reconstructed_images = ddpm.denoise_from_xt(xt, start_step=t_show, deterministic=True)

    panel = torch.cat([fixed_images, xt, reconstructed_images], dim=0)
    panel = (panel.clamp(-1.0, 1.0) + 1.0) / 2.0

    save_image(panel, output_path, nrow=batch_size, padding=2)

    print(f"Reconstruction image saved to: {output_path}")
    print("Row 1: original fixed digits")
    print("Row 2: noisy digits")
    print("Row 3: reconstructed digits")


# ============================================================
# 9. 保存随机生成样本
# ============================================================
@torch.no_grad()
def save_generated_samples(ddpm, device, output_path, num_samples=16):
    ddpm.eval()

    samples = ddpm.sample(batch_size=num_samples, image_size=28, image_channels=1, device=device)
    samples = (samples + 1.0) / 2.0

    save_image(samples, output_path, nrow=4, padding=2)
    print(f"Generated samples saved to: {output_path}")


# ============================================================
# 10. 训练模型
# ============================================================
def train(ddpm, train_loader, optimizer, device, fixed_images, num_epochs=10, t_show=150, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        ddpm.train()
        running_loss = 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for batch_index, (images, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = ddpm.training_loss(images)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_index % 100 == 0:
                if device.type == "cuda":
                    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved_memory = torch.cuda.memory_reserved() / 1024 ** 3
                    print(f"Epoch: {epoch + 1:02d}/{num_epochs:02d}, Batch: {batch_index:04d}/{len(train_loader):04d}, Loss: {loss.item():.6f}, Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB")
                else:
                    print(f"Epoch: {epoch + 1:02d}/{num_epochs:02d}, Batch: {batch_index:04d}/{len(train_loader):04d}, Loss: {loss.item():.6f}")

        average_loss = running_loss / len(train_loader)

        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"Epoch: {epoch + 1:02d}, Average loss: {average_loss:.6f}, Peak GPU memory: {peak_memory:.2f} GB")
        else:
            print(f"Epoch: {epoch + 1:02d}, Average loss: {average_loss:.6f}")

        reconstruction_path = os.path.join(output_dir, f"reconstruction_epoch_{epoch + 1:02d}.png")
        save_fixed_reconstruction(ddpm, fixed_images, device, reconstruction_path, t_show=t_show)

        checkpoint_path = os.path.join(output_dir, f"ddpm_epoch_{epoch + 1:02d}.pth")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": ddpm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "average_loss": average_loss
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")


# ============================================================
# 11. 加载模型检查点
# ============================================================
def load_checkpoint(ddpm, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    average_loss = checkpoint.get("average_loss", None)

    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Completed epoch: {epoch}")

    if average_loss is not None:
        print(f"Stored average loss: {average_loss:.6f}")

    return epoch


# ============================================================
# 12. 主程序
# ============================================================
def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    base_channels = 64
    time_dim = 128
    num_steps = 300
    num_epochs = 2
    learning_rate = 2e-4
    num_workers = 2
    t_show = 150
    output_dir = "./outputs"

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    os.makedirs(output_dir, exist_ok=True)

    train_loader, test_dataset = create_datasets(batch_size=batch_size, num_workers=num_workers)
    fixed_images, fixed_labels = get_fixed_digit_samples(test_dataset, device)

    print(f"Fixed digits: {fixed_labels.tolist()}")

    fixed_reference_path = os.path.join(output_dir, "fixed_reference.png")
    save_fixed_reference(fixed_images, fixed_reference_path)

    noise_predictor = TinyUNet(image_channels=1, base_channels=base_channels, time_dim=time_dim)
    ddpm = DDPM(model=noise_predictor, num_steps=num_steps, beta_start=1e-4, beta_end=0.02).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=learning_rate, weight_decay=1e-4)

    total_parameters = sum(parameter.numel() for parameter in ddpm.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in ddpm.parameters() if parameter.requires_grad)

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    train(ddpm=ddpm, train_loader=train_loader, optimizer=optimizer, device=device, fixed_images=fixed_images, num_epochs=num_epochs, t_show=t_show, output_dir=output_dir)

    final_reconstruction_path = os.path.join(output_dir, "final_reconstruction.png")
    save_fixed_reconstruction(ddpm, fixed_images, device, final_reconstruction_path, t_show=t_show)

    generated_samples_path = os.path.join(output_dir, "generated_samples.png")
    save_generated_samples(ddpm, device, generated_samples_path, num_samples=16)

if __name__ == "__main__":
    main()















