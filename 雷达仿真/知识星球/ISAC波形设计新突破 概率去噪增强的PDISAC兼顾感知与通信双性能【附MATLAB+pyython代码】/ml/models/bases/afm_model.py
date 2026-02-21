import math

import torch
import torch.nn as nn
import torch.nn.init as init


class AFM_PRBS(nn.Module):
    def __init__(self, fq_bound=1.0, domain="freq", device="cuda"):
        super().__init__()

        self.device = device
        self.radius_factor_set = torch.arange(0.01, 1.01, 0.01).to(self.device)
        self.domain = domain
        self.fq_bound = fq_bound
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        ).to(self.device)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        ).to(self.device)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        ).to(self.device)

        self.dropout = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)
        self.fclayer_v1 = nn.Linear(64, 256).to(self.device)
        self.fclayer_last = nn.Linear(
            256, len(self.radius_factor_set) * len(self.radius_factor_set)
        ).to(self.device)
        self.leaky_relu = nn.LeakyReLU()
        self.temperature = 0.1

        # tmp
        self.value_set = 0.0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _preprocess(self, x):
        return torch.fft.fft(x, dim=1)

    def _postprocess(self, x):
        return torch.fft.ifft(x, dim=1)

    def convert_to_complex(self, x):
        return torch.complex(x[:, 0, ...], x[:, 1, ...])

    def reverse_from_complex(self, x):
        real_part = x.real
        imag_part = x.imag
        return torch.stack([real_part, imag_part], dim=1)

    def _complex_multiply(self, a, b):
        """
        Multiply two complex tensors with compatible shapes, handling broadcasting.
        a: Shape (batch, 2, h, 1) or (batch, 2, h, w)
        b: Shape (batch, 2, h, w)
        Returns: Tensor of shape (batch, 2, h, w)
        """
        real_a = a[:, 0, :, :]  # (batch, h, 1) or (batch, h, w)
        imag_a = a[:, 1, :, :]  # (batch, h, 1) or (batch, h, w)
        real_b = b[:, 0, :, :]  # (batch, h, w)
        imag_b = b[:, 1, :, :]  # (batch, h, w)
        real = real_a * real_b - imag_a * imag_b  # Broadcasting (h, 1) to (h, w)
        imag = real_a * imag_b + imag_a * real_b
        return torch.stack([real, imag], dim=1)  # (batch, 2, h, w)

    def forward(self, z_clean_fq, y_noise_fq, prbs_fq):
        B, C, H, W = y_noise_fq.size()

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H / 2) ** 2 + (b - W / 2) ** 2)
        dist = dist.to(y_noise_fq.device)
        max_radius = math.sqrt(H * H + W * W) / 2

        if self.domain == "time":
            z_clean_fq = self.reverse_from_complex(
                self._preprocess(self.convert_to_complex(z_clean_fq))
            )
            y_noise_fq = self.reverse_from_complex(
                self._preprocess(self.convert_to_complex(y_noise_fq))
            )
            prbs_fq = self.reverse_from_complex(
                self._preprocess(self.convert_to_complex(prbs_fq))
            )

        z_clean_fq_c = self.convert_to_complex(z_clean_fq)
        y_noise_fq_c = self.convert_to_complex(y_noise_fq)

        filter_input = torch.stack(
            [
                torch.abs(self.convert_to_complex(y_noise_fq)),
                torch.log10(torch.abs(y_noise_fq_c) + 1e-10),
                torch.abs(self.convert_to_complex(z_clean_fq)),
                torch.log10(torch.abs(z_clean_fq_c) + 1e-10),
            ],
            dim=1,
        )

        # filter_input = torch.cat([noisy, clean], dim=1)
        y = self.conv1(filter_input)
        y = self.relu(y)
        y = self.down1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.down2(y)
        y = self.conv3(y)
        y = self.relu(y)

        y = self.avgpool(y)
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        # using softmax
        value_prob = self.fclayer_last(self.fclayer_v1(y))
        value_prob = value_prob.view(B, 100, 100)
        value_prob = self.soft(value_prob * self.temperature) * (
            self.radius_factor_set.unsqueeze(0).unsqueeze(1)
        )
        value_prob = value_prob.sum(dim=-1)
        value_prob = value_prob.squeeze(-1)
        value_set = (value_prob * self.fq_bound).to(self.device)

        radius_set = max_radius * self.radius_factor_set

        mask = []
        zero_mask = torch.zeros_like(dist).to(self.device)
        one_mask = torch.ones_like(dist).to(self.device)
        for i in range(len(radius_set)):
            if i == 0:
                mask.append(torch.where((dist < radius_set[i]), one_mask, zero_mask))
            else:
                mask.append(
                    torch.where(
                        (dist < radius_set[i]) & (dist >= radius_set[i - 1]),
                        one_mask,
                        zero_mask,
                    )
                )

        fq_mask_set = torch.stack(mask, dim=0)
        fq_mask = value_set.unsqueeze(-1).unsqueeze(-1) * fq_mask_set.unsqueeze(0)
        fq_mask = torch.sum(fq_mask, dim=1)
        fq_mask = fq_mask.unsqueeze(1)

        # Compute prbs_fq_conj
        prbs_fq_conj = torch.stack(
            [prbs_fq[:, 0, :, :], -prbs_fq[:, 1, :, :]], dim=1
        )  # (batch, 2, h, 1)

        # Compute S^* / |S|^2
        magnitude_squared = (
            prbs_fq[:, 0, :, :] ** 2 + prbs_fq[:, 1, :, :] ** 2
        )  # (batch, h, 1)
        magnitude_squared = magnitude_squared.clamp(min=1e-10)  # Avoid division by zero
        s_conj_over_magnitude = prbs_fq_conj / magnitude_squared.unsqueeze(
            1
        )  # (batch, 2, h, 1)

        replaced_z_freq = z_clean_fq_c + self.convert_to_complex(
            self._complex_multiply(prbs_fq_conj, y_noise_fq - z_clean_fq * fq_mask)
        )

        replaced_y_noise_freq = y_noise_fq_c + self.convert_to_complex(
            self._complex_multiply(
                s_conj_over_magnitude, z_clean_fq - y_noise_fq * fq_mask
            )
        )

        if self.domain == "time":
            replaced_z_freq = self._postprocess(replaced_z_freq)
            replaced_y_noise_freq = self._postprocess(replaced_y_noise_freq)

        replaced_z_freq = self.reverse_from_complex(replaced_z_freq)
        replaced_y_noise_freq = self.reverse_from_complex(replaced_y_noise_freq)

        return replaced_z_freq, replaced_y_noise_freq, fq_mask


if __name__ == "__main__":
    from configs.ml import device

    model = AFM_PRBS(fq_bound=1.0, device=device)
    z_clean = torch.randn(3, 2, 255, 256).to(device)
    y_noise = torch.randn(3, 2, 255, 256).to(device)
    prbs_fq = torch.randn(3, 2, 255, 1).to(device)

    replaced_z_fq, replaced_y_noise_fq, value_set = model(z_clean, y_noise, prbs_fq)
    print(replaced_z_fq.shape, replaced_y_noise_fq.shape, value_set.shape)
