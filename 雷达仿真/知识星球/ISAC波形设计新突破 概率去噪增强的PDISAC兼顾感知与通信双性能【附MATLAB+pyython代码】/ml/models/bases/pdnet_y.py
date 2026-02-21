import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=2, hidden_channel=32, level=3):
        super().__init__()
        self.level = level
        self.encoders = nn.ModuleList()

        # Build encoder blocks from z1 to zL
        for i in range(level):
            out_ch = hidden_channel
            self.encoders.append(ConvBlock(in_ch if i == 0 else hidden_channel, out_ch))

        # q(z_i|z_{i-1}) for each level i from 1 to L
        self.q_z_blocks = nn.ModuleList()
        for i in range(level):
            in_ch = hidden_channel
            out_ch = hidden_channel * 2  # *2 for mu and logv
            self.q_z_blocks.append(ConvBlock(in_ch, out_ch))

    def forward(self, Z):
        # Encode through all levels
        encoded_features = []
        x = Z

        for i, enc in enumerate(self.encoders):
            x = enc(x)
            encoded_features.append(x)

        # Get q(z_i|z_{i-1}) distributions for each level
        q_distributions = []
        for i, (feature, q_block) in enumerate(zip(encoded_features, self.q_z_blocks)):
            mu, logv = q_block(feature).chunk(2, dim=1)
            logv = torch.clamp(logv, min=-10, max=8)
            q_distributions.append((mu, logv))

        return q_distributions, encoded_features


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, hidden_channel=32, level=3):
        super().__init__()
        self.level = level

        # Decoder blocks for upsampling
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, level)):
            in_ch = hidden_channel
            out_ch = hidden_channel
            self.decoders.append(DecoderBlock(in_ch, out_ch))

        # p(z_{i-1}|z_i) - top-down prior
        self.p_z_blocks = nn.ModuleList()
        for i in reversed(range(1, level)):
            in_ch = hidden_channel
            out_ch = hidden_channel * 2  # *2 for mu and logv
            self.p_z_blocks.append(ConvBlock(in_ch, out_ch))

        # q(z_{i-1}|z_i, encoder_features) - posterior with skip connections
        self.q_z_skip_blocks = nn.ModuleList()
        for i in reversed(range(1, level)):
            # Concatenate decoder output + encoder skip connection
            in_ch = hidden_channel * 2  # decoder + skip
            out_ch = hidden_channel * 2  # *2 for mu and logv
            self.q_z_skip_blocks.append(ConvBlock(in_ch, out_ch))

        # Final reconstruction p(Ẑ|z1)
        self.reconstruct = nn.Sequential(
            DecoderBlock(hidden_channel, hidden_channel),
            ConvBlock(hidden_channel, 2),
        )

    def kl_normal(self, mu_q, logv_q, mu_p=None, logv_p=None):
        if mu_p is None:
            mu_p = torch.zeros_like(mu_q)
        if logv_p is None:
            logv_p = torch.zeros_like(logv_q)

        # Clamp log variances
        logv_q = torch.clamp(logv_q, min=-10, max=8)
        logv_p = torch.clamp(logv_p, min=-10, max=8)

        vq, vp = torch.exp(logv_q), torch.exp(logv_p)
        vp = vp + 1e-8  # Prevent division by zero

        kl = 0.5 * (logv_p - logv_q + (vq + (mu_q - mu_p).pow(2)) / vp - 1)
        kl = torch.clamp(kl, min=-1e6, max=1e6)

        return kl.view(kl.size(0), -1).sum(1).mean()

    def reparameterize(self, mu, logv):
        logv = torch.clamp(logv, min=-10, max=8)
        std = torch.exp(0.5 * logv) + 1e-6
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, z_top, encoder_features=None, distributions=None):
        kls = []
        x = z_top

        # Hierarchical decoding from top level down
        for i, decoder in enumerate(self.decoders):
            # Upsample current level
            x = decoder(x)
            if distributions is not None:
                distributions[
                    rf"$p(\vec{{z}}_{{{self.level - i - 1}}}\mid \vec{{z}}_{{{self.level - i}}})$"
                ] = x

            # Get top-down prior p(z_{i-1}|z_i)
            mu_p, logv_p = self.p_z_blocks[i](x).chunk(2, dim=1)

            if encoder_features is not None:
                # During training: use posterior q(z_{i-1}|z_i, encoder_features)
                skip_idx = (
                    self.level - 2 - i
                )  # Map decoder index to encoder feature index
                if skip_idx >= 0 and skip_idx < len(encoder_features):
                    # Concatenate decoder output with encoder skip connection
                    combined = torch.cat([x, encoder_features[skip_idx]], dim=1)
                    mu_q, logv_q = self.q_z_skip_blocks[i](combined).chunk(2, dim=1)

                    # KL divergence: KL(q(z_{i-1}|z_i, encoder) || p(z_{i-1}|z_i))
                    kls.append(self.kl_normal(mu_q, logv_q, mu_p, logv_p))

        # Final reconstruction
        output = self.reconstruct(x)

        if len(kls) == 0:
            return output

        return output, kls


class PDNet(nn.Module):
    def __init__(self, input_shape=(2, 256, 256), hidden_channel=32, level=4):
        super().__init__()

        c, h, w = input_shape
        self.level = level
        self.encoder = Encoder(in_ch=c, hidden_channel=hidden_channel, level=level)
        self.decoder = Decoder(hidden_channel=hidden_channel, level=level)

        # Encoders for Y and S to get top-level latent distribution
        self.encY = nn.Sequential(
            ConvBlock(2, hidden_channel, 3, 1, 1),
            ConvBlock(hidden_channel, hidden_channel * 2, 3, 1, 1),
        )

        self.extract_z = nn.Sequential(
            ConvBlock(hidden_channel * 2, hidden_channel, 3, 1, 1),
            ConvBlock(hidden_channel, hidden_channel, 3, 1, 1),
        )

    def reparameterize(self, mu, logv):
        logv = torch.clamp(logv, min=-10, max=8)
        std = torch.exp(0.5 * logv) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def combine_experts(self, muY, logvY, muS, logvS):
        """Product of Experts: p(zL|Y,S) ∝ p(zL|Y)p(zL|S)p(zL)"""
        logvY = torch.clamp(logvY, min=-10, max=8)
        logvS = torch.clamp(logvS, min=-10, max=8)

        vY, vS = torch.exp(logvY), torch.exp(logvS)
        invY, invS = 1 / (vY + 1e-8), 1 / (vS + 1e-8)

        # Product of experts
        v_c = 1 / (invY + invS + 1e-8)
        mu_c = v_c * (muY * invY + muS * invS)
        logv_c = torch.clamp(torch.log(v_c + 1e-8), min=-10, max=8)

        return mu_c, logv_c

    def kl_normal(self, mu_q, logv_q, mu_p=None, logv_p=None):
        if mu_p is None:
            mu_p = torch.zeros_like(mu_q)
        if logv_p is None:
            logv_p = torch.zeros_like(logv_q)

        logv_q = torch.clamp(logv_q, min=-10, max=8)
        logv_p = torch.clamp(logv_p, min=-10, max=8)

        vq, vp = torch.exp(logv_q), torch.exp(logv_p)
        vp = vp + 1e-8

        kl = 0.5 * (logv_p - logv_q + (vq + (mu_q - mu_p).pow(2)) / vp - 1)
        kl = torch.clamp(kl, min=-1e6, max=1e6)

        return kl.view(kl.size(0), -1).sum(1).mean()

    def forward(self, Y, Z):
        # Get encoder distributions q(z_i|Z) for all levels
        q_distributions, encoder_features = self.encoder(Z)

        # Get top-level prior from encoder: q(zL|Z)
        mu_encoder, logv_encoder = q_distributions[-1]  # Top level

        # Get expert distributions from Y and S
        b, c, h, w = Y.shape
        enc_Y = self.encY(Y)

        mu_Y, logv_Y = enc_Y.chunk(2, dim=1)

        # KL divergence for top level: KL(q(zL|Y) || q(zL|Z))
        kl_top = self.kl_normal(mu_Y, logv_Y, mu_encoder, logv_encoder)

        z_top = self.extract_z(enc_Y)

        # Hierarchical decoding with skip connections
        output, kls_hierarchical = self.decoder(z_top, encoder_features[:-1])

        # Combine all KL divergences
        all_kls = [kl_top] + kls_hierarchical

        return output, all_kls

    def predict(self, Y):
        """Inference without encoder Z"""
        b, c, h, w = Y.shape

        # Get expert distributions
        enc_Y = self.encY(Y)

        z_top = self.extract_z(enc_Y)

        # Hierarchical decoding without skip connections (inference mode)
        output = self.decoder(z_top)

        return output

    def distribution(self, Y, Z, distributions={}):
        q_distributions, encoder_features = self.encoder(Z)
        for i, encoder_feature in enumerate(encoder_features):
            distributions[rf"$p(\vec{{z}}_{{{i + 1}}}\mid \vec{{z}}_{{{i}}})$"] = (
                encoder_feature
            )

        # Get top-level prior from encoder: q(zL|Z)
        mu_encoder, logv_encoder = q_distributions[-1]  # Top level

        # Get expert distributions from Y and S
        b, c, h, w = Y.shape
        enc_Y = self.encY(Y)

        mu_Y, logv_Y = enc_Y.chunk(2, dim=1)

        # KL divergence for top level: KL(q(zL|Y,S) || q(zL|Z))
        kl_top = self.kl_normal(mu_Y, logv_Y, mu_encoder, logv_encoder)

        z_top = self.extract_z(enc_Y)
        distributions[rf"$p(\vec{{z}}_{{{self.level}}})$"] = z_top

        # Hierarchical decoding with skip connections
        output, kls_hierarchical = self.decoder(
            z_top, encoder_features[:-1], distributions
        )

        # Combine all KL divergences
        all_kls = [kl_top] + kls_hierarchical

        return output, all_kls
