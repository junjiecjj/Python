import torch
import torch.nn as nn
import numpy as np


class ComplexKLLoss(nn.Module):
    def __init__(self, phase_bins=100, epsilon=1e-10, w_amplitude=0.5, w_phase=0.5):
        """
        Custom loss for KL Divergence between two complex matrices based on amplitude and phase.
        Input matrices have shape (batch, 2, H, W), where channel 0 is real, channel 1 is imaginary.

        Args:
            phase_bins (int): Number of bins for phase histogram.
            epsilon (float): Small constant to avoid numerical issues.
            w_amplitude (float): Weight for amplitude KL loss.
            w_phase (float): Weight for phase KL loss.
        """
        super(ComplexKLLoss, self).__init__()
        self.phase_bins = phase_bins
        self.epsilon = epsilon
        self.w_amplitude = w_amplitude
        self.w_phase = w_phase
        self.phase_bin_edges = torch.linspace(
            -np.pi, np.pi, phase_bins + 1, device="cpu"
        )

    def to_complex(self, x):
        """
        Convert tensor of shape (batch, 2, H, W) to complex tensor of shape (batch, H, W).

        Args:
            x (torch.Tensor): Input tensor with real (x[:,0,:,:]) and imaginary (x[:,1,:,:]) parts.

        Returns:
            torch.Tensor: Complex tensor of shape (batch, H, W).
        """
        return torch.complex(x[:, 0, :, :], x[:, 1, :, :])

    def compute_kl_divergence(self, p, q):
        """
        Compute KL Divergence between two distributions.

        Args:
            p (torch.Tensor): Normalized distribution P.
            q (torch.Tensor): Normalized distribution Q.

        Returns:
            torch.Tensor: KL Divergence value.
        """
        return torch.sum(
            p * torch.log((p + self.epsilon) / (q + self.epsilon)), dim=(-2, -1)
        )

    def amplitude_kl_loss(self, A, B):
        """
        Compute KL Divergence for amplitudes of complex matrices.

        Args:
            A (torch.Tensor): Input tensor A (shape: [batch, 2, H, W]).
            B (torch.Tensor): Input tensor B (shape: [batch, 2, H, W]).

        Returns:
            torch.Tensor: Amplitude KL Divergence (shape: [batch]).
        """
        # Convert to complex
        A_complex = self.to_complex(A)
        B_complex = self.to_complex(B)

        # Extract amplitudes
        A_abs = torch.abs(A_complex)
        B_abs = torch.abs(B_complex)

        # Normalize to form probability distributions
        S_A = torch.sum(A_abs, dim=(-2, -1), keepdim=True)
        S_B = torch.sum(B_abs, dim=(-2, -1), keepdim=True)
        P = A_abs / (S_A + self.epsilon)
        Q = B_abs / (S_B + self.epsilon)

        # Compute KL Divergence
        return self.compute_kl_divergence(P, Q)

    def phase_kl_loss(self, A, B):
        """
        Compute KL Divergence for phases of complex matrices using histograms.

        Args:
            A (torch.Tensor): Input tensor A (shape: [batch, 2, H, W]).
            B (torch.Tensor): Input tensor B (shape: [batch, 2, H, W]).

        Returns:
            torch.Tensor: Phase KL Divergence (shape: [batch]).
        """
        # Convert to complex
        A_complex = self.to_complex(A)
        B_complex = self.to_complex(B)

        # Extract phases
        A_phase = torch.angle(A_complex)
        B_phase = torch.angle(B_complex)

        # Move bin edges to the same device as input tensors
        bin_edges = self.phase_bin_edges.to(A_phase.device)

        # Compute histograms for phases
        batch_size = A_phase.shape[0]
        P_phase = []
        Q_phase = []
        for i in range(batch_size):
            hist_A, _ = torch.histogram(
                A_phase[i].flatten(), bins=bin_edges, density=True
            )
            hist_B, _ = torch.histogram(
                B_phase[i].flatten(), bins=bin_edges, density=True
            )
            P_phase.append(hist_A)
            Q_phase.append(hist_B)

        P_phase = torch.stack(P_phase)
        Q_phase = torch.stack(Q_phase)

        # Normalize histograms
        P_phase = P_phase / (torch.sum(P_phase, dim=-1, keepdim=True) + self.epsilon)
        Q_phase = Q_phase / (torch.sum(Q_phase, dim=-1, keepdim=True) + self.epsilon)

        # Compute KL Divergence
        return self.compute_kl_divergence(P_phase, Q_phase)

    def forward(self, A, B):
        """
        Compute the total KL Divergence loss.

        Args:
            A (torch.Tensor): Input tensor A (shape: [batch, 2, H, W]).
            B (torch.Tensor): Input tensor B (shape: [batch, 2, H, W]).

        Returns:
            torch.Tensor: Weighted sum of amplitude and phase KL Divergences (shape: [batch]).
        """
        amplitude_loss = self.amplitude_kl_loss(A, B)
        phase_loss = self.phase_kl_loss(A, B)
        return self.w_amplitude * amplitude_loss + self.w_phase * phase_loss
