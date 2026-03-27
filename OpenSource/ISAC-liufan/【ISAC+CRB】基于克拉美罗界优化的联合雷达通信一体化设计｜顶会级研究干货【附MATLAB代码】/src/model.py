"""
CRB-ISAC Beamforming Model
============================
Mathematical model for the CRB-minimizing dual-functional
radar-communication (DFRC) beamforming problem.

Paper: "Cramer-Rao Bound Optimization for Joint Radar-Communication Design"
       Fan Liu et al., IEEE TSP 2022

Key formulations:
- Problem (P1): CRB minimization + per-user SINR constraints
- Problem (P2): Extended target CRB minimization
"""

from __future__ import annotations
import numpy as np
from typing import NamedTuple


class SystemParams(NamedTuple):
    """System parameters extracted from paper Section VII."""
    N_t: int = 16       # Number of transmit antennas
    N_p: int = 20       # Number of receive antennas
    K: int = 4          # Number of communication users
    L: int = 30         # Frame length (fast-time snapshots)
    P_dBm: float = 30.0 # Total transmit power (dBm)
    sigma2_dBm: float = 0.0  # Noise variance (dBm)
    gamma_dB: float = 15.0  # SINR threshold (dB) per user


def db_to_linear(x_db: float) -> float:
    """dB to linear conversion."""
    return 10 ** (x_db / 10)


def linear_to_db(x: float) -> float:
    """Linear to dB conversion."""
    return 10 * np.log10(x + 1e-12)


class ULASteeringVector:
    """Uniform Linear Array (ULA) steering vector."""

    def __init__(self, N: int, spacing: float = 0.5, wavelength: float = 2.0):
        self.N = N
        self.spacing = spacing
        self.wavelength = wavelength
        self.k = 2 * np.pi * spacing / wavelength

    def compute(self, angle_deg: float) -> np.ndarray:
        """
        Compute steering vector for given angle.

        Args:
            angle_deg: Angle in degrees (-90 to 90)

        Returns:
            Steering vector of shape (N_t,)
        """
        theta = np.deg2rad(angle_deg)
        a = np.exp(1j * self.k * np.arange(self.N) * np.sin(theta))
        return a

    def compute_range(self, n_points: int = 361) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute steering vectors over all angles.

        Returns:
            (angles, A): angles in degrees, array of shape (n_points, N_t)
        """
        angles = np.linspace(-90, 90, n_points)
        thetas = np.deg2rad(angles)
        A = np.exp(
            1j * self.k * np.arange(self.N)[None, :] * np.sin(thetas)[:, None]
        )
        return angles, A


class CRBModel:
    """
    CRB model for MIMO DFRC system.

    Supports:
    - Point target: CRB(θ) = 1 / (2·L·SNR_eff·|α|²)
    - Extended target: CRB(G) = tr(J^{-1}) / L = σ² / (|α|²·L·N_p)
    """

    def __init__(self, params: SystemParams):
        self.N_t = params.N_t
        self.N_p = params.N_p
        self.K = params.K
        self.L = params.L
        self.P_total = db_to_linear(params.P_dBm - 30)  # relative to 1 mW
        self.sigma2 = db_to_linear(params.sigma2_dBm - 30)
        self.gamma_db = params.gamma_dB
        self.gamma = db_to_linear(params.gamma_dB)
        self.ula_tx = ULASteeringVector(params.N_t)
        self.ula_rx = ULASteeringVector(params.N_p)

    def crb_point_target(
        self,
        W: np.ndarray,
        h: np.ndarray,
        alpha2: float = 1.0,
    ) -> float:
        """
        Compute CRB for point target angle estimation.

        Args:
            W: Transmit beamforming matrix (N_t x K)
            h: Channel vector to target (N_t,) — steering vector at target angle
            alpha2: Signal gain |α|²

        Returns:
            CRB(θ) value
        """
        # Effective SNR at radar receiver
        SNR_radar = self._compute_radar_snr(W, h, alpha2)
        if SNR_radar < 1e-12:
            return 1e6
        crb = 1.0 / (2.0 * self.L * SNR_radar + 1e-12)
        return crb

    def crb_extended_target(
        self,
        W: np.ndarray,
        alpha2: float = 1.0,
    ) -> float:
        """
        Compute CRB for extended target response matrix G.

        Args:
            W: Transmit beamforming matrix (N_t x K)
            alpha2: Signal gain |α|²

        Returns:
            CRB(G) value (MSE of G estimation)
        """
        P_radar = self._compute_radar_power(W)
        SNR_eff = P_radar / self.sigma2
        crb = self.sigma2 / (alpha2 * self.L * self.N_p * SNR_eff + 1e-12)
        return crb

    def compute_user_sinr(self, W: np.ndarray, h_k: np.ndarray) -> float:
        """
        Compute SINR for user k.

        SINR_k = |h_k^H w_k|² / (Σ_{j≠k} |h_k^H w_j|² + σ²)
        """
        signal = np.abs(h_k.conj().T @ W[:, 0]) ** 2
        interference = sum(
            np.abs(h_k.conj().T @ W[:, j]) ** 2
            for j in range(1, self.K)
        ) if W.shape[1] > 1 else 0.0
        denominator = interference + self.sigma2
        if denominator < 1e-12:
            return 1e12
        return signal / denominator

    def _compute_radar_snr(
        self,
        W: np.ndarray,
        h: np.ndarray,
        alpha2: float,
    ) -> float:
        """Compute effective radar SNR at receiver."""
        # Transmit power in target direction
        P_tx = np.linalg.norm(W, "fro") ** 2
        # SNR at radar receiver (simplified: proportional to transmit power)
        SNR = alpha2 * P_tx / self.sigma2
        return SNR

    def _compute_radar_power(self, W: np.ndarray) -> float:
        """Compute total transmit power."""
        return np.linalg.norm(W, "fro") ** 2

    def power_sharing_fraction(self, W: np.ndarray) -> tuple[float, float]:
        """
        Compute power sharing between radar and communication.

        Returns:
            (comm_fraction, radar_fraction): fractions of total power
        """
        P_total = np.linalg.norm(W, "fro") ** 2
        if P_total < 1e-12:
            return 0.0, 0.0
        # Communication power: proportional to SINR requirement
        P_comm_needed = self.gamma * self.sigma2 * self.K
        comm_frac = min(P_comm_needed / self.P_total, 1.0)
        radar_frac = max(1.0 - comm_frac, 0.0)
        return comm_frac, radar_frac

    def verify_sinr_constraints(
        self,
        W: np.ndarray,
        H: np.ndarray,
        tol: float = 1e-3,
    ) -> dict:
        """
        Verify per-user SINR constraints are satisfied.

        Args:
            W: Beamforming matrix (N_t x K)
            H: Channel matrix (N_t x K), columns are user channels
            tol: Relative tolerance

        Returns:
            dict with per-user SINR and constraint satisfaction
        """
        results = {}
        for k in range(self.K):
            sinr_k = self.compute_user_sinr(W, H[:, k])
            satisfied = sinr_k >= self.gamma * (1 - tol)
            results[f"user_{k}"] = {
                "sinr_linear": sinr_k,
                "sinr_db": linear_to_db(sinr_k),
                "threshold_db": self.gamma_db,
                "satisfied": satisfied,
            }
        return results
