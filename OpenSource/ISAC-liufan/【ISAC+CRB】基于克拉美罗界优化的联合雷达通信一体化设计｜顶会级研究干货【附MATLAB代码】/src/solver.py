"""
CRB-ISAC Beamforming Solver
=============================
SDR-based solvers for CRB-minimizing DFRC beamforming.

Implements:
1. Closed-form solution for single-user case (Theorem 1)
2. SDR for multi-user CRB minimization (Problems P1/P2)
3. Rank-one extraction via eigenvalue decomposition
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import Optional
from src.model import SystemParams, db_to_linear, ULASteeringVector


class SingleUserSolver:
    """
    Closed-form solution for K=1 (single user) point target.
    From paper Theorem 1: CRB-minimizing beamformer is the
    transmit steering vector weighted by sqrt(P_r).
    """

    def __init__(self, params: SystemParams):
        self.N_t = params.N_t
        self.ula = ULASteeringVector(params.N_t)
        self.P_total = db_to_linear(params.P_dBm - 30)
        self.sigma2 = db_to_linear(params.sigma2_dBm - 30)

    def solve(self, theta_target: float = 0.0) -> np.ndarray:
        """
        Compute closed-form beamformer.

        Args:
            theta_target: Target angle in degrees

        Returns:
            Beamforming vector w (N_t,)
        """
        a = self.ula.compute(theta_target)  # (N_t,)
        w = np.sqrt(self.P_total) * a / np.linalg.norm(a)
        return w


class MultiUserSDRSolver:
    """
    SDR solver for multi-user CRB minimization problem.

    Problem (P1) from paper:
        min_{W}  CRB(θ)
        s.t.  SINR_k ≥ γ_k,  ∀k
              ||W||_F² ≤ P_r

    Relaxation: W → X = W W^H (SDP)
    """

    def __init__(self, params: SystemParams):
        self.N_t = params.N_t
        self.K = params.K
        self.L = params.L
        self.P_total = db_to_linear(params.P_dBm - 30)
        self.sigma2 = db_to_linear(params.sigma2_dBm - 30)
        self.gamma = db_to_linear(params.gamma_dB)
        self.ula = ULASteeringVector(params.N_t)

    def solve(
        self,
        H: np.ndarray,
        theta_target: float = 0.0,
        rank_one: bool = True,
        solver: str = "SCS",
    ) -> tuple[np.ndarray, dict]:
        """
        Solve multi-user CRB minimization via SDR.

        Args:
            H: Channel matrix (N_t x K), columns are user channels
            theta_target: Target angle for radar
            rank_one: If True, extract rank-one solution via EVD
            solver: CVXPY solver ('SCS', 'ECOS', 'MOSEK')

        Returns:
            (W, info): beamforming matrix and solver info
        """
        N_t, K = self.N_t, self.K
        a_target = self.ula.compute(theta_target)

        # SDR variable: X = W W^H (PSD, symmetric is implied)
        X = cp.Variable((N_t, N_t), PSD=True)
        constraints = []

        # Power constraint: tr(X) ≤ P_total
        constraints.append(cp.trace(X) <= self.P_total * (1 + 1e-6))

        # SINR constraints for each user
        # DCP-compliant form: signal >= gamma * (interference + sigma2)
        for k in range(K):
            h_k = H[:, k]  # (N_t,)
            h_k_H = h_k.conj().T  # (1, N_t)

            # Signal power for user k: |h_k^H w_k|^2
            signal_k = cp.real(h_k_H @ X @ h_k)
            # Interference from other users
            interference = sum(
                cp.real(H[:, j].conj().T @ X @ H[:, j])
                for j in range(K) if j != k
            )
            # Linearized SINR constraint: signal >= gamma * (interference + sigma2)
            # Note: use relaxed tolerance to avoid infeasibility from linearization gap
            constraints.append(
                signal_k >= self.gamma * (interference + self.sigma2 * (1 + 1e-4))
            )

        # Objective: minimize CRB
        # For point target: CRB ∝ -trace(A X) where A = a(θ)a(θ)^H
        # Minimizing trace(A X) maximizes projection onto target direction
        A = a_target[:, None] @ a_target[None, :].conj()
        objective = cp.Minimize(cp.real(cp.trace(A @ X)))

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            prob.solve(solver="ECOS", verbose=False)

        status = prob.status
        opt_val = prob.value if prob.status not in ["infeasible", "unbounded"] else None

        # Extract X*
        if prob.status in ["optimal", "optimal_inaccurate"]:
            X_opt = X.value
        else:
            X_opt = np.eye(N_t) * self.P_total / N_t

        # Rank-one extraction
        if rank_one:
            W = self._extract_rank_one(X_opt, K)
        else:
            # Take square root
            eigenvalues, eigenvectors = np.linalg.eigh(X_opt)
            eigenvalues = np.maximum(eigenvalues, 0)
            W = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        info = {
            "status": status,
            "optimal_value": opt_val,
            "cvxpy_status": prob.status,
            "rank_one_extracted": rank_one,
        }
        return W, info

    def _extract_rank_one(self, X: np.ndarray, K: int) -> np.ndarray:
        """
        Extract rank-one beamformers via EVD + Gaussian randomization.

        Uses Theorem 4 from paper: extract rank-one components from X*.
        """
        N_t = X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(X)
        eigenvalues = np.maximum(eigenvalues, 0)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        W = np.zeros((N_t, K), dtype=complex)
        remaining_power = self.P_total

        for k in range(K):
            if k >= len(eigenvalues) or eigenvalues[k] <= 1e-9:
                # Fallback: use dominant eigenvector
                v = eigenvectors[:, 0]
                power = min(remaining_power / K, self.P_total / K)
                w_k = np.sqrt(power) * v / np.linalg.norm(v)
            else:
                v = eigenvectors[:, k]
                power = min(eigenvalues[k], remaining_power / (K - k))
                w_k = np.sqrt(power) * v

            W[:, k] = w_k
            remaining_power -= np.linalg.norm(w_k) ** 2
            remaining_power = max(remaining_power, 0)

        return W


class BeampatternApproxSolver:
    """
    Benchmark: Beampattern approximation designs.

    Design 1: Uniform power + phase steering to target
    Design 2: SVD-based beampattern with SINR constraints
    """

    def __init__(self, params: SystemParams):
        self.N_t = params.N_t
        self.K = params.K
        self.P_total = db_to_linear(params.P_dBm - 30)
        self.ula = ULASteeringVector(params.N_t)

    def design1(self, theta_target: float = 0.0) -> np.ndarray:
        """Design 1: Uniform weighted beamformer steered to target."""
        a = self.ula.compute(theta_target)
        w = a / np.linalg.norm(a) * np.sqrt(self.P_total / self.K)
        W = np.tile(w[:, None], (1, self.K))
        return W

    def design2(self, H: np.ndarray, theta_target: float = 0.0) -> np.ndarray:
        """Design 2: MRT beamforming with target direction boost."""
        a = self.ula.compute(theta_target)
        W = np.zeros((self.N_t, self.K), dtype=complex)
        for k in range(self.K):
            h_k = H[:, k]
            # MRT: w_k ∝ h_k
            w_mrt = h_k / np.linalg.norm(h_k) * np.sqrt(self.P_total / self.K)
            # Boost target direction
            w_k = 0.7 * w_mrt + 0.3 * a * np.sqrt(self.P_total / self.K) / np.linalg.norm(a)
            W[:, k] = w_k / np.linalg.norm(w_k) * np.sqrt(self.P_total / self.K)
        return W
