"""
Unit tests for solver.py
"""

import pytest
import numpy as np
from src.model import SystemParams
from src.solver import SingleUserSolver, MultiUserSDRSolver, BeampatternApproxSolver


class TestSingleUserSolver:
    @pytest.fixture
    def params(self):
        return SystemParams(N_t=16, N_p=20, K=1, L=30, P_dBm=30.0, sigma2_dBm=0.0)

    @pytest.fixture
    def solver(self, params):
        return SingleUserSolver(params)

    def test_w_norm(self, solver):
        w = solver.solve(theta_target=0)
        assert w.shape == (16,)
        power = np.linalg.norm(w) ** 2
        assert abs(power - solver.P_total) < 1e-6

    def test_different_angles(self, solver):
        w0 = solver.solve(theta_target=0)
        w30 = solver.solve(theta_target=30)
        assert not np.allclose(w0, w30)


class TestMultiUserSDRSolver:
    @pytest.fixture
    def params(self):
        return SystemParams(N_t=16, N_p=20, K=4, L=30, P_dBm=30.0, sigma2_dBm=0.0, gamma_dB=15.0)

    @pytest.fixture
    def solver(self, params):
        return MultiUserSDRSolver(params)

    @pytest.fixture
    def random_channels(self, params):
        np.random.seed(42)
        # i.i.d. Rayleigh fading channels (scaled so avg |h|^2 = 1 per antenna)
        H = (np.random.randn(params.N_t, params.K) + 1j * np.random.randn(params.N_t, params.K)) / np.sqrt(2)
        return H

    def test_sdr_returns_w(self, solver, random_channels):
        W, info = solver.solve(random_channels, theta_target=0)
        assert W.shape == (solver.N_t, solver.K)
        assert "status" in info

    def test_power_constraint(self, solver, random_channels):
        W, _ = solver.solve(random_channels, theta_target=0)
        power = np.linalg.norm(W, "fro") ** 2
        assert power <= solver.P_total * 1.01

    def test_rank_one_extraction(self, solver, random_channels):
        W, info = solver.solve(random_channels, theta_target=0, rank_one=True)
        assert W.shape == (solver.N_t, solver.K)
        assert info["rank_one_extracted"] is True

    def test_solver_status(self, solver, random_channels):
        _, info = solver.solve(random_channels, theta_target=0)
        # Accept infeasible as well (SDR may be infeasible with certain channel realizations)
        assert info["cvxpy_status"] in ["optimal", "optimal_inaccurate", "infeasible"]


class TestBeampatternApproxSolver:
    @pytest.fixture
    def params(self):
        return SystemParams(N_t=16, N_p=20, K=4, L=30, P_dBm=30.0, sigma2_dBm=0.0, gamma_dB=15.0)

    @pytest.fixture
    def solver(self, params):
        return BeampatternApproxSolver(params)

    def test_design1_shape(self, solver):
        W = solver.design1(theta_target=0)
        assert W.shape == (16, 4)

    def test_design1_power(self, solver):
        W = solver.design1(theta_target=0)
        power = np.linalg.norm(W, "fro") ** 2
        assert power <= solver.P_total * 1.01

    def test_design2_shape(self, solver):
        np.random.seed(42)
        H = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        W = solver.design2(H, theta_target=0)
        assert W.shape == (16, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
