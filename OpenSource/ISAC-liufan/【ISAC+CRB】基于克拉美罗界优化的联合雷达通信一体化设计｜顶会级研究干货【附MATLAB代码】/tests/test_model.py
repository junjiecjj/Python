"""
Unit tests for model.py
"""

import pytest
import numpy as np
from src.model import (
    SystemParams, db_to_linear, linear_to_db,
    ULASteeringVector, CRBModel,
)


class TestDBConversions:
    def test_db_to_linear(self):
        assert abs(db_to_linear(0) - 1.0) < 1e-8
        assert abs(db_to_linear(10) - 10.0) < 1e-6
        assert abs(db_to_linear(30) - 1000.0) < 1e-3

    def test_linear_to_db(self):
        assert abs(linear_to_db(1.0) - 0.0) < 1e-8
        assert abs(linear_to_db(10.0) - 10.0) < 1e-6
        assert abs(linear_to_db(1000.0) - 30.0) < 1e-3

    def test_roundtrip(self):
        for db in [-10, 0, 10, 20, 30, 40]:
            assert abs(db_to_linear(linear_to_db(db_to_linear(db))) - db_to_linear(db)) < 1e-10


class TestULASteeringVector:
    def test_shape(self):
        ula = ULASteeringVector(N=8)
        a = ula.compute(0)
        assert a.shape == (8,)

    def test_norm_equals_sqrtN(self):
        ula = ULASteeringVector(N=8)
        a = ula.compute(0)
        assert abs(np.linalg.norm(a) - np.sqrt(8)) < 1e-8

    def test_boresight_nonzero(self):
        ula = ULASteeringVector(N=8)
        a = ula.compute(0)
        assert np.abs(a[0]) > 0

    def test_range_shape(self):
        ula = ULASteeringVector(N=8)
        angles, A = ula.compute_range(n_points=181)
        assert A.shape == (181, 8)
        assert len(angles) == 181


class TestCRBModel:
    @pytest.fixture
    def model(self):
        params = SystemParams(N_t=16, N_p=20, K=4, L=30, P_dBm=30.0, sigma2_dBm=0.0, gamma_dB=15.0)
        return CRBModel(params)

    def test_init(self, model):
        assert model.N_t == 16
        assert model.K == 4
        assert model.L == 30

    def test_crb_point_target(self, model):
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        h = np.random.randn(16) + 1j * np.random.randn(16)
        crb = model.crb_point_target(W, h)
        assert crb > 0

    def test_crb_extended_target(self, model):
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        crb = model.crb_extended_target(W)
        assert crb > 0

    def test_sinr_positive(self, model):
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        h = np.random.randn(16) + 1j * np.random.randn(16)
        sinr = model.compute_user_sinr(W, h)
        assert sinr > 0

    def test_power_sharing(self, model):
        W = np.random.randn(16, 4) + 1j * np.random.randn(16, 4)
        comm_frac, radar_frac = model.power_sharing_fraction(W)
        assert 0 <= comm_frac <= 1
        assert 0 <= radar_frac <= 1
        assert abs(comm_frac + radar_frac - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
