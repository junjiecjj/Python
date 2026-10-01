"""
通信速率计算函数模块
===================
对应论文 Eq.(4) [Case 1] 和 Eq.(13)-(14) [Case 2]。

用法:
    from comm_rate import compute_rate_case1, compute_rate_case2
    rate1, sinr1 = compute_rate_case1(Rc, h, sigma2_c)
    rate2, sinr2 = compute_rate_case2(Rc, Rs, h, sigma2_c)
"""

import numpy as np


def compute_rate_case1(Rc, h, sigma2_c):
    """
    Compute communication SINR and rate at CU  [Eq.(4)].
    Case 1: Gaussian signals only (no interference).

    gamma_1 = h^H * R_c * h / sigma2_c
    R_1     = log2(1 + gamma_1)

    Args:
        Rc: Transmit covariance matrix (Mt x Mt)
        h: BS -> CU channel vector (Mt, 1)
        sigma2_c: CU noise power (scalar)

    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR value (linear scale)
    """
    sinr = (h.conj().T @ Rc @ h).real / sigma2_c
    rate = np.log2(1 + max(sinr, 0))
    return rate, sinr


def compute_rate_case2(Rc, Rs, h, sigma2_c):
    """
    Compute communication SINR and rate at CU  [Eq.(13)-(14)].
    Case 2: Superposition of Gaussian + deterministic signals.

    gamma_2 = h^H * R_c * h / (h^H * R_s * h + sigma2_c)
    R_2     = log2(1 + gamma_2)

    Args:
        Rc: Information signal covariance matrix (Mt x Mt)
        Rs: Deterministic sensing signal covariance matrix (Mt x Mt)
        h: BS -> CU channel vector (Mt, 1)
        sigma2_c: CU noise power (scalar)

    Returns:
        rate: Achievable rate (bps/Hz)
        sinr: SINR value (linear scale)
    """
    signal   = (h.conj().T @ Rc @ h).real
    interfer = (h.conj().T @ Rs @ h).real
    sinr = signal / (interfer + sigma2_c)
    rate = np.log2(1 + max(sinr, 0))
    return rate, sinr
