"""
CRB 计算函数模块
===============
对应论文 Eq.(22) [Case 1] 和 Eq.(45) [Case 2]。

用法:
    from crb_calc import compute_crb_case1, compute_crb_case2
    crb1 = compute_crb_case1(theta, Rc, a, b, b_dot, alpha_sq, sigma2_s, T)
    crb2 = compute_crb_case2(theta, Rc, Rs, a, b, b_dot, alpha_sq, sigma2_s, T)
"""

import numpy as np


def compute_crb_case1(theta, Rc, a, b, b_dot, alpha_sq, sigma2_s, T):
    """
    Compute CRB for target DoA estimation with Gaussian signals only  [Eq.(22)].

    CRB_1(theta) = sigma2_s / (2 * T * |alpha|^2 * a^H R_c a * ||b_dot||^2)
                  * (1 + sigma2_s / (|alpha|^2 * a^H R_c a * ||b||^2))

    Args:
        theta: Target DoA (rad)
        Rc: Transmit covariance matrix (Mt x Mt)
        a: BS steering vector toward phi_target (Mt, 1)
        b: RX steering vector toward theta_target (Mr, 1)
        b_dot: Derivative of b w.r.t. theta (Mr, 1)
        alpha_sq: |alpha|^2 (scalar), channel coefficient
        sigma2_s: Sensing receiver noise power (scalar)
        T: Number of symbols

    Returns:
        crb: CRB value (rad^2), or 1e10 if no power in sensing direction
    """
    aH_Rc_a = (a.conj().T @ Rc @ a).real

    if aH_Rc_a <= 1e-20:
        return 1e10

    norm_b    = np.linalg.norm(b)
    norm_bdot = np.linalg.norm(b_dot)

    crb = (sigma2_s / (2 * T * alpha_sq * aH_Rc_a * norm_bdot**2)
           * (1 + sigma2_s / (alpha_sq * aH_Rc_a * norm_b**2)))

    return crb


def compute_crb_case2(theta, Rc, Rs, a, b, b_dot, alpha_sq, sigma2_s, T):
    """
    Compute CRB for target DoA estimation with superposition signals  [Eq.(45)].

    CRB_2(theta) = sigma2_s / (2 * T * |alpha|^2 * F)

    where
        F = A_s + gamma_ran / (1 + gamma_ran) * A_c
        A_s = a^H R_s a * ||b_dot||^2
        A_c = a^H R_c a * ||b_dot||^2
        gamma_ran = |alpha|^2 * a^H R_c a * ||b||^2 / sigma2_s

    Args:
        theta: Target DoA (rad)
        Rc: Information signal covariance matrix (Mt x Mt)
        Rs: Deterministic sensing signal covariance matrix (Mt x Mt)
        a: BS steering vector toward phi_target (Mt, 1)
        b: RX steering vector toward theta_target (Mr, 1)
        b_dot: Derivative of b w.r.t. theta (Mr, 1)
        alpha_sq: |alpha|^2 (scalar), channel coefficient
        sigma2_s: Sensing receiver noise power (scalar)
        T: Number of symbols

    Returns:
        crb: CRB value (rad^2), or 1e10 if no power in sensing direction
    """
    aH_Rc_a = (a.conj().T @ Rc @ a).real
    aH_Rs_a = (a.conj().T @ Rs @ a).real

    if aH_Rc_a <= 1e-20 and aH_Rs_a <= 1e-20:
        return 1e10

    norm_b    = np.linalg.norm(b)
    norm_bdot = np.linalg.norm(b_dot)

    gamma_ran = alpha_sq * aH_Rc_a * norm_b**2 / sigma2_s

    A_s = aH_Rs_a * norm_bdot**2
    A_c = aH_Rc_a * norm_bdot**2

    if gamma_ran > 0:
        F = A_s + (gamma_ran / (1 + gamma_ran)) * A_c
    else:
        F = A_s

    if F <= 1e-20:
        return 1e10

    crb = sigma2_s / (2 * T * alpha_sq * F)

    return crb


def compute_crb_deterministic(theta, R, a, b, b_dot, alpha_sq, sigma2_s, T):
    """
    Compute CRB for target DoA estimation with deterministic signals  [Eq.(23)].
    Used for: "ISAC with given realizations of information signals" benchmark.

    CRB_d(theta) = sigma2_s / (2 * T * |alpha|^2 * a^H R a * ||b_dot||^2)

    Note: no (1 + 1/gamma_ran) penalty term — the receiver knows the signal realizations.

    Args:
        theta: Target DoA (rad)
        R: Transmit covariance matrix (Mt x Mt)
        a: BS steering vector toward phi_target (Mt, 1)
        b: RX steering vector toward theta_target (Mr, 1)
        b_dot: Derivative of b w.r.t. theta (Mr, 1)
        alpha_sq: |alpha|^2 (scalar), channel coefficient
        sigma2_s: Sensing receiver noise power (scalar)
        T: Number of symbols

    Returns:
        crb: CRB value (rad^2), or 1e10 if no power in sensing direction
    """
    aH_R_a = (a.conj().T @ R @ a).real

    if aH_R_a <= 1e-20:
        return 1e10

    norm_bdot = np.linalg.norm(b_dot)

    crb = sigma2_s / (2 * T * alpha_sq * aH_R_a * norm_bdot**2)

    return crb
