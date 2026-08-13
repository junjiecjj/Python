"""
波束赋形优化求解器
=================
对应论文 Proposition 1 [Eq.(34)-(36)]:SINR-Constrained CRB Minimization 的闭式最优解。

用法示例:
    from beamforming_opt import solve_p2_optimal
    Rc_opt, status = solve_p2_optimal(gamma_0, h, a, sigma2_c, P, Mt)
"""

import numpy as np

##case1中的R_c_opt求解器
def solve_p2_optimal(gamma_0, h, a, sigma2_c, P, Mt):
    """
    SINR-Constrained CRB Minimization — 闭式最优解  [Proposition 1].

    maximize     a^T R_c a*
    subject to   h^H R_c h / sigma2_c >= gamma_0    (SINR constraint)
                 R_c >= 0                             (PSD)
                 tr(R_c) <= P                          (Power)

    解的形式 [Eq.(34)]:
    如果 P * |h^H a*|^2 >= Mt * gamma_0 * sigma2_c  → MRT toward target
    否则 → rank-2 解（在 h 和 a* 张成的子空间内）

    Args:
        gamma_0: SINR threshold (linear scale, not dB)
        h: BS → CU channel vector (Mt, 1)  [Eq.(62)]
        a: BS steering vector toward target (Mt, 1)  [Eq.(5a)]
        sigma2_c: CU noise power (scalar)
        P: BS max transmit power (scalar, Watts)
        Mt: Number of BS antennas

    Returns:
        Rc_opt: Optimal transmit covariance matrix (Mt x Mt)
        status: 'MRT-target' | 'rank-2' | 'infeasible: reason'
    """
    # Flatten to 1D for linear algebra
    h = h.flatten()
    a = a.flatten()

    # Check fundamental feasibility
    max_SINR = P * np.linalg.norm(h)**2 / sigma2_c
    if gamma_0 > max_SINR:
        return None, f"infeasible: gamma_0={10*np.log10(gamma_0):.1f}dB exceeds max possible SINR ({10*np.log10(max_SINR):.1f}dB)"

    # Proposition 1 condition
    hH_astar_sq = np.abs(h.conj() @ a.conj())**2
    MRT_satisfies = P * hH_astar_sq >= Mt * gamma_0 * sigma2_c

    if MRT_satisfies:
        # Case 1: MRT toward target  [Eq.(34) top]
        Rc_opt = P * np.outer(a.conj(), a) / np.linalg.norm(a)**2
        return Rc_opt, "MRT-target"

    else:
        # Case 2: Rank-2 solution  [Eq.(34) bottom]

        # Orthonormal basis  [Eq.(35)]
        u1 = h / np.linalg.norm(h)
        proj = (u1.conj() @ a.conj()) * u1
        u2 = a.conj() - proj
        u2_norm = np.linalg.norm(u2)
        if u2_norm < 1e-15:
            return None, "infeasible: a* collinear with h (no spatial DoF)"
        u2 = u2 / u2_norm

        # Lambda matrix elements  [Eq.(36)]
        lambda1 = gamma_0 * sigma2_c / np.linalg.norm(h)**2
        lambda2 = P - lambda1
        if lambda2 < 0:
            return None, f"infeasible: insufficient power ({lambda2:.2e} < 0)"

        u1H_astar = (u1.conj() @ a.conj())
        lambda12 = 0.0
        if abs(u1H_astar) > 1e-15:
            lambda12 = np.sqrt(lambda1 * lambda2) * u1H_astar / abs(u1H_astar)

        Lambda = np.array([[lambda1, lambda12],
                           [lambda12.conj(), lambda2]])

        U = np.column_stack([u1, u2])
        Rc_opt = U @ Lambda @ U.conj().T

        # Force Hermitian symmetry
        Rc_opt = (Rc_opt + Rc_opt.conj().T) / 2

        return Rc_opt, "rank-2"



