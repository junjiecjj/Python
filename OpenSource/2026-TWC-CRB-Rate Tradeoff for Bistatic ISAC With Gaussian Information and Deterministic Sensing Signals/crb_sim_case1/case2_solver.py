"""
Case 2 波束赋形优化求解器 (SCA)
=============================
对应论文 Algorithm 1 + Problem (P4)。

问题 (P4): 非凸 → SCA 迭代求解
    max  A_s + gamma_ran/(1+gamma_ran) * A_c
    s.t. h^H R_c h / (h^H R_s h + sigma2_c) >= gamma_0   (SINR)
         R_c >= 0, R_s >= 0                              (PSD)
         tr(R_c) + tr(R_s) <= P                           (Power)

用法:
    from case2_solver import solve_p4_sca
    Rc_opt, Rs_opt, info = solve_p4_sca(gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq)
"""

import numpy as np
import cvxpy as cp


def solve_p4_sca(gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq,
                 max_iter=50, tol=1e-4):
    """
    Solve Problem (P4) using SCA  [Algorithm 1].

    Args:
        gamma_0: SINR threshold (linear scale, not dB)
        h: BS -> CU channel vector (Mt, 1)
        a: BS steering vector toward target (Mt, 1)
        sigma2_c: CU noise power (scalar)
        sigma2_s: Sensing receiver noise power (scalar)
        P: BS max transmit power (scalar, Watts)
        Mt, Mr: Number of antennas
        b: RX steering vector (Mr, 1)
        b_dot: Derivative of b (Mr, 1)
        alpha_sq: |alpha|^2 (scalar)
        max_iter: Maximum SCA iterations
        tol: Convergence tolerance

    Returns:
        Rc_opt: Optimal information covariance (Mt x Mt)
        Rs_opt: Optimal sensing covariance (Mt x Mt)
        info: dict with convergence info
    """
    h_flat = h.flatten()
    a_flat = a.flatten()
    b_flat = b.flatten()

    # ========================================================================
    # SINR: h^H R_c h >= gamma_0 * (h^H R_s h + sigma2_c)
    #   h_tilde = h / sqrt(sigma2_c),
    #   h_tilde^H R_c h_tilde >= gamma_0 * (h_tilde^H R_s h_tilde + 1)
    # 这一步主要是解决求解器容差问题，扩大数值范围，避免过小的数值导致求解器误判不可行或不收敛。
    # ========================================================================
    sigma_c_sqrt = np.sqrt(sigma2_c)
    h_tilde = h_flat / sigma_c_sqrt
    h_norm_sq = np.linalg.norm(h_tilde)**2

    # ========================================================================
    # Feasibility check
    # ========================================================================
    max_SINR = P * np.linalg.norm(h_flat)**2 / sigma2_c  # = P * h_norm_sq
    if gamma_0 > max_SINR:
        return None, None, {"status": f"infeasible: gamma_0={10*np.log10(gamma_0):.1f}dB > max SINR ({10*np.log10(max_SINR):.1f}dB)"}

    # ========================================================================
    # Constants for SCA
    # ========================================================================
    C = alpha_sq * np.linalg.norm(b_flat)**2 / sigma2_s       # Eq.(45): 变量打包一下 gamma_ran = C * a^H R_c a
    H_mat = np.outer(h_tilde, h_tilde.conj())                  # h_tilde @ h_tilde^H (normalized)
    M_mat = np.outer(a_flat.conj(), a_flat)                    # a* @ a^T  (so tr(R @ M) = a^T R a*)

    # ========================================================================
    # Initialization: ensure feasible (account for Rs interference at CU)
    # ========================================================================
    a_norm = a_flat / np.linalg.norm(a_flat)
    h_a_corr_sq = np.abs(h_tilde.conj() @ a_norm)**2          # |a_norm^H h_tilde|^2

    # Rc must overcome both noise AND Rs interference
    # Solve: Rc_power * h_norm_sq >= gamma_0 * ((P - Rc_power) * h_a_corr_sq + 1)
    # => Rc_power * (h_norm_sq + gamma_0 * h_a_corr_sq) >= gamma_0 * (P * h_a_corr_sq + 1)
    denom = h_norm_sq + gamma_0 * h_a_corr_sq
    numer = gamma_0 * (P * h_a_corr_sq + 1)
    Rc_power = min(numer / denom, P * 0.95)
    Rc = Rc_power * np.outer(h_flat, h_flat.conj()) / np.linalg.norm(h_flat)**2

    Rs_power = P - Rc_power
    if Rs_power <= 0:
        Rs = np.zeros((Mt, Mt), dtype=complex)
    else:
        Rs = Rs_power * np.outer(a_norm, a_norm.conj())

    # ========================================================================
    # SCA iteration
    # ========================================================================
    history = []
    status = "max_iter_reached"

    for k in range(max_iter):
        Rc_old = Rc.copy()
        Rs_old = Rs.copy()

        # ---- Step 1: Compute Taylor expansion coefficients  [Eq.(59)] ----
        x_k = (a_flat.T @ Rc @ a_flat.conj()).real               # a^H R_c a (current)
        Cx = C * x_k                                               # gamma_ran(current)
        if Cx > 1e-15:
            h_prime = C * (2 + Cx) / (1 + Cx)**2
        else:
            h_prime = C

        # ---- Step 2: Build and solve convex subproblem (P4,k)  [Eq.(60)] ----
        Rc_var = cp.Variable((Mt, Mt), complex=True)
        Rs_var = cp.Variable((Mt, Mt), complex=True)

        # Objective: maximize  tr(Rs @ M) + h_prime * tr(Rc @ M)
        obj = cp.real(cp.trace(Rs_var @ M_mat) + h_prime * cp.trace(Rc_var @ M_mat))

        # Normalized SINR constraint: h_tilde^H R_c h_tilde >= gamma_0 * (h_tilde^H R_s h_tilde + 1)
        constraints = [
            Rc_var >> 0,
            Rs_var >> 0,
            cp.real(cp.trace(Rc_var) + cp.trace(Rs_var)) <= P,
            cp.real(cp.trace(Rc_var @ H_mat)) >= gamma_0 * (cp.real(cp.trace(Rs_var @ H_mat)) + 1.0),
        ]

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=10000)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            if k == 0:
                obj_val = (a_flat.T @ Rs @ a_flat.conj()).real + h_prime * (a_flat.T @ Rc @ a_flat.conj()).real
                return Rc, Rs, {"status": "initial_only", "obj_val": obj_val,
                                "iters": 0, "gamma_ran": Cx}
            break

        Rc_new = Rc_var.value
        Rs_new = Rs_var.value

        # ---- Step 3: Check convergence ----
        Rc_change = np.linalg.norm(Rc_new - Rc_old) / (np.linalg.norm(Rc_old) + 1e-15)
        Rs_change = np.linalg.norm(Rs_new - Rs_old) / (np.linalg.norm(Rs_old) + 1e-15)
        obj_val = (a_flat.T @ Rs_new @ a_flat.conj()).real + h_prime * (a_flat.T @ Rc_new @ a_flat.conj()).real

        Rc, Rs = Rc_new, Rs_new
        history.append({"iter": k, "obj": obj_val, "Rc_change": Rc_change, "gamma_ran": Cx})

        if max(Rc_change, Rs_change) < tol:
            status = f"converged in {k+1} iters"
            break

    Rc = (Rc + Rc.conj().T) / 2
    Rs = (Rs + Rs.conj().T) / 2

    return Rc, Rs, {"status": status, "history": history, "iters": k+1}
