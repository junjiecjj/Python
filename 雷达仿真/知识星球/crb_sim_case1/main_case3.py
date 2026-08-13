"""
CRB-Rate Tradeoff — Case 3: ISAC with Given Realizations (Deterministic CRB)
============================================================================
基于: "CRB-Rate Tradeoff for Bistatic ISAC..." (Song, Yu, Xu, Ng, TWC 2026)

此场景对应论文中的 benchmark: "ISAC with Given Realizations of
Information Signal [22]"。发射端发送高斯信息信号，但感知接收端
已知信号的具体实现（充分长的感知时间/收发端信息交换），
因而 CRB 适用确定性信号公式 Eq.(23)，没有高斯信号的 (1+1/gamma_ran) 惩罚项。

优化问题同 Case 1 (P2)，用 Proposition 1 闭式解。

用法:
    python main_case3.py
"""
import os, time, numpy as np
from steering_vectors import steering_vector, steering_vector_derivative
from channels       import generate_rician_channel, compute_alpha_sq
from crb_calc       import compute_crb_deterministic
from comm_rate      import compute_rate_case1 as compute_rate
from beamforming_opt import solve_p2_optimal
from plot_results   import plot_comparison

# ========================================================================
# 全局参数（和 Case 1 & 2 完全一致）
# ========================================================================
Mt = 32
Mr = 32
T  = 1024

P_dBm = 30
P     = 10**((P_dBm - 30) / 10)
sigma2_c_dBm = -80
sigma2_s_dBm = -80
sigma2_c = 10**((sigma2_c_dBm - 30) / 10)
sigma2_s = 10**((sigma2_s_dBm - 30) / 10)

theta_target = 0.0
phi_target   = 0.0

d_bt = 200.0
d_tr = 200.0
d_bc = 1000.0

K0      = -30
alpha0  = 2.5
d0      = 1.0
CAL_ALPHA = 1.0e-32

Kc     = 1.0
phi_cu = 0.3

N_gamma        = 40
gamma_0_dB_min = -10.0
gamma_0_dB_max =  19.0

h_seed = 46    # 和 Case 1 & 2 一致

# ========================================================================
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("CRB-Rate Tradeoff (Case 3: Given realizations — Deterministic CRB)")
    print("=" * 60)

    # --- 信道生成 ---
    h = generate_rician_channel(Mt, phi_cu, Kc, d_bc,
                                K0, alpha0, d0, h_seed)
    a = steering_vector(Mt, phi_target)
    b = steering_vector(Mr, theta_target)
    b_dot = steering_vector_derivative(Mr, theta_target)
    alpha_sq = compute_alpha_sq(d_bt, d_tr, 1.0,
                                K0, alpha0, d0, CAL_ALPHA)

    print(f"\nMt={Mt}, Mr={Mr}, T={T}, P={P_dBm} dBm")
    print(f"theta_target={theta_target:.1f} rad, phi_cu={phi_cu:.2f} rad")
    print(f"|alpha|^2 = {alpha_sq:.3e}")
    print(f"h_seed = {h_seed}")

    # --- 扫描 gamma_0 ---
    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, N_gamma)
    results = []

    print(f"\nSweeping {N_gamma} SINR thresholds...")
    print(f"{'gamma_0(dB)':>10} {'Status':>14} {'CRB(rad^2)':>14} "
          f"{'Rate(bps/Hz)':>14} {'SINR(dB)':>8}")

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        # 优化问题同 Case 1 (P2), Proposition 1 闭式解
        R_opt, status = solve_p2_optimal(gamma_0, h, a, sigma2_c, P, Mt)

        if R_opt is None:
            results.append((gamma_0, None, None))
            print(f"{g0_dB:>10.2f} {status:>14} {'---':>14} {'---':>14} {'---':>8}")
            continue

        # 确定性信号 CRB (Eq.23) — 无高斯惩罚项
        rate, sinr = compute_rate(R_opt, h, sigma2_c)
        crb = compute_crb_deterministic(theta_target, R_opt, a, b, b_dot,
                                        alpha_sq, sigma2_s, T)

        rate_f = float(rate.item()) if hasattr(rate, 'item') else float(rate)
        crb_f  = float(crb.item()) if hasattr(crb, 'item') else float(crb)
        sinr_f = float(sinr.item()) if hasattr(sinr, 'item') else float(sinr)

        results.append((gamma_0, crb_f, rate_f))

        sinr_dB = 10*np.log10(sinr_f) if sinr_f > 0 else -np.inf
        print(f"{g0_dB:>10.2f} {status:>14} {crb_f:>14.3e} "
              f"{rate_f:>14.4f} {sinr_dB:>8.2f}")

    # --- 保存数据 ---
    valid = [(g, c, r) for g, c, r in results if c is not None]
    if len(valid) < 3:
        print("\nERROR: Too few feasible points.")
        return

    gamma_arr, crb_arr, rate_arr = zip(*valid)
    gamma_arr = np.array(gamma_arr)
    crb_arr   = np.array(crb_arr)
    rate_arr  = np.array(rate_arr)

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    # 保存 Case 3 数据
    data_path = os.path.join(out_dir, 'case3_data.npz')
    np.savez(data_path, gamma=gamma_arr, crb=crb_arr, rate=rate_arr)
    print(f"\nData saved to: {data_path}")

    # --- 画三条曲线对比 ---
    # Case 1
    c1_path = os.path.join(out_dir, 'case1_data.npz')
    if os.path.exists(c1_path):
        c1 = np.load(c1_path)
        data1 = {'rate': c1['rate'], 'crb': c1['crb'], 'gamma': c1['gamma']}
    else:
        print("Warning: case1_data.npz not found.")
        data1 = None

    # Case 2
    c2_path = os.path.join(out_dir, 'case2_data.npz')
    if os.path.exists(c2_path):
        c2 = np.load(c2_path)
        data2 = {'rate': c2['rate'], 'crb': c2['crb'], 'gamma': c2['gamma']}
    else:
        print("Warning: case2_data.npz not found.")
        data2 = None

    # Case 3
    data3 = {'rate': rate_arr, 'crb': crb_arr, 'gamma': gamma_arr}

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(out_dir, f'crb_rate_comparison_{timestamp}.png')
    plot_comparison(data1, data2, data3, save_path=fig_path)
    print("Done.")


if __name__ == '__main__':
    main()
