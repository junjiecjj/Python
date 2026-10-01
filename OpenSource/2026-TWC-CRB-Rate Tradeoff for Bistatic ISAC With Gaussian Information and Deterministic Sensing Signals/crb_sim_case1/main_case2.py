"""
CRB-Rate Tradeoff — Case 2: Superposition of Gaussian + Deterministic Signals
=============================================================================
基于: "CRB-Rate Tradeoff for Bistatic ISAC..." (Song, Yu, Xu, Ng, TWC 2026)

用法:
    python main_case2.py

Case 2 发送叠加信号 x(t) = s(t) + x0(t)：
  - s(t) ~ CN(0, Rc)：高斯信息信号（通信为主）
  - x0(t) = 确定性感知信号，协方差 Rs

优化问题 (P4)：非凸 → SCA 迭代 (Algorithm 1)
对比 Case 2 vs Case 1 的两条 tradeoff 曲线。
"""
import os, time, numpy as np
from steering_vectors import steering_vector, steering_vector_derivative
from channels       import generate_rician_channel, compute_alpha_sq
from crb_calc       import compute_crb_case2
from comm_rate      import compute_rate_case2
from case2_solver  import solve_p4_sca
from plot_results   import plot_comparison

# ========================================================================
# 全局参数（和 Case 1 完全一致）
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

h_seed = 46    # 和 Case 1 一致的低起点种子

# ========================================================================
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("CRB-Rate Tradeoff (Case 2: Gaussian + Deterministic signals)")
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

    print(f"\nSweeping {N_gamma} SINR thresholds (SCA, max 50 iter each)...")
    print(f"{'gamma_0(dB)':>10} {'Status':>20} {'CRB(rad^2)':>14} "
          f"{'Rate(bps/Hz)':>14} {'SINR(dB)':>8} {'SCA_iter':>8}")

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        t_start = time.time()
        Rc_opt, Rs_opt, info = solve_p4_sca(
            gamma_0, h, a, sigma2_c, sigma2_s, P, Mt, Mr, b, b_dot, alpha_sq)
        t_elapsed = time.time() - t_start

        if Rc_opt is None:
            results.append((gamma_0, None, None))
            print(f"{g0_dB:>10.2f} {info['status']:>20} {'---':>14} {'---':>14} {'---':>8} {'---':>8}")
            continue

        rate, sinr = compute_rate_case2(Rc_opt, Rs_opt, h, sigma2_c)
        crb = compute_crb_case2(theta_target, Rc_opt, Rs_opt, a, b, b_dot,
                                alpha_sq, sigma2_s, T)

        rate_f = float(rate.item()) if hasattr(rate, 'item') else float(rate)
        crb_f  = float(crb.item()) if hasattr(crb, 'item') else float(crb)
        sinr_f = float(sinr.item()) if hasattr(sinr, 'item') else float(sinr)

        results.append((gamma_0, crb_f, rate_f))

        sinr_dB = 10*np.log10(sinr_f) if sinr_f > 0 else -np.inf
        sca_iters = info.get('iters', '?')
        print(f"{g0_dB:>10.2f} {info['status']:>20} {crb_f:>14.3e} "
              f"{rate_f:>14.4f} {sinr_dB:>8.2f} {sca_iters:>8}")

    # --- 绘图 ---
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
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fig_path = os.path.join(out_dir, f'crb_rate_comparison_{timestamp}.png')

    # 读取 Case 1 数据做对比
    c1_path = os.path.join(out_dir, 'case1_data.npz')
    if os.path.exists(c1_path):
        c1 = np.load(c1_path)
        data1 = {'rate': c1['rate'], 'crb': c1['crb'], 'gamma': c1['gamma']}
    else:
        print("Warning: case1_data.npz not found, plotting Case 2 only.")
        data1 = None

    data2 = {'rate': rate_arr, 'crb': crb_arr, 'gamma': gamma_arr}

    # 保存 Case 2 数据供对比用
    np.savez(os.path.join(out_dir, 'case2_data.npz'), **data2)
    print("Data saved to:", os.path.join(out_dir, 'case2_data.npz'))

    plot_comparison(data1, data2, save_path=fig_path)
    print("Done.")


if __name__ == '__main__':
    main()
