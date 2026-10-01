"""
CRB-Rate Tradeoff — Case 1: Gaussian Signals Only  (主脚本)
===========================================================
基于: "CRB-Rate Tradeoff for Bistatic ISAC..." (Song, Yu, Xu, Ng, TWC 2026)

用法:
    python main.py

全局参数在此定义。各功能模块在独立的 .py 文件中，
可像 MATLAB .m 函数一样单独 import 调用。
"""
import os, time, numpy as np
from steering_vectors import steering_vector, steering_vector_derivative
from channels       import generate_rician_channel, compute_alpha_sq
from crb_calc       import compute_crb_case1 as compute_crb
from comm_rate      import compute_rate_case1 as compute_rate
from beamforming_opt import solve_p2_optimal
from plot_results   import plot_tradeoff

# ========================================================================
# 全局参数
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

# ========================================================================
# 主程序
# ========================================================================
def main():
    print("=" * 60)
    print("CRB-Rate Tradeoff Simulation (Case 1: Gaussian signals only)")
    print("=" * 60)

    h_seed = 46
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

    gamma_0_dB_vals = np.linspace(gamma_0_dB_min, gamma_0_dB_max, N_gamma)
    results = []

    print(f"\nSweeping {N_gamma} SINR thresholds...")
    print(f"{'gamma_0(dB)':>10} {'Status':>14} {'CRB(rad^2)':>14} "
          f"{'Rate(bps/Hz)':>14} {'SINR(dB)':>8}")

    for g0_dB in gamma_0_dB_vals:
        gamma_0 = 10**(g0_dB / 10)

        Rc_opt, status = solve_p2_optimal(gamma_0, h, a, sigma2_c, P, Mt)

        if Rc_opt is None:
            results.append((gamma_0, None, None))
            print(f"{g0_dB:>10.2f} {status:>14} {'---':>14} {'---':>14} {'---':>8}")
            continue

        rate_np, sinr_np = compute_rate(Rc_opt, h, sigma2_c)
        crb_np = compute_crb(theta_target, Rc_opt, a, b, b_dot,
                             alpha_sq, sigma2_s, T)
        rate = float(rate_np.item())
        crb  = float(crb_np.item())
        sinr = float(sinr_np.item()) if hasattr(sinr_np, 'item') else float(sinr_np)

        results.append((gamma_0, crb, rate))

        sinr_dB = 10*np.log10(sinr) if sinr > 0 else -np.inf
        print(f"{g0_dB:>10.2f} {status:>14} {crb:>14.3e} "
              f"{rate:>14.4f} {sinr_dB:>8.2f}")

    # —— 画图 ——
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
    fig_path = os.path.join(out_dir, f'crb_rate_case1_{timestamp}.png')

    plot_tradeoff(gamma_arr, crb_arr, rate_arr, fig_path)

    # 保存数据供 Case 2 对比用
    data_path = os.path.join(out_dir, 'case1_data.npz')
    np.savez(data_path, gamma=gamma_arr, crb=crb_arr, rate=rate_arr)
    print(f"Data saved to: {data_path}")
    print("Done.")


if __name__ == '__main__':
    main()
