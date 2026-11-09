#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图1：单目标距离RMSE，16-PSK/16-QAM连续ML估计与平均条件CRB。"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter
from DigiCommPy.modem import PSKModem, QAMModem

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置：沿用原程序
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["legend.fontsize"] = 16

#%%>>>>>>>>>>>>>>>>>>>>>>>>  物理参数与蒙特卡洛参数
N = 4
M = 2
Ncp = N // 4
Q = 1
QAM_mod = 16
Iter = 10000
SYMBOL_SEED = 42
NOISE_SEED = 20261109
delta_f = 960e3
fc = 30e9
c = 3e8
dt = 1 / (N * delta_f)
Tcp = Ncp * dt
Tsym = (N + Ncp) * dt
SNR_dB = np.arange(-10, 31, 5)
target_pos = 20.0
target_speed = 40.0
alpha = 1.0 + 0.0j                    # 只用于生成回波和计算CRB，估计器不知道alpha
modem_dict = {"psk": PSKModem, "qam": QAMModem}

# 零填充只用于ML搜索初值；最终估计值由连续优化给出，不受该网格限制。
FFT_RANGE_FACTOR = 8
FFT_DOPPLER_FACTOR = 8
ML_STARTS = 3                        # 多个FFT局部峰初值，最终取ML准则最大的结果
ML_MAXITER = 100
PRINT_EVERY = 1000
SHOW_FIGURE = True
OUTPUT_STEM = "Fig1_SingleTarget_Range_ML_CRB"

#%%>>>>>>>>>>>>>>>>>>>>>>>>  参数映射与索引：保持X为(M,N)，按行展开
range_cell = c / (2 * N * delta_f)
velocity_cell = c / (2 * fc * M * Tsym)
range_max = c * Tcp / 2
tau_bin = target_pos / range_cell
nu_bin = target_speed / velocity_cell
k = np.tile(np.arange(N), M)
m = np.repeat(np.arange(M), N)
A = np.exp(-2j * np.pi * k * tau_bin / N + 2j * np.pi * m * nu_bin / M)
N_fft = FFT_RANGE_FACTOR * N
M_fft = FFT_DOPPLER_FACTOR * M
range_grid = np.arange(FFT_RANGE_FACTOR * Ncp + 1) / FFT_RANGE_FACTOR
doppler_grid = np.fft.fftshift(np.fft.fftfreq(M_fft)) * M
ML_BOUNDS = [(0.0, float(Ncp)), (-M / 2, M / 2)]
# 中心化索引只改变相关量的整体相位，不改变其模平方或ML估计。
omega = np.vstack((2 * np.pi * (k - (N - 1) / 2) / N,
                   -2 * np.pi * (m - (M - 1) / 2) / M))

def check_settings():
    if Q != 1 or N < 2 or M < 2 or not 0 < Ncp < N or Iter < 2:
        raise ValueError("本程序要求Q=1、N,M>=2、0<Ncp<N、Iter>=2。")
    if not 0 < tau_bin < Ncp or not -M / 2 < nu_bin < M / 2:
        raise ValueError("目标必须位于CP覆盖范围和无模糊速度区间内部。")
    if FFT_RANGE_FACTOR < 1 or FFT_DOPPLER_FACTOR < 1 or ML_STARTS < 1:
        raise ValueError("FFT加密因子和ML_STARTS必须为正整数。")
    if abs(alpha) == 0:
        raise ValueError("alpha不能为零。")
    print(f"N={N}, M={M}, Q={Q}, Iter={Iter}")
    print(f"目标：R={target_pos:.6f} m, v={target_speed:.6f} m/s")
    print(f"归一化参数：tau_bin={tau_bin:.8f}, nu_bin={nu_bin:.8f}")
    print(f"CP覆盖距离={range_max:.6f} m；距离cell={range_cell:.6f} m")
    print(f"速度cell={velocity_cell:.6f} m/s；未知有效增益alpha={alpha}")
    phase = 2 * np.pi * abs(2 * fc * target_speed / c) / delta_f
    print(f"符号内多普勒相位变化={phase:.6f} rad；数据直接按论文的子载波模型生成。")

def conditional_range_crb(X):
    """sigma_W^2=1时的完整4x4 FIM；增益实部、虚部同时作为未知参数。"""
    Bmat = X.reshape(-1) * A
    D_tau = -2j * np.pi / N * k * Bmat * alpha
    D_nu = 2j * np.pi / M * m * Bmat * alpha
    G = np.column_stack((D_tau, D_nu, Bmat, 1j * Bmat))
    J = 2 * np.real(G.conj().T @ G)
    J = (J + J.T) / 2
    np.linalg.cholesky(J)
    C = np.linalg.solve(J, np.eye(4))   # 不用pinv掩盖不可辨识或奇异问题
    return range_cell**2 * C[0, 0]

def ml_objective(theta, product, total_energy):
    """集中ML的负目标函数及解析梯度；product=conj(X)*Y=|X|^2*Z。"""
    z = product * np.exp(1j * (theta @ omega))
    correlation = np.sum(z)
    derivative = 1j * (omega @ z)
    scale = total_energy**2
    value = -abs(correlation)**2 / scale
    gradient = -2 * np.real(np.conj(correlation) * derivative) / scale
    return float(value), gradient

def estimate_ml(Y, X):
    """FFT网格上选取初值，再在完整可行域内连续优化；不使用真实目标参数。"""
    product_matrix = np.conj(X) * Y
    total_energy = float(np.sum(abs(X)**2))
    profile = np.fft.ifft(product_matrix, n=N_fft, axis=1)
    profile = np.fft.fftshift(np.fft.fft(profile, n=M_fft, axis=0), axes=0)
    power = abs(profile[:, :len(range_grid)])**2
    local_max = power == maximum_filter(power, size=3, mode=("wrap", "nearest"))
    candidates = np.flatnonzero(local_max)
    order = np.argsort(power.reshape(-1)[candidates])[::-1]
    candidates = candidates[order[:ML_STARTS]]
    product = product_matrix.reshape(-1)
    best_theta = None
    best_value = np.inf
    best_converged = False
    for index in candidates:
        row, col = np.unravel_index(index, power.shape)
        theta0 = np.array([range_grid[col], doppler_grid[row]])
        value0, _ = ml_objective(theta0, product, total_energy)
        if value0 < best_value:
            best_value, best_theta, best_converged = value0, theta0.copy(), False
        result = minimize(ml_objective, theta0, args=(product, total_energy),
                          method="L-BFGS-B", jac=True, bounds=ML_BOUNDS,
                          options={"maxiter": ML_MAXITER, "ftol": 1e-13,
                                   "gtol": 1e-9, "maxls": 40})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and result.fun <= best_value:
            best_value = float(result.fun)
            best_theta = result.x.copy()
            best_converged = bool(result.success)
    if best_theta is None:
        raise RuntimeError("ML初值或目标函数出现非有限值。")
    return float(best_theta[0]), best_converged

def validate_formulas(X):
    """核对完整FIM、单目标闭式CRB、解析梯度和无噪声连续搜索。"""
    p = abs(X.reshape(-1))**2
    k_bar, m_bar = np.dot(p, k) / p.sum(), np.dot(p, m) / p.sum()
    Skk = np.dot(p, (k - k_bar)**2)
    Smm = np.dot(p, (m - m_bar)**2)
    Skm = np.dot(p, (k - k_bar) * (m - m_bar))
    closed = range_cell**2 * N**2 * Smm / (
        8 * np.pi**2 * abs(alpha)**2 * (Skk * Smm - Skm**2))
    np.testing.assert_allclose(conditional_range_crb(X), closed, rtol=1e-10)
    Y = alpha * X * A.reshape(M, N)
    product = (np.conj(X) * Y).reshape(-1)
    theta = np.array([0.37 * Ncp, 0.13 * M])
    _, gradient = ml_objective(theta, product, p.sum())
    numerical = np.empty(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = 1e-6
        plus = ml_objective(theta + step, product, p.sum())[0]
        minus = ml_objective(theta - step, product, p.sum())[0]
        numerical[i] = (plus - minus) / (2e-6)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)
    estimate, _ = estimate_ml(Y, X)
    if abs(estimate - tau_bin) > 1e-5:
        raise RuntimeError("无噪声ML未恢复真实时延，请检查搜索设置。")

def main():
    check_settings()
    samples = {}
    rmse_curves = {}
    crb_curves = {}
    rmse_se = {}
    errors = {}
    converged = {}
    symbols_mean_energy = {}
    for MOD_TYPE in ["psk", "qam"]:
        modem = modem_dict[MOD_TYPE](QAM_mod)
        Es = np.mean(abs(modem.modulate(np.arange(QAM_mod)))**2)
        rng_symbols = np.random.default_rng(SYMBOL_SEED)
        d = rng_symbols.integers(0, QAM_mod, (Iter, M * N))
        X_frames = np.asarray(modem.modulate(d.reshape(-1))).reshape(Iter, M, N) / np.sqrt(Es)
        if np.any(abs(X_frames) == 0):
            raise ValueError("除法接收要求所有发送符号非零。")
        symbols_mean_energy[MOD_TYPE] = float(np.mean(abs(X_frames)**2))
        validate_formulas(X_frames[0])
        print(f"\n{QAM_mod}-{MOD_TYPE.upper()}: 原星座Es={Es:.8f}, "
              f"数据平均能量={symbols_mean_energy[MOD_TYPE]:.8f}；不做逐帧归一化。")
        nFrame = 1 if MOD_TYPE == "psk" else Iter
        samples[MOD_TYPE] = np.array([conditional_range_crb(X_frames[t]) for t in range(nFrame)])
        if MOD_TYPE == "psk":
            closed_cm = 3 * c**2 / (
                8 * np.pi**2 * abs(alpha)**2 * N * M * (N**2 - 1) * delta_f**2)
            np.testing.assert_allclose(samples[MOD_TYPE][0], closed_cm, rtol=1e-10)
        crb_curves[MOD_TYPE] = np.sqrt(samples[MOD_TYPE].mean() * 10**(-SNR_dB / 10))
        errors[MOD_TYPE] = np.empty((len(SNR_dB), Iter))
        converged[MOD_TYPE] = np.zeros(len(SNR_dB), dtype=int)
        rmse_curves[MOD_TYPE] = np.empty(len(SNR_dB))
        rmse_se[MOD_TYPE] = np.empty(len(SNR_dB))
        for snr_index, snr_db in enumerate(SNR_dB):
            sigma_W2 = 10**(-snr_db / 10)  # 归一化平均符号能量为1
            rng_noise = np.random.default_rng(NOISE_SEED + snr_index)
            for trial in range(Iter):
                X = X_frames[trial]
                W = np.sqrt(sigma_W2 / 2) * (
                    rng_noise.standard_normal((M, N)) + 1j * rng_noise.standard_normal((M, N)))
                # 两种星座共用去除符号相位后的高斯噪声，以降低性能差值的MC波动。
                # 单位相位旋转保持W|X ~ CN(0,sigma_W2*I)，不改变任何噪声方差。
                W *= X / abs(X)
                Y = alpha * X * A.reshape(M, N) + W
                estimate_tau, success = estimate_ml(Y, X)
                errors[MOD_TYPE][snr_index, trial] = range_cell * estimate_tau - target_pos
                converged[MOD_TYPE][snr_index] += int(success)
                if (trial + 1) % PRINT_EVERY == 0:
                    print(f"{MOD_TYPE.upper()}, SNR={snr_db:g} dB: {trial + 1}/{Iter}", flush=True)
            error = errors[MOD_TYPE][snr_index]
            mse = np.mean(error**2)
            rmse = np.sqrt(mse)
            se = np.std(error**2, ddof=1) / (2 * np.sqrt(Iter) * rmse)
            rmse_curves[MOD_TYPE][snr_index] = rmse
            rmse_se[MOD_TYPE][snr_index] = se
            ratio = mse / crb_curves[MOD_TYPE][snr_index]**2
            print(f"  RMSE={rmse:.8e} m, sqrt(mean CRB)={crb_curves[MOD_TYPE][snr_index]:.8e} m, "
                  f"MSE/CRB={ratio:.5f}, bias={error.mean():.3e} m, RMSE标准误差≈{se:.3e} m, "
                  f"未报告收敛={Iter-converged[MOD_TYPE][snr_index]}/{Iter}", flush=True)
        # 所有估计均计入RMSE，包括错误峰、边界解和未报告收敛的数值结果。
    qam_mean = samples["qam"].mean()
    psk_value = samples["psk"][0]
    gap = 100 * (qam_mean / psk_value - 1)
    gap_se = 100 * samples["qam"].std(ddof=1) / (np.sqrt(Iter) * psk_value)
    print(f"\n理论距离MSE下界差距：QAM相对PSK高{gap:.6f}%，MC标准误差={gap_se:.6f}个百分点。")
    print(f"理论距离RMSE下界差距：{100 * (np.sqrt(qam_mean / psk_value) - 1):.6f}%。")
    print("低SNR下存在阈值效应和偏差；有限次RMSE曲线不强制排序，也不删除异常估计。")

    #%%>>>>>>>>>>>>>>>>>>>>>>>>  保存原始误差、CRB样本和曲线数据
    np.savez_compressed(OUTPUT_STEM + ".npz", SNR_dB=SNR_dB, N=N, M=M, Ncp=Ncp,
                        Q=Q, Iter=Iter, QAM_mod=QAM_mod, delta_f=delta_f, fc=fc, c=c,
                        target_pos=target_pos, target_speed=target_speed, alpha=alpha,
                        SYMBOL_SEED=SYMBOL_SEED, NOISE_SEED=NOISE_SEED,
                        errors_psk=errors["psk"], errors_qam=errors["qam"],
                        crb_unit_psk=samples["psk"], crb_unit_qam=samples["qam"],
                        rmse_psk=rmse_curves["psk"], rmse_qam=rmse_curves["qam"],
                        sqrt_mean_crb_psk=crb_curves["psk"], sqrt_mean_crb_qam=crb_curves["qam"],
                        rmse_se_psk=rmse_se["psk"], rmse_se_qam=rmse_se["qam"],
                        converged_psk=converged["psk"], converged_qam=converged["qam"])
    table = np.column_stack((SNR_dB, rmse_curves["psk"], crb_curves["psk"],
                             rmse_curves["qam"], crb_curves["qam"],
                             rmse_se["psk"], rmse_se["qam"]))
    np.savetxt(OUTPUT_STEM + ".csv", table, delimiter=",", comments="",
               header="SNR_dB,PSK_ML_RMSE_m,PSK_sqrt_mean_CRB_m,QAM_ML_RMSE_m,"
                      "QAM_sqrt_mean_CRB_m,PSK_RMSE_MC_SE_m,QAM_RMSE_MC_SE_m")

    #%%>>>>>>>>>>>>>>>>>>>>>>>>  单图四条曲线：颜色、字体、边框沿用原程序
    colors = ["#F65314", "#00A1F1"]
    styles = ["o-", "s--"]
    font1 = FontProperties(family="Times New Roman", style="normal", size=16)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for i, MOD_TYPE in enumerate(["psk", "qam"]):
        name = f"{QAM_mod}-{MOD_TYPE.upper()}"
        ax.semilogy(SNR_dB, rmse_curves[MOD_TYPE], styles[i], color=colors[i],
                    markerfacecolor="none", label=name + ", ML")
        ax.semilogy(SNR_dB, crb_curves[MOD_TYPE], linestyle=":" if i == 0 else "-.",
                    color=colors[i], label=name + ", CRB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Range RMSE (m)")
    ax.set_xlim(SNR_dB[0], SNR_dB[-1])
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, which="both")
    ax.legend(edgecolor="black", framealpha=1, prop=font1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.tick_params(direction="in", top=True, right=True, labelsize=16, width=1)
    fig.savefig(OUTPUT_STEM + ".pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_STEM + ".png", dpi=600, bbox_inches="tight")
    print(f"Saved {OUTPUT_STEM}.pdf/.png/.csv/.npz")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)
    return table

if __name__ == "__main__":
    main()
