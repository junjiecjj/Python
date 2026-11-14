#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单目标测距：16-PSK/16-QAM二维FFT、连续ML的RMSE与平均条件CRB的平方根。"""
import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from Modulations import modulator

np.random.seed(42)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置
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
N = 16
M = 4
Ncp = N // 4
QAM_mod = 16
Iter = 10000
delta_f = 960e3
fc = 30e9
c = 3e8
target_pos = 20.0
target_speed = 40.0
alpha = 1.0 + 0.0j                   # 固定未知增益；仅用于生成回波和计算CRB
SNR_dB = np.arange(-10, 31, 5)        # 平均符号能量归一化为1，SNR=1/sigma_W2
FFT_RANGE_FACTOR = 8                  # 仅用于ML初值的加密FFT
FFT_DOPPLER_FACTOR = 8                # 仅用于ML初值的加密FFT
ML_STARTS = 3
ML_MAXITER = 100

#%%>>>>>>>>>>>>>>>>>>>>>>>>  归一化坐标与固定目标响应
dt = 1 / (N * delta_f)
Tcp = Ncp * dt
Tsym = (N + Ncp) * dt
range_cell = c / (2 * N * delta_f)
velocity_cell = c / (2 * fc * M * Tsym)
tau_bin = target_pos / range_cell
nu_bin = target_speed / velocity_cell
if N < 2 or M < 2 or not 0 < Ncp < N or Iter < 1 or abs(alpha) == 0:
    raise ValueError("要求N,M>=2、0<Ncp<N、Iter>=1且alpha非零。")
if not 0 < tau_bin < Ncp or not -M / 2 < nu_bin < M / 2:
    raise ValueError("目标应位于CP覆盖范围和多普勒无模糊区间内部。")

# S、Y、H、W均为(N,M)：S[k,m]的行对应子载波，列对应OFDM符号。
k = np.arange(N)[:, None]
m = np.arange(M)[None, :]
H = alpha * np.exp(-2j * np.pi * k * tau_bin / N + 2j * np.pi * m * nu_bin / M)
N_fft = FFT_RANGE_FACTOR * N
M_fft = FFT_DOPPLER_FACTOR * M
tau_grid = np.arange(FFT_RANGE_FACTOR * Ncp + 1) / FFT_RANGE_FACTOR
nu_grid = np.fft.fftshift(np.fft.fftfreq(M_fft)) * M
ML_BOUNDS = [(0.0, float(Ncp)), (-M / 2, M / 2)]
fft_tau_grid = np.arange(Ncp + 1)
fft_nu_grid = np.fft.fftshift(np.fft.fftfreq(M)) * M

# 列向量化：第mN+k个元素对应S[k,m]，与论文中的vec(S)一致。
k_vec = np.tile(np.arange(N), M)
m_vec = np.repeat(np.arange(M), N)
omega = np.vstack((2 * np.pi * k_vec / N, -2 * np.pi * m_vec / M))
print(f"N={N}, M={M}, Iter={Iter}；目标R={target_pos:g} m，v={target_speed:g} m/s。")
print(f"归一化目标：tau_bin={tau_bin:.8f}，nu_bin={nu_bin:.8f}。")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  单目标归一化参数的解析CRB
def single_target_crb(S, sigma_W2, alpha, MOD_TYPE):
    """返回时延bin和多普勒bin的条件CRB；复增益作为未知参数处理。"""
    rho = abs(alpha)**2 / sigma_W2
    if MOD_TYPE == "psk":
        crb_tau = 3 * N / (2 * np.pi**2 * rho * M * (N**2 - 1))
        crb_nu = 3 * M / (2 * np.pi**2 * rho * N * (M**2 - 1))
    else:
        p = np.abs(S)**2
        energy = np.sum(p)
        k_bar = np.sum(p * k) / energy
        m_bar = np.sum(p * m) / energy
        Skk = np.sum(p * (k - k_bar)**2)
        Smm = np.sum(p * (m - m_bar)**2)
        Skm = np.sum(p * (k - k_bar) * (m - m_bar))
        determinant = Skk * Smm - Skm**2
        if determinant <= 0:
            raise ValueError("当前发送帧的单目标等效FIM不可逆。")
        crb_tau = N**2 * Smm / (8 * np.pi**2 * rho * determinant)
        crb_nu = M**2 * Skk / (8 * np.pi**2 * rho * determinant)
    return float(crb_tau), float(crb_nu)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  普通二维FFT估计：不补零、不插值
def estimate_fft(Y, S):
    """先除去发送符号，再在原始N×M网格上联合选择距离、多普勒峰值。"""
    divider_matrix = Y / S
    profile = np.fft.ifft(divider_matrix, axis=0)
    profile = np.fft.fftshift(np.fft.fft(profile, axis=1), axes=1)
    power = np.abs(profile[:len(fft_tau_grid), :])**2
    row, col = np.unravel_index(np.argmax(power), power.shape)
    return float(fft_tau_grid[row]), float(fft_nu_grid[col])

#%%>>>>>>>>>>>>>>>>>>>>>>>>  集中ML目标函数与连续参数估计
def ml_objective(theta, product, total_energy):
    """消去未知复增益后的负ML准则及解析梯度；theta=[tau_bin,nu_bin]。"""
    z = product * np.exp(1j * (theta @ omega))
    correlation = np.sum(z)
    derivative = 1j * (omega @ z)
    value = -abs(correlation)**2 / total_energy
    gradient = -2 * np.real(np.conj(correlation) * derivative) / total_energy
    return float(value), gradient

def estimate_ml(Y, S):
    """加密FFT提供初值，连续优化返回时延bin、多普勒bin；不使用目标真值。"""
    product_matrix = np.conj(S) * Y
    total_energy = float(np.sum(np.abs(S)**2))
    profile = np.fft.ifft(product_matrix, n=N_fft, axis=0)
    profile = np.fft.fftshift(np.fft.fft(profile, n=M_fft, axis=1), axes=1)
    power = np.abs(profile[:len(tau_grid), :])**2
    local_max = power == maximum_filter(power, size=3, mode=("nearest", "wrap"))
    candidates = np.flatnonzero(local_max.ravel(order="F"))
    order = np.argsort(power.ravel(order="F")[candidates])[::-1]
    candidates = candidates[order[:ML_STARTS]]
    product = product_matrix.ravel(order="F")
    best_theta = None
    best_value = np.inf
    for index in candidates:
        row, col = np.unravel_index(index, power.shape, order="F")
        theta0 = np.array([tau_grid[row], nu_grid[col]])
        value0, _ = ml_objective(theta0, product, total_energy)
        if value0 < best_value:
            best_value, best_theta = value0, theta0.copy()
        result = minimize(ml_objective, theta0, args=(product, total_energy), method="L-BFGS-B", jac=True, bounds=ML_BOUNDS, options={"maxiter": ML_MAXITER, "ftol": 1e-13, "gtol": 1e-9, "maxls": 40})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and result.fun < best_value:
            best_value, best_theta = float(result.fun), result.x.copy()
    if best_theta is None:
        raise RuntimeError("ML搜索未得到有限的估计结果。")
    return float(best_theta[0]), float(best_theta[1])

#%%>>>>>>>>>>>>>>>>>>>>>>>>  蒙特卡洛：内部统计归一化时延MSE和CRB
fft_mse = np.zeros((2, len(SNR_dB)))
ml_mse = np.zeros((2, len(SNR_dB)))
crb_mean = np.zeros((2, len(SNR_dB)))
for mod_index, MOD_TYPE in enumerate(["psk", "qam"]):
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    _, demap_table = modem.getMappTable()
    Es = float(Es)
    for snr_index, snr_db in enumerate(SNR_dB):
        sigma_W2 = 10**(-snr_db / 10)
        fft_squared_error_sum = 0.0
        squared_error_sum = 0.0
        crb_sum = 0.0
        if MOD_TYPE == "psk":
            crb_tau, _ = single_target_crb(None, sigma_W2, alpha, MOD_TYPE)
        for trial in range(Iter):
            bits = np.random.randint(0, 2, M * N * bps)
            S = modem.modulate(bits).reshape(N, M, order="F") / np.sqrt(Es)
            W = np.sqrt(sigma_W2 / 2) * (np.random.randn(N * M) + 1j * np.random.randn(N * M)).reshape(N, M, order="F")
            Y = S * H + W
            tau_fft_hat, nu_fft_hat = estimate_fft(Y, S)
            tau_hat, nu_hat = estimate_ml(Y, S)
            fft_squared_error_sum += (tau_fft_hat - tau_bin)**2
            squared_error_sum += (tau_hat - tau_bin)**2
            if MOD_TYPE == "qam":
                crb_tau, _ = single_target_crb(S, sigma_W2, alpha, MOD_TYPE)
                crb_sum += crb_tau
            if (trial + 1) % 1000 == 0:
                print(f"{QAM_mod}-{MOD_TYPE.upper()}, SNR={snr_db:g} dB：{trial + 1}/{Iter}", flush=True)
        fft_mse[mod_index, snr_index] = fft_squared_error_sum / Iter
        ml_mse[mod_index, snr_index] = squared_error_sum / Iter
        crb_mean[mod_index, snr_index] = crb_tau if MOD_TYPE == "psk" else crb_sum / Iter
        print(f"  归一化时延：FFT MSE={fft_mse[mod_index, snr_index]:.6e}，"
              f"ML MSE={ml_mse[mod_index, snr_index]:.6e}，"
              f"平均CRB={crb_mean[mod_index, snr_index]:.6e}", flush=True)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  最后换算为物理单位：距离RMSE与CRB平方根
range_fft_rmse = range_cell * np.sqrt(fft_mse)
range_rmse = range_cell * np.sqrt(ml_mse)
range_crb = range_cell * np.sqrt(crb_mean)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  画图并保存PDF
output_dir = os.path.join(os.getcwd(), "fig_OFDM_ML_CRB")
os.makedirs(output_dir, exist_ok=True)
colors = ["#F65314", "#00A1F1"]
styles = ["o", "s"]
fft_styles = ["^--", "D:"]

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for i, MOD_TYPE in enumerate(["psk", "qam"]):
    name = f"{QAM_mod}-{MOD_TYPE.upper()}"
    ax.semilogy(SNR_dB, range_fft_rmse[i], fft_styles[i], color=colors[i], markerfacecolor="none", label=name + ", FFT")
    ax.semilogy(SNR_dB, range_rmse[i], styles[i], linestyle='none', color=colors[i], mfc="none",mew = 1.5, label=name + ", ML")
    ax.semilogy(SNR_dB, range_crb[i], linestyle="--" if i == 0 else ":", color=colors[i], label=name + ", CRB")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (m)")
ax.set_xlim(SNR_dB[0], SNR_dB[-1])
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, which="both")
font1 = FontProperties(family="Times New Roman", style="normal", size=12)
ax.legend(edgecolor="black", framealpha=1, prop=font1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.tick_params(direction="in", top=True, right=True, labelsize=16, width=1)
filename = f"Fig1_Range_FFT_ML_CRB_N{N}_M{M}_{QAM_mod}PSK_{QAM_mod}QAM.pdf"
fig.savefig(os.path.join(output_dir, filename), format="pdf", bbox_inches="tight")
plt.show()
plt.close(fig)
