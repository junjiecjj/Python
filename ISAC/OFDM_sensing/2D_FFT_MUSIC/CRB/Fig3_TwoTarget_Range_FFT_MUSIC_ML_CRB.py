#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双目标测距：16-PSK/16-QAM的FFT、MUSIC、联合ML及完整条件CRB。"""
import os
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.linalg import eigh
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, linear_sum_assignment
from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations
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
N = 128
M = 32
Ncp = N // 4
Q = 2
QAM_mod = 16
Iter = 1000
delta_f = 960e3
fc = 30e9
c = 3e8
target_pos = np.array([20.0, 30.0])
target_speed = np.array([200.0, -200.0])
alpha = np.array([1.0 + 0.0j, 0.8 * np.exp(0.3j)])
SNR_dB = np.arange(-10, 31, 5)        # 平均发送符号能量为1，SNR=1/sigma_W2
Ns = 16                             # MUSIC子阵的子载波数
Ms = 8                              # MUSIC子阵的OFDM符号数
GRID_FACTOR_TAU = 8                  # MUSIC搜索和ML初值网格的加密倍数
GRID_FACTOR_NU = 8
MUSIC_STARTS = 4
MUSIC_MAXITER = 100
ML_PEAKS = 4
ML_STARTS = 3
ML_MAXITER = 100

#%%>>>>>>>>>>>>>>>>>>>>>>>>  归一化参数、列向量化与固定双目标回波
dt = 1 / (N * delta_f)
Tsym = (N + Ncp) * dt
range_cell = c / (2 * N * delta_f)
velocity_cell = c / (2 * fc * M * Tsym)
tau_bin = target_pos / range_cell
nu_bin = target_speed / velocity_cell
truth = np.column_stack((tau_bin, nu_bin))
k_vec = np.tile(np.arange(N), M)
m_vec = np.repeat(np.arange(M), N)
if Q != 2 or len(target_pos) != Q or len(target_speed) != Q or len(alpha) != Q:
    raise ValueError("本程序用于双目标，距离、速度和增益数组均须包含两个元素。")
if not 0 < Ncp < N or not 2 <= Ns <= N or not 2 <= Ms <= M or Iter < 1:
    raise ValueError("检查CP、MUSIC子阵尺寸和蒙特卡洛次数。")
if np.any(tau_bin <= 0) or np.any(tau_bin >= Ncp) or np.any(np.abs(nu_bin) >= M / 2) or np.any(np.abs(alpha) == 0):
    raise ValueError("目标须在CP覆盖范围和无模糊速度区间内部，且增益非零。")
if Ns * Ms <= Q or (N - Ns + 1) * (M - Ms + 1) < Q:
    raise ValueError("MUSIC子阵维数须大于目标数，平滑快拍数须不少于目标数。")

def steering_matrix(tau, nu):
    """返回NM×Q导向矩阵；第mN+k行对应论文中的资源位置(k,m)。"""
    return np.exp(-2j * np.pi * k_vec[:, None] * np.asarray(tau)[None, :] / N + 2j * np.pi * m_vec[:, None] * np.asarray(nu)[None, :] / M)

A = steering_matrix(tau_bin, nu_bin)
H = (A @ alpha).reshape(N, M, order="F")
N_grid = GRID_FACTOR_TAU * N
M_grid = GRID_FACTOR_NU * M
tau_grid = np.arange(GRID_FACTOR_TAU * Ncp + 1) / GRID_FACTOR_TAU
nu_grid = np.fft.fftshift(np.fft.fftfreq(M_grid)) * M
fft_tau_grid = np.arange(Ncp + 1)
fft_nu_grid = np.fft.fftshift(np.fft.fftfreq(M)) * M
BOUNDS = [(0.0, float(Ncp)), (-M / 2, M / 2)]
ML_BOUNDS = [(0.0, float(Ncp))] * Q + [(-M / 2, M / 2)] * Q
P = Ns * Ms
sub_k = np.tile(np.arange(Ns), Ms)
sub_m = np.repeat(np.arange(Ms), Ns)
music_omega = np.vstack((2 * np.pi * sub_k / N, -2 * np.pi * sub_m / M))
music_b = np.exp(-2j * np.pi * np.arange(Ns)[:, None] * tau_grid[None, :] / N)
music_d = np.exp(2j * np.pi * np.arange(Ms)[:, None] * nu_grid[None, :] / M)
print(f"N={N}, M={M}, Q={Q}, Iter={Iter}；S[k,m]为N×M矩阵。")
print(f"归一化时延={tau_bin}；归一化多普勒={nu_bin}。")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  完整FIM求逆：返回两个目标的平均归一化CRB
def conditional_crb(S, sigma_W2):
    B = S.ravel(order="F")[:, None] * A
    D_tau = -2j * np.pi / N * k_vec[:, None] * B * alpha[None, :]
    D_nu = 2j * np.pi / M * m_vec[:, None] * B * alpha[None, :]
    G = np.column_stack((D_tau, D_nu, B, 1j * B))
    J = 2 / sigma_W2 * np.real(G.conj().T @ G)
    J = (J + J.T) / 2
    np.linalg.cholesky(J)
    C = np.linalg.solve(J, np.eye(4 * Q))
    return np.array([np.trace(C[:Q, :Q]) / Q, np.trace(C[Q:2 * Q, Q:2 * Q]) / Q])

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维局部峰选择：仅使用谱和预设保护间隔
def select_peaks(power, tau_axis, nu_axis, count, guard_tau, guard_nu):
    maxima = power == maximum_filter(power, size=3, mode=("nearest", "wrap"))
    indices = np.flatnonzero(maxima.ravel(order="F"))
    indices = indices[np.argsort(power.ravel(order="F")[indices])[::-1]]
    chosen = []
    for index in indices:
        row, col = np.unravel_index(index, power.shape, order="F")
        theta = np.array([tau_axis[row], nu_axis[col]])
        separated = True
        for old in chosen:
            delta_tau = theta[0] - old[0]
            delta_nu = (theta[1] - old[1] + M / 2) % M - M / 2
            if (delta_tau / guard_tau)**2 + (delta_nu / guard_nu)**2 < 1:
                separated = False
                break
        if separated:
            chosen.append(theta)
        if len(chosen) == count:
            break
    if len(chosen) < Q:
        raise RuntimeError("二维谱未能提供两个分离的候选峰。")
    return np.array(chosen)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  普通二维FFT：原始N×M网格，不补零、不插值
def estimate_fft(Y, S):
    Z = Y / S
    profile = np.fft.ifft(Z, axis=0)
    profile = np.fft.fftshift(np.fft.fft(profile, axis=1), axes=1)
    power = np.abs(profile[:len(fft_tau_grid), :])**2
    return select_peaks(power, fft_tau_grid, fft_nu_grid, Q, 0.75, 0.75)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC：滑动子阵、Lambda归一化和连续谱峰优化
def estimate_music(Y, S):
    Z = Y / S
    windows = sliding_window_view(Z, (Ns, Ms))
    snapshots = windows.reshape(-1, Ns, Ms).transpose(0, 2, 1).reshape(-1, P).T
    inv_power = sliding_window_view(1 / np.abs(S)**2, (Ns, Ms))
    noise_scale = np.sqrt(inv_power.mean(axis=(0, 1))).ravel(order="F")
    snapshots = snapshots / noise_scale[:, None]
    R = snapshots @ snapshots.conj().T / snapshots.shape[1]
    R = (R + R.conj().T) / 2
    _, Esignal = eigh(R, subset_by_index=(P - Q, P - 1), check_finite=False)
    coefficients = (Esignal.conj().T / noise_scale[None, :]).reshape(Q, Ms, Ns)
    projection = np.einsum("qmk,kr,mv->qrv", coefficients, music_b, music_d, optimize=True)
    norm2 = float(np.sum(1 / noise_scale**2))
    denominator = np.maximum(1 - np.sum(np.abs(projection)**2, axis=0) / norm2, 1e-12)
    guard_tau, guard_nu = 0.4 * N / Ns, 0.4 * M / Ms
    starts = select_peaks(1 / denominator, tau_grid, nu_grid, MUSIC_STARTS, guard_tau, guard_nu)

    def objective(theta):
        a = np.exp(-1j * (theta @ music_omega)) / noise_scale
        projected = Esignal.conj().T @ a
        derivative = Esignal.conj().T @ (-1j * music_omega.T * a[:, None])
        value = 1 - np.vdot(projected, projected).real / norm2
        gradient = -2 * np.real(projected.conj() @ derivative) / norm2
        return float(value), gradient

    candidates = []
    for start in starts:
        candidates.append((objective(start)[0], start.copy()))
        result = minimize(objective, start, method="L-BFGS-B", jac=True, bounds=BOUNDS, options={"maxiter": MUSIC_MAXITER, "ftol": 1e-14, "gtol": 1e-10, "maxls": 40})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
            candidates.append((float(result.fun), result.x.copy()))
    chosen = []
    for value, theta in sorted(candidates, key=lambda item: item[0]):
        if all(((theta[0] - old[0]) / guard_tau)**2 + (((theta[1] - old[1] + M / 2) % M - M / 2) / guard_nu)**2 >= 1 for old in chosen):
            chosen.append(theta)
        if len(chosen) == Q:
            return np.array(chosen)
    raise RuntimeError("MUSIC优化后未能提取两个不同目标。")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  独立ML模块：最小二乘消去两个复增益，联合优化四个参数
def ml_objective(theta, s, y, y_energy):
    """theta=[tau_1,tau_2,nu_1,nu_2]；返回归一化残差能量及解析梯度。"""
    B = s[:, None] * steering_matrix(theta[:Q], theta[Q:])
    gain, _, rank, _ = np.linalg.lstsq(B, y, rcond=None)
    if rank < Q:
        return 1e6, np.zeros(2 * Q)
    residual = y - B @ gain
    D_tau = -2j * np.pi / N * k_vec[:, None] * B * gain[None, :]
    D_nu = 2j * np.pi / M * m_vec[:, None] * B * gain[None, :]
    value = np.vdot(residual, residual).real / y_energy
    gradient = -2 * np.real(residual.conj() @ np.column_stack((D_tau, D_nu))) / y_energy
    return float(value), gradient

def estimate_ml(Y, S):
    """内部生成多组初值并进行四维联合优化，不使用真实目标参数或MUSIC输出。"""
    s = S.ravel(order="F")
    y = Y.ravel(order="F")
    y_energy = max(float(np.vdot(y, y).real), np.finfo(float).tiny)
    profile = np.fft.ifft(np.conj(S) * Y, n=N_grid, axis=0)
    profile = np.fft.fftshift(np.fft.fft(profile, n=M_grid, axis=1), axes=1)
    power = np.abs(profile[:len(tau_grid), :])**2
    peaks = select_peaks(power, tau_grid, nu_grid, ML_PEAKS, 0.75, 0.75)
    starts = list(combinations(range(len(peaks)), Q))[:ML_STARTS]
    best_value, best_theta = np.inf, None
    for pair in starts:
        initial = peaks[list(pair)]
        theta0 = np.r_[initial[:, 0], initial[:, 1]]
        value0, _ = ml_objective(theta0, s, y, y_energy)
        if value0 < best_value:
            best_value, best_theta = value0, theta0.copy()
        result = minimize(ml_objective, theta0, args=(s, y, y_energy), method="L-BFGS-B", jac=True, bounds=ML_BOUNDS, options={"maxiter": ML_MAXITER, "ftol": 1e-13, "gtol": 1e-9, "maxls": 40})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and result.fun < best_value:
            best_value, best_theta = float(result.fun), result.x.copy()
    if best_theta is None or best_value >= 1e6:
        raise RuntimeError("双目标ML未得到有效结果。")
    return np.column_stack((best_theta[:Q], best_theta[Q:]))

#%%>>>>>>>>>>>>>>>>>>>>>>>>  估计完成后匹配目标：返回两个目标的平均归一化平方误差
def target_mse(estimates):
    if np.shape(estimates) != (Q, 2) or not np.all(np.isfinite(estimates)):
        raise ValueError("估计结果须为有限的Q×2数组：[时延bin，多普勒bin]。")
    cost = np.sum((truth[:, None, :] - estimates[None, :, :])**2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    ordered = np.empty_like(estimates)
    ordered[rows] = estimates[cols]
    return np.mean((ordered - truth)**2, axis=0)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  蒙特卡洛：算法共用同一帧S和同一个Y
range_fft_mse = np.zeros((2, len(SNR_dB)))
range_music_mse = np.zeros((2, len(SNR_dB)))
range_ml_mse = np.full((2, len(SNR_dB)), np.nan)
range_crb_mean = np.zeros((2, len(SNR_dB)))
for mod_index, MOD_TYPE in enumerate(["psk", "qam"]):
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    _, demap_table = modem.getMappTable()
    Es = float(Es)
    for snr_index, snr_db in enumerate(SNR_dB):
        sigma_W2 = 10**(-snr_db / 10)
        fft_error_sum, music_error_sum, ml_error_sum, crb_sum = 0.0, 0.0, 0.0, 0.0
        ml_count = 0
        if MOD_TYPE == "psk":
            crb_value = conditional_crb(np.ones((N, M), dtype=complex), sigma_W2)[0]
        for trial in range(Iter):
            bits = np.random.randint(0, 2, N * M * bps)
            S = modem.modulate(bits).reshape(N, M, order="F") / np.sqrt(Es)
            W = np.sqrt(sigma_W2 / 2) * (np.random.randn(N * M) + 1j * np.random.randn(N * M)).reshape(N, M, order="F")
            Y = S * H + W
            fft_error_sum += target_mse(estimate_fft(Y, S))[0]
            music_error_sum += target_mse(estimate_music(Y, S))[0]
            if MOD_TYPE == "qam":
                crb_sum += conditional_crb(S, sigma_W2)[0]

            # ---------- ML调用开始：可将本段try/except整体注释，不影响FFT、MUSIC和CRB ----------
            try:
                ml_estimate = estimate_ml(Y, S)
                ml_error_sum += target_mse(ml_estimate)[0]
                ml_count += 1
            except Exception as error:
                if ml_count == trial:
                    print(f"ML本SNR首次失败：{error}", flush=True)
            # ---------- ML调用结束 ----------

            if (trial + 1) % 100 == 0:
                print(f"{QAM_mod}-{MOD_TYPE.upper()}, SNR={snr_db:g} dB：{trial + 1}/{Iter}", flush=True)
        range_fft_mse[mod_index, snr_index] = fft_error_sum / Iter
        range_music_mse[mod_index, snr_index] = music_error_sum / Iter
        range_crb_mean[mod_index, snr_index] = crb_value if MOD_TYPE == "psk" else crb_sum / Iter
        if ml_count == Iter:
            range_ml_mse[mod_index, snr_index] = ml_error_sum / Iter
        else:
            print(f"ML有效试验数={ml_count}/{Iter}，该SNR的ML曲线点留空。", flush=True)
        print(f"归一化MSE：FFT={range_fft_mse[mod_index, snr_index]:.6e}，MUSIC={range_music_mse[mod_index, snr_index]:.6e}，ML={range_ml_mse[mod_index, snr_index]:.6e}；平均CRB={range_crb_mean[mod_index, snr_index]:.6e}", flush=True)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  最后换算为距离RMSE与平均CRB的平方根
range_fft_rmse = range_cell * np.sqrt(range_fft_mse)
range_music_rmse = range_cell * np.sqrt(range_music_mse)
range_ml_rmse = range_cell * np.sqrt(range_ml_mse)
range_crb_root = range_cell * np.sqrt(range_crb_mean)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  画图：停用ML后自动仅画其余六条曲线，只保存PDF
output_dir = os.path.join(os.getcwd(), "fig_OFDM_MultiTarget_CRB")
os.makedirs(output_dir, exist_ok=True)
colors = ["#F65314", "#00A1F1"]
font1 = FontProperties(family="Times New Roman", style="normal", size=12)
fig, ax = plt.subplots(num=3, figsize=(6, 4), constrained_layout=True)
for i, MOD_TYPE in enumerate(["psk", "qam"]):
    name = f"{QAM_mod}-{MOD_TYPE.upper()}"
    # PSK实线作底，QAM虚线叠加；交错显示三角标记，避免FFT重合后相互遮挡。
    ax.semilogy(SNR_dB, range_fft_rmse[i], color=colors[i], linestyle="-" if i == 0 else (0, (5, 3)), linewidth=2.2 if i == 0 else 1.6, marker="^" if i == 0 else "v", markersize=7, markerfacecolor="white", markeredgewidth=1.4, markevery=(i, 2), zorder=3, label=name + ", FFT")
    ax.semilogy(SNR_dB, range_music_rmse[i], "s-.", color=colors[i], markerfacecolor="none", label=name + ", MUSIC", zorder=3)
    # ML只画标记：PSK在第0、2、4…个SNR点画圆，QAM在第1、3、5…个点画方块。
    if np.any(np.isfinite(range_ml_mse[i])):
        ax.semilogy(SNR_dB, range_ml_rmse[i], linestyle="None", marker="o" if i == 0 else "s", color=colors[i], markersize=8, markerfacecolor="white", markeredgewidth=1.6, markevery=(i, 2), zorder=5, label=name + ", ML")
    # 两条CRB位于ML标记下方；虚线间隙中仍能看见实线。
    ax.semilogy(SNR_dB, range_crb_root[i], color=colors[i], linestyle="-" if i == 0 else (0, (5, 3)), linewidth=2.6 if i == 0 else 1.7, zorder=2, label=name + ", CRB")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (m)")
ax.set_xlim(SNR_dB[0], SNR_dB[-1])
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, which="both")
ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.93), ncol=2, edgecolor="black", framealpha=1, prop=font1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.tick_params(direction="in", top=True, right=True, labelsize=16, width=1)
filename = f"Fig3_Range_FFT_MUSIC_ML_CRB_N{N}_M{M}_Q{Q}_{QAM_mod}PSK_{QAM_mod}QAM.pdf"
fig.savefig(os.path.join(output_dir, filename), format="pdf", bbox_inches="tight")
plt.show()
plt.close(fig)
