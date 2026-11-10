#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:09:06 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.3：双目标测距RMSE，16-PSK/16-QAM的二维FFT、二维MUSIC及完整条件CRB。"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, linear_sum_assignment
from numpy.lib.stride_tricks import sliding_window_view
try:
    from DigiCommPy.modem import PSKModem, QAMModem
except ImportError:
    PSKModem = QAMModem = None

#%%>>>>>>>>>>>>>>>>>>>>>>>>  绘图设置：沿用前两幅图
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14,
                     "axes.labelsize": 18, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "axes.unicode_minus": False,
                     "lines.linewidth": 2, "lines.markersize": 6,
                     "figure.facecolor": "white", "legend.fontsize": 12})

#%%>>>>>>>>>>>>>>>>>>>>>>>>  物理参数、目标与蒙特卡洛参数
N = 256
M = 16
Ncp = N // 4
QAM_mod = 16
Iter = 1000
SYMBOL_SEED = 42
NOISE_SEED = 20261109
delta_f = 960e3
fc = 30e9
c = 3e8
dt = 1 / (N * delta_f)
Tcp = Ncp * dt
Tsym = (N + Ncp) * dt
SNR_dB = np.arange(-10, 31, 5)
target_pos = np.array([20.0, 30.0])
target_speed = np.array([200.0, -200.0])
alpha = np.array([1.0 + 0.0j, 0.8 * np.exp(0.3j)])
Q = len(target_pos)

# 数值搜索设置；N、M决定物理分辨率，零填充仅加密搜索网格。
FFT_RANGE_FACTOR = 8
FFT_DOPPLER_FACTOR = 8
Pn = 16                             # MUSIC子阵：子载波方向长度
Pm = 8                              # MUSIC子阵：慢时间方向长度
MUSIC_STARTS = 4 * Q
MUSIC_MAXITER = 150
PRINT_EVERY = 100
SHOW_FIGURE = True
OUTPUT_STEM = "Fig3_MultiTarget_Range_FFT_MUSIC_CRB"

#%%>>>>>>>>>>>>>>>>>>>>>>>>  X与Y均为(M,N)，按行展开
range_cell = c / (2 * N * delta_f)
velocity_cell = c / (2 * fc * M * Tsym)
range_max = c * Tcp / 2
tau_bin = target_pos / range_cell
nu_bin = target_speed / velocity_cell
k = np.tile(np.arange(N), M)
m = np.repeat(np.arange(M), N)
A = np.exp(-2j * np.pi * k[:, None] * tau_bin / N
           + 2j * np.pi * m[:, None] * nu_bin / M)
echo_channel = (A @ alpha).reshape(M, N)
N_fft = FFT_RANGE_FACTOR * N
M_fft = FFT_DOPPLER_FACTOR * M
range_grid = np.arange(FFT_RANGE_FACTOR * Ncp + 1) / FFT_RANGE_FACTOR
doppler_grid = np.fft.fftshift(np.fft.fftfreq(M_fft)) * M
BOUNDS = [(0.0, float(Ncp)), (-M / 2, M / 2)]
sub_k = np.tile(np.arange(Pn), Pm)
sub_m = np.repeat(np.arange(Pm), Pn)
omega = np.vstack((2 * np.pi * sub_k / N, -2 * np.pi * sub_m / M))

def check_settings():
    if Iter < 2 or Q < 1 or len(target_speed) != Q or len(alpha) != Q:
        raise ValueError("要求Iter>=2，且距离、速度和复增益数组长度一致。")
    if not 0 < Ncp < N or not 1 < Pn <= N or not 1 < Pm <= M:
        raise ValueError("请检查CP和MUSIC子阵长度。")
    if min(Pn * Pm, (N - Pn + 1) * (M - Pm + 1)) <= Q:
        raise ValueError("MUSIC子阵维数和空间平滑快拍数必须大于目标数。")
    if np.any(tau_bin <= 0) or np.any(tau_bin >= Ncp):
        raise ValueError("所有目标必须在CP覆盖距离内部。")
    if np.any(abs(nu_bin) >= M / 2) or np.any(abs(alpha) == 0):
        raise ValueError("目标须在无模糊速度区间内，且复增益非零。")
    print(f"N={N}, M={M}, Q={Q}, Iter={Iter}, CP覆盖距离={range_max:.4f} m")
    print(f"距离cell={range_cell:.6f} m，速度cell={velocity_cell:.6f} m/s")
    print(f"FFT网格间隔={range_cell/FFT_RANGE_FACTOR:.6f} m，"
          f"{velocity_cell/FFT_DOPPLER_FACTOR:.6f} m/s")
    print(f"MUSIC子阵={Pm}x{Pn}，平滑快拍数={(M-Pm+1)*(N-Pn+1)}")
    print("采用忽略符号内ICI的理想频域模型；SNR=1/sigma_W^2，不按回波总功率重定义。")

def constellation(MOD_TYPE):
    """按星座平均能量归一化，不对随机OFDM帧逐帧归一化。"""
    if PSKModem is not None:
        modem = {"psk": PSKModem, "qam": QAMModem}[MOD_TYPE](QAM_mod)
        points = np.asarray(modem.modulate(np.arange(QAM_mod)), dtype=complex)
    elif MOD_TYPE == "psk":
        points = np.exp(2j * np.pi * np.arange(QAM_mod) / QAM_mod)
    elif MOD_TYPE == "qam":
        side = int(np.sqrt(QAM_mod))
        if side**2 != QAM_mod:
            raise ValueError("内置QAM映射要求调制阶数为完全平方数。")
        levels = 2 * np.arange(side) - side + 1
        points = (levels[:, None] + 1j * levels[None, :]).reshape(-1)
    else:
        raise ValueError("MOD_TYPE只能是psk或qam。")
    points = points / np.sqrt(np.mean(abs(points)**2))
    if np.any(abs(points) == 0):
        raise ValueError("Y/X接收要求星座点非零。")
    return points

def draw_frame(rng, points):
    return points[rng.integers(0, len(points), (M, N))]

def conditional_range_crb(X):
    """噪声方差为1时，联合估计全部时延、多普勒和未知复增益的4Q维FIM。"""
    B = X.reshape(-1, 1) * A
    # 中心化导数仅在未知复增益参数中引入重参数化，不改变时延/多普勒CRB。
    D_tau = -2j * np.pi / N * (k[:, None] - (N - 1) / 2) * B * alpha
    D_nu = 2j * np.pi / M * (m[:, None] - (M - 1) / 2) * B * alpha
    G = np.column_stack((D_tau, D_nu, B, 1j * B))
    J = 2 * np.real(G.conj().T @ G)
    J = (J + J.T) / 2
    np.linalg.cholesky(J)
    C = np.linalg.solve(J, np.eye(4 * Q))
    return range_cell**2 * np.diag(C)[:Q]

def spectrum_transform(Z):
    response = np.fft.ifft(Z, n=N_fft, axis=1) * N_fft
    return np.fft.fftshift(np.fft.fft(response, n=M_fft, axis=0), axes=0)

def peak_indices(power, count, guard_range, guard_doppler):
    """在整个可行域找峰；保护间隔不使用真实目标参数。"""
    maxima = power == maximum_filter(power, size=3, mode=("wrap", "nearest"))
    indices = np.flatnonzero(maxima)
    indices = indices[np.argsort(power.ravel()[indices])[::-1]]
    chosen = []
    for index in indices:
        row, col = np.unravel_index(index, power.shape)
        theta = np.array([range_grid[col], doppler_grid[row]])
        separated = True
        for old in chosen:
            delta = theta - old
            delta[1] = (delta[1] + M / 2) % M - M / 2
            if (delta[0] / guard_range)**2 + (delta[1] / guard_doppler)**2 < 1:
                separated = False
                break
        if separated:
            chosen.append(theta)
        if len(chosen) == count:
            return np.array(chosen)
    raise RuntimeError("可分离局部峰不足，请检查目标数、搜索网格或保护间隔。")

def estimate_fft(Z):
    power = abs(spectrum_transform(Z)[:, :len(range_grid)])**2
    theta = peak_indices(power, Q, 1.0, 1.0)
    return theta * np.array([range_cell, velocity_cell])

def signal_subspace(Z):
    """单帧二维滑窗平滑；每列为一个Pm x Pn子阵快拍。"""
    windows = sliding_window_view(Z, (Pm, Pn))
    H = np.ascontiguousarray(windows.reshape(-1, Pm * Pn).T)
    R = H @ H.conj().T / H.shape[1]
    R = (R + R.conj().T) / 2
    _, U = eigh(R, subset_by_index=(Pm * Pn - Q, Pm * Pn - 1), check_finite=False)
    return U                         # eigh升序返回；所选为最大的Q个特征值对应向量

def music_grid_cost(U):
    projection = np.zeros((M_fft, len(range_grid)))
    for q in range(Q):
        response = spectrum_transform(U[:, q].reshape(Pm, Pn))
        projection += abs(response[:, :len(range_grid)])**2
    return np.maximum(1 - projection / (Pm * Pn), 0.0)

def music_objective(theta, U):
    a = np.exp(-1j * (theta @ omega))
    projection = U.conj().T @ a
    derivative = U.conj().T @ (-1j * omega.T * a[:, None])
    value = 1 - np.vdot(projection, projection).real / (Pm * Pn)
    gradient = -2 * np.real(projection.conj() @ derivative) / (Pm * Pn)
    return float(value), gradient

def estimate_music(Z):
    U = signal_subspace(Z)
    cost = music_grid_cost(U)
    guard_range, guard_doppler = 0.4 * N / Pn, 0.4 * M / Pm
    starts = peak_indices(1 / np.maximum(cost, 1e-12), MUSIC_STARTS,
                          guard_range, guard_doppler)
    candidates = []
    unsuccessful = 0
    for start in starts:
        result = minimize(music_objective, start, args=(U,), method="L-BFGS-B",
                          jac=True, bounds=BOUNDS, options={"maxiter": MUSIC_MAXITER,
                          "ftol": 1e-14, "gtol": 1e-10, "maxls": 40})
        initial_cost = music_objective(start, U)[0]
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and result.fun <= initial_cost:
            candidates.append((float(result.fun), result.x.copy()))
        else:
            candidates.append((initial_cost, start.copy()))
        unsuccessful += int(not result.success)
    candidates.extend((music_objective(t, U)[0], t.copy()) for t in starts)
    chosen = []
    for _, theta in sorted(candidates, key=lambda item: item[0]):
        theta[1] = (theta[1] + M / 2) % M - M / 2
        if all(((theta[0] - old[0]) / guard_range)**2
               + (((theta[1] - old[1] + M / 2) % M - M / 2) / guard_doppler)**2 >= 1
               for old in chosen):
            chosen.append(theta)
        if len(chosen) == Q:
            return np.array(chosen) * np.array([range_cell, velocity_cell]), unsuccessful
    raise RuntimeError("MUSIC未能提取足够的不同目标。")

def match_targets(estimates):
    """真值只在估计完成后的误差统计中使用，匹配距离-速度对。"""
    truth = np.column_stack((target_pos, target_speed))
    delta = (truth[:, None, :] - estimates[None, :, :]) / [range_cell, velocity_cell]
    rows, cols = linear_sum_assignment(np.sum(delta**2, axis=2))
    ordered = np.empty_like(estimates)
    ordered[rows] = estimates[cols]
    return ordered

# def main():
check_settings()
algorithms = ["FFT", "MUSIC"]
errors, crb_samples, curves, standard_errors = {}, {}, {}, {}
for MOD_TYPE in ["psk", "qam"]:
    points = constellation(MOD_TYPE)
    rng = np.random.default_rng(SYMBOL_SEED)
    X_frames = np.array([draw_frame(rng, points) for _ in range(Iter)])
    crb_samples[MOD_TYPE] = np.array([conditional_range_crb(X) for X in X_frames])
    curves[(MOD_TYPE, "CRB")] = np.sqrt(crb_samples[MOD_TYPE].mean() * 10**(-SNR_dB / 10))
    print(f"\n{QAM_mod}-{MOD_TYPE.upper()}: 数据平均能量={np.mean(abs(X_frames)**2):.6f}, "
          f"E[1/|X|^2]={np.mean(1/abs(points)**2):.6f}", flush=True)
    for algorithm in algorithms:
        errors[(MOD_TYPE, algorithm)] = np.empty((len(SNR_dB), Iter, Q))
        curves[(MOD_TYPE, algorithm)] = np.empty(len(SNR_dB))
        standard_errors[(MOD_TYPE, algorithm)] = np.empty(len(SNR_dB))
    for snr_index, snr_db in enumerate(SNR_dB):
        sigma_W2 = 10**(-snr_db / 10)
        rng_noise = np.random.default_rng(NOISE_SEED + snr_index)
        unsuccessful = 0
        for trial, X in enumerate(X_frames):
            W = np.sqrt(sigma_W2 / 2) * (rng_noise.standard_normal((M, N))
                                        + 1j * rng_noise.standard_normal((M, N)))
            W *= X / abs(X)     # 单位相位旋转不改变W|X的白噪声分布
            Y = X * echo_channel + W
            Z = Y / X           # Var(W/X | X)=sigma_W2/|X|^2
            fft_estimate = estimate_fft(Z)
            music_estimate, failed = estimate_music(Z)
            unsuccessful += failed
            for algorithm, estimate in zip(algorithms, [fft_estimate, music_estimate]):
                ordered = match_targets(estimate)
                errors[(MOD_TYPE, algorithm)][snr_index, trial] = ordered[:, 0] - target_pos
            if (trial + 1) % PRINT_EVERY == 0:
                print(f"{MOD_TYPE.upper()}, SNR={snr_db:g} dB: {trial+1}/{Iter}", flush=True)
        for algorithm in algorithms:
            trial_mse = np.mean(errors[(MOD_TYPE, algorithm)][snr_index]**2, axis=1)
            rmse = np.sqrt(trial_mse.mean())
            curves[(MOD_TYPE, algorithm)][snr_index] = rmse
            standard_errors[(MOD_TYPE, algorithm)][snr_index] = (
                trial_mse.std(ddof=1) / (2 * np.sqrt(Iter) * max(rmse, 1e-300)))
        print(f"SNR={snr_db:5g} dB: FFT={curves[(MOD_TYPE,'FFT')][snr_index]:.6e} m, "
              f"MUSIC={curves[(MOD_TYPE,'MUSIC')][snr_index]:.6e} m, "
              f"CRB={curves[(MOD_TYPE,'CRB')][snr_index]:.6e} m, "
              f"MUSIC初值优化未报告收敛={unsuccessful}/{Iter*MUSIC_STARTS}", flush=True)
    del X_frames


#%%>>>>>>>>>>>>>>>>>>>>>>>>  单图六条曲线
fig, ax = plt.subplots(num=3, figsize=(6, 4), constrained_layout=True)
colors = ["#F65314", "#00A1F1"]
for i, mod in enumerate(["psk", "qam"]):
    label = f"{QAM_mod}-{mod.upper()}"
    ax.semilogy(SNR_dB, curves[(mod, "FFT")], "o-" if i == 0 else "o--",
                color=colors[i], markerfacecolor="none", label=label + ", FFT")
    ax.semilogy(SNR_dB, curves[(mod, "MUSIC")], "s-" if i == 0 else "s--",
                color=colors[i], markerfacecolor="none", label=label + ", MUSIC")
    ax.semilogy(SNR_dB, curves[(mod, "CRB")], ":" if i == 0 else "-.",
                color=colors[i], label=label + ", CRB")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (m)")
ax.set_xlim(SNR_dB[0], SNR_dB[-1])
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, which="both")
ax.legend(edgecolor="black", framealpha=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.tick_params(direction="in", top=True, right=True, labelsize=16, width=1)
fig.savefig(OUTPUT_STEM + ".pdf", bbox_inches="tight")
fig.savefig(OUTPUT_STEM + ".png", dpi=600, bbox_inches="tight")
print(f"Saved {OUTPUT_STEM}.pdf/.png/.csv/.npz")
if SHOW_FIGURE:
    plt.show()
plt.close(fig)
# return table

