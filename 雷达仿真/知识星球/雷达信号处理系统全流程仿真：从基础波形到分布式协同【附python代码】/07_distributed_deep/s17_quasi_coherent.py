"""s17：准相参 vs 全相参 vs 非相参积累性能对比。

验证目标：
  - 对比三种信号积累方式的 SNR 增益：全相参、准相参、非相参
  - 验证相干因子 γ 与相位误差的关系
  - 量化相位同步误差对检测性能的影响
  - 为分布式雷达系统设计提供性能上界参考

原理：
  分布式雷达系统中，多个站点协同工作以提高检测性能。
  积累方式的选择取决于站点间相位同步的精度：

  1) 全相参积累（Coherent Integration）：
     各站信号直接相加，假设完美相位同步。
     SNR 增益 = N 倍 = 10*log10(N) dB。
     这是分布式雷达性能的理论上界。

  2) 准相参积累（Quasi-Coherent Integration）：
     先估计各站相位并补偿，但估计存在残余误差 φ_k。
     相干因子 γ = |mean(exp(jφ))|，其中 φ 为相位误差。
     SNR 增益 = 1 + (N-1)*γ² 倍。
     推导：E[|Σ(A+n_k)·e^{-jφ_k}|²] = A²·(N + N(N-1)γ²) + N·σ²
     → SNR = A²·(1 + (N-1)γ²)/σ²，其中 γ = E[e^{jφ}]。

  3) 非相参积累（Non-Coherent Integration）：
     各站先取功率再相加，不依赖相位信息。
     SNR 增益 = A²/σ² = 单站 SNR（无额外增益，0 dB）。
     积累损失 = 全相参增益 - 非相参增益 = 10*log10(N) dB。

  物理直觉：
    全相参利用相位信息使信号幅度相干叠加（增益 N²），
    而噪声仅功率叠加（增益 N），因此 SNR 提升 N 倍。
    准相参因相位补偿不完美，信号叠加效率降低为 γ²，
    但仍优于非相参（γ=0 时退化为非相参）。
    非相参丢弃相位信息，信号和噪声均功率叠加，SNR 无改善。

对应知识库：分布式雷达信号处理
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gamma as gamma_dist

from lib.signal_utils import power_to_db, db_to_power
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
# Parseval: sum(|x|^2) = (1/N) * sum(|X|^2)


# ============================================================
# 核心函数
# ============================================================


def coherent_integration(signals: np.ndarray) -> complex:
    """全相参积累：多站信号直接相加。

    假设各站完美同步，信号在同相位叠加。
    输出 SNR 增益 = N 倍 = 10*log10(N) dB。

    推导：
      输出 = Σ(s_k + n_k) = N·A + Σn_k
      |输出|² = N²A² + 噪声交叉项 + |Σn_k|²
      E[|输出|²] = N²A² + N·σ²
      SNR_out = N²A² / (N·σ²) = N·A²/σ² = N·SNR_in

    Args:
        signals: (N,) 复数数组，各站接收信号（含目标回波 + 噪声）

    Returns:
        积累后的复数信号（标量）
    """
    return np.sum(signals)


def quasi_coherent_integration(
    signals: np.ndarray, phase_errors_rad: np.ndarray
) -> complex:
    """准相参积累：先估计补偿各站相位，再叠加。

    各站信号先乘以 exp(-j·φ_k) 来补偿相位，
    其中 φ_k 是估计的相位误差（残余偏差）。
    积累后 SNR 增益 = 1 + (N-1)·γ² 倍。

    推导：
      输出 = Σ(A + n_k)·e^{-jφ_k}
      E[|输出|²] = A²·(N + N(N-1)γ²) + N·σ²
      SNR_out/SNR_in = 1 + (N-1)·γ²

      其中 γ = |E[e^{jφ}]| = exp(-σ_φ²/2)（高斯近似）

    Args:
        signals:          (N,) 复数数组，各站接收信号
        phase_errors_rad: (N,) 相位误差（弧度），估计的残余相位偏差

    Returns:
        积累后的复数信号（标量）
    """
    compensated = signals * np.exp(-1j * phase_errors_rad)
    return np.sum(compensated)


def non_coherent_integration(signals: np.ndarray) -> float:
    """非相参积累：各站取功率后相加。

    不依赖相位信息，直接叠加各站功率。
    SNR 增益 = 1（0 dB），无额外增益。

    推导：
      输出 = Σ|s_k + n_k|²
      E[输出] = N·(A² + σ²)
      SNR_out = N·A² / (N·σ²) = A²/σ² = SNR_in

    Args:
        signals: (N,) 复数数组，各站接收信号

    Returns:
        积累后的功率值（实数，线性单位）
    """
    return np.sum(np.abs(signals) ** 2)


def coherence_factor(phase_errors_rad: np.ndarray) -> float:
    """计算相干因子 γ。

    γ = |mean(exp(j·φ))|，衡量各站相位的一致性。

    - γ = 1：所有相位误差为 0，完全相干（全相参）
    - γ → 0：相位误差均匀分布，完全不相干

    对于零均值高斯相位误差：
      γ = exp(-σ² / 2)

    Args:
        phase_errors_rad: (N,) 相位误差数组（弧度）

    Returns:
        相干因子 γ ∈ [0, 1]（实数）
    """
    return float(np.abs(np.mean(np.exp(1j * phase_errors_rad))))


def snr_gain_vs_phase_error(
    n_stations: int,
    phase_error_stds_rad: np.ndarray,
    rng: np.random.Generator,
    n_mc: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算三种积累方式的 SNR 增益随相位误差的变化（Monte Carlo）。

    使用向量化 Monte Carlo 仿真：
    - 生成 N 个站的信号（目标 + 噪声）
    - 对每组相位误差，计算三种积累的输出 SNR
    - 统计平均 SNR 增益

    SNR 增益定义：E[|输出|²]/(N·σ²) - 1，即信号分量贡献 / 噪声基底。

    Args:
        n_stations:            站数 N
        phase_error_stds_rad:  相位误差标准差数组（弧度）
        rng:                   随机数生成器
        n_mc:                  每个相位误差级别的 Monte Carlo 试验数

    Returns:
        (gain_coherent_dB, gain_quasi_dB, gain_non_coherent_dB)
        每个元素长度 = len(phase_error_stds_rad)
    """
    n_points = len(phase_error_stds_rad)
    gain_coherent = np.zeros(n_points)
    gain_quasi = np.zeros(n_points)
    gain_non_coherent = np.zeros(n_points)

    # 单站 SNR = 0 dB 作为参考（A=1, σ²=1）
    signal_amplitude = 1.0
    noise_power = 1.0

    for idx, sigma_rad in enumerate(phase_error_stds_rad):
        # 向量化生成所有 MC 试验的噪声: (n_mc, n_stations)
        noise = (
            rng.standard_normal((n_mc, n_stations))
            + 1j * rng.standard_normal((n_mc, n_stations))
        ) * np.sqrt(noise_power / 2)

        # 各站信号（幅度相同，假设完美幅度同步）
        received = signal_amplitude + noise  # (n_mc, n_stations)

        # --- 全相参积累 ---
        s_coh = np.abs(np.sum(received, axis=1)) ** 2  # (n_mc,)
        gain_coherent[idx] = np.mean(s_coh) / (n_stations * noise_power) - 1.0

        # --- 准相参积累 ---
        phase_err = rng.normal(0, sigma_rad, (n_mc, n_stations))
        s_qc = np.abs(
            np.sum(received * np.exp(-1j * phase_err), axis=1)
        ) ** 2  # (n_mc,)
        gain_quasi[idx] = np.mean(s_qc) / (n_stations * noise_power) - 1.0

        # --- 非相参积累 ---
        p_nc = np.sum(np.abs(received) ** 2, axis=1)  # (n_mc,)
        gain_non_coherent[idx] = np.mean(p_nc) / (n_stations * noise_power) - 1.0

    # 转换为 dB
    eps = 1e-40
    gain_coherent_dB = power_to_db(np.maximum(gain_coherent, eps))
    gain_quasi_dB = power_to_db(np.maximum(gain_quasi, eps))
    gain_non_coherent_dB = power_to_db(np.maximum(gain_non_coherent, eps))

    return gain_coherent_dB, gain_quasi_dB, gain_non_coherent_dB


def snr_gain_theoretical(
    n_stations: int, phase_error_stds_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """计算三种积累方式的理论 SNR 增益。

    全相参：增益 = N → 10*log10(N) dB
    准相参：增益 = 1 + (N-1)·γ²，γ = exp(-σ²/2)
    非相参：增益 = 1 → 0 dB

    Args:
        n_stations:            站数 N
        phase_error_stds_rad:  相位误差标准差数组（弧度）

    Returns:
        (gain_coherent_dB, gain_quasi_dB, gain_nc_dB)
    """
    gain_coherent_dB = 10 * np.log10(n_stations) * np.ones_like(phase_error_stds_rad)

    # 准相参：γ = exp(-σ²/2)，SNR 增益 = 1 + (N-1)·γ²
    gamma = np.exp(-phase_error_stds_rad**2 / 2.0)
    gain_quasi_linear = 1.0 + (n_stations - 1) * gamma**2
    gain_quasi_dB = 10 * np.log10(np.maximum(gain_quasi_linear, 1e-40))

    # 非相参：SNR 增益 = 1（0 dB），无额外增益
    gain_nc_dB = 0.0

    return gain_coherent_dB, gain_quasi_dB, gain_nc_dB


def compute_pd_vs_snr(
    n_stations: int,
    snr_single_dB: float,
    pfa: float,
    phase_error_std_rad: float,
    rng: np.random.Generator,
    n_mc: int = 10000,
) -> tuple[float, float, float]:
    """Monte Carlo 估计三种积累方式在给定单站 SNR 下的检测概率 Pd。

    信号模型：
      - 目标：复幅度 A（各站相同），单站 SNR = A²/σ²
      - 噪声：各站独立复高斯 CN(0, σ²)，σ² = 1
      - 检测统计量：积累后取功率 |输出|² 或 Σ|输出_k|²
      - 门限：由噪声-only 分布和 Pfa 解析确定

    Args:
        n_stations:           站数 N
        snr_single_dB:        单站 SNR (dB)
        pfa:                  虚警概率
        phase_error_std_rad:  准相参相位误差标准差（弧度）
        rng:                  随机数生成器
        n_mc:                 Monte Carlo 试验数

    Returns:
        (pd_coherent, pd_quasi, pd_non_coherent) 检测概率
    """
    noise_power = 1.0
    signal_amp = np.sqrt(db_to_power(snr_single_dB) * noise_power)

    # --- 解析门限 ---
    # 全相参/准相参：|Σ n_k|² ~ Exp(N·σ²)
    #   P(|Σ n_k|² > th) = exp(-th/(N·σ²)) = Pfa
    #   → th = -N·σ²·ln(Pfa)
    threshold_coh = -n_stations * noise_power * np.log(pfa)

    # 准相参门限与全相参相同（相位旋转不改变复高斯分布）
    threshold_qc = threshold_coh

    # 非相参：Σ|n_k|² ~ Gamma(N, σ²)
    #   → th = Gamma.ppf(1-Pfa, shape=N, scale=σ²)
    threshold_nc = gamma_dist.ppf(1 - pfa, a=n_stations, scale=noise_power)

    # --- 向量化 Monte Carlo 检测 ---
    noise = (
        rng.standard_normal((n_mc, n_stations))
        + 1j * rng.standard_normal((n_mc, n_stations))
    ) * np.sqrt(noise_power / 2)
    received = signal_amp + noise  # (n_mc, n_stations)

    # 全相参
    s_coh = np.abs(np.sum(received, axis=1)) ** 2
    pd_coherent = float(np.mean(s_coh > threshold_coh))

    # 准相参
    phase_err = rng.normal(0, phase_error_std_rad, (n_mc, n_stations))
    s_qc = np.abs(np.sum(received * np.exp(-1j * phase_err), axis=1)) ** 2
    pd_quasi = float(np.mean(s_qc > threshold_qc))

    # 非相参
    p_nc = np.sum(np.abs(received) ** 2, axis=1)
    pd_non_coherent = float(np.mean(p_nc > threshold_nc))

    return pd_coherent, pd_quasi, pd_non_coherent


# ============================================================
# 绘图
# ============================================================


def plot_quasi_coherent(
    n_stations: int,
    snr_single_db: float,
    pfa: float,
    rng: np.random.Generator,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制准相参 vs 全相参 vs 非相参性能对比（3 子图）。

    子图 1：三种积累方式的 SNR 增益 vs 相位误差标准差
      - 全相参：恒定 10*log10(N) dB
      - 准相参：从全相参水平随相位误差增大而下降至 0 dB
      - 非相参：恒定 0 dB

    子图 2：相干因子 γ vs 相位标准差（理论曲线 vs Monte Carlo 仿真）
      - 理论：γ = exp(-σ²/2)
      - 仿真：γ = |mean(exp(jφ))|

    子图 3：检测概率 Pd vs 单站 SNR（三种积累方式对比，固定 Pfa）

    Args:
        n_stations:    站数 N
        snr_single_db: 单站 SNR (dB)
        pfa:           虚警概率
        rng:           随机数生成器
        output_dir:    输出目录
    """
    # ---- 子图 1 数据：SNR 增益 vs 相位误差 ----
    phase_stds_deg = np.linspace(0, 60, 25)
    phase_stds_rad = np.deg2rad(phase_stds_deg)

    n_mc_gain = 5000
    gain_coh_sim, gain_qc_sim, gain_nc_sim = snr_gain_vs_phase_error(
        n_stations, phase_stds_rad, rng, n_mc=n_mc_gain,
    )
    gain_coh_theory, gain_qc_theory, gain_nc_theory = snr_gain_theoretical(
        n_stations, phase_stds_rad,
    )

    # ---- 子图 2 数据：相干因子 γ ----
    gamma_sim = np.zeros(len(phase_stds_rad))
    n_mc_gamma = 50000
    for idx, sigma_rad in enumerate(phase_stds_rad):
        phis = rng.normal(0, sigma_rad, n_mc_gamma)
        gamma_sim[idx] = coherence_factor(phis)
    gamma_theory = np.exp(-phase_stds_rad**2 / 2.0)

    # ---- 子图 3 数据：Pd vs SNR ----
    snr_range_dB = np.linspace(-5, 20, 18)
    phase_error_for_pd = np.deg2rad(15.0)  # 准相参相位误差 15°
    n_mc_pd = 10000

    pd_coh = np.zeros(len(snr_range_dB))
    pd_qc = np.zeros(len(snr_range_dB))
    pd_nc = np.zeros(len(snr_range_dB))

    for idx, snr_db in enumerate(snr_range_dB):
        pd_coh[idx], pd_qc[idx], pd_nc[idx] = compute_pd_vs_snr(
            n_stations, snr_db, pfa, phase_error_for_pd, rng, n_mc=n_mc_pd,
        )

    # ---- 绘图 ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # 子图 1：SNR 增益 vs 相位误差
    ax1 = axes[0]
    ax1.plot(phase_stds_deg, gain_coh_theory, "b--", linewidth=2, label="全相参（理论）")
    ax1.plot(phase_stds_deg, gain_qc_theory, "r-", linewidth=2, label="准相参（理论）")
    ax1.axhline(y=gain_nc_theory, color="g", linestyle="--", linewidth=2, label="非相参（理论）")
    ax1.plot(phase_stds_deg, gain_coh_sim, "b^", markersize=5, alpha=0.6, label="全相参（仿真）")
    ax1.plot(phase_stds_deg, gain_qc_sim, "ro", markersize=5, alpha=0.6, label="准相参（仿真）")
    ax1.plot(phase_stds_deg, gain_nc_sim, "gs", markersize=5, alpha=0.6, label="非相参（仿真）")
    ax1.set_xlabel("相位误差标准差 (度)", fontsize=12)
    ax1.set_ylabel("SNR 增益 (dB)", fontsize=12)
    ax1.set_title(
        f"三种积累方式的 SNR 增益 vs 相位误差 (N={n_stations})",
        fontsize=13,
    )
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 60])

    # 子图 2：相干因子 γ
    ax2 = axes[1]
    ax2.plot(phase_stds_deg, gamma_theory, "r-", linewidth=2, label="理论: γ = exp(-σ²/2)")
    ax2.plot(phase_stds_deg, gamma_sim, "b^", markersize=5, alpha=0.6, label="Monte Carlo 仿真")
    ax2.set_xlabel("相位误差标准差 (度)", fontsize=12)
    ax2.set_ylabel("相干因子 γ", fontsize=12)
    ax2.set_title("相干因子 γ vs 相位误差标准差", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 60])
    ax2.set_ylim([0, 1.05])

    # 子图 3：Pd vs SNR
    ax3 = axes[2]
    ax3.plot(snr_range_dB, pd_coh, "b-o", linewidth=2, markersize=5, label="全相参积累")
    ax3.plot(
        snr_range_dB, pd_qc, "r-s", linewidth=2, markersize=5,
        label=f"准相参积累 (σ={np.rad2deg(phase_error_for_pd):.0f}°)",
    )
    ax3.plot(snr_range_dB, pd_nc, "g-^", linewidth=2, markersize=5, label="非相参积累")
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Pd = 0.5")
    ax3.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="Pd = 0.9")
    ax3.set_xlabel("单站 SNR (dB)", fontsize=12)
    ax3.set_ylabel("检测概率 Pd", fontsize=12)
    ax3.set_title(
        f"检测概率 Pd vs SNR (N={n_stations}, Pfa={pfa:.0e})",
        fontsize=13,
    )
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlim([snr_range_dB[0], snr_range_dB[-1]])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s17_quasi_coherent.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s17_quasi_coherent.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_stations: int,
    snr_single_db: float,
    pfa: float,
    rng: np.random.Generator,
) -> bool:
    """验证准相参 vs 全相参性能对比的正确性。

    验证项：
      a. 全相参 SNR 增益 = 10*log10(N) dB（误差 < 0.5 dB）
      b. 相干因子：相位误差为 0 时 γ = 1
      c. 准相参退化：相位误差极大时准相参增益接近非相参（0 dB）
      d. 积累损失：非相参积累损失 = 10*log10(N) dB（误差 < 0.8 dB）
    """
    results = []
    noise_power = 1.0
    signal_amp = 1.0
    n_mc = 20000

    # --- 验证 a：全相参 SNR 增益 = 10*log10(N) dB ---
    snr_coh_vals = np.zeros(n_mc)
    noise_a = (
        rng.standard_normal((n_mc, n_stations))
        + 1j * rng.standard_normal((n_mc, n_stations))
    ) * np.sqrt(noise_power / 2)
    received_a = signal_amp + noise_a
    s_coh_a = np.abs(np.sum(received_a, axis=1)) ** 2
    mean_snr_coh = np.mean(s_coh_a) / (n_stations * noise_power)
    coherent_gain_dB = power_to_db(max(mean_snr_coh - 1.0, 1e-40))
    theoretical_gain_dB = 10 * np.log10(n_stations)

    results.append(verify(
        name="全相参 SNR 增益",
        theoretical=theoretical_gain_dB,
        simulated=coherent_gain_dB,
        tolerance=0.5,
        unit="dB",
    ))

    # --- 验证 b：相位误差为 0 时 γ = 1 ---
    phase_errors_zero = np.zeros(100)
    gamma_zero = coherence_factor(phase_errors_zero)

    results.append(verify(
        name="相干因子（相位误差为 0 时 γ = 1）",
        theoretical=1.0,
        simulated=gamma_zero,
        tolerance=0.001,
        unit="",
    ))

    # --- 验证 c：准相参退化 —— 相位误差极大时接近非相参 ---
    # σ = 150°：γ ≈ exp(-2.618²/2) ≈ 0.033，增益 ≈ 0.03 dB ≈ 0 dB
    large_phase_std_rad = np.deg2rad(150.0)

    noise_c = (
        rng.standard_normal((n_mc, n_stations))
        + 1j * rng.standard_normal((n_mc, n_stations))
    ) * np.sqrt(noise_power / 2)
    received_c = signal_amp + noise_c

    phase_err_c = rng.normal(0, large_phase_std_rad, (n_mc, n_stations))
    s_qc_c = np.abs(
        np.sum(received_c * np.exp(-1j * phase_err_c), axis=1)
    ) ** 2
    gain_qc_large = np.mean(s_qc_c) / (n_stations * noise_power) - 1.0
    gain_qc_large_dB = power_to_db(max(gain_qc_large, 1e-40))

    # 非相参增益（作为参考基线）
    p_nc_c = np.sum(np.abs(received_c) ** 2, axis=1)
    gain_nc = np.mean(p_nc_c) / (n_stations * noise_power) - 1.0
    gain_nc_dB = power_to_db(max(gain_nc, 1e-40))

    results.append(verify(
        name="准相参退化（σ=150°时接近非相参 0 dB）",
        theoretical=0.0,
        simulated=gain_qc_large_dB,
        tolerance=1.0,
        unit="dB",
    ))

    # --- 验证 d：非相参积累损失 = 10*log10(N) dB ---
    # 积累损失 = 全相参增益 - 非相参增益
    integration_loss_dB = coherent_gain_dB - gain_nc_dB

    results.append(verify(
        name="非相参积累损失",
        theoretical=theoretical_gain_dB,
        simulated=integration_loss_dB,
        tolerance=0.8,
        unit="dB",
    ))

    return print_validation("s17 准相参 vs 全相参性能对比", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s17 准相参 vs 全相参性能对比仿真与验证。"""
    print("=" * 60)
    print("s17：准相参 vs 全相参 vs 非相参积累性能对比")
    print("=" * 60)

    # 仿真参数
    n_stations = 8           # 站数 N
    snr_single_db = 10.0     # 单站 SNR (dB)
    pfa = 1e-6               # 虚警概率
    seed = 42                # 随机种子

    rng = np.random.default_rng(seed)

    print(f"\n仿真参数:")
    print(f"  站数 N              = {n_stations}")
    print(f"  单站 SNR            = {snr_single_db} dB")
    print(f"  虚警概率 Pfa        = {pfa:.0e}")
    print(f"  随机种子            = {seed}")
    print(f"  全相参理论 SNR 增益 = {10*np.log10(n_stations):.2f} dB")
    print(f"  非相参理论增益      = 0 dB（无 SNR 改善）")
    print(f"  非相参积累损失      = {10*np.log10(n_stations):.2f} dB")

    # 核心计算示例
    print(f"\n核心函数验证:")
    signals_example = rng.standard_normal(n_stations) + 1j * rng.standard_normal(n_stations)
    s_coh = coherent_integration(signals_example)
    p_nc = non_coherent_integration(signals_example)
    phase_err_example = rng.normal(0, np.deg2rad(10), n_stations)
    s_qc = quasi_coherent_integration(signals_example, phase_err_example)
    gamma = coherence_factor(phase_err_example)

    print(f"  全相参输出（幅度）  = {np.abs(s_coh):.4f}")
    print(f"  准相参输出（幅度）  = {np.abs(s_qc):.4f}")
    print(f"  非相参输出（功率）  = {p_nc:.4f}")
    print(f"  相干因子 γ          = {gamma:.4f}（σ=10°）")

    # 绘图
    print(f"\n绘制性能对比图...")
    plot_quasi_coherent(n_stations, snr_single_db, pfa, rng)

    # 验证
    print(f"\n运行验证...")
    rng_val = np.random.default_rng(seed + 100)  # 独立的随机种子用于验证
    all_passed = validate(n_stations, snr_single_db, pfa, rng_val)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
