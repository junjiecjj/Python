"""s06：多站相参/非相参积累 SNR 增益仿真。

模拟 N 个接收站对同一目标的回波，做相参/非相参积累，验证 SNR 增益。

核心物理模型：
  相参积累：N 个站的回波电压相加后再取功率
    → 信号电压相干叠加：|Σ s_j|² = N² · |s|²（信号功率增益 N²）
    → 噪声非相干叠加：|Σ n_j|² 的期望 = N · σ_n²，标准差 = N · σ_n²
    → SNR 增益 = N² / (N · σ_n²) / (1 / σ_n²) = N

  非相参积累：N 个站的功率求和
    → 信号功率求和：Σ |s_j|² = N · |s|²（信号功率增益 N）
    → 噪声功率求和：Σ |n_j|² 的期望 = N · σ_n²，标准差 = √N · σ_n²
    → SNR 增益 = N / (√N · σ_n²) / (1 / σ_n²) = √N

  SNR 定义（偏差系数 / deflection coefficient）：
    SNR = (E[T|H1] - E[T|H0]) / Std[T|H0]
    其中 T 为检测统计量，H0 为纯噪声假设，H1 为信号+噪声假设

  有相位误差时的相参积累：
    各站回波引入随机相位 φ_j ~ N(0, σ_φ²)
    幅度相干因子 ρ = E[exp(jφ)] = exp(-σ_φ²/2)
    |ρ|² = exp(-σ_φ²)
    相参 SNR 增益 = [1 + (N-1)·|ρ|²] （大 N 近似 → |ρ|²）
    SNR 损失 ≈ -10·log10(|ρ|²) = 10·log10(e)·σ_φ² ≈ 4.343·σ_φ² dB

  对应知识库：s07 中的相干因子定义
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.radar_params import RadarParams
from lib.signal_utils import power_to_db
from lib.validation import verify, print_validation, ValidationResult

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 核心仿真函数
# ---------------------------------------------------------------------------


def simulate_coherent_integration(
    params: RadarParams,
    n_stations: int,
    phase_error_std_rad: float,
    noise_seed: int = 42,
    n_trials: int = 10000,
    noise_power: float = 0.1,
) -> dict:
    """模拟多站相参/非相参积累，统计功率输出。

    仿真流程：
      1. 生成归一化目标回波信号（复数幅度 = 1）
      2. 对每个接收站：回波 × 相位旋转 + 热噪声
      3. 相参积累：各站电压求和 → 取功率  |Σ(s_j + n_j)|²
      4. 非相参积累：各站功率求和  Σ|s_j + n_j|²
      5. 蒙特卡洛多次试验，取中值估计

    物理含义：
      - 相参积累利用了信号的相位信息，电压相干叠加后功率增益 N²
      - 非相参积累丢弃相位信息，仅叠加功率，增益较低
      - 相位误差破坏相干性，降低相参积累增益

    Args:
        params:                雷达参数（本函数未直接使用，保留接口一致性）
        n_stations:            接收站数量 N
        phase_error_std_rad:   各站相位误差标准差 σ_φ (rad)，0 表示理想相参
        noise_seed:            噪声随机种子（保证可复现）
        n_trials:              蒙特卡洛试验次数
        noise_power:           每站噪声功率 σ_n²（信号归一化为 1）

    Returns:
        dict: {
            "coherent_power":     相参积累功率中值 |Σ(s_j + n_j)|²
            "incoherent_power":   非相参积累功率中值 Σ|s_j + n_j|²
            "single_station_power": 单站功率中值 |s_1 + n_1|²
            "n_stations":         接收站数
            "phase_error_std_rad":相位误差标准差
        }
    """
    rng = np.random.default_rng(noise_seed)
    sigma_n = np.sqrt(noise_power)

    # 归一化目标回波信号（复数幅度 = 1）
    target_signal = np.complex128(1.0 + 0.0j)

    # 蒙特卡洛存储
    coherent_powers = np.zeros(n_trials)
    incoherent_powers = np.zeros(n_trials)
    single_station_powers = np.zeros(n_trials)

    for trial in range(n_trials):
        station_signals = np.zeros(n_stations, dtype=np.complex128)

        for j in range(n_stations):
            # 相位误差：φ_j ~ N(0, σ_φ²)，旋转回波信号
            if phase_error_std_rad > 0:
                phase_error = rng.normal(0.0, phase_error_std_rad)
                station_signals[j] = target_signal * np.exp(1j * phase_error)
            else:
                station_signals[j] = target_signal

            # 加入热噪声（复高斯，每维方差 σ_n²/2，总方差 σ_n²）
            noise = sigma_n * (
                rng.standard_normal() + 1j * rng.standard_normal()
            ) / np.sqrt(2)
            station_signals[j] += noise

        # 相参积累：各站电压求和 → 取功率
        coherent_sum = np.sum(station_signals)
        coherent_powers[trial] = np.abs(coherent_sum) ** 2

        # 非相参积累：各站功率求和
        incoherent_powers[trial] = np.sum(np.abs(station_signals) ** 2)

        # 单站功率
        single_station_powers[trial] = np.abs(station_signals[0]) ** 2

    return {
        "coherent_power": float(np.median(coherent_powers)),
        "incoherent_power": float(np.median(incoherent_powers)),
        "single_station_power": float(np.median(single_station_powers)),
        "n_stations": n_stations,
        "phase_error_std_rad": phase_error_std_rad,
    }


def compute_snr_gain(
    measured_power: float,
    noise_mean: float,
    noise_std: float,
) -> float:
    """从蒙特卡洛测量功率计算 SNR（偏差系数定义）。

    SNR = (measured_power - noise_mean) / noise_std

    物理含义：
      偏差系数（deflection coefficient）衡量检测统计量在 H1 和 H0 之间的
      可分离程度。对于复高斯噪声：
        - 单站：|s+n|² 的噪声分布为 Exp(σ²)，mean=std=σ²
        - 相参和：|Σn_j|² 的噪声分布为 Exp(Nσ²)，mean=std=Nσ²
        - 功率和：Σ|n_j|² 的噪声分布为 Gamma(N,σ²)，mean=Nσ²，std=√N·σ²

    Args:
        measured_power: 蒙特卡洛中值估计的功率
        noise_mean:     纯噪声假设下的期望功率 E[T|H0]
        noise_std:      纯噪声假设下的功率标准差 Std[T|H0]

    Returns:
        SNR 的线性值
    """
    return (measured_power - noise_mean) / noise_std


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_integration_results(
    results_list: list[dict],
    noise_power: float = 0.1,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制相参/非相参积累结果的三子图。

    子图布局：
      (1) 各站信号幅度示意（展示相位误差的影响）
      (2) 相参 vs 非相参积累 SNR 增益 vs 站数
      (3) 有相位误差时的 SNR 损失衰减

    Args:
        results_list:  simulate_coherent_integration 返回的结果列表
        noise_power:   噪声功率 σ_n²（用于 SNR 计算）
        output_dir:    图像输出目录
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- 子图 1：各站信号幅度示意 ---
    ax1 = axes[0]
    rng_demo = np.random.default_rng(123)
    n_show = 8
    phases_ideal = np.zeros(n_show)
    phases_err = rng_demo.normal(0.0, 0.3, n_show)

    # 理想情况：所有站信号同相
    ax1.scatter(
        range(n_show), np.cos(phases_ideal), s=100, c="blue",
        marker="o", label="理想（无相位误差）", zorder=3,
    )
    # 有相位误差：信号旋转
    ax1.scatter(
        range(n_show), np.cos(phases_err), s=100, c="red",
        marker="^", label="σ_φ = 0.3 rad", zorder=3,
    )

    # 画矢量箭头
    for j in range(n_show):
        ax1.annotate(
            "", xy=(j, np.cos(phases_ideal[j])), xytext=(j, 0),
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5),
        )
        ax1.annotate(
            "", xy=(j, np.cos(phases_err[j])), xytext=(j, 0),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
        )

    ax1.set_xlabel("接收站编号", fontsize=12)
    ax1.set_ylabel("信号实部 (V)", fontsize=12)
    ax1.set_title("各站回波信号幅度\n（相位误差导致电压分量减小）", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(n_show))

    # --- 子图 2：SNR 增益 vs 站数（理想情况） ---
    ax2 = axes[1]
    ideal_results = [
        r for r in results_list if r["phase_error_std_rad"] == 0.0
    ]
    ideal_results.sort(key=lambda r: r["n_stations"])

    n_stations_arr = np.array([r["n_stations"] for r in ideal_results])

    # 用偏差系数计算 SNR 增益
    coherent_gain_arr = []
    incoherent_gain_arr = []
    for r in ideal_results:
        n_sta = r["n_stations"]
        # 单站 SNR：Exp(σ²) 分布，mean=std=σ²
        snr_single = compute_snr_gain(
            r["single_station_power"], noise_power, noise_power,
        )
        # 相参 SNR：Exp(Nσ²) 分布，mean=std=Nσ²
        noise_mean_coh = n_sta * noise_power
        noise_std_coh = n_sta * noise_power
        snr_coh = compute_snr_gain(
            r["coherent_power"], noise_mean_coh, noise_std_coh,
        )
        # 非相参 SNR：Gamma(N,σ²) 分布，mean=Nσ²，std=√N·σ²
        noise_mean_incoh = n_sta * noise_power
        noise_std_incoh = np.sqrt(n_sta) * noise_power
        snr_incoh = compute_snr_gain(
            r["incoherent_power"], noise_mean_incoh, noise_std_incoh,
        )

        if snr_single > 0:
            coherent_gain_arr.append(snr_coh / snr_single)
            incoherent_gain_arr.append(snr_incoh / snr_single)
        else:
            coherent_gain_arr.append(np.nan)
            incoherent_gain_arr.append(np.nan)

    coherent_gain_arr = np.array(coherent_gain_arr)
    incoherent_gain_arr = np.array(incoherent_gain_arr)

    ax2.plot(
        n_stations_arr, coherent_gain_arr, "b-o", linewidth=2,
        markersize=8, label="相参积累（仿真）",
    )
    ax2.plot(
        n_stations_arr, incoherent_gain_arr, "r-s", linewidth=2,
        markersize=8, label="非相参积累（仿真）",
    )

    # 理论曲线
    n_theory = np.linspace(1, max(n_stations_arr), 100)
    ax2.plot(n_theory, n_theory, "b--", alpha=0.5, label="理论：相参增益 = N")
    ax2.plot(
        n_theory, np.sqrt(n_theory), "r--", alpha=0.5,
        label="理论：非相参增益 = √N",
    )

    ax2.set_xlabel("接收站数 N", fontsize=12)
    ax2.set_ylabel("SNR 增益（相对单站）", fontsize=12)
    ax2.set_title("相参 vs 非相参积累 SNR 增益\n（理想情况，无相位误差）", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # --- 子图 3：相位误差对 SNR 的影响 ---
    ax3 = axes[2]
    unique_n = sorted(set(r["n_stations"] for r in results_list))
    colors = ["blue", "green", "red", "purple", "orange"]

    for idx, n_sta in enumerate(unique_n):
        # 筛选该站数下有相位误差的结果
        n_results = [
            r for r in results_list
            if r["n_stations"] == n_sta and r["phase_error_std_rad"] > 0
        ]
        n_results.sort(key=lambda r: r["phase_error_std_rad"])

        if not n_results:
            continue

        phase_errors = [r["phase_error_std_rad"] for r in n_results]

        # SNR 损失 = 10·log10(P_coh(有误差) / P_coh(无误差))
        # 噪声贡献相同，在比值中抵消，因此功率比 = SNR 比
        ideal_ref = [
            r for r in results_list
            if r["n_stations"] == n_sta and r["phase_error_std_rad"] == 0.0
        ]
        if not ideal_ref:
            continue
        ref_power = ideal_ref[0]["coherent_power"]

        snr_loss_db = [
            power_to_db(r["coherent_power"] / ref_power) for r in n_results
        ]

        ax3.plot(
            phase_errors, snr_loss_db, "-o",
            color=colors[idx % len(colors)],
            linewidth=2, markersize=6, label=f"N={n_sta}（仿真）",
        )

    # 理论曲线：SNR 损失 ≈ 10·log10(e)·σ_φ² dB（大 N 近似）
    phase_theory = np.linspace(0.01, 1.0, 100)
    snr_loss_theory = 10.0 * np.log10(np.exp(1)) * phase_theory**2
    ax3.plot(
        phase_theory, snr_loss_theory, "k--", linewidth=2, alpha=0.7,
        label="理论：10·log₁₀(e)·σ_φ²",
    )

    ax3.set_xlabel("相位误差标准差 σ_φ (rad)", fontsize=12)
    ax3.set_ylabel("SNR 损失 (dB)", fontsize=12)
    ax3.set_title("相位误差对相参积累 SNR 的影响", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.subplots_adjust(wspace=0.3)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "s06_coherent_integration.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图像已保存: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def validate(results_list: list[dict], noise_power: float = 0.1) -> bool:
    """验证相参/非相参积累的 SNR 增益是否符合理论预期。

    SNR 定义（偏差系数）：
      SNR = (E[T|H1] - E[T|H0]) / Std[T|H0]

    验证项：
      a. 理想相参增益：SNR_coherent / SNR_single ≈ N（容差 1 dB）
      b. 理想非相参增益：SNR_incoherent / SNR_single ≈ √N（容差 1 dB）
      c. 相干因子：有误差时 SNR_loss ≈ 10·log10(e)·σ_φ² dB（容差 1 dB）
      d. Parseval：非相参功率 ≈ N × (1 + σ_n²)（能量守恒）

    物理判据：
      - 相参积累 SNR 增益 = N（功率增益 N² / 噪声标准差增益 N）
      - 非相参积累 SNR 增益 = √N（功率增益 N / 噪声标准差增益 √N）
      - 相位误差使相参增益下降，损失量 ∝ σ_φ²

    Args:
        results_list:  所有仿真结果
        noise_power:   噪声功率 σ_n²
    """
    results: list[ValidationResult] = []

    # --- 验证 a & b：理想情况下的 SNR 增益 ---
    ideal_results = [
        r for r in results_list if r["phase_error_std_rad"] == 0.0
    ]

    for r in ideal_results:
        n_sta = r["n_stations"]

        # 单站 SNR：Exp(σ²) 分布，mean = std = σ²
        snr_single = compute_snr_gain(
            r["single_station_power"], noise_power, noise_power,
        )

        # 相参 SNR：Exp(Nσ²) 分布，mean = std = Nσ²
        noise_mean_coh = n_sta * noise_power
        noise_std_coh = n_sta * noise_power
        snr_coherent = compute_snr_gain(
            r["coherent_power"], noise_mean_coh, noise_std_coh,
        )

        # 非相参 SNR：Gamma(N,σ²) 分布，mean = Nσ²，std = √N·σ²
        noise_mean_incoh = n_sta * noise_power
        noise_std_incoh = np.sqrt(n_sta) * noise_power
        snr_incoherent = compute_snr_gain(
            r["incoherent_power"], noise_mean_incoh, noise_std_incoh,
        )

        if snr_single <= 0:
            continue

        # 验证 a：相参增益 ≈ N
        gain_coherent = snr_coherent / snr_single
        results.append(verify(
            name=f"相参增益 N={n_sta}",
            theoretical=float(n_sta),
            simulated=gain_coherent,
            tolerance=1.0,
            unit="",
        ))

        # 验证 b：非相参增益 ≈ √N
        gain_incoherent = snr_incoherent / snr_single
        results.append(verify(
            name=f"非相参增益 N={n_sta}",
            theoretical=np.sqrt(n_sta),
            simulated=gain_incoherent,
            tolerance=1.0,
            unit="",
        ))

    # --- 验证 c：相干因子（有相位误差时的 SNR 损失） ---
    # 取 N=8 的结果来验证相位误差影响（N 越大，近似越准）
    n8_results = [r for r in results_list if r["n_stations"] == 8]
    ideal_n8 = [r for r in n8_results if r["phase_error_std_rad"] == 0.0]

    if ideal_n8:
        ref_power = ideal_n8[0]["coherent_power"]
        for r in n8_results:
            if r["phase_error_std_rad"] == 0.0:
                continue

            sigma_phi = r["phase_error_std_rad"]

            # SNR 损失 = 10·log10(P_coh(有误差) / P_coh(无误差))
            # 噪声贡献相同（同一随机种子），在比值中抵消
            snr_loss_sim = power_to_db(r["coherent_power"] / ref_power)

            # 理论值：SNR_loss ≈ -10·log10(|ρ|²) = 10·log10(e)·σ_φ²
            snr_loss_theory = -10.0 * np.log10(np.exp(-sigma_phi**2))

            results.append(verify(
                name=f"相干因子 σ_φ={sigma_phi:.1f}rad",
                theoretical=snr_loss_theory,
                simulated=abs(snr_loss_sim),
                tolerance=1.0,
                unit="dB",
            ))

    # --- 验证 d：Parseval 能量守恒 ---
    # 非相参功率（求和）应接近 N × (1 + σ_n²)
    # 其中 1 = |s|² 为信号功率，σ_n² 为每站噪声功率
    for r in ideal_results:
        n_sta = r["n_stations"]
        expected_power = n_sta * (1.0 + noise_power)
        results.append(verify(
            name=f"Parseval 非相参 N={n_sta}",
            theoretical=expected_power,
            simulated=r["incoherent_power"],
            tolerance=expected_power * 0.05,  # 5% 容差
            unit="",
        ))

    return print_validation("s06 相参/非相参积累", results)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> int:
    """运行 s06 相参/非相参积累仿真与验证。"""
    print("=" * 60)
    print("s06：多站相参/非相参积累 SNR 增益仿真")
    print("=" * 60)

    params = RadarParams()

    # 仿真参数
    n_stations_list = [2, 4, 8]
    phase_errors = [0.0, 0.1, 0.3, 1.0]
    noise_power = 0.1   # 每站噪声功率 σ_n²
    n_trials = 10000     # 蒙特卡洛试验次数
    noise_seed = 42      # 噪声种子

    print(f"\n仿真参数:")
    print(f"  接收站数 N     = {n_stations_list}")
    print(f"  相位误差 σ_φ   = {phase_errors} rad")
    print(f"  噪声功率 σ_n²  = {noise_power}")
    print(f"  蒙特卡洛试验数 = {n_trials}")
    print(f"  随机种子       = {noise_seed}")

    # 运行仿真
    print(f"\n运行仿真...")
    all_results: list[dict] = []

    for n_sta in n_stations_list:
        for phase_err in phase_errors:
            result = simulate_coherent_integration(
                params=params,
                n_stations=n_sta,
                phase_error_std_rad=phase_err,
                noise_seed=noise_seed,
                n_trials=n_trials,
                noise_power=noise_power,
            )
            all_results.append(result)

    # 打印结果表格（功率比，即功率增益）
    print(f"\n{'='*80}")
    print(f"{'N':>4} {'σ_φ(rad)':>10} {'P_coh':>10} {'P_incoh':>10} "
          f"{'P_single':>10} {'P比_coh':>10} {'P比_incoh':>10}")
    print(f"{'='*80}")

    for r in all_results:
        gain_coh = r["coherent_power"] / r["single_station_power"]
        gain_incoh = r["incoherent_power"] / r["single_station_power"]
        print(
            f"{r['n_stations']:>4d} "
            f"{r['phase_error_std_rad']:>10.1f} "
            f"{r['coherent_power']:>10.3f} "
            f"{r['incoherent_power']:>10.3f} "
            f"{r['single_station_power']:>10.3f} "
            f"{gain_coh:>10.2f} "
            f"{gain_incoh:>10.3f}"
        )

    # SNR 增益理论对比（用偏差系数计算）
    print(f"\n--- SNR 增益理论对比 ---")
    for r in all_results:
        n_sta = r["n_stations"]
        sigma_phi = r["phase_error_std_rad"]

        # 单站 SNR
        snr_single = compute_snr_gain(
            r["single_station_power"], noise_power, noise_power,
        )
        # 相参 SNR
        snr_coh = compute_snr_gain(
            r["coherent_power"], n_sta * noise_power, n_sta * noise_power,
        )
        # 非相参 SNR
        snr_incoh = compute_snr_gain(
            r["incoherent_power"],
            n_sta * noise_power,
            np.sqrt(n_sta) * noise_power,
        )

        gain_coh = snr_coh / snr_single if snr_single > 0 else 0
        gain_incoh = snr_incoh / snr_single if snr_single > 0 else 0

        if sigma_phi == 0.0:
            print(
                f"  N={n_sta}, σ_φ=0: "
                f"相参增益 仿真={gain_coh:.2f} 理论={n_sta}, "
                f"非相参增益 仿真={gain_incoh:.2f} 理论={np.sqrt(n_sta):.2f}"
            )
        else:
            # 有误差时的理论增益：1 + (N-1)·exp(-σ_φ²)
            theory_gain = 1.0 + (n_sta - 1) * np.exp(-sigma_phi**2)
            print(
                f"  N={n_sta}, σ_φ={sigma_phi:.1f}: "
                f"相参增益 仿真={gain_coh:.2f} 理论≈{theory_gain:.2f}"
            )

    # 绘图
    print(f"\n绘制结果...")
    plot_integration_results(all_results, noise_power=noise_power)

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(all_results, noise_power=noise_power)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
