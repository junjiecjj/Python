"""s01：雷达方程与 SNR 计算。

验证目标：
  - 给定雷达参数，计算最大探测距离 R_max
  - 绘制 SNR 随距离变化的曲线
  - 用 validate() 对比理论值和仿真值

雷达方程（单基地脉冲雷达）：
  SNR = (Pt * G² * λ² * σ * T) / ((4π)³ * R⁴ * k * T₀ * F * B)

其中：
  Pt  = 峰值发射功率 (W)
  G   = 天线增益（线性值，收发共用天线故 G²）
  λ   = 波长 (m)
  σ   = 目标 RCS (m²)
  T   = 脉冲宽度 (s)  — 单脉冲能量 E = Pt * T
  R   = 目标距离 (m)
  k   = 玻尔兹曼常数 1.38e-23 J/K
  T₀  = 标准噪声温度 290 K
  F   = 接收机噪声系数（线性值）
  L   = 系统损耗（本例简化为 1，即无损耗）

物理含义：
  SNR 与 R⁴ 成反比 — 距离翻倍，SNR 降 12 dB（信号按球面扩散衰减）。
  最大探测距离 R_max 定义为 SNR 刚好等于最小可检测 SNR（通常 13 dB）时的距离。

对应知识库：radar-knowledge-base/基础/01-雷达测量原理/00-雷达方程与系统概述.md
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.radar_params import RadarParams, SPEED_OF_LIGHT, BOLTZMANN, STANDARD_TEMP
from lib.validation import verify, print_validation
from lib.signal_utils import power_to_db, db_to_power

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def calculate_snr(params: RadarParams, range_m: float) -> float:
    """计算指定距离处的接收 SNR（线性值）。

    雷达方程推导：
      接收功率 Pr = (Pt * G * σ * Ae) / ((4π)² * R⁴)
      其中 Ae = G * λ² / (4π) 是有效接收孔径面积
      代入得 Pr = (Pt * G² * λ² * σ) / ((4π)³ * R⁴)

      噪声功率 Pn = k * T₀ * F * B

      单脉冲 SNR = Pr * T / Pn = (Pt * G² * λ² * σ * T) / ((4π)³ * R⁴ * k * T₀ * F * B)

    Args:
        params:   雷达参数
        range_m:  目标距离 (m)

    Returns:
        SNR 的线性值（无量纲）
    """
    # 发射功率 × 天线增益² × 波长² × RCS × 脉宽
    numerator = (params.pt
                 * params.gain_linear ** 2
                 * params.wavelength_m ** 2
                 * params.target_rcs_m2
                 * params.pulse_width_s)

    # (4π)³ × R⁴ × k × T₀ × F × B
    denominator = ((4 * np.pi) ** 3
                   * range_m ** 4
                   * BOLTZMANN * STANDARD_TEMP
                   * params.noise_figure_linear
                   * params.bandwidth_hz)

    return numerator / denominator


def calculate_max_range(params: RadarParams, min_snr_db: float = 13.0) -> float:
    """计算最大探测距离 R_max。

    R_max 定义为 SNR = min_snr 时的距离。
    从雷达方程反解：
      R_max = [ (Pt * G² * λ² * σ * T) / ((4π)³ * k * T₀ * F * B * SNR_min) ] ^ (1/4)

    Args:
        params:      雷达参数
        min_snr_db:  最小可检测 SNR (dB)，默认 13 dB（经典值）

    Returns:
        最大探测距离 (m)
    """
    min_snr_linear = db_to_power(min_snr_db)

    numerator = (params.pt
                 * params.gain_linear ** 2
                 * params.wavelength_m ** 2
                 * params.target_rcs_m2
                 * params.pulse_width_s)

    denominator = ((4 * np.pi) ** 3
                   * BOLTZMANN * STANDARD_TEMP
                   * params.noise_figure_linear
                   * params.bandwidth_hz
                   * min_snr_linear)

    return (numerator / denominator) ** 0.25


def plot_snr_vs_range(params: RadarParams, output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")):
    """绘制 SNR 随距离变化的曲线。

    图中标注：
      - 目标距离处的 SNR 值
      - 最大探测距离（SNR = 13 dB 的位置）

    Args:
        params:      雷达参数
        output_dir:  图像输出目录
    """
    # 距离范围：从 1 km 到 R_max 的 1.5 倍
    r_max = calculate_max_range(params)
    ranges_km = np.linspace(1e3, r_max * 1.5, 1000)

    # 计算每个距离的 SNR（dB）
    snr_values = np.array([calculate_snr(params, r) for r in ranges_km])
    snr_db = power_to_db(snr_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # SNR 曲线
    ax.plot(ranges_km / 1e3, snr_db, "b-", linewidth=2, label="SNR vs 距离")

    # 最大探测距离线（SNR = 13 dB）
    ax.axhline(y=13, color="r", linestyle="--", alpha=0.7, label="最小可检测 SNR = 13 dB")
    ax.axvline(x=r_max / 1e3, color="r", linestyle="--", alpha=0.7)
    ax.annotate(
        f"R_max = {r_max / 1e3:.1f} km",
        xy=(r_max / 1e3, 13),
        xytext=(r_max / 1e3 + (r_max / 1e3) * 0.15, 20),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=12, color="red",
    )

    # 目标距离处的 SNR 标注
    target_snr_db = power_to_db(calculate_snr(params, params.target_range_m))
    ax.plot(params.target_range_m / 1e3, target_snr_db, "go", markersize=10)
    ax.annotate(
        f"目标: {params.target_range_m / 1e3:.0f} km\nSNR = {target_snr_db:.1f} dB",
        xy=(params.target_range_m / 1e3, target_snr_db),
        xytext=(params.target_range_m / 1e3 + (r_max / 1e3) * 0.15, target_snr_db + 5),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=11, color="green",
    )

    ax.set_xlabel("距离 (km)", fontsize=13)
    ax.set_ylabel("SNR (dB)", fontsize=13)
    ax.set_title(
        f"雷达方程：SNR vs 距离\n"
        f"Pt={params.pt / 1e6:.0f}MW, G={params.gain_db}dB, "
        f"f={params.freq_hz / 1e9:.1f}GHz, B={params.bandwidth_hz / 1e6:.0f}MHz, "
        f"σ={params.target_rcs_m2:.0f}m²",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "s01_snr_vs_range.png"), dpi=150, bbox_inches="tight")
    print(f"  图像已保存: {output_dir}/s01_snr_vs_range.png")
    plt.close(fig)


def validate(params: RadarParams) -> bool:
    """验证雷达方程计算的正确性。

    验证策略：
      用 calculate_snr() 和 calculate_max_range() 的输出，
      对比独立的理论计算（手工或用 numpy 直接代入公式）。
      容差设为 1%（相对误差），因为这些是确定性计算，不涉及随机性。

    物理判据：
      - 最大探测距离：R_max 应随 Pt^(1/4)、G^(1/2)、σ^(1/4) 变化
      - SNR 与 R⁴ 成反比：距离加倍，SNR 降 12.04 dB
      - SNR 与 Pt 成正比：功率加倍，SNR 升 3 dB
    """
    results = []

    # --- 验证 1：最大探测距离 ---
    # 独立计算（不用 calculate_max_range，用 numpy 直接代入雷达方程反解）
    snr_min = 20  # 13 dB = 20 倍
    r4_theory = (params.pt
                 * params.gain_linear ** 2
                 * params.wavelength_m ** 2
                 * params.target_rcs_m2
                 * params.pulse_width_s) / (
        (4 * np.pi) ** 3
        * BOLTZMANN * STANDARD_TEMP
        * params.noise_figure_linear
        * params.bandwidth_hz
        * snr_min
    )
    r_max_theory = r4_theory ** 0.25
    r_max_sim = calculate_max_range(params)
    results.append(verify(
        name="最大探测距离",
        theoretical=r_max_theory,
        simulated=r_max_sim,
        tolerance=r_max_theory * 0.001,  # 0.1% 容差
        unit="m",
    ))

    # --- 验证 2：目标距离处的 SNR ---
    snr_theory = (params.pt
                  * params.gain_linear ** 2
                  * params.wavelength_m ** 2
                  * params.target_rcs_m2
                  * params.pulse_width_s) / (
        (4 * np.pi) ** 3
        * params.target_range_m ** 4
        * BOLTZMANN * STANDARD_TEMP
        * params.noise_figure_linear
        * params.bandwidth_hz
    )
    snr_sim = calculate_snr(params, params.target_range_m)
    results.append(verify(
        name="SNR@目标距离",
        theoretical=snr_theory,
        simulated=snr_sim,
        tolerance=snr_theory * 0.001,
        unit="",
    ))

    # --- 验证 3：SNR 与 R⁴ 的关系 ---
    # 将距离加倍，SNR 应精确下降 12.04 dB（10*log10(2⁴) = 12.04 dB）
    r_test_1 = 20e3
    r_test_2 = 40e3
    snr_1 = power_to_db(calculate_snr(params, r_test_1))
    snr_2 = power_to_db(calculate_snr(params, r_test_2))
    snr_drop = snr_1 - snr_2
    results.append(verify(
        name="SNR R⁴衰减（20→40km）",
        theoretical=12.04,  # 10*log10(2⁴) = 12.04 dB
        simulated=snr_drop,
        tolerance=0.01,
        unit="dB",
    ))

    # --- 验证 4：功率加倍 → SNR +3 dB ---
    params_2x = RadarParams(
        pt=params.pt * 2,
        gain_db=params.gain_db,
        freq_hz=params.freq_hz,
        bandwidth_hz=params.bandwidth_hz,
        pulse_width_s=params.pulse_width_s,
        prf_hz=params.prf_hz,
        noise_figure_db=params.noise_figure_db,
        target_range_m=params.target_range_m,
        target_rcs_m2=params.target_rcs_m2,
    )
    snr_base = power_to_db(calculate_snr(params, params.target_range_m))
    snr_2x = power_to_db(calculate_snr(params_2x, params.target_range_m))
    results.append(verify(
        name="功率加倍→SNR+3dB",
        theoretical=3.01,  # 10*log10(2) = 3.01 dB
        simulated=snr_2x - snr_base,
        tolerance=0.01,
        unit="dB",
    ))

    return print_validation("s01 雷达方程", results)


def main():
    """运行 s01 雷达方程仿真与验证。"""
    print("=" * 60)
    print("s01：雷达方程与 SNR 计算")
    print("=" * 60)

    # S 波段远程搜索雷达参数（演示用，探测距离更直观）
    # 典型场景：搜索雷达探测中型飞机（RCS≈5m²）
    params = RadarParams(
        pt=2e6,             # 峰值功率 2 MW
        gain_db=40.0,       # 天线增益 40 dB（10000 倍）
        freq_hz=3e9,        # S 波段 3 GHz（波长 10 cm）
        bandwidth_hz=5e6,   # 带宽 5 MHz（距离分辨率 30 m）
        pulse_width_s=200e-6,  # 脉宽 200 μs（长脉冲，高能量）
        prf_hz=500,         # PRF 500 Hz（不模糊距离 300 km）
        noise_figure_db=3.0,   # 噪声系数 3 dB
        target_range_m=30e3,   # 目标距离 30 km
        target_rcs_m2=5.0,     # 目标 RCS 5 m²（中型飞机）
    )

    print(f"\n雷达参数（S 波段搜索雷达）:")
    print(f"  峰值功率 Pt = {params.pt / 1e6:.0f} MW")
    print(f"  天线增益 G  = {params.gain_db} dB ({params.gain_linear:.0f} 倍)")
    print(f"  载波频率 f  = {params.freq_hz / 1e9:.0f} GHz (波长 λ = {params.wavelength_m * 100:.0f} cm)")
    print(f"  信号带宽 B  = {params.bandwidth_hz / 1e6:.0f} MHz")
    print(f"  脉冲宽度 T  = {params.pulse_width_s * 1e6:.0f} μs")
    print(f"  噪声系数 F  = {params.noise_figure_db} dB")
    print(f"  目标距离 R  = {params.target_range_m / 1e3:.0f} km")
    print(f"  目标 RCS  σ = {params.target_rcs_m2:.0f} m²")

    # 噪声功率
    noise_power_w = params.noise_power_w
    print(f"\n中间量:")
    print(f"  噪声功率 Pn  = k·T₀·F·B = {noise_power_w:.2e} W ({power_to_db(noise_power_w):.1f} dBW)")
    print(f"  距离分辨率 ΔR = c/(2B) = {params.range_resolution_m:.1f} m")
    print(f"  不模糊距离    = {params.max_unambiguous_range_m / 1e3:.0f} km")

    # 计算并显示结果
    r_max = calculate_max_range(params)
    snr_target = power_to_db(calculate_snr(params, params.target_range_m))
    print(f"\n计算结果:")
    print(f"  最大探测距离  = {r_max / 1e3:.1f} km (SNR_min = 13 dB)")
    print(f"  目标处 SNR    = {snr_target:.1f} dB")
    print(f"  距离余量      = {r_max / params.target_range_m:.1f} 倍")

    # 绘制 SNR vs 距离曲线
    print(f"\n绘制 SNR vs 距离曲线...")
    plot_snr_vs_range(params)

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(params)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
