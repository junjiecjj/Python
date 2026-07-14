"""s07：时间/频率同步误差对相参积累性能的影响。

验证目标：
  - 量化时间同步误差 Δt 引入的相位误差对相干因子的影响
  - 量化频率同步误差 Δf 引入的相位累积对相干因子的影响
  - 参数化扫描误差空间，绘制 SNR 损失热力图

核心物理模型：
  时间同步误差 Δt → 相位误差 σ_φ = 2π·f_c·Δt（f_c 为载波频率）
  频率同步误差 Δf → 相位误差随 CPI 累积：σ_φ = 2π·Δf·T_int
  相干因子 ρ = exp(-σ_φ²/2)（幅度相干因子）
  SNR 损失 = -10·log10(|ρ|²) = 10·log10(e)·σ_φ² ≈ 4.343·σ_φ² dB

  两种误差独立时，总相位方差 σ_φ_total² = σ_φ_time² + σ_φ_freq²

X 波段（10 GHz）精度要求：
  - 时间同步 < 0.1 ns（实际要求更严，约 0.013 ns 才能保证 < 3 dB 损失）
  - 频率同步 < 0.1 Hz
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
from lib.validation import verify, verify_relative, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 核心计算函数
# ---------------------------------------------------------------------------


def compute_coherent_gain(
    time_error_ns: float,
    freq_error_hz: float,
    carrier_freq_hz: float,
    integration_time_s: float,
) -> tuple[float, float]:
    """计算给定同步误差下的相干因子和 SNR 损失。

    物理模型：
      时间误差引入固定相位偏差：σ_φ_t = 2π·f_c·Δt
      频率误差在 CPI 内线性累积相位：σ_φ_f = 2π·Δf·T_int
      两者的总相位方差：σ_φ² = σ_φ_t² + σ_φ_f²
      幅度相干因子：ρ = exp(-σ_φ²/2)
      SNR 损失：L = -10·log10(|ρ|²) = 10·log10(e)·σ_φ² ≈ 4.343·σ_φ² dB

    Args:
        time_error_ns:     时间同步误差 (ns)
        freq_error_hz:     频率同步误差 (Hz)
        carrier_freq_hz:   载波频率 (Hz)
        integration_time_s:相参积累时间 (s)

    Returns:
        (coherent_factor, snr_loss_db)
        coherent_factor: 相干因子 ρ ∈ [0, 1]，1 表示完美相参
        snr_loss_db:     SNR 损失 (dB)，0 表示无损失
    """
    # 时间误差 → 相位误差 (rad)
    time_error_s = time_error_ns * 1e-9
    phase_err_time = 2.0 * np.pi * carrier_freq_hz * time_error_s

    # 频率误差 → 累积相位误差 (rad)
    phase_err_freq = 2.0 * np.pi * freq_error_hz * integration_time_s

    # 总相位方差（两种误差独立，方差相加）
    sigma_phi_sq = phase_err_time**2 + phase_err_freq**2

    # 幅度相干因子 ρ = exp(-σ_φ²/2)
    # SNR 损失 = -10·log10(|ρ|²) = -10·log10(exp(-σ_φ²)) = 10·log10(e)·σ_φ²
    coherent_factor = np.exp(-sigma_phi_sq / 2.0)

    # SNR 损失 (dB)
    # 使用 epsilon 避免 log(0)
    snr_loss_db = -10.0 * np.log10(max(coherent_factor**2, 1e-40))

    return coherent_factor, snr_loss_db


def sweep_sync_errors(
    carrier_freq_hz: float,
    time_errors_ns: np.ndarray,
    freq_errors_hz: np.ndarray,
    integration_time_s: float,
) -> np.ndarray:
    """对时间/频率误差参数空间做二维扫描。

    Args:
        carrier_freq_hz:    载波频率 (Hz)
        time_errors_ns:     时间误差数组 (ns)
        freq_errors_hz:     频率误差数组 (Hz)
        integration_time_s: 相参积累时间 (s)

    Returns:
        snr_loss_map: 2D 数组，shape = (len(freq_errors_hz), len(time_errors_ns))
                      行对应频率误差，列对应时间误差
    """
    n_freq = len(freq_errors_hz)
    n_time = len(time_errors_ns)
    snr_loss_map = np.zeros((n_freq, n_time))

    for i, df in enumerate(freq_errors_hz):
        for j, dt in enumerate(time_errors_ns):
            _, loss_db = compute_coherent_gain(
                dt, df, carrier_freq_hz, integration_time_s
            )
            snr_loss_map[i, j] = loss_db

    return snr_loss_map


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def plot_sync_impact(
    carrier_freq_hz: float,
    time_errors_ns: np.ndarray,
    freq_errors_hz: np.ndarray,
    integration_time_s: float,
    snr_loss_map: np.ndarray,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制同步误差对 SNR 损失的三子图。

    子图布局：
      (1) 时间误差 vs SNR 损失曲线（频率误差 = 0）
      (2) 频率误差 vs SNR 损失曲线（时间误差 = 0）
      (3) 时间×频率误差二维热力图

    Args:
        carrier_freq_hz:    载波频率 (Hz)
        time_errors_ns:     时间误差数组 (ns)
        freq_errors_hz:     频率误差数组 (Hz)
        integration_time_s: 相参积累时间 (s)
        snr_loss_map:       sweep_sync_errors 返回的 2D 数组
        output_dir:         图像输出目录
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- 子图 1：时间误差 vs SNR 损失（频率误差 = 0）---
    ax1 = axes[0]
    snr_vs_time = np.array([
        compute_coherent_gain(dt, 0.0, carrier_freq_hz, integration_time_s)[1]
        for dt in time_errors_ns
    ])
    ax1.semilogx(time_errors_ns, snr_vs_time, "b-", linewidth=2)
    ax1.axhline(y=3.0, color="r", linestyle="--", alpha=0.7, label="3 dB 损失门限")
    # 标注 3 dB 交叉点
    idx_3db = np.where(snr_vs_time >= 3.0)[0]
    if len(idx_3db) > 0:
        dt_3db = time_errors_ns[idx_3db[0]]
        ax1.axvline(x=dt_3db, color="g", linestyle=":", alpha=0.7)
        ax1.annotate(
            f"Δt = {dt_3db:.3f} ns",
            xy=(dt_3db, 3.0),
            xytext=(dt_3db * 3, 8),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=10, color="green",
        )
    ax1.set_xlabel("时间同步误差 Δt (ns)", fontsize=12)
    ax1.set_ylabel("SNR 损失 (dB)", fontsize=12)
    ax1.set_title("时间同步误差 vs SNR 损失\n(频率误差 = 0)", fontsize=12)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- 子图 2：频率误差 vs SNR 损失（时间误差 = 0）---
    ax2 = axes[1]
    snr_vs_freq = np.array([
        compute_coherent_gain(0.0, df, carrier_freq_hz, integration_time_s)[1]
        for df in freq_errors_hz
    ])
    ax2.semilogx(freq_errors_hz, snr_vs_freq, "r-", linewidth=2)
    ax2.axhline(y=3.0, color="r", linestyle="--", alpha=0.7, label="3 dB 损失门限")
    idx_3db_f = np.where(snr_vs_freq >= 3.0)[0]
    if len(idx_3db_f) > 0:
        df_3db = freq_errors_hz[idx_3db_f[0]]
        ax2.axvline(x=df_3db, color="g", linestyle=":", alpha=0.7)
        ax2.annotate(
            f"Δf = {df_3db:.2f} Hz",
            xy=(df_3db, 3.0),
            xytext=(df_3db * 3, 8),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=10, color="green",
        )
    ax2.set_xlabel("频率同步误差 Δf (Hz)", fontsize=12)
    ax2.set_ylabel("SNR 损失 (dB)", fontsize=12)
    ax2.set_title(
        f"频率同步误差 vs SNR 损失\n(T_int = {integration_time_s} s)",
        fontsize=12,
    )
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # --- 子图 3：二维热力图 ---
    ax3 = axes[2]
    # 使用对数坐标显示 SNR 损失（限制最大显示值为 30 dB）
    display_map = np.clip(snr_loss_map, 0, 30)
    im = ax3.pcolormesh(
        time_errors_ns,
        freq_errors_hz,
        display_map,
        shading="auto",
        cmap="hot_r",
        vmin=0,
        vmax=30,
    )
    fig.colorbar(im, ax=ax3, label="SNR 损失 (dB)")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("时间同步误差 Δt (ns)", fontsize=12)
    ax3.set_ylabel("频率同步误差 Δf (Hz)", fontsize=12)
    ax3.set_title(
        f"同步误差 SNR 损失热力图\n"
        f"(f_c = {carrier_freq_hz / 1e9:.0f} GHz, T_int = {integration_time_s} s)",
        fontsize=12,
    )
    # 绘制 3 dB 等高线
    ax3.contour(
        time_errors_ns,
        freq_errors_hz,
        snr_loss_map,
        levels=[3.0],
        colors="cyan",
        linewidths=2,
        linestyles="--",
    )

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "s07_sync_error_impact.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  图像已保存: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def validate(
    carrier_freq_hz: float,
    integration_time_s: float,
) -> bool:
    """验证同步误差模型的正确性。

    验证项：
      a. 零误差 → SNR 损失 = 0 dB（无误差时应完全相参）
      b. 小误差近似：SNR_loss ≈ 4.343·σ_φ² dB（相位误差 < 0.1 rad 时容差 10%）
      c. X 波段临界值：Δt = 0.1 ns 时使用小误差近似公式 SNR 损失 < 3 dB
         （精确公式下 σ_φ = 2π ≈ 6.28 rad 已超出近似范围，此处验证近似公式的适用边界）

    Args:
        carrier_freq_hz:    载波频率 (Hz)
        integration_time_s: 相参积累时间 (s)

    Returns:
        全部通过返回 True
    """
    results = []

    # --- 验证 a：零误差 → SNR 损失 = 0 dB ---
    coh_zero, loss_zero = compute_coherent_gain(
        0.0, 0.0, carrier_freq_hz, integration_time_s
    )
    results.append(verify(
        name="零误差→相干因子=1",
        theoretical=1.0,
        simulated=coh_zero,
        tolerance=1e-12,
    ))
    results.append(verify(
        name="零误差→SNR损失=0dB",
        theoretical=0.0,
        simulated=loss_zero,
        tolerance=1e-10,
        unit="dB",
    ))

    # --- 验证 b：小误差近似 SNR_loss ≈ 4.343·σ_φ² dB ---
    # 使用 σ_φ ≈ 0.05 rad 的小误差（时间误差约 0.0008 ns @ 10 GHz）
    sigma_phi_small = 0.05  # rad
    dt_small_ns = sigma_phi_small / (2.0 * np.pi * carrier_freq_hz) * 1e9
    _, loss_exact = compute_coherent_gain(
        dt_small_ns, 0.0, carrier_freq_hz, integration_time_s
    )
    loss_approx = 4.343 * sigma_phi_small**2
    results.append(verify_relative(
        name="小误差近似(σ_φ=0.05rad)",
        theoretical=loss_approx,
        simulated=loss_exact,
        rel_tolerance=0.10,  # 10% 容差
        unit="dB",
    ))

    # 再用 σ_φ ≈ 0.1 rad 验证一次
    sigma_phi_small2 = 0.1  # rad
    dt_small2_ns = sigma_phi_small2 / (2.0 * np.pi * carrier_freq_hz) * 1e9
    _, loss_exact2 = compute_coherent_gain(
        dt_small2_ns, 0.0, carrier_freq_hz, integration_time_s
    )
    loss_approx2 = 4.343 * sigma_phi_small2**2
    results.append(verify_relative(
        name="小误差近似(σ_φ=0.1rad)",
        theoretical=loss_approx2,
        simulated=loss_exact2,
        rel_tolerance=0.10,
        unit="dB",
    ))

    # --- 验证 c：X 波段 Δt = 0.1 ns 的 SNR 损失 ---
    # σ_φ = 2π·f_c·Δt = 2π·10e9·0.1e-9 = 2π ≈ 6.283 rad
    # SNR_loss = 10·log10(e)·σ_φ² = 4.343·(2π)² ≈ 171.4 dB
    # 远超 3 dB 门限，说明 X 波段对时间同步要求极高
    sigma_phi_01ns = 2.0 * np.pi * carrier_freq_hz * 0.1e-9
    _, loss_01ns = compute_coherent_gain(
        0.1, 0.0, carrier_freq_hz, integration_time_s
    )
    loss_01ns_theory = 4.343 * sigma_phi_01ns**2
    results.append(verify(
        name="Δt=0.1ns精确SNR损失",
        theoretical=loss_01ns_theory,
        simulated=loss_01ns,
        tolerance=loss_01ns_theory * 0.01,  # 1% 容差
        unit="dB",
    ))

    # 验证 3 dB 门限对应的临界时间误差
    # 3 = 4.343 · (2π·f_c·Δt)² → Δt = sqrt(3/4.343) / (2π·f_c)
    dt_crit_s = np.sqrt(3.0 / 4.343) / (2.0 * np.pi * carrier_freq_hz)
    dt_crit_ns = dt_crit_s * 1e9
    _, loss_crit = compute_coherent_gain(
        dt_crit_ns, 0.0, carrier_freq_hz, integration_time_s
    )
    results.append(verify(
        name="3dB临界时间误差",
        theoretical=3.0,
        simulated=loss_crit,
        tolerance=0.01,
        unit="dB",
    ))

    # 验证 3 dB 门限对应的临界频率误差
    df_crit_hz = np.sqrt(3.0 / 4.343) / (2.0 * np.pi * integration_time_s)
    _, loss_crit_f = compute_coherent_gain(
        0.0, df_crit_hz, carrier_freq_hz, integration_time_s
    )
    results.append(verify(
        name="3dB临界频率误差",
        theoretical=3.0,
        simulated=loss_crit_f,
        tolerance=0.01,
        unit="dB",
    ))

    # 验证两种误差的方差可加性
    dt_mix_ns = 0.01
    df_mix_hz = 1.0
    _, loss_mix = compute_coherent_gain(
        dt_mix_ns, df_mix_hz, carrier_freq_hz, integration_time_s
    )
    _, loss_t = compute_coherent_gain(
        dt_mix_ns, 0.0, carrier_freq_hz, integration_time_s
    )
    _, loss_f = compute_coherent_gain(
        0.0, df_mix_hz, carrier_freq_hz, integration_time_s
    )
    # 方差可加：σ_total² = σ_t² + σ_f²
    # 即 loss_mix = loss_t + loss_f（dB 域相加，因为 SNR_loss ∝ σ_φ²）
    loss_mix_theory = loss_t + loss_f
    results.append(verify(
        name="方差可加性",
        theoretical=loss_mix_theory,
        simulated=loss_mix,
        tolerance=loss_mix_theory * 0.01,
        unit="dB",
    ))

    return print_validation("s07 同步误差影响", results)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> int:
    """运行 s07 同步误差影响仿真与验证。"""
    print("=" * 60)
    print("s07：时间/频率同步误差对相参积累性能的影响")
    print("=" * 60)

    # X 波段参数
    carrier_freq_hz = 10e9       # 10 GHz
    integration_time_s = 0.1     # CPI 时间 0.1 s

    print(f"\n系统参数:")
    print(f"  载波频率    f_c  = {carrier_freq_hz / 1e9:.0f} GHz (X 波段)")
    print(f"  积累时间    T_int = {integration_time_s} s")
    print(f"  波长        λ    = {3e8 / carrier_freq_hz * 100:.1f} cm")

    # 计算临界误差值
    dt_crit_ns = np.sqrt(3.0 / 4.343) / (2.0 * np.pi * carrier_freq_hz) * 1e9
    df_crit_hz = np.sqrt(3.0 / 4.343) / (2.0 * np.pi * integration_time_s)
    print(f"\n3 dB 损失临界值:")
    print(f"  时间误差: Δt = {dt_crit_ns:.4f} ns")
    print(f"  频率误差: Δf = {df_crit_hz:.4f} Hz")

    # 参数扫描范围
    time_errors_ns = np.logspace(-2, 1, 200)    # 0.01 ~ 10 ns
    freq_errors_hz = np.logspace(-2, 2, 200)    # 0.01 ~ 100 Hz

    print(f"\n参数扫描:")
    print(f"  时间误差: {time_errors_ns[0]:.2f} ~ {time_errors_ns[-1]:.0f} ns ({len(time_errors_ns)} 点)")
    print(f"  频率误差: {freq_errors_hz[0]:.2f} ~ {freq_errors_hz[-1]:.0f} Hz ({len(freq_errors_hz)} 点)")

    # 二维扫描
    print(f"\n执行二维误差扫描...")
    snr_loss_map = sweep_sync_errors(
        carrier_freq_hz, time_errors_ns, freq_errors_hz, integration_time_s
    )

    # 统计摘要
    print(f"\n扫描结果摘要:")
    print(f"  SNR 损失范围: {snr_loss_map.min():.2f} ~ {snr_loss_map.max():.1f} dB")

    # 绘图
    print(f"\n绘制同步误差影响图...")
    plot_sync_impact(
        carrier_freq_hz,
        time_errors_ns,
        freq_errors_hz,
        integration_time_s,
        snr_loss_map,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(carrier_freq_hz, integration_time_s)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
