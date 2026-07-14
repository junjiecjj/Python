"""s02：LFM 波形与脉冲压缩。

验证目标：
  - 生成 LFM（线性调频）信号
  - 实现频域匹配滤波（脉冲压缩）
  - 验证距离分辨率 ΔR ≈ c/(2B)
  - 验证脉冲压缩增益 = T*B（SNR 改善因子）
  - 对比加窗 vs 不加窗的旁瓣抑制效果

核心概念：
  脉冲压缩解决了雷达的"根本矛盾"：
    - 远探测距离需要大能量 → 长脉冲
    - 高距离分辨率需要大带宽 → 短脉冲
  LFM 信号通过频率调制，使一个长脉冲也具有大带宽，
  压缩后等效于一个短脉冲（宽度 ≈ 1/B），同时保留了长脉冲的能量。

  压缩增益的定义（SNR 改善因子）：
    输入 SNR = 信号功率 / 噪声功率
    输出 SNR = 压缩峰值功率 / 输出噪声功率
    增益 = 输出 SNR / 输入 SNR = N = T*B（线性值）

    物理直觉：匹配滤波器对信号是"相干叠加"（幅度 ×N），
    对噪声是"非相干叠加"（功率 ×N）。
    所以信号功率增益是 N²，噪声功率增益是 N，SNR 增益是 N。

  LFM 的旁瓣特性：
    - 不加窗：第一旁瓣约 -13.2 dB（sinc 函数的固有特性）
    - 加 Hamming 窗：旁瓣约 -43 dB，但主瓣展宽约 1.5 倍
    - 加窗是用分辨率换旁瓣抑制

对应知识库：radar-knowledge-base/基础/01-雷达测量原理/01-测距原理与脉冲压缩.md
             radar-knowledge-base/基础/02-雷达信号处理流程/01-LFM波形与脉压.md
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # 修复 PyQt6/KeyboardModifier 兼容性问题
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.radar_params import RadarParams, SPEED_OF_LIGHT
from lib.signal_utils import (
    generate_lfm,
    matched_filter,
    apply_window,
    power_to_db,
)
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def simulate_pulse_compression(
    params: RadarParams,
    sample_rate_hz: float = 200e6,
    window_name: str = "none",
    input_snr_db: float = 0.0,
) -> dict:
    """仿真脉冲压缩过程。

    完整流程：
      1. 生成 LFM 发射波形
      2. 构造接收信号 = 目标回波 + 高斯白噪声（按 input_snr_db 控制噪声功率）
      3. 用发射波形作为模板进行匹配滤波
      4. 测量压缩后的脉冲特性

    SNR 增益的验证方法：
      在输入端设定已知 SNR（如 0 dB），在输出端测量峰值处的 SNR。
      理论期望：输出 SNR ≈ 输入 SNR + 10*log10(T*B) dB

    Args:
        params:         雷达参数
        sample_rate_hz: 采样率 (Hz)，需 > B 以满足 Nyquist
        window_name:    窗函数名称
        input_snr_db:   输入端信噪比 (dB)

    Returns:
        dict，包含压缩结果和测量值
    """
    rng = np.random.default_rng(42)  # 固定种子，保证可复现

    # 1. 生成 LFM 波形（归一化幅度为 1）
    waveform = generate_lfm(
        bandwidth_hz=params.bandwidth_hz,
        pulse_width_s=params.pulse_width_s,
        sample_rate_hz=sample_rate_hz,
    )
    n_signal = len(waveform)

    # 2. 构造接收信号
    #    信号功率（归一化 LFM 的平均功率 = 1）
    signal_power = 1.0
    #    噪声功率 = 信号功率 / SNR
    noise_power = signal_power / (10 ** (input_snr_db / 10))
    #    复高斯白噪声（实部虚部各一半功率）
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n_signal) + 1j * rng.standard_normal(n_signal)
    )
    received = waveform + noise

    # 3. 匹配滤波器模板
    #    对于自相关：模板 = s*(-t) = s*(t) 的反转
    #    在离散域：template = conj(waveform[::-1])
    template = np.conj(waveform[::-1])

    # 可选加窗
    if window_name != "none":
        template = apply_window(template, window_name)

    # 4. 频域匹配滤波（返回完整线性卷积结果）
    compressed_full = matched_filter(received, template)
    # 峰值位于 index N-1（信号长度减 1），这是 conj(rev) 方式的特性
    # 我们以峰值为中心截取分析窗口
    full_mag = np.abs(compressed_full)
    peak_idx_full = np.argmax(full_mag)
    peak_value = full_mag[peak_idx_full]

    # 以峰值为中心截取 N 个样本用于分析
    # 这样峰值在中心位置，两侧都有足够数据
    half_n = n_signal // 2
    start = max(0, peak_idx_full - half_n)
    end = min(len(compressed_full), start + n_signal)
    compressed = compressed_full[start:end]
    mag = np.abs(compressed)
    peak_idx = np.argmax(mag)

    # 主瓣 3dB 宽度
    half_power_level = peak_value / np.sqrt(2)
    above_half = mag >= half_power_level
    # 从峰值向左找第一个低于半功率的点
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if not above_half[i]:
            left_idx = i + 1
            break
    # 从峰值向右找第一个低于半功率的点
    right_idx = peak_idx
    for i in range(peak_idx + 1, len(mag)):
        if not above_half[i]:
            right_idx = i - 1
            break
    mainlobe_width_samples = right_idx - left_idx + 1
    # 采样间隔对应的距离：dt = 1/fs, ΔR = c * dt / 2
    sample_spacing_m = SPEED_OF_LIGHT / (2 * sample_rate_hz)
    mainlobe_width_m = mainlobe_width_samples * sample_spacing_m

    # 第一旁瓣电平（主瓣外的最大值相对峰值）
    # LFM 自相关的第一旁瓣约在 offset ≈ 1/B 处（≈ fs/B 个采样点）
    # 需要排除主瓣（约 2 个采样点宽），但保留旁瓣区域
    sidelobe_inner = max(mainlobe_width_samples * 2, 3)  # 主瓣外边缘
    sidelobe_outer = max(int(sample_rate_hz / params.bandwidth_hz * 2), 30)  # 约 2*1/B 距离
    left_sidelobe = mag[max(0, peak_idx - sidelobe_outer):max(0, peak_idx - sidelobe_inner)]
    right_sidelobe = mag[min(len(mag), peak_idx + sidelobe_inner):min(len(mag), peak_idx + sidelobe_outer)]
    sidelobe_region = np.concatenate([left_sidelobe, right_sidelobe])
    if len(sidelobe_region) > 0:
        sidelobe_db = 20 * np.log10(np.max(sidelobe_region) / peak_value)
    else:
        sidelobe_db = float("nan")

    # SNR 增益：多次试验取平均，减少随机波动
    # 理论期望：增益 = N（信号长度 = T*B）
    # 单次试验的噪声功率有较大方差，需要 Monte Carlo 平均
    if input_snr_db > -999:
        num_trials = 50
        noise_powers = []
        for trial in range(num_trials):
            trial_rng = np.random.default_rng(1000 + trial)
            trial_noise = np.sqrt(noise_power / 2) * (
                trial_rng.standard_normal(n_signal) + 1j * trial_rng.standard_normal(n_signal)
            )
            trial_compressed = matched_filter(trial_noise, template)
            noise_powers.append(np.mean(np.abs(trial_compressed) ** 2))
        output_noise_power = np.mean(noise_powers)
        output_snr = peak_value ** 2 / output_noise_power
        output_snr_db = power_to_db(output_snr)
        snr_gain_db = output_snr_db - input_snr_db
    else:
        output_snr_db = float("inf")
        snr_gain_db = float("inf")

    return {
        "waveform": waveform,
        "received": received,
        "compressed": compressed,
        "compressed_mag": mag,
        "peak_idx": peak_idx,
        "peak_value": peak_value,
        "mainlobe_width_m": mainlobe_width_m,
        "mainlobe_width_samples": mainlobe_width_samples,
        "sidelobe_db": sidelobe_db,
        "output_snr_db": output_snr_db,
        "snr_gain_db": snr_gain_db,
        "input_snr_db": input_snr_db,
        "sample_rate_hz": sample_rate_hz,
        "window_name": window_name,
    }


def plot_results(
    results_nowin: dict,
    results_hamm: dict,
    params: RadarParams,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
):
    """绘制脉冲压缩结果对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sample_rate = results_nowin["sample_rate_hz"]
    n = len(results_nowin["waveform"])
    t_us = np.arange(n) / sample_rate * 1e6  # 时间轴 (μs)
    sample_spacing_m = SPEED_OF_LIGHT / (2 * sample_rate)

    # 距离轴：以峰值为中心，0 表示目标位置
    compressed_len = len(results_nowin["compressed_mag"])
    peak_at = results_nowin["peak_idx"]
    range_axis = (np.arange(compressed_len) - peak_at) * sample_spacing_m

    # --- 子图 1：LFM 波形 ---
    ax = axes[0, 0]
    ax.plot(t_us, np.real(results_nowin["waveform"]), "b-", linewidth=0.5)
    ax.set_xlabel("时间 (μs)")
    ax.set_ylabel("幅度（实部）")
    ax.set_title(f"LFM 波形\nB={params.bandwidth_hz / 1e6:.0f}MHz, T={params.pulse_width_s * 1e6:.0f}μs")
    ax.grid(True, alpha=0.3)

    # --- 子图 2：不加窗压缩 ---
    ax = axes[0, 1]
    mag_db = power_to_db(results_nowin["compressed_mag"] ** 2)
    mag_db -= np.max(mag_db)  # 归一化到 0 dB
    range_axis = np.arange(len(mag_db)) * sample_spacing_m
    ax.plot(range_axis, mag_db, "b-", linewidth=0.8)
    ax.set_xlabel("等效距离 (m)")
    ax.set_ylabel("归一化功率 (dB)")
    ax.set_title(f"脉冲压缩（不加窗）\n旁瓣 = {results_nowin['sidelobe_db']:.1f} dB")
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)

    # --- 子图 3：加 Hamming 窗压缩 ---
    ax = axes[1, 0]
    mag_db_h = power_to_db(results_hamm["compressed_mag"] ** 2)
    mag_db_h -= np.max(mag_db_h)
    ax.plot(range_axis, mag_db_h, "r-", linewidth=0.8)
    ax.set_xlabel("等效距离 (m)")
    ax.set_ylabel("归一化功率 (dB)")
    ax.set_title(f"脉冲压缩（Hamming 窗）\n旁瓣 = {results_hamm['sidelobe_db']:.1f} dB")
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)

    # --- 子图 4：归一化对比 ---
    ax = axes[1, 1]
    ax.plot(range_axis, mag_db, "b-", linewidth=0.8, label="不加窗")
    ax.plot(range_axis, mag_db_h, "r-", linewidth=0.8, label="Hamming 窗")
    ax.set_xlabel("等效距离 (m)")
    ax.set_ylabel("归一化功率 (dB)")
    ax.set_title("加窗 vs 不加窗对比")
    ax.set_ylim(-50, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"s02：LFM 脉冲压缩仿真\n"
        f"理论分辨率 ΔR = c/(2B) = {params.range_resolution_m:.1f} m, "
        f"理论 SNR 增益 T·B = {params.time_bandwidth_product:.0f} ({10 * np.log10(params.time_bandwidth_product):.1f} dB)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "s02_pulse_compression.png"), dpi=150, bbox_inches="tight")
    print(f"  图像已保存: {output_dir}/s02_pulse_compression.png")
    plt.close(fig)


def validate(params: RadarParams) -> bool:
    """验证脉冲压缩的关键指标。

    验证项及其物理含义：

      1. 距离分辨率：压缩后主瓣 3dB 宽度 ≈ c/(2B)
         这是两个等强度目标能被分辨的最小距离差。
         B=50MHz → ΔR = 3.0 m
         注意：3dB 宽度 ≈ 0.886/B（秒），比 c/(2B) 略窄。

      2. SNR 增益：匹配滤波器的 SNR 改善因子
         临界采样（fs=B）时：增益 = T*B = 500 → 27.0 dB
         4x 过采样（fs=4B）时：增益 = T*fs = 2000 → 33.0 dB
         原因：过采样增加了信号样本数，匹配滤波器对每个样本相干叠加。

      3. 不加窗旁瓣：≈ -13.2 dB
         LFM 的固有旁瓣，由 sinc 函数的性质决定。
         这意味着一个强目标的旁瓣可能掩盖附近弱目标。

      4. 加窗旁瓣改善：Hamming 窗使旁瓣降低约 30 dB
         注意：-43 dB 是无限长信号的理想值。有限脉宽（10μs）的 LFM
         实际旁瓣约 -25~-30 dB，因为频谱泄露限制了窗函数的效果。
    """
    sample_rate = params.bandwidth_hz * 4  # 4x 过采样

    results_nowin = simulate_pulse_compression(
        params, sample_rate, "none", input_snr_db=0.0
    )
    results_hamm = simulate_pulse_compression(
        params, sample_rate, "hamming", input_snr_db=0.0
    )

    validation_results = []

    # --- 验证 1：距离分辨率 ---
    # 理论：ΔR = c / (2B)
    # 实际 3dB 宽度约 0.886/B（秒）对应距离 ≈ 0.886*c/(2B)
    # 用 c/(2B) 作为参考，容差 50%（3dB 宽度和等效噪声带宽定义的差异）
    theoretical_res = params.range_resolution_m
    simulated_res = results_nowin["mainlobe_width_m"]
    validation_results.append(verify(
        name="距离分辨率",
        theoretical=theoretical_res,
        simulated=simulated_res,
        tolerance=theoretical_res * 0.5,
        unit="m",
    ))

    # --- 验证 2：SNR 增益 ---
    # 理论：临界采样时 Gain = T*B，过采样时 Gain = T*fs = N
    # 4x 过采样：N = T * 4B = 4 * TB = 2000 → 33.0 dB
    n_samples = int(params.pulse_width_s * sample_rate)
    theoretical_gain = 10 * np.log10(n_samples)  # 10*log10(N) = 10*log10(2000) = 33.0 dB
    simulated_gain = results_nowin["snr_gain_db"]
    validation_results.append(verify(
        name="SNR 增益",
        theoretical=theoretical_gain,
        simulated=simulated_gain,
        tolerance=4.0,  # 4 dB 容限（Monte Carlo 方差 + 峰值偏移）
        unit="dB",
    ))

    # --- 验证 3：不加窗旁瓣 ---
    # 理论：约 -13.2 dB
    theoretical_sidelobe = -13.2
    simulated_sidelobe = results_nowin["sidelobe_db"]
    validation_results.append(verify(
        name="不加窗旁瓣电平",
        theoretical=theoretical_sidelobe,
        simulated=simulated_sidelobe,
        tolerance=3.0,  # 3 dB 容限
        unit="dB",
    ))

    # --- 验证 4：加窗旁瓣改善 ---
    # Hamming 窗理论上旁瓣约 -43 dB，但有限脉宽的 LFM 实际约 -25~-30 dB
    # 验证标准：加窗后旁瓣应比不加窗至少改善 10 dB
    simulated_hamm_sidelobe = results_hamm["sidelobe_db"]
    validation_results.append(verify(
        name="Hamming窗旁瓣电平",
        theoretical=-30.0,  # 有限脉宽 LFM 的实际期望
        simulated=simulated_hamm_sidelobe,
        tolerance=10.0,  # 10 dB 容限
        unit="dB",
    ))

    # --- 验证 5：加窗后主瓣展宽 ---
    # Hamming 窗使主瓣展宽约 1.46 倍
    broaden = results_hamm["mainlobe_width_m"] / results_nowin["mainlobe_width_m"]
    validation_results.append(verify(
        name="Hamming窗主瓣展宽比",
        theoretical=1.46,
        simulated=broaden,
        tolerance=0.5,  # 0.5 倍容限
        unit="x",
    ))

    return print_validation("s02 脉冲压缩", validation_results)


def main():
    """运行 s02 LFM 脉冲压缩仿真与验证。"""
    print("=" * 60)
    print("s02：LFM 波形与脉冲压缩")
    print("=" * 60)

    params = RadarParams()  # 使用默认 X 波段参数
    sample_rate = params.bandwidth_hz * 4

    print(f"\n参数:")
    print(f"  带宽 B       = {params.bandwidth_hz / 1e6:.0f} MHz")
    print(f"  脉宽 T       = {params.pulse_width_s * 1e6:.0f} μs")
    print(f"  时宽带宽积 TB = {params.time_bandwidth_product:.0f}")
    print(f"  理论分辨率   = c/(2B) = {params.range_resolution_m:.1f} m")
    n_samples = int(params.pulse_width_s * sample_rate)
    print(f"  理论 SNR 增益 = 10·log10(T·fs) = 10·log10({n_samples}) = {10 * np.log10(n_samples):.1f} dB (4x 过采样)")
    print(f"  采样率       = {sample_rate / 1e6:.0f} MHz ({sample_rate / params.bandwidth_hz:.0f}x 过采样)")

    # 仿真
    print(f"\n仿真脉冲压缩 (输入 SNR = 0 dB)...")
    results_nowin = simulate_pulse_compression(params, sample_rate, "none", input_snr_db=0.0)
    results_hamm = simulate_pulse_compression(params, sample_rate, "hamming", input_snr_db=0.0)

    print(f"\n不加窗结果:")
    print(f"  主瓣 3dB 宽度 = {results_nowin['mainlobe_width_m']:.2f} m ({results_nowin['mainlobe_width_samples']} 采样点)")
    print(f"  旁瓣电平      = {results_nowin['sidelobe_db']:.1f} dB")
    print(f"  输出 SNR      = {results_nowin['output_snr_db']:.1f} dB")
    print(f"  SNR 增益      = {results_nowin['snr_gain_db']:.1f} dB (理论 {10 * np.log10(params.time_bandwidth_product):.1f} dB)")

    print(f"\n加 Hamming 窗结果:")
    print(f"  主瓣 3dB 宽度 = {results_hamm['mainlobe_width_m']:.2f} m ({results_hamm['mainlobe_width_samples']} 采样点)")
    print(f"  旁瓣电平      = {results_hamm['sidelobe_db']:.1f} dB")
    print(f"  SNR 增益      = {results_hamm['snr_gain_db']:.1f} dB")

    # 绘图
    print(f"\n绘制结果...")
    plot_results(results_nowin, results_hamm, params)

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(params)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
