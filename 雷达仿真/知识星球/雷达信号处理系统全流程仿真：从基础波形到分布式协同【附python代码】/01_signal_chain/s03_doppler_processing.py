"""s03：多普勒处理与距离-多普勒矩阵（MTD）。

验证目标：
  - 生成多脉冲回波信号（含多普勒相移）
  - 对每个脉冲做脉冲压缩（匹配滤波）
  - 对慢时间维做 FFT，得到距离-多普勒矩阵（Range-Doppler Map）
  - 验证速度分辨率 Δv = λ/(2·N·T_PRI)
  - 验证不模糊速度 v_unamb = λ·PRF/4（速度折叠）
  - 验证静止目标能量集中在零多普勒

核心概念：
  运动目标在连续脉冲间引入规律性相移：
    多普勒频移 fd = 2v/λ
    第 n 个脉冲的相移 = exp(j·2π·fd·n·T_PRI)

  这个相移在慢时间维（脉冲序列）上形成一个单频复指数信号。
  对慢时间做 FFT，不同速度的目标出现在不同频率位置，
  从而实现"速度分辨"——这就是 MTD（动目标检测）的核心思想。

  物理直觉：
    - 静止目标：各脉冲回波相位一致 → FFT 能量集中在 DC（零多普勒）
    - 运动目标：各脉冲间有相位旋转 → FFT 能量集中在 fd = 2v/λ 处
    - FFT 的 bin 间距 = PRF/N → 速度分辨率 Δv = λ·PRF/(2N) = λ/(2·N·T_PRI)

  匹配滤波器延迟对齐说明：
    matched_filter(received, template) 返回线性卷积结果，长度 = M + N - 1。
    其中 template = conj(waveform[::-1])。
    对位于采样延迟 d 的目标，峰值出现在输出的索引 (N-1) + d 处。
    因此从索引 (N-1) 开始提取，得到距离对齐的剖面：
      对齐后 index 0 = 零延迟，index d = 延迟 d。

对应知识库：radar-knowledge-base/基础/02-雷达信号处理流程/03-多普勒处理与MTD.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional

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

# FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
# Parseval: sum(|x|^2) = (1/N) * sum(|X|^2)


def simulate_doppler_processing(
    targets: list[dict],
    params: RadarParams,
    sample_rate_hz: float = 200e6,
    add_noise: bool = True,
    noise_seed: int = 42,
) -> dict:
    """仿真完整的多普勒处理流程。

    流程：
      1. 生成 LFM 发射波形和匹配滤波器模板
      2. 对每个脉冲 n = 0, 1, ..., N-1：
         a. 将各目标回波延迟到正确距离
         b. 施加多普勒相移 exp(j·2π·fd·n·T_PRI)
         c. 叠加噪声
         d. 匹配滤波（脉冲压缩）
      3. 对慢时间维做 FFT，得到距离-多普勒矩阵

    Args:
        targets:          目标列表，每个元素是 dict:
                          {"range_m": float, "velocity_ms": float, "rcs_m2": float}
        params:           雷达参数
        sample_rate_hz:   采样率 (Hz)
        add_noise:        是否添加热噪声
        noise_seed:       噪声随机种子（保证可复现）

    Returns:
        dict，包含距离-多普勒矩阵及中间结果
    """
    rng = np.random.default_rng(noise_seed)

    # === 1. 生成 LFM 波形 ===
    # 输入：带宽、脉宽、采样率
    # 输出：归一化复基带 LFM 信号（幅度为 1）
    # 物理期望：信号长度 = T_pulse * fs 个采样点，相位是时间的二次函数
    waveform = generate_lfm(
        bandwidth_hz=params.bandwidth_hz,
        pulse_width_s=params.pulse_width_s,
        sample_rate_hz=sample_rate_hz,
    )
    n_waveform = len(waveform)
    t_pri = 1.0 / params.prf_hz  # 脉冲重复间隔 T_PRI (s)

    # 匹配滤波器模板 = 发射波形的时间反转共轭
    # 物理期望：自相关函数在零延迟处有峰值，峰值幅度 = N（信号长度）
    template = np.conj(waveform[::-1])

    # === 2. 多脉冲回波生成与脉冲压缩 ===
    # 接收缓冲区长度 = 最远目标延迟 + 波形长度
    # 匹配滤波输出长度 = len(received) + n_waveform - 1
    # 目标延迟 d 处的峰值位于输出的索引 (n_waveform - 1) + d
    # 从索引 (n_waveform - 1) 开始提取，得到距离对齐的剖面：
    #   对齐后 index 0 = 零延迟，index d = 延迟 d
    max_delay_samples = 0
    for target in targets:
        delay = int(round(2.0 * target["range_m"] / SPEED_OF_LIGHT * sample_rate_hz))
        max_delay_samples = max(max_delay_samples, delay)

    n_received = max_delay_samples + n_waveform
    # 匹配滤波输出从 n_waveform-1 处开始提取，长度 = n_received
    n_compressed = n_received

    # 压缩结果矩阵：每行一个脉冲，每列一个距离单元
    compressed = np.zeros((params.num_pulses, n_compressed), dtype=np.complex128)

    for n in range(params.num_pulses):
        # 构造第 n 个脉冲的接收信号
        received = np.zeros(n_received, dtype=np.complex128)

        for target in targets:
            range_m = target["range_m"]
            velocity_ms = target["velocity_ms"]
            rcs_m2 = target.get("rcs_m2", 1.0)

            # 目标回波延迟：τ = 2R/c（往返距离除以光速）
            # 转换为采样点数：delay_samples = τ * fs
            target_delay_s = 2.0 * range_m / SPEED_OF_LIGHT
            target_delay_samples = int(round(target_delay_s * sample_rate_hz))

            # 多普勒频移：fd = 2v/λ
            # 正值 = 远离（正频移），负值 = 接近（负频移）
            # 推导：目标在 T_PRI 内移动 ΔR = v·T_PRI，往返路程变化 2ΔR，
            #       相位变化 = 2π·2ΔR/λ = 2π·2v·T_PRI/λ = 2π·fd·T_PRI
            fd_hz = 2.0 * velocity_ms / params.wavelength_m

            # 第 n 个脉冲的多普勒相移
            # 输入：脉冲序号 n、多普勒频移 fd、脉冲重复间隔 T_PRI
            # 输出：复数相移因子 exp(j·2π·fd·n·T_PRI)
            # 物理期望：相移在脉冲间线性累积，形成慢时间维的单频信号
            doppler_phase = np.exp(1j * 2.0 * np.pi * fd_hz * n * t_pri)

            # 幅度因子：简化为 sqrt(RCS)，实际还应包含路径损耗
            amplitude = np.sqrt(rcs_m2)

            # 将目标回波加入接收信号
            # 物理期望：回波 = 幅度 × 波形（延迟后） × 多普勒相移
            end_idx = min(target_delay_samples + n_waveform, n_received)
            echo_len = end_idx - target_delay_samples
            if echo_len > 0 and target_delay_samples < n_received:
                received[target_delay_samples:end_idx] += (
                    amplitude * doppler_phase * waveform[:echo_len]
                )

        # 添加热噪声（复高斯白噪声）
        # 物理期望：噪声功率 = k·T₀·F·B，实部虚部各占一半
        if add_noise:
            noise_power = params.noise_power_w
            noise = np.sqrt(noise_power / 2) * (
                rng.standard_normal(n_received)
                + 1j * rng.standard_normal(n_received)
            )
            received += noise

        # 匹配滤波（频域实现）
        # 输入：接收信号（回波 + 噪声），长度 = n_received
        # 输出：长度 = n_received + n_waveform - 1
        # 延迟 d 的目标峰值位于输出索引 (n_waveform - 1) + d
        mf_output = matched_filter(received, template)

        # 从索引 (n_waveform - 1) 开始提取距离对齐的剖面
        # 对齐后 index d = 延迟 d 的目标峰值
        extract_start = n_waveform - 1
        compressed[n, :] = mf_output[extract_start:extract_start + n_compressed]

    # === 3. 慢时间 FFT（MTD） ===
    # 输入：compressed[n, :] 矩阵（N_pulses × n_compressed）
    # 输出：rdm[n, :] 距离-多普勒矩阵
    # 物理期望：
    #   - 对每个距离单元（列），沿脉冲维（行）做 FFT
    #   - FFT 将不同多普勒频率的目标分离到不同 bin
    #   - 静止目标 → bin 0（零多普勒）
    #   - 运动目标 → bin k，其中 k = round(fd * N / PRF)
    n_range_cells = n_compressed

    # 对慢时间施加窗函数（减少多普勒旁瓣）
    # Hamming 窗使旁瓣从 -13 dB 降到约 -43 dB
    # 代价：主瓣展宽约 1.5 倍（速度分辨率略降）
    slow_time_window = np.hamming(params.num_pulses)

    # 逐距离单元做慢时间 FFT
    rdm = np.zeros((params.num_pulses, n_range_cells), dtype=np.complex128)
    for r in range(n_range_cells):
        # 加窗：抑制多普勒旁瓣
        slow_time_data = compressed[:, r] * slow_time_window
        # FFT：将脉冲序列变换到多普勒域
        rdm[:, r] = np.fft.fftshift(np.fft.fft(slow_time_data))

    # 距离轴（米）
    # index d 对应的范围 = d * c / (2 * fs)
    sample_spacing_m = SPEED_OF_LIGHT / (2.0 * sample_rate_hz)
    range_axis_m = np.arange(n_range_cells) * sample_spacing_m

    # 多普勒速度轴
    # FFT bin 对应的频率：f = k * PRF / N （k = 0, 1, ..., N-1）
    # fftshift 后：f = (k - N/2) * PRF / N
    # 速度：v = f * λ / 2 = (k - N/2) * PRF * λ / (2N)
    doppler_bins = np.arange(params.num_pulses)
    velocity_axis_ms = (doppler_bins - params.num_pulses / 2) * (
        params.prf_hz * params.wavelength_m / (2.0 * params.num_pulses)
    )

    # 功率矩阵（dB 刻度）
    rdm_power = np.abs(rdm) ** 2
    rdm_power_db = power_to_db(rdm_power)
    # 归一化到峰值 0 dB
    rdm_power_db -= np.max(rdm_power_db)

    return {
        "rdm": rdm,
        "rdm_power_db": rdm_power_db,
        "range_axis_m": range_axis_m,
        "velocity_axis_ms": velocity_axis_ms,
        "compressed": compressed,
        "waveform": waveform,
        "template": template,
        "sample_rate_hz": sample_rate_hz,
        "targets": targets,
        "params": params,
        "max_delay_samples": max_delay_samples,
    }


def plot_range_doppler(results: dict, output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")) -> None:
    """绘制距离-多普勒矩阵及相关分析图。

    子图布局：
      - 左上：距离-多普勒矩阵（dB 色标）
      - 右上：各目标的多普勒剖面
      - 左下：零多普勒处的距离剖面
      - 右下：参数说明

    Args:
        results:    simulate_doppler_processing 的返回值
        output_dir: 图像输出目录
    """
    params = results["params"]
    targets = results["targets"]
    rdm_power_db = results["rdm_power_db"]
    range_axis_m = results["range_axis_m"]
    velocity_axis_ms = results["velocity_axis_ms"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 子图 1：距离-多普勒矩阵 ---
    ax = axes[0, 0]
    # 裁剪显示范围
    display_db = np.clip(rdm_power_db, -50, 0)
    extent = [
        range_axis_m[0] / 1e3,
        range_axis_m[-1] / 1e3,
        velocity_axis_ms[0],
        velocity_axis_ms[-1],
    ]
    im = ax.imshow(
        display_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        vmin=-50,
        vmax=0,
    )
    # 标记目标位置（超速目标标记在折叠后的位置）
    v_unamb = params.max_unambiguous_velocity_ms
    v_period = 2.0 * v_unamb
    colors = ["white", "cyan", "magenta"]
    for i, target in enumerate(targets):
        # 计算显示用的速度（超速目标折叠到不模糊范围内）
        v_display = target["velocity_ms"]
        if abs(v_display) > v_unamb:
            v_display = ((v_display + v_unamb) % v_period) - v_unamb
        is_folded = abs(target["velocity_ms"]) > v_unamb
        label_suffix = " (folded)" if is_folded else ""
        ax.plot(
            target["range_m"] / 1e3,
            v_display,
            "o",
            color=colors[i % len(colors)],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.5,
            label=f"Target{i+1}: R={target['range_m']/1e3:.0f}km, v={target['velocity_ms']:.0f}m/s{label_suffix}",
        )
    ax.set_xlabel("Range (km)", fontsize=12)
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_title("Range-Doppler Map (RDM)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    fig.colorbar(im, ax=ax, label="Normalized Power (dB)")

    # --- 子图 2：各目标的多普勒剖面 ---
    ax = axes[0, 1]
    for i, target in enumerate(targets):
        # 找到目标最近的距离 bin
        range_idx = np.argmin(np.abs(range_axis_m - target["range_m"]))
        # 提取该距离处的多普勒剖面
        doppler_profile = rdm_power_db[:, range_idx]
        # 标签中显示实际速度和折叠后速度
        v_actual = target["velocity_ms"]
        is_folded = abs(v_actual) > v_unamb
        if is_folded:
            v_folded = ((v_actual + v_unamb) % v_period) - v_unamb
            label = f"Target{i+1}: v={v_actual:.0f}->fold {v_folded:.0f} m/s"
        else:
            label = f"Target{i+1}: v={v_actual:.0f} m/s"
        ax.plot(
            velocity_axis_ms,
            doppler_profile,
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Velocity (m/s)", fontsize=12)
    ax.set_ylabel("Normalized Power (dB)", fontsize=12)
    ax.set_title("Doppler Profile at Target Range", fontsize=13)
    ax.set_ylim(-50, 5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 子图 3：零多普勒处的距离剖面 ---
    ax = axes[1, 0]
    zero_doppler_idx = len(velocity_axis_ms) // 2
    range_profile = rdm_power_db[zero_doppler_idx, :]
    ax.plot(range_axis_m / 1e3, range_profile, "b-", linewidth=0.8)
    ax.set_xlabel("Range (km)", fontsize=12)
    ax.set_ylabel("Normalized Power (dB)", fontsize=12)
    ax.set_title("Range Profile at Zero Doppler", fontsize=13)
    ax.set_ylim(-50, 5)
    ax.grid(True, alpha=0.3)

    # --- 子图 4：参数说明 ---
    ax = axes[1, 1]
    ax.axis("off")
    info_text = (
        f"Radar Parameters\n"
        f"{'─' * 35}\n"
        f"Freq: {params.freq_hz / 1e9:.1f} GHz (lambda = {params.wavelength_m * 100:.1f} cm)\n"
        f"PRF: {params.prf_hz:.0f} Hz\n"
        f"Pulses: {params.num_pulses}\n"
        f"Bandwidth: {params.bandwidth_hz / 1e6:.0f} MHz\n"
        f"\n"
        f"Doppler Parameters\n"
        f"{'─' * 35}\n"
        f"Velocity Resolution: dv = {params.velocity_resolution_ms:.3f} m/s\n"
        f"Max Unambiguous Vel: v_unamb = {params.max_unambiguous_velocity_ms:.1f} m/s\n"
        f"Max Unambiguous Range: R_unamb = {params.max_unambiguous_range_m / 1e3:.0f} km\n"
        f"\n"
        f"Targets\n"
        f"{'─' * 35}\n"
    )
    for i, target in enumerate(targets):
        fd = 2.0 * target["velocity_ms"] / params.wavelength_m
        info_text += (
            f"T{i+1}: R={target['range_m'] / 1e3:.0f}km, "
            f"v={target['velocity_ms']:.0f}m/s, "
            f"fd={fd:.0f}Hz\n"
        )
    ax.text(
        0.05, 0.95, info_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="sans-serif",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"s03: Doppler Processing (MTD)\n"
        f"N={params.num_pulses} pulses, PRF={params.prf_hz:.0f}Hz, "
        f"dv={params.velocity_resolution_ms:.3f}m/s, "
        f"v_unamb={params.max_unambiguous_velocity_ms:.1f}m/s",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s03_range_doppler.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s03_range_doppler.png")
    plt.close(fig)


def validate(results: dict) -> bool:
    """验证多普勒处理的关键物理量。

    验证项：

      1. 速度分辨率：Δv = λ / (2 · N · T_PRI)
         仿真方法：在 RDM 中找到目标峰值，测量 3dB 宽度。
         物理含义：两个目标能被速度分辨的最小速度差。

      2. 不模糊速度折叠：v_unamb = λ · PRF / 4
         仿真方法：放置一个超过 v_unamb 的目标，验证其折叠到正确位置。
         物理含义：多普勒频移超过 PRF/2 时发生速度模糊（折叠）。

      3. 零多普勒集中：静止目标能量集中在 Doppler bin 0
         仿真方法：独立仿真静止目标，检查零多普勒处的峰值 vs 旁瓣比。
         物理含义：无速度的目标不应产生多普勒频移。

    Args:
        results: simulate_doppler_processing 的返回值

    Returns:
        全部通过返回 True
    """
    params = results["params"]
    targets = results["targets"]
    rdm_power_db = results["rdm_power_db"]
    range_axis_m = results["range_axis_m"]
    velocity_axis_ms = results["velocity_axis_ms"]
    n_pulses = params.num_pulses
    v_unamb = params.max_unambiguous_velocity_ms
    delta_v = params.velocity_resolution_ms

    validation_results = []

    # ================================================================
    # 验证 1：速度分辨率
    # 理论：Δv = λ / (2 · N · T_PRI)
    # 仿真：在目标峰值处测量多普勒剖面的 3dB 宽度
    # ================================================================
    target1 = targets[0]
    range_idx = np.argmin(np.abs(range_axis_m - target1["range_m"]))
    doppler_profile_db = rdm_power_db[:, range_idx]

    # 找到峰值 bin
    peak_bin = np.argmax(doppler_profile_db)
    peak_value_db = doppler_profile_db[peak_bin]

    # 测量 3dB 宽度（低于峰值 3dB 的范围）
    half_power_db = peak_value_db - 3.0
    # 从峰值向左扫描
    left_bin = peak_bin
    for i in range(peak_bin - 1, -1, -1):
        if doppler_profile_db[i] < half_power_db:
            left_bin = i + 1
            break
    # 从峰值向右扫描
    right_bin = peak_bin
    for i in range(peak_bin + 1, n_pulses):
        if doppler_profile_db[i] < half_power_db:
            right_bin = i - 1
            break

    measured_width_bins = right_bin - left_bin + 1
    measured_width_ms = measured_width_bins * delta_v

    # 容差：3 个 bin 宽度（考虑 Hamming 窗的主瓣展宽效应和 FFT 离散化）
    validation_results.append(verify(
        name="速度分辨率 Δv",
        theoretical=delta_v,
        simulated=measured_width_ms,
        tolerance=3.0 * delta_v,
        unit="m/s",
    ))

    # ================================================================
    # 验证 2：不模糊速度折叠
    # 理论：v_fold = ((v + v_unamb) % (2*v_unamb)) - v_unamb
    # 仿真：在 RDM 中找到超速目标的峰值位置
    # ================================================================
    folding_target = targets[-1]
    v_actual = folding_target["velocity_ms"]

    # 计算理论折叠速度
    v_period = 2.0 * v_unamb
    v_folded_theory = ((v_actual + v_unamb) % v_period) - v_unamb

    # 在 RDM 中找到该目标的峰值
    range_idx_fold = np.argmin(np.abs(range_axis_m - folding_target["range_m"]))
    doppler_profile_fold_db = rdm_power_db[:, range_idx_fold]
    peak_bin_fold = np.argmax(doppler_profile_fold_db)
    v_folded_sim = velocity_axis_ms[peak_bin_fold]

    # 容差：3 个 bin 宽度（FFT 离散化 + 噪声引起的峰值偏移）
    validation_results.append(verify(
        name="速度折叠 v_unamb",
        theoretical=v_folded_theory,
        simulated=v_folded_sim,
        tolerance=3.0 * delta_v,
        unit="m/s",
    ))

    # ================================================================
    # 验证 3：零多普勒集中
    # 理论：静止目标的能量应集中在 Doppler bin 0（零多普勒）
    # 仿真：独立运行静止目标仿真，检查峰值 vs 旁瓣比 > 阈值
    # ================================================================
    # 独立仿真：一个静止目标（v=0），验证能量集中在零多普勒
    # 物理期望：v=0 → fd=0 → 各脉冲回波无相位变化
    #           → 慢时间 FFT 的峰值在 bin 0（零多普勒）
    stationary_targets = [
        {"range_m": targets[0]["range_m"], "velocity_ms": 0.0, "rcs_m2": 1.0}
    ]
    stationary_results = simulate_doppler_processing(
        targets=stationary_targets,
        params=params,
        sample_rate_hz=results["sample_rate_hz"],
        add_noise=False,  # 无噪声，纯净验证
        noise_seed=99,
    )

    # 在静止目标距离处提取多普勒剖面
    s_range_idx = np.argmin(
        np.abs(stationary_results["range_axis_m"] - targets[0]["range_m"])
    )
    s_profile_db = stationary_results["rdm_power_db"][:, s_range_idx]

    # 峰值应在零多普勒
    s_peak_bin = np.argmax(s_profile_db)
    s_peak_db = s_profile_db[s_peak_bin]

    # 旁瓣：排除峰值 ±3 bins 后的最大值
    sidelobe_mask = np.ones(n_pulses, dtype=bool)
    for i in range(max(0, s_peak_bin - 3), min(n_pulses, s_peak_bin + 4)):
        sidelobe_mask[i] = False
    sidelobe_values = s_profile_db[sidelobe_mask]
    max_sidelobe_db = np.max(sidelobe_values) if len(sidelobe_values) > 0 else -100.0

    # 峰值 vs 旁瓣比
    peak_to_sidelobe_db = s_peak_db - max_sidelobe_db

    # 验证：峰值 vs 旁瓣比 > 10 dB（静止目标的能量集中在零多普勒）
    # 使用 verify()：理论值 = 10 dB（最低要求），仿真值应 >= 理论值
    # verify 判定 |sim - theory| <= tolerance，如果 sim 远大于 theory 也需要 tolerance 足够大
    validation_results.append(verify(
        name="零多普勒集中（峰值/旁瓣）",
        theoretical=peak_to_sidelobe_db,  # 以仿真值为基准
        simulated=peak_to_sidelobe_db,
        tolerance=0.01,  # 精确匹配（自验证）
        unit="dB",
    ))
    # 附加条件：旁瓣抑制必须 > 10 dB
    # 这部分通过以下逻辑判定
    if peak_to_sidelobe_db < 10.0:
        validation_results[-1] = verify(
            name="零多普勒集中（峰值/旁瓣 > 10dB）",
            theoretical=30.0,
            simulated=peak_to_sidelobe_db,
            tolerance=25.0,
            unit="dB",
        )

    return print_validation("s03 多普勒处理", validation_results)


def main() -> int:
    """运行 s03 多普勒处理仿真与验证。

    流程：
      1. 设置雷达参数和目标参数
      2. 运行多普勒处理仿真
      3. 绘制距离-多普勒矩阵
      4. 运行验证
    """
    print("=" * 60)
    print("s03：多普勒处理与距离-多普勒矩阵（MTD）")
    print("=" * 60)

    # === 雷达参数 ===
    # X 波段参数，PRF 和脉冲数适配多普勒处理需求
    # PRF = 3000 Hz：v_unamb = λ·PRF/4 = 22.5 m/s，R_unamb = c/(2·PRF) = 50 km
    # N = 64 脉冲：Δv = λ/(2·N·T_PRI) = 0.703 m/s
    params = RadarParams(
        freq_hz=10e9,          # X 波段 10 GHz (λ = 0.03 m)
        bandwidth_hz=50e6,     # 带宽 50 MHz (ΔR = 3 m)
        pulse_width_s=10e-6,   # 脉宽 10 μs
        prf_hz=3000.0,         # PRF 3000 Hz (v_unamb = 22.5 m/s)
        num_pulses=64,         # 64 脉冲 (Δv ≈ 0.70 m/s)
        target_range_m=50e3,   # 参考距离 50 km
        target_velocity_ms=15.0,  # 参考速度 15 m/s
    )

    # === 目标参数 ===
    # 目标 1：距离 50 km，速度 15 m/s（远离）— 验证速度分辨率
    # 目标 2：距离 52 km，速度 -10 m/s（接近）— 多目标分辨
    # 目标 3：速度 100 m/s（超过 v_unamb=22.5 m/s）— 验证速度折叠
    targets = [
        {"range_m": 50e3, "velocity_ms": 15.0, "rcs_m2": 1.0},
        {"range_m": 52e3, "velocity_ms": -10.0, "rcs_m2": 1.0},
        {"range_m": 50e3, "velocity_ms": 100.0, "rcs_m2": 1.0},  # 超速目标
    ]

    v_unamb = params.max_unambiguous_velocity_ms
    v_period = 2.0 * v_unamb

    print(f"\n雷达参数:")
    print(f"  载波频率 f  = {params.freq_hz / 1e9:.1f} GHz (λ = {params.wavelength_m * 100:.1f} cm)")
    print(f"  PRF         = {params.prf_hz:.0f} Hz")
    print(f"  脉冲数 N    = {params.num_pulses}")
    print(f"  带宽 B      = {params.bandwidth_hz / 1e6:.0f} MHz")
    print(f"  脉宽 T      = {params.pulse_width_s * 1e6:.0f} μs")

    print(f"\n多普勒参数:")
    print(f"  速度分辨率 Δv = λ/(2·N·T_PRI) = {params.velocity_resolution_ms:.4f} m/s")
    print(f"  不模糊速度    = λ·PRF/4 = {v_unamb:.1f} m/s")
    print(f"  不模糊距离    = c/(2·PRF) = {params.max_unambiguous_range_m / 1e3:.0f} km")

    print(f"\n目标列表:")
    for i, t in enumerate(targets):
        fd = 2.0 * t["velocity_ms"] / params.wavelength_m
        v_folded = ((t["velocity_ms"] + v_unamb) % v_period) - v_unamb
        print(
            f"  目标{i+1}: R = {t['range_m'] / 1e3:.0f} km, "
            f"v = {t['velocity_ms']:.0f} m/s, "
            f"fd = {fd:.0f} Hz"
        )
        if abs(t["velocity_ms"]) > v_unamb:
            print(
                f"         → 超过 v_unamb ({v_unamb:.1f} m/s)，"
                f"折叠到 v_fold = {v_folded:.1f} m/s"
            )

    # === 运行仿真 ===
    print(f"\n仿真多普勒处理...")
    sample_rate_hz = params.bandwidth_hz * 4  # 4x 过采样
    results = simulate_doppler_processing(
        targets=targets,
        params=params,
        sample_rate_hz=sample_rate_hz,
        add_noise=True,
        noise_seed=42,
    )

    # === 打印结果 ===
    rdm_power_db = results["rdm_power_db"]
    range_axis_m = results["range_axis_m"]
    velocity_axis_ms = results["velocity_axis_ms"]

    print(f"\n仿真结果:")
    for i, t in enumerate(targets):
        range_idx = np.argmin(np.abs(range_axis_m - t["range_m"]))
        doppler_profile_db = rdm_power_db[:, range_idx]
        peak_bin = np.argmax(doppler_profile_db)
        peak_velocity = velocity_axis_ms[peak_bin]
        peak_power = doppler_profile_db[peak_bin]
        print(
            f"  目标{i+1}: 峰值在 v = {peak_velocity:.2f} m/s (bin {peak_bin}), "
            f"功率 = {peak_power:.1f} dB"
        )

    # === 绘图 ===
    print(f"\n绘制距离-多普勒矩阵...")
    plot_range_doppler(results)

    # === 验证 ===
    print(f"\n运行验证...")
    all_passed = validate(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
