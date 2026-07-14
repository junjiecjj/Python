"""s05：完整雷达信号处理链（发射→回波→脉压→MTD→CFAR→检测）。

将 s01~s04 串联为端到端仿真：
  发射 LFM 波形
    → 多目标回波（距离延迟 + 多普勒相移 + 热噪声）
    → 脉冲压缩（matched_filter）
    → MTD（慢时间 FFT）
    → CA-CFAR 检测
    → 点迹输出（距离、多普勒、SNR 估计）

验证目标：
  a. 端到端检测率：所有注入目标应被检测到（±5 bin 容差）
  b. 检测位置精度：距离误差 < 2 bin，多普勒误差 < 2 bin
  c. SNR 趋势：远距离/弱目标的 SNR 应低于近距离/强目标
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass

from lib.radar_params import RadarParams, SPEED_OF_LIGHT
from lib.signal_utils import generate_lfm, matched_filter, power_to_db
from lib.validation import verify, print_validation
from s04_cfar_detection import ca_cfar_2d

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
# Parseval: sum(|x|^2) = (1/N) * sum(|X|^2)


@dataclass
class Detection:
    """单次检测结果。

    Attributes:
        range_bin:    检测到的距离 bin 索引
        doppler_bin:  检测到的多普勒 bin 索引
        range_m:      估计距离 (m)
        velocity_ms:  估计径向速度 (m/s)
        snr_db:       估计信噪比 (dB)
    """
    range_bin: int
    doppler_bin: int
    range_m: float
    velocity_ms: float
    snr_db: float


def _find_significant_peaks(
    rdm_power: np.ndarray,
    detections_mask: np.ndarray,
    max_peaks: int = 20,
    min_separation_doppler: int = 5,
    min_separation_range: int = 10,
) -> list[tuple[int, int]]:
    """从 CFAR 检测掩码中提取显著峰值。

    对于大尺寸 RDM，直接使用 CFAR 输出会导致大量虚警。
    本函数通过以下策略提取真正的目标峰值：
      1. 在 CFAR 检测区域内，按功率降序排列所有 bin
      2. 依次选取功率最高的 bin 作为候选峰值
      3. 新峰值必须与已选峰值保持最小间距（避免重复检测同一目标）

    Args:
        rdm_power:              RDM 功率矩阵（线性功率）
        detections_mask:        CFAR 检测布尔矩阵
        max_peaks:              最大峰值数
        min_separation_doppler: 多普勒维最小间距 (bin)
        min_separation_range:   距离维最小间距 (bin)

    Returns:
        峰值位置列表 [(doppler_bin, range_bin), ...]
    """
    # 在 CFAR 检测区域内，按功率降序选取峰值
    det_indices = np.argwhere(detections_mask)
    if len(det_indices) == 0:
        return []

    # 获取检测点的功率值并排序
    det_powers = rdm_power[det_indices[:, 0], det_indices[:, 1]]
    sorted_order = np.argsort(det_powers)[::-1]

    selected_peaks = []
    for idx in sorted_order:
        if len(selected_peaks) >= max_peaks:
            break

        d_idx = int(det_indices[idx, 0])
        r_idx = int(det_indices[idx, 1])

        # 检查与已选峰值的最小间距
        is_far_enough = True
        for sd, sr in selected_peaks:
            if (abs(d_idx - sd) < min_separation_doppler and
                    abs(r_idx - sr) < min_separation_range):
                is_far_enough = False
                break

        if is_far_enough:
            selected_peaks.append((d_idx, r_idx))

    return selected_peaks


def simulate_full_chain(
    params: RadarParams,
    targets: list[dict],
    sample_rate_hz: float,
    noise_seed: int = 42,
    max_range_m: float = 100e3,
) -> dict:
    """端到端雷达信号处理链仿真。

    流程：
      1. 生成 LFM 发射波形及匹配滤波模板
      2. 对每个脉冲 n = 0, 1, ..., N-1：
         a. 各目标回波叠加（距离延迟 + 多普勒相移 + 幅度加权）
         b. 添加复高斯白噪声
         c. 匹配滤波（脉冲压缩）
      3. 慢时间 FFT 生成距离-多普勒矩阵（RDM）
      4. CA-CFAR 检测 + 局部峰值提取，输出点迹

    目标回波模型：
      第 n 个脉冲的第 k 个目标回波为：
        echo_k(n, t) = A_k * s(t - tau_k) * exp(j * 2*pi * fd_k * n * T_PRI)
      其中：
        A_k = sqrt(RCS_k)     幅度因子（简化模型，不含路径损耗）
        tau_k = 2*R_k / c     往返延迟
        fd_k = 2*v_k / lambda 多普勒频移
        T_PRI = 1 / PRF       脉冲重复间隔

    匹配滤波延迟对齐：
      matched_filter(received, template) 返回线性卷积，长度 = len(received) + len(template) - 1。
      对延迟 d 的目标，峰值位于输出索引 (len(template) - 1) + d。
      从索引 (len(template) - 1) 开始提取，得到距离对齐的剖面。

    Args:
        params:          雷达系统参数
        targets:         目标列表，每个元素为 dict：
                         {"range_m": float, "velocity_ms": float, "rcs_m2": float}
        sample_rate_hz:  采样率 (Hz)，需 > bandwidth 满足 Nyquist
        noise_seed:      噪声随机种子（保证可复现）
        max_range_m:     最大仿真距离 (m)，限制 RDM 大小

    Returns:
        dict，包含：
          rdm_power:       距离-多普勒功率矩阵（线性功率，N_pulses x n_range_cells）
          rdm_power_db:    归一化功率矩阵 (dB)
          detections:      CA-CFAR 检测布尔矩阵
          detected_targets: 检测结果列表 [Detection, ...]
          range_axis_m:    距离轴 (m)
          velocity_axis_ms: 速度轴 (m/s)
          compressed_last: 最后一个脉冲的压缩结果（用于绘图）
          params:          雷达参数
          targets:         目标参数
          sample_rate_hz:  采样率
    """
    rng = np.random.default_rng(noise_seed)

    # === 1. 生成 LFM 波形 ===
    waveform = generate_lfm(
        bandwidth_hz=params.bandwidth_hz,
        pulse_width_s=params.pulse_width_s,
        sample_rate_hz=sample_rate_hz,
    )
    n_waveform = len(waveform)
    t_pri = 1.0 / params.prf_hz  # 脉冲重复间隔 T_PRI (s)

    # 匹配滤波模板 = 发射波形的时间反转共轭
    template = np.conj(waveform[::-1])

    # === 2. 确定接收缓冲区长度 ===
    # 只覆盖到 max_range_m，限制 RDM 大小
    max_delay_samples = int(round(2.0 * max_range_m / SPEED_OF_LIGHT * sample_rate_hz))
    n_received = max_delay_samples + n_waveform
    n_compressed = max_delay_samples  # 距离对齐后只保留 max_delay 个距离 bin

    # === 3. 多脉冲回波生成与脉冲压缩 ===
    compressed = np.zeros((params.num_pulses, n_compressed), dtype=np.complex128)

    for n in range(params.num_pulses):
        # 构造第 n 个脉冲的接收信号
        received = np.zeros(n_received, dtype=np.complex128)

        for target in targets:
            range_m = target["range_m"]
            velocity_ms = target["velocity_ms"]
            rcs_m2 = target.get("rcs_m2", 1.0)

            # 目标回波延迟：tau = 2R/c
            target_delay_s = 2.0 * range_m / SPEED_OF_LIGHT
            target_delay_samples = int(round(target_delay_s * sample_rate_hz))

            # 多普勒频移：fd = 2v/lambda
            fd_hz = 2.0 * velocity_ms / params.wavelength_m

            # 第 n 个脉冲的多普勒相移
            doppler_phase = np.exp(1j * 2.0 * np.pi * fd_hz * n * t_pri)

            # 幅度因子
            amplitude = np.sqrt(rcs_m2)

            # 将回波叠加到接收信号
            end_idx = min(target_delay_samples + n_waveform, n_received)
            echo_len = end_idx - target_delay_samples
            if echo_len > 0 and target_delay_samples < n_received:
                received[target_delay_samples:end_idx] += (
                    amplitude * doppler_phase * waveform[:echo_len]
                )

        # 添加热噪声
        noise_power = params.noise_power_w
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(n_received) + 1j * rng.standard_normal(n_received)
        )
        received += noise

        # 匹配滤波
        mf_output = matched_filter(received, template)

        # 距离对齐：从索引 (n_waveform - 1) 开始提取 n_compressed 个样本
        extract_start = n_waveform - 1
        compressed[n, :] = mf_output[extract_start:extract_start + n_compressed]

    # === 4. 慢时间 FFT（MTD） ===
    # 对每个距离单元沿脉冲维做 FFT，将不同多普勒频率分离到不同 bin
    n_range_cells = n_compressed

    # 慢时间加 Hamming 窗（抑制多普勒旁瓣）
    slow_time_window = np.hamming(params.num_pulses)

    rdm = np.zeros((params.num_pulses, n_range_cells), dtype=np.complex128)
    for r in range(n_range_cells):
        slow_data = compressed[:, r] * slow_time_window
        rdm[:, r] = np.fft.fftshift(np.fft.fft(slow_data))

    # 距离轴和速度轴
    sample_spacing_m = SPEED_OF_LIGHT / (2.0 * sample_rate_hz)
    range_axis_m = np.arange(n_range_cells) * sample_spacing_m

    doppler_bins = np.arange(params.num_pulses)
    velocity_axis_ms = (doppler_bins - params.num_pulses / 2) * (
        params.prf_hz * params.wavelength_m / (2.0 * params.num_pulses)
    )

    # 功率矩阵
    rdm_power = np.abs(rdm) ** 2
    rdm_power_db = power_to_db(rdm_power)
    rdm_power_db_normalized = rdm_power_db - np.max(rdm_power_db)

    # === 5. CA-CFAR 检测 ===
    guard_cells = (2, 2)
    training_cells = (8, 8)
    pfa = 1e-6

    detections_mask, threshold = ca_cfar_2d(
        rdm_power, guard_cells, training_cells, pfa
    )

    # === 6. 局部峰值提取 + 点迹输出 ===
    # CFAR 输出包含大量相邻检测点，需提取局部峰值
    peak_positions = _find_significant_peaks(
        rdm_power, detections_mask,
        max_peaks=10,
        min_separation_doppler=10,
        min_separation_range=100,
    )

    noise_floor = np.median(rdm_power)
    detected_targets = []
    for d_idx, r_idx in peak_positions:
        peak_power = rdm_power[d_idx, r_idx]
        snr_linear = peak_power / noise_floor if noise_floor > 0 else 0.0
        snr_db = power_to_db(snr_linear)

        detected_targets.append(Detection(
            range_bin=r_idx,
            doppler_bin=d_idx,
            range_m=range_axis_m[r_idx],
            velocity_ms=velocity_axis_ms[d_idx],
            snr_db=snr_db,
        ))

    return {
        "rdm_power": rdm_power,
        "rdm_power_db": rdm_power_db_normalized,
        "detections": detections_mask,
        "detected_targets": detected_targets,
        "range_axis_m": range_axis_m,
        "velocity_axis_ms": velocity_axis_ms,
        "compressed_last": compressed[-1, :],
        "params": params,
        "targets": targets,
        "sample_rate_hz": sample_rate_hz,
    }


def plot_chain_results(
    results: dict,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制完整处理链的 4 子图结果。

    子图布局：
      (1) 脉冲压缩结果：最后一个脉冲的压缩幅度（dB）
      (2) Range-Doppler Map：RDM 功率矩阵
      (3) CFAR 检测叠加：RDM 上叠加检测标记
      (4) 检测 vs 真实目标对比：距离-速度散点图

    Args:
        results:    simulate_full_chain 的返回值
        output_dir: 图像输出目录
    """
    params = results["params"]
    targets = results["targets"]
    rdm_power_db = results["rdm_power_db"]
    detections = results["detections"]
    detected_targets = results["detected_targets"]
    range_axis_m = results["range_axis_m"]
    velocity_axis_ms = results["velocity_axis_ms"]
    compressed_last = results["compressed_last"]
    sample_rate_hz = results["sample_rate_hz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 子图 1：脉冲压缩结果 ---
    ax = axes[0, 0]
    sample_spacing_m = SPEED_OF_LIGHT / (2.0 * sample_rate_hz)
    comp_mag = np.abs(compressed_last)
    comp_mag_db = power_to_db(comp_mag ** 2)
    comp_mag_db -= np.max(comp_mag_db)
    range_profile = np.arange(len(comp_mag_db)) * sample_spacing_m / 1e3
    ax.plot(range_profile, comp_mag_db, "b-", linewidth=0.8)
    ax.set_xlabel("距离 (km)")
    ax.set_ylabel("归一化功率 (dB)")
    ax.set_title("脉冲压缩结果（最后一个脉冲）")
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    # 标记目标位置
    for t in targets:
        ax.axvline(t["range_m"] / 1e3, color="red", linestyle="--",
                    alpha=0.5, linewidth=0.8)

    # --- 子图 2：Range-Doppler Map ---
    ax = axes[0, 1]
    display_db = np.clip(rdm_power_db, -50, 0)
    extent = [
        range_axis_m[0] / 1e3, range_axis_m[-1] / 1e3,
        velocity_axis_ms[0], velocity_axis_ms[-1],
    ]
    im = ax.imshow(
        display_db, aspect="auto", origin="lower",
        extent=extent, cmap="jet", vmin=-50, vmax=0,
    )
    # 标记真实目标
    colors = ["white", "cyan", "magenta"]
    for i, t in enumerate(targets):
        ax.plot(
            t["range_m"] / 1e3, t["velocity_ms"],
            "o", color=colors[i % len(colors)],
            markersize=10, markeredgecolor="black", markeredgewidth=1.5,
            label=f"真实目标{i+1}",
        )
    ax.set_xlabel("距离 (km)")
    ax.set_ylabel("速度 (m/s)")
    ax.set_title("Range-Doppler Map (MTD)")
    ax.legend(fontsize=9, loc="upper right")
    fig.colorbar(im, ax=ax, label="归一化功率 (dB)")

    # --- 子图 3：CFAR 检测叠加 ---
    ax = axes[1, 0]
    ax.imshow(
        display_db, aspect="auto", origin="lower",
        extent=extent, cmap="gray", vmin=-50, vmax=0, alpha=0.6,
    )
    # 只绘制检测到的峰值点（而非所有 CFAR 超门限点）
    if detected_targets:
        det_ranges = [d.range_m / 1e3 for d in detected_targets]
        det_vels = [d.velocity_ms for d in detected_targets]
        ax.scatter(
            det_ranges, det_vels,
            c="red", s=40, marker="x", linewidths=2,
            label=f"检测峰值 ({len(detected_targets)})",
            zorder=5,
        )
    for i, t in enumerate(targets):
        ax.plot(
            t["range_m"] / 1e3, t["velocity_ms"],
            "g^", markersize=12,
            markeredgecolor="black", markeredgewidth=1,
        )
    ax.set_xlabel("距离 (km)")
    ax.set_ylabel("速度 (m/s)")
    ax.set_title("CA-CFAR 检测结果（局部峰值）")
    ax.legend(fontsize=9)

    # --- 子图 4：检测 vs 真实目标对比 ---
    ax = axes[1, 1]
    # 真实目标
    true_ranges = [t["range_m"] / 1e3 for t in targets]
    true_vels = [t["velocity_ms"] for t in targets]
    ax.scatter(
        true_ranges, true_vels, c="green", s=200, marker="^",
        edgecolors="black", linewidths=1.5, zorder=5, label="真实目标",
    )
    # 检测结果
    if detected_targets:
        det_ranges = [d.range_m / 1e3 for d in detected_targets]
        det_vels = [d.velocity_ms for d in detected_targets]
        det_snrs = [d.snr_db for d in detected_targets]
        sc = ax.scatter(
            det_ranges, det_vels, c=det_snrs, cmap="hot",
            s=30, marker="x", vmin=0, vmax=max(det_snrs) + 5,
            label="检测点",
        )
        fig.colorbar(sc, ax=ax, label="SNR (dB)")

    # 标注真实目标信息
    for i, t in enumerate(targets):
        ax.annotate(
            f"T{i+1}: R={t['range_m']/1e3:.0f}km\nv={t['velocity_ms']:.0f}m/s",
            (t["range_m"] / 1e3, t["velocity_ms"]),
            textcoords="offset points", xytext=(10, 10),
            fontsize=8, color="green",
        )
    ax.set_xlabel("距离 (km)")
    ax.set_ylabel("速度 (m/s)")
    ax.set_title("检测 vs 真实目标对比")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"s05：完整雷达信号处理链\n"
        f"N={params.num_pulses} pulses, PRF={params.prf_hz:.0f}Hz, "
        f"B={params.bandwidth_hz/1e6:.0f}MHz, "
        f"targets={len(targets)}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s05_full_chain.png"),
        dpi=150, bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s05_full_chain.png")
    plt.close(fig)


def validate(
    params: RadarParams,
    true_targets: list[dict],
    detected_targets: list[Detection],
    rdm_power: np.ndarray,
    range_axis_m: np.ndarray,
    velocity_axis_ms: np.ndarray,
) -> bool:
    """验证完整处理链的端到端性能。

    验证项：

      a. 端到端检测率：所有注入目标应被检测到（±5 bin 容差）
         对每个真实目标，在检测列表中查找距离和速度均在容差内的匹配项。

      b. 检测位置精度：距离误差 < 2 bin，多普勒误差 < 2 bin
         对成功匹配的检测，验证其距离和多普勒估计精度。

      c. SNR 趋势：远距离/弱目标的 SNR 应低于近距离/强目标
         验证 RCS 小或距离远的目标 SNR 低于 RCS 大或距离近的目标。

    Args:
        params:            雷达参数
        true_targets:      真实目标列表
        detected_targets:  检测结果列表
        rdm_power:         RDM 功率矩阵（线性）
        range_axis_m:      距离轴
        velocity_axis_ms:  速度轴

    Returns:
        全部通过返回 True
    """
    validation_results = []

    # 距离和速度的 bin 间距
    range_bin_width_m = range_axis_m[1] - range_axis_m[0] if len(range_axis_m) > 1 else 1.0
    vel_bin_width_ms = velocity_axis_ms[1] - velocity_axis_ms[0] if len(velocity_axis_ms) > 1 else 1.0

    # === 验证 a：端到端检测率 ===
    detected_count = 0
    matched_pairs = []  # (true_idx, detection) 匹配对

    for i, t in enumerate(true_targets):
        t_range_m = t["range_m"]
        t_vel_ms = t["velocity_ms"]

        # 在检测列表中查找最近的匹配（容差 ±5 bin）
        range_tol_m = 5.0 * range_bin_width_m
        vel_tol_ms = 5.0 * vel_bin_width_ms

        best_det = None
        best_dist = float("inf")
        for det in detected_targets:
            dr = abs(det.range_m - t_range_m)
            dv = abs(det.velocity_ms - t_vel_ms)
            if dr < range_tol_m and dv < vel_tol_ms:
                # 综合距离度量
                dist = dr / range_bin_width_m + dv / vel_bin_width_ms
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

        if best_det is not None:
            detected_count += 1
            matched_pairs.append((i, best_det))

    detection_rate = detected_count / len(true_targets) if true_targets else 1.0
    validation_results.append(verify(
        name="端到端检测率",
        theoretical=1.0,
        simulated=detection_rate,
        tolerance=0.01,
        unit="",
    ))

    # === 验证 b：检测位置精度 ===
    range_errors = []
    doppler_errors = []

    for true_idx, det in matched_pairs:
        t = true_targets[true_idx]
        # 距离误差（bin 数）
        range_err_bins = abs(det.range_m - t["range_m"]) / range_bin_width_m
        # 多普勒误差（bin 数）
        doppler_err_bins = abs(det.velocity_ms - t["velocity_ms"]) / vel_bin_width_ms
        range_errors.append(range_err_bins)
        doppler_errors.append(doppler_err_bins)

    # 距离精度：平均误差 < 2 bin
    mean_range_err = np.mean(range_errors) if range_errors else 999.0
    validation_results.append(verify(
        name="距离估计精度（平均误差）",
        theoretical=0.0,
        simulated=mean_range_err,
        tolerance=2.0,
        unit="bin",
    ))

    # 多普勒精度：平均误差 < 2 bin
    mean_doppler_err = np.mean(doppler_errors) if doppler_errors else 999.0
    validation_results.append(verify(
        name="多普勒估计精度（平均误差）",
        theoretical=0.0,
        simulated=mean_doppler_err,
        tolerance=2.0,
        unit="bin",
    ))

    # === 验证 c：SNR 趋势 ===
    # 按 "RCS / R^4" 排序（雷达方程接收功率 ∝ RCS / R^4）
    # 验证 SNR 潜力高的目标实际 SNR 也更高
    if len(matched_pairs) >= 2:
        snr_potentials = []
        snr_actuals = []
        for true_idx, det in matched_pairs:
            t = true_targets[true_idx]
            rcs = t.get("rcs_m2", 1.0)
            r = t["range_m"]
            potential = rcs / (r ** 4)
            snr_potentials.append(potential)
            snr_actuals.append(det.snr_db)

        sorted_indices = np.argsort(snr_potentials)
        snr_sorted = [snr_actuals[i] for i in sorted_indices]

        # 单调性检查
        monotonic_violations = 0
        for k in range(len(snr_sorted) - 1):
            potential_diff = snr_potentials[sorted_indices[k + 1]] - snr_potentials[sorted_indices[k]]
            snr_diff = snr_sorted[k + 1] - snr_sorted[k]
            if potential_diff > 0 and snr_diff < -3.0:
                monotonic_violations += 1
            elif potential_diff < 0 and snr_diff > 3.0:
                monotonic_violations += 1

        validation_results.append(verify(
            name="SNR 趋势一致性",
            theoretical=0.0,
            simulated=float(monotonic_violations),
            tolerance=0.5,
            unit="violations",
        ))
    else:
        validation_results.append(verify(
            name="SNR 趋势一致性（跳过：匹配对不足）",
            theoretical=0.0,
            simulated=0.0,
            tolerance=0.01,
            unit="",
        ))

    return print_validation("s05 完整处理链", validation_results)


def main() -> int:
    """运行 s05 完整雷达信号处理链仿真与验证。"""
    print("=" * 60)
    print("s05：完整雷达信号处理链")
    print("=" * 60)

    # === 雷达参数 ===
    # 使用 RadarParams 默认参数（X 波段）
    params = RadarParams()
    sample_rate_hz = params.bandwidth_hz * 2  # 2x 过采样（Nyquist 满足即可）

    # === 目标参数 ===
    # 3 个不同距离/速度/RCS 的目标
    # 速度需在不模糊范围内：v_unamb = λ·PRF/4 = 7.5 m/s
    targets = [
        {"range_m": 30e3, "velocity_ms": 3.0, "rcs_m2": 10.0},    # 近距、大 RCS
        {"range_m": 50e3, "velocity_ms": -2.0, "rcs_m2": 1.0},    # 中距、中 RCS
        {"range_m": 80e3, "velocity_ms": 5.0, "rcs_m2": 0.1},     # 远距、小 RCS
    ]

    # 限制最大仿真距离
    max_range_m = 100e3  # 100 km

    print(f"\n雷达参数:")
    print(f"  载波频率 f  = {params.freq_hz / 1e9:.1f} GHz (λ = {params.wavelength_m * 100:.1f} cm)")
    print(f"  带宽 B      = {params.bandwidth_hz / 1e6:.0f} MHz")
    print(f"  脉宽 T      = {params.pulse_width_s * 1e6:.0f} μs")
    print(f"  PRF         = {params.prf_hz:.0f} Hz")
    print(f"  脉冲数 N    = {params.num_pulses}")
    print(f"  采样率 fs   = {sample_rate_hz / 1e6:.0f} MHz ({sample_rate_hz / params.bandwidth_hz:.0f}x 过采样)")
    print(f"  最大仿真距离 = {max_range_m / 1e3:.0f} km")
    print(f"  距离分辨率   = c/(2B) = {params.range_resolution_m:.1f} m")
    print(f"  速度分辨率   = {params.velocity_resolution_ms:.4f} m/s")
    print(f"  不模糊距离   = {params.max_unambiguous_range_m / 1e3:.0f} km")
    print(f"  不模糊速度   = {params.max_unambiguous_velocity_ms:.1f} m/s")

    print(f"\n目标列表:")
    for i, t in enumerate(targets):
        fd = 2.0 * t["velocity_ms"] / params.wavelength_m
        rcs_db = 10 * np.log10(max(t["rcs_m2"], 1e-40))
        print(
            f"  目标{i+1}: R={t['range_m']/1e3:.0f} km, "
            f"v={t['velocity_ms']:.0f} m/s, "
            f"RCS={t['rcs_m2']:.1f} m² ({rcs_db:.0f} dBsm), "
            f"fd={fd:.0f} Hz"
        )

    # === 运行完整处理链 ===
    print(f"\n运行完整处理链仿真...")
    results = simulate_full_chain(
        params=params,
        targets=targets,
        sample_rate_hz=sample_rate_hz,
        noise_seed=42,
        max_range_m=max_range_m,
    )

    # === 打印检测结果 ===
    detected_targets = results["detected_targets"]
    range_axis_m = results["range_axis_m"]
    velocity_axis_ms = results["velocity_axis_ms"]
    rdm_power = results["rdm_power"]

    print(f"\n检测结果:")
    print(f"  总检测数: {len(detected_targets)}")
    print(f"  {'序号':>4}  {'距离(km)':>8}  {'速度(m/s)':>10}  {'SNR(dB)':>8}  {'距离bin':>8}  {'多普勒bin':>10}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
    for i, det in enumerate(detected_targets):
        print(
            f"  {i+1:>4}  {det.range_m/1e3:>8.1f}  {det.velocity_ms:>10.1f}  "
            f"{det.snr_db:>8.1f}  {det.range_bin:>8}  {det.doppler_bin:>10}"
        )

    # === 绘图 ===
    print(f"\n绘制结果...")
    plot_chain_results(results)

    # === 验证 ===
    print(f"\n运行验证...")
    all_passed = validate(
        params=params,
        true_targets=targets,
        detected_targets=detected_targets,
        rdm_power=rdm_power,
        range_axis_m=range_axis_m,
        velocity_axis_ms=velocity_axis_ms,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
