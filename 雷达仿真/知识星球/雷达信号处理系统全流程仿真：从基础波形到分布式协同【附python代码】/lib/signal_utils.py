"""信号处理工具函数。

提供雷达信号生成、FFT 处理、窗函数等基础操作。
FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
  Parseval: sum(|x|²) = (1/N) * sum(|X|²)
"""

import numpy as np
from typing import Optional


def generate_lfm(
    bandwidth_hz: float,
    pulse_width_s: float,
    sample_rate_hz: float,
    up_chirp: bool = True,
) -> np.ndarray:
    """生成线性调频（LFM）复基带信号。

    LFM 是最常用的雷达波形。频率随时间线性变化（上调频或下调频），
    使得压缩后能同时获得大时宽（高能量）和大带宽（高分辨率）。

    数学表达式：s(t) = rect(t/T) * exp(j * π * K * t²)
      其中 K = B/T 是调频斜率，B 是带宽，T 是脉宽。

    Args:
        bandwidth_hz:  信号带宽 (Hz)，决定距离分辨率 ΔR = c/(2B)
        pulse_width_s: 脉冲宽度 (s)，决定发射能量
        sample_rate_hz:采样率 (Hz)，需满足 Nyquist: fs > B
        up_chirp:      True 为上调频（频率从低到高），False 为下调频

    Returns:
        复数数组，长度 = round(pulse_width_s * sample_rate_hz)
        归一化幅度为 1（不包含发射功率，功率在雷达方程中计算）

    物理期望：
        - 信号相位是时间的二次函数
        - 瞬时频率从 fc-B/2 线性变化到 fc+B/2（上调频时）
        - 时宽带宽积 TB 决定了脉冲压缩的理论增益
    """
    num_samples = int(round(pulse_width_s * sample_rate_hz))
    t = np.arange(num_samples) / sample_rate_hz  # 时间轴 [0, T)

    # 调频斜率：K = ±B/T（正负取决于上下调频）
    sign = 1.0 if up_chirp else -1.0
    chirp_rate = sign * bandwidth_hz / pulse_width_s

    # LFM 信号：相位 = π * K * t²
    # 这是一个瞬时频率随时间线性变化的信号
    phase = np.pi * chirp_rate * t**2
    return np.exp(1j * phase)


def matched_filter(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """频域匹配滤波。

    匹配滤波器是白噪声背景下检测已知波形的最优滤波器。
    输出在目标位置产生峰值，峰值高度正比于输入 SNR。

    频域实现：Y(f) = X(f) * H*(f)
      其中 X(f) 是信号频谱，H(f) 是模板频谱，* 表示共轭。

    Args:
        signal:   接收信号（可能包含目标回波 + 噪声 + 杂波）
        template: 匹配滤波器模板（通常是发射波形的时域反转共轭）

    Returns:
        压缩后的信号。脉冲位置出现尖峰，旁瓣电平取决于信号的自相关特性。

    物理期望：
        - 压缩后脉冲宽度 ≈ 1/B（B 为信号带宽），而非原始脉宽 T
        - 峰值幅度 ≈ T*B * (输入信号幅度)，即获得 TB 增益
        - LFM 的距离旁瓣约 -13.2 dB（sinc 包络的第一旁瓣）
    """
    # 零填充到足够长度，避免循环卷积的混叠
    n_fft = len(signal) + len(template) - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))  # 补到 2 的幂次，加速 FFT

    # 频域匹配滤波：Y(f) = X(f) × H(f)
    # 注意：template 已经是时域反转共轭（conj(s[::-1])），
    # 其 FFT 天然包含了共轭，不需要再取 conj。
    S = np.fft.fft(signal, n_fft)
    H = np.fft.fft(template, n_fft)
    result = np.fft.ifft(S * H)

    # 返回线性卷积的前 N+M-1 个样本（不截断，保留完整结果）
    return result[:len(signal) + len(template) - 1]


def apply_window(data: np.ndarray, window_name: str = "hamming") -> np.ndarray:
    """对数据施加窗函数。

    窗函数用于抑制频谱泄露和降低旁瓣。代价是主瓣展宽（分辨率略降）。

    常用窗函数的旁瓣抑制能力：
      - 矩形窗（无窗）：-13.2 dB
      - Hamming：-43 dB
      - Hann：-31 dB
      - Taylor (nbar=4, SLL=-30dB)：-30 dB，主瓣较窄

    Args:
        data:         输入数据（一维数组）
        window_name:  窗函数名称，支持 "hamming", "hann", "blackman", "none"

    Returns:
        加窗后的数据。需要补偿相干增益以保持能量守恒。

    注意：加窗后应补偿相干增益（除以窗函数均值），否则 SNR 估计会偏低。
    """
    n = len(data)
    if window_name == "none":
        return data.copy()

    windows = {
        "hamming": np.hamming(n),
        "hann": np.hanning(n),
        "blackman": np.blackman(n),
    }
    if window_name not in windows:
        raise ValueError(f"未知窗函数: {window_name}，可选: {list(windows.keys()) + ['none']}")

    return data * windows[window_name]


def coherent_gain_compensation(windowed_signal: np.ndarray,
                                window_name: str) -> np.ndarray:
    """补偿窗函数的相干增益损失。

    窗函数降低了信号幅度，需要除以窗函数均值来恢复。

    Args:
        windowed_signal: 已加窗的信号
        window_name:     窗函数名称

    Returns:
        补偿后的信号
    """
    if window_name == "none":
        return windowed_signal
    windows = {
        "hamming": np.hamming(len(windowed_signal)),
        "hann": np.hanning(len(windowed_signal)),
        "blackman": np.blackman(len(windowed_signal)),
    }
    return windowed_signal / np.mean(windows[window_name])


def power_to_db(power: float, ref: float = 1.0, epsilon: float = 1e-40) -> float:
    """线性功率转换为 dB。

    使用 epsilon 避免 log(0) 导致 -inf。

    Args:
        power:    线性功率值
        ref:      参考功率（默认 1.0，即 dBW）
        epsilon:  防止 log(0) 的最小值

    Returns:
        功率的 dB 值
    """
    return 10 * np.log10(np.maximum(power, epsilon) / ref)


def db_to_power(db: float) -> float:
    """dB 转换为线性功率。"""
    return 10 ** (db / 10)
