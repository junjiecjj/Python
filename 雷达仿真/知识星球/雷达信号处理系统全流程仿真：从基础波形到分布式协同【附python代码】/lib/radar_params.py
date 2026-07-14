"""雷达系统参数定义。

使用 dataclass 封装雷达系统的核心参数，统一物理量的命名和单位。
所有距离单位：米(m)，功率单位：瓦(W)，频率单位：赫兹(Hz)，时间单位：秒(s)。
"""

from dataclasses import dataclass, field
import numpy as np


# 物理常量
SPEED_OF_LIGHT = 3e8  # 光速 m/s
BOLTZMANN = 1.38e-23  # 玻尔兹曼常数 J/K
STANDARD_TEMP = 290   # 标准噪声温度 K


@dataclass
class RadarParams:
    """雷达系统参数。

    Attributes:
        pt:          峰值发射功率 (W)
        gain_db:     天线增益 (dB)
        freq_hz:     载波频率 (Hz)
        bandwidth_hz:信号带宽 (Hz)
        pulse_width_s:脉冲宽度 (s)
        prf_hz:      脉冲重复频率 (Hz)
        noise_figure_db: 接收机噪声系数 (dB)
        num_pulses:  相参处理间隔内的脉冲数
        target_range_m: 目标距离 (m)
        target_rcs_m2:  目标雷达截面积 (m²)
        target_velocity_ms: 目标径向速度 (m/s)，正值远离
    """
    # 发射参数
    pt: float = 1e6                # 峰值功率 1 MW
    gain_db: float = 30.0          # 天线增益 30 dB (1000 倍线性)
    freq_hz: float = 10e9          # X 波段 10 GHz
    bandwidth_hz: float = 50e6     # 带宽 50 MHz
    pulse_width_s: float = 10e-6   # 脉宽 10 μs
    prf_hz: float = 1000.0         # PRF 1 kHz

    # 接收参数
    noise_figure_db: float = 3.0   # 噪声系数 3 dB

    # 信号处理参数
    num_pulses: int = 64           # 相参处理间隔内脉冲数（应为 2 的幂次，便于 FFT）

    # 目标参数
    target_range_m: float = 50e3   # 目标距离 50 km
    target_rcs_m2: float = 1.0     # 目标 RCS 1 m²
    target_velocity_ms: float = 100.0  # 目标速度 100 m/s

    @property
    def wavelength_m(self) -> float:
        """波长 λ = c / f (m)"""
        return SPEED_OF_LIGHT / self.freq_hz

    @property
    def gain_linear(self) -> float:
        """天线增益的线性值"""
        return 10 ** (self.gain_db / 10)

    @property
    def noise_figure_linear(self) -> float:
        """噪声系数的线性值"""
        return 10 ** (self.noise_figure_db / 10)

    @property
    def noise_power_w(self) -> float:
        """接收机噪声功率 P_n = k * T * F * B (W)

        这是接收机带宽内的总噪声功率，是 SNR 计算的分母。
        """
        return BOLTZMANN * STANDARD_TEMP * self.noise_figure_linear * self.bandwidth_hz

    @property
    def time_bandwidth_product(self) -> float:
        """时宽带宽积 T*B（无量纲）

        脉冲压缩的理论增益。对于 LFM 信号，压缩后 SNR 增益 ≈ T*B。
        例如 T=10μs, B=50MHz → T*B=500 → 增益约 27 dB。
        """
        return self.pulse_width_s * self.bandwidth_hz

    @property
    def range_resolution_m(self) -> float:
        """距离分辨率 ΔR = c / (2B) (m)

        带宽越大，分辨率越好。50 MHz 带宽 → 3 m 分辨率。
        """
        return SPEED_OF_LIGHT / (2 * self.bandwidth_hz)

    @property
    def max_unambiguous_range_m(self) -> float:
        """最大不模糊距离 R_unamb = c / (2 * PRF) (m)

        PRF 太高会导致距离模糊。1 kHz PRF → 150 km 不模糊距离。
        """
        return SPEED_OF_LIGHT / (2 * self.prf_hz)

    @property
    def velocity_resolution_ms(self) -> float:
        """速度分辨率 Δv = λ / (2 * N * T_PRI) (m/s)

        N 个脉冲做 FFT 的速度分辨率。取决于脉冲数和 PRF。
        """
        t_pri = 1.0 / self.prf_hz
        return self.wavelength_m / (2 * self.num_pulses * t_pri)

    @property
    def max_unambiguous_velocity_ms(self) -> float:
        """最大不模糊速度 v_unamb = λ * PRF / 4 (m/s)

        多普勒频移超过 PRF/2 时速度会折叠。
        """
        return self.wavelength_m * self.prf_hz / 4
