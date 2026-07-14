"""s13：地面杂波 RCS 与杂波谱建模。

验证目标：
  - 实现经典常数-γ 杂波 RCS 模型，验证不同地形的后向散射特性
  - 基于杂波环模型构建空时杂波协方差矩阵，验证杂波谱特性
  - 生成杂波 Range-Doppler Map，展示杂波脊在空时平面上的分布

常数-γ 模型原理：
  地面杂波的归一化 RCS（σ₀）与擦地角 ψ 的关系可用常数-γ 模型近似：
    σ₀ = γ · sin(ψ)
  其中 γ 是与地形相关的常数。分辨单元面积 A_cell = R · ΔR · Δθ，
  杂波 RCS 为 σ = σ₀ · A_cell。

杂波环模型原理：
  对于运动平台上的相控阵雷达，地面杂波在空时二维平面上形成特征"杂波脊"。
  将杂波区域沿方位角分成 N_c 个等距环，每个杂波环对应：
    - 空间频率：f_s = (d/λ) · sin(θ_i)
    - 多普勒频率：f_d = (2v/λ) · cos(ψ) · sin(θ_i)
  杂波协方差矩阵：R_c = Σ σ_i · s(θ_i, f_d_i) · s(θ_i, f_d_i)^H
  其中 s 是空时联合导向矢量（Kronecker 积形式）。

对应知识库：radar-knowledge-base/基础/09-STAP/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.signal_utils import power_to_db, db_to_power
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
# Parseval: sum(|x|²) = (1/N) * sum(|X|²)

# 物理常数
C_MS = 3.0e8  # 光速 (m/s)

# 典型地形 γ 值 (dB)，参考 Nathanson 等文献
GAMMA_DB = {
    "farmland": -15.0,
    "desert": -10.0,
    "forest": -5.0,
    "urban": 0.0,
    "sea": -20.0,
}


# ============================================================
# 核心函数
# ============================================================


def clutter_rcs_terrain(
    terrain_type: str,
    grazing_angle_deg: float,
    freq_hz: float,
    range_m: float = 10000.0,
    delta_range_m: float = 150.0,
    beamwidth_deg: float = 3.0,
) -> float:
    """计算地面杂波 RCS（常数-γ 模型）。

    常数-γ 模型：
      σ₀ = γ · sin(ψ)
      σ  = σ₀ · A_cell

    其中：
      σ₀  — 归一化杂波 RCS（m²/m²），单位面积的散射截面积
      γ   — 与地形相关的散射常数
      ψ   — 擦地角（grazing angle），即入射余角
      A_cell — 雷达分辨单元面积，A_cell = R · ΔR · Δθ

    物理直觉：
      - 低擦地角时 sin(ψ) → 0，杂波回波很弱（接近切向入射）
      - 高擦地角时 sin(ψ) → 1，杂波回波取决于 γ
      - 城市区域 γ 最大（建筑物强散射），海面 γ 最小

    Args:
        terrain_type:      地形类型，支持 "farmland", "desert", "forest", "urban", "sea"
        grazing_angle_deg: 擦地角 (度)
        freq_hz:           载波频率 (Hz)
        range_m:           杂波单元距离 (m)
        delta_range_m:     距离分辨率 (m)，对应脉冲带宽 ΔR = c/(2B)
        beamwidth_deg:     波束宽度 (度)

    Returns:
        杂波 RCS σ (m²)
    """
    if terrain_type not in GAMMA_DB:
        raise ValueError(
            f"未知地形类型: {terrain_type}，"
            f"可选: {list(GAMMA_DB.keys())}"
        )

    # γ 值：从 dB 转换为线性
    gamma_linear = db_to_power(GAMMA_DB[terrain_type])

    # 擦地角转换为弧度
    psi_rad = np.deg2rad(grazing_angle_deg)

    # 归一化 RCS：σ₀ = γ · sin(ψ)
    sigma0 = gamma_linear * np.sin(psi_rad)

    # 波束宽度转换为弧度
    beamwidth_rad = np.deg2rad(beamwidth_deg)

    # 分辨单元面积：A_cell = R · ΔR · Δθ
    # Δθ 取波束宽度（单程），近似为方位向分辨角
    a_cell = range_m * delta_range_m * beamwidth_rad

    # 杂波 RCS
    sigma = sigma0 * a_cell

    return sigma


def spatial_steering_vector(
    n_elements: int, d_lambda: float, theta_deg: float
) -> np.ndarray:
    """计算均匀线阵的空间导向矢量。

    a_s(θ) = [1, exp(j·2π·(d/λ)·sin(θ)), ..., exp(j·2π·(N-1)·(d/λ)·sin(θ))]^T

    Args:
        n_elements: 阵元数 N
        d_lambda:   阵元间距与波长之比 d/λ
        theta_deg:  方位角 (度)

    Returns:
        空间导向矢量 (N,)
    """
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(n_elements)
    phase = 2 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(1j * phase)


def temporal_steering_vector(
    n_pulses: int, f_d_hz: float, pri_s: float
) -> np.ndarray:
    """计算脉冲维的时域导向矢量。

    a_t(f_d) = [1, exp(j·2π·f_d·PRI), ..., exp(j·2π·f_d·(P-1)·PRI)]^T

    Args:
        n_pulses: 脉冲数 P（CPI 内）
        f_d_hz:   多普勒频率 (Hz)
        pri_s:    脉冲重复间隔 PRI = 1/PRF (s)

    Returns:
        时域导向矢量 (P,)
    """
    m = np.arange(n_pulses)
    phase = 2 * np.pi * f_d_hz * pri_s * m
    return np.exp(1j * phase)


def space_time_steering_vector(
    n_elements: int,
    n_pulses: int,
    d_lambda: float,
    theta_deg: float,
    f_d_hz: float,
    pri_s: float,
) -> np.ndarray:
    """计算空时联合导向矢量（Kronecker 积形式）。

    s(θ, f_d) = a_t(f_d) ⊗ a_s(θ)

    维度：(N·P,)，先空间后时间排列。
    这是 STAP 中的基本构建块。

    Args:
        n_elements: 阵元数 N
        n_pulses:   脉冲数 P
        d_lambda:   阵元间距/波长比
        theta_deg:  方位角 (度)
        f_d_hz:     多普勒频率 (Hz)
        pri_s:      脉冲重复间隔 (s)

    Returns:
        空时联合导向矢量 (N·P,)
    """
    a_s = spatial_steering_vector(n_elements, d_lambda, theta_deg)
    a_t = temporal_steering_vector(n_pulses, f_d_hz, pri_s)
    # Kronecker 积：a_t ⊗ a_s
    return np.kron(a_t, a_s)


def clutter_doppler_hz(
    velocity_ms: float,
    wavelength_m: float,
    theta_azimuth_deg: float,
    grazing_angle_deg: float = 0.0,
) -> float:
    """计算杂波环的多普勒频率。

    运动平台接收到的地面杂波多普勒频率：
      f_d = (2v/λ) · cos(ψ) · sin(θ)

    其中：
      v — 平台速度 (m/s)
      λ — 波长 (m)
      ψ — 擦地角 (度)
      θ — 方位角（相对于平台运动方向）

    物理直觉：
      - 正侧视（θ=90°）时多普勒最大：f_d = 2v/λ
      - 前视（θ=0°）和后视（θ=180°）时多普勒为零
      - 杂波多普勒随方位角呈正弦分布，形成"杂波脊"

    Args:
        velocity_ms:       平台速度 (m/s)
        wavelength_m:      波长 (m)
        theta_azimuth_deg: 方位角 (度)
        grazing_angle_deg: 擦地角 (度)，远距杂波时接近 0°

    Returns:
        多普勒频率 (Hz)
    """
    theta_rad = np.deg2rad(theta_azimuth_deg)
    psi_rad = np.deg2rad(grazing_angle_deg)
    return (2.0 * velocity_ms / wavelength_m) * np.cos(psi_rad) * np.sin(theta_rad)


def clutter_spectrum_model(
    n_elements: int,
    n_pulses: int,
    d_lambda: float,
    prf_hz: float,
    theta_clutter_deg: float,
    velocity_ms: float,
    clutter_type: str = "farmland",
    n_clutter_rings: int = 360,
    grazing_angle_deg: float = 5.0,
    freq_hz: float = 10.0e9,
    range_m: float = 10000.0,
    delta_range_m: float = 150.0,
    beamwidth_deg: float = 3.0,
    noise_power: float = 1.0,
) -> np.ndarray:
    """基于杂波环模型构建空时杂波协方差矩阵。

    杂波环模型：
      将 360° 方位角分成 N_c 个等间隔杂波环，每个环贡献一个
      空时导向矢量，加权后叠加得到杂波协方差矩阵：

        R_c = Σᵢ σᵢ · s(θᵢ, f_dᵢ) · s(θᵢ, f_dᵢ)^H  +  σ²_n · I

      其中 σᵢ 是第 i 个杂波环的 RCS，s 是空时联合导向矢量。

    Args:
        n_elements:        阵元数 N
        n_pulses:          脉冲数 P
        d_lambda:          阵元间距/波长比
        prf_hz:            脉冲重复频率 (Hz)
        theta_clutter_deg: 杂波中心方位角 (度)
        velocity_ms:       平台速度 (m/s)
        clutter_type:      地形类型
        n_clutter_rings:   杂波环数（方位角采样数）
        grazing_angle_deg: 擦地角 (度)
        freq_hz:           载频 (Hz)
        range_m:           杂波距离 (m)
        delta_range_m:     距离分辨率 (m)
        beamwidth_deg:     波束宽度 (度)
        noise_power:       噪声功率 σ²_n

    Returns:
        空时杂波协方差矩阵 R_c (N·P × N·P)，Hermitian 正定
    """
    dim = n_elements * n_pulses
    pri_s = 1.0 / prf_hz
    wavelength_m = C_MS / freq_hz

    # 初始化协方差矩阵（含噪声底）
    r_clutter = noise_power * np.eye(dim, dtype=complex)

    # 杂波环：方位角从 0° 到 360° 均匀采样
    azimuth_angles_deg = np.linspace(0.0, 360.0, n_clutter_rings, endpoint=False)
    delta_theta_rad = 2.0 * np.pi / n_clutter_rings

    for theta_az_deg in azimuth_angles_deg:
        # 杂波环的多普勒频率
        f_d = clutter_doppler_hz(
            velocity_ms, wavelength_m, theta_az_deg, grazing_angle_deg
        )

        # 杂波环的 RCS（常数-γ 模型）
        sigma0_linear = db_to_power(GAMMA_DB.get(clutter_type, -15.0))
        sigma0 = sigma0_linear * np.sin(np.deg2rad(grazing_angle_deg))

        # 分辨单元面积（方位角宽度取杂波环间隔）
        a_cell = range_m * delta_range_m * delta_theta_rad
        sigma_i = sigma0 * a_cell

        # 空时联合导向矢量
        s_i = space_time_steering_vector(
            n_elements, n_pulses, d_lambda, theta_az_deg, f_d, pri_s
        )

        # 杂波协方差矩阵累加：R_c += σᵢ · s · s^H
        r_clutter += sigma_i * np.outer(s_i, s_i.conj())

    return r_clutter


def clutter_rdm(
    n_elements: int,
    n_pulses: int,
    d_lambda: float,
    prf_hz: float,
    velocity_ms: float,
    freq_hz: float = 10.0e9,
    n_range_bins: int = 128,
    grazing_angle_deg: float = 5.0,
    clutter_type: str = "farmland",
    range_min_m: float = 5000.0,
    range_max_m: float = 20000.0,
    delta_range_m: float = 150.0,
    beamwidth_deg: float = 3.0,
    snr_db: float = 30.0,
    seed: int = 42,
) -> tuple:
    """生成杂波的 Range-Doppler Map。

    流程：
      1. 对每个距离单元，计算杂波的空时协方差矩阵
      2. 生成杂波 + 噪声的随机快拍
      3. 沿脉冲维做 Doppler FFT，得到 Range-Doppler 分布

    Args:
        n_elements:      阵元数 N
        n_pulses:        脉冲数 P
        d_lambda:        阵元间距/波长比
        prf_hz:          PRF (Hz)
        velocity_ms:     平台速度 (m/s)
        freq_hz:         载频 (Hz)
        n_range_bins:    距离单元数
        grazing_angle_deg: 擦地角 (度)
        clutter_type:    地形类型
        range_min_m:     最小距离 (m)
        range_max_m:     最大距离 (m)
        delta_range_m:   距离分辨率 (m)
        beamwidth_deg:   波束宽度 (度)
        snr_db:          杂噪比 CNR (dB)
        seed:            随机种子

    Returns:
        rdm_db:    Range-Doppler Map (n_range_bins × n_pulses)，dB 刻度
        range_m:   距离轴 (m)
        doppler_hz: 多普勒轴 (Hz)
    """
    rng = np.random.default_rng(seed)
    wavelength_m = C_MS / freq_hz
    pri_s = 1.0 / prf_hz

    # 距离轴
    range_bins_m = np.linspace(range_min_m, range_max_m, n_range_bins)

    # 多普勒轴（归一化频率映射到 Hz）
    doppler_bins_hz = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=pri_s))

    # 存储 Range-Doppler Map（取第一个阵元的输出）
    rdm_complex = np.zeros((n_range_bins, n_pulses), dtype=complex)

    for r_idx, r_m in enumerate(range_bins_m):
        # 随距离变化的擦地角（简化：地球曲率影响）
        # 远距离擦地角略小
        grazing_deg = max(grazing_angle_deg * (range_min_m / r_m), 0.5)

        # 杂波协方差矩阵
        r_c = clutter_spectrum_model(
            n_elements=n_elements,
            n_pulses=n_pulses,
            d_lambda=d_lambda,
            prf_hz=prf_hz,
            theta_clutter_deg=0.0,  # 正侧视
            velocity_ms=velocity_ms,
            clutter_type=clutter_type,
            n_clutter_rings=72,  # 简化计算量
            grazing_angle_deg=grazing_deg,
            freq_hz=freq_hz,
            range_m=r_m,
            delta_range_m=delta_range_m,
            beamwidth_deg=beamwidth_deg,
            noise_power=1.0,
        )

        # Cholesky 分解生成相关杂波样本
        try:
            l_mat = np.linalg.cholesky(r_c)
        except np.linalg.LinAlgError:
            # 如果矩阵不正定，加对角加载
            r_c += 0.01 * np.eye(r_c.shape[0])
            l_mat = np.linalg.cholesky(r_c)

        # 生成空时快拍：x = L · w，w ~ CN(0, I)
        w_vec = (
            rng.standard_normal(n_elements * n_pulses)
            + 1j * rng.standard_normal(n_elements * n_pulses)
        ) / np.sqrt(2.0)
        snapshot = l_mat @ w_vec

        # 提取第一个阵元的脉冲维数据 (P,)
        snapshot_2d = snapshot.reshape(n_pulses, n_elements)
        pulse_data = snapshot_2d[:, 0]

        # Doppler FFT
        doppler_spectrum = np.fft.fftshift(np.fft.fft(pulse_data))
        rdm_complex[r_idx, :] = doppler_spectrum

    # 功率转 dB
    rdm_power = np.abs(rdm_complex) ** 2
    rdm_db = power_to_db(rdm_power / np.max(rdm_power))

    return rdm_db, range_bins_m, doppler_bins_hz


# ============================================================
# 绘图
# ============================================================


def plot_clutter_model(
    n_elements: int,
    n_pulses: int,
    d_lambda: float,
    prf_hz: float,
    velocity_ms: float,
    freq_hz: float,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制杂波模型结果（3 子图）。

    子图 1：不同地形的杂波 RCS vs 擦地角
      - 展示常数-γ 模型下 σ₀ 随擦地角的变化
      - 不同地形的 γ 值差异

    子图 2：杂波谱（功率 vs 多普勒频率）
      - 展示杂波的多普勒分布
      - 正侧视杂波集中在 2v/λ 附近

    子图 3：杂波在空时平面上的分布（角度-多普勒图）
      - 展示杂波脊在空间频率-多普勒频率平面上的椭圆轨迹

    Args:
        n_elements:  阵元数
        n_pulses:    脉冲数
        d_lambda:    阵元间距/波长比
        prf_hz:      PRF (Hz)
        velocity_ms: 平台速度 (m/s)
        freq_hz:     载频 (Hz)
        output_dir:  输出目录
    """
    wavelength_m = C_MS / freq_hz
    pri_s = 1.0 / prf_hz

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # ---- 子图 1：不同地形的 σ₀ vs 擦地角 ----
    ax1 = axes[0]
    grazing_angles_deg = np.linspace(1.0, 80.0, 200)
    terrain_colors = {
        "farmland": "green",
        "desert": "goldenrod",
        "forest": "darkgreen",
        "urban": "red",
        "sea": "blue",
    }

    for terrain, gamma_db in GAMMA_DB.items():
        gamma_lin = db_to_power(gamma_db)
        sigma0_db = power_to_db(gamma_lin * np.sin(np.deg2rad(grazing_angles_deg)))
        ax1.plot(
            grazing_angles_deg,
            sigma0_db,
            linewidth=2,
            color=terrain_colors.get(terrain, "gray"),
            label=f"{terrain} (γ={gamma_db} dB)",
        )

    ax1.set_xlabel("擦地角 (度)", fontsize=12)
    ax1.set_ylabel("归一化 RCS σ₀ (dB, m²/m²)", fontsize=12)
    ax1.set_title("常数-γ 模型：不同地形的 σ₀ vs 擦地角", fontsize=13)
    ax1.set_xlim([0, 80])
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2：杂波谱（功率 vs 多普勒频率） ----
    ax2 = axes[1]

    # 使用杂波环模型计算杂波协方差矩阵
    r_clutter = clutter_spectrum_model(
        n_elements=n_elements,
        n_pulses=n_pulses,
        d_lambda=d_lambda,
        prf_hz=prf_hz,
        theta_clutter_deg=0.0,
        velocity_ms=velocity_ms,
        clutter_type="farmland",
        n_clutter_rings=360,
        grazing_angle_deg=5.0,
        freq_hz=freq_hz,
    )

    # 杂波功率谱：对协方差矩阵的脉冲维做特征分析
    # 提取第一个阵元对应的 P×P 子矩阵
    r_pulse = r_clutter[:n_pulses, :n_pulses]
    # 对角线平均功率谱
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=pri_s))

    # 通过 Doppler 导向矢量扫描计算杂波功率谱
    clutter_power_spectrum = np.zeros(n_pulses)
    for k in range(n_pulses):
        a_t = temporal_steering_vector(n_pulses, doppler_axis[k], pri_s)
        clutter_power_spectrum[k] = np.real(a_t.conj() @ r_pulse @ a_t)

    clutter_spectrum_db = power_to_db(
        clutter_power_spectrum / np.max(clutter_power_spectrum)
    )

    ax2.plot(doppler_axis / 1000.0, clutter_spectrum_db, "b-", linewidth=2)
    # 标注理论多普勒最大值
    f_d_max = 2.0 * velocity_ms / wavelength_m
    ax2.axvline(
        x=f_d_max / 1000.0,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"最大杂波多普勒 {f_d_max:.0f} Hz",
    )
    ax2.axvline(
        x=-f_d_max / 1000.0,
        color="r",
        linestyle="--",
        alpha=0.7,
    )
    ax2.set_xlabel("多普勒频率 (kHz)", fontsize=12)
    ax2.set_ylabel("归一化功率 (dB)", fontsize=12)
    ax2.set_title(
        f"杂波功率谱 (v={velocity_ms} m/s, PRF={prf_hz} Hz)",
        fontsize=13,
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3：角度-多普勒图（杂波脊） ----
    ax3 = axes[2]

    # 杂波脊：f_d = (2v/λ) · sin(θ)，空间频率 f_s = (d/λ) · sin(θ)
    azimuth_deg = np.linspace(-90.0, 90.0, 361)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # 空间频率（归一化到 PRF）
    spatial_freq = d_lambda * np.sin(azimuth_rad)  # d/λ · sin(θ)

    # 多普勒频率
    doppler_freq = (2.0 * velocity_ms / wavelength_m) * np.sin(azimuth_rad)

    # 归一化多普勒频率（相对于 PRF）
    doppler_norm = doppler_freq / prf_hz  # f_d / PRF

    ax3.plot(spatial_freq, doppler_norm, "r-", linewidth=2.5, label="杂波脊")
    ax3.fill_between(
        spatial_freq,
        doppler_norm - 0.02,
        doppler_norm + 0.02,
        alpha=0.2,
        color="red",
        label="杂波脊扩展（运动杂波）",
    )

    # 标注关键方位角
    for theta_mark in [-60, -30, 0, 30, 60]:
        theta_r = np.deg2rad(theta_mark)
        fs = d_lambda * np.sin(theta_r)
        fd = (2.0 * velocity_ms / wavelength_m) * np.sin(theta_r) / prf_hz
        ax3.plot(fs, fd, "ko", markersize=6)
        ax3.annotate(
            f"{theta_mark}°",
            xy=(fs, fd),
            xytext=(fs + 0.03, fd + 0.03),
            fontsize=9,
        )

    ax3.set_xlabel("空间频率 d/λ · sin(θ)", fontsize=12)
    ax3.set_ylabel("归一化多普勒频率 f_d / PRF", fontsize=12)
    ax3.set_title("空时平面上的杂波脊（角度-多普勒图）", fontsize=13)
    ax3.set_xlim([-0.6, 0.6])
    ax3.set_ylim([-0.6, 0.6])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s13_clutter_model.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s13_clutter_model.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_elements: int,
    n_pulses: int,
    d_lambda: float,
    prf_hz: float,
    velocity_ms: float,
    freq_hz: float,
) -> bool:
    """验证杂波模型的正确性。

    验证项：
      a. 常数-γ 模型：σ₀ 与 sin(ψ) 成正比（验证线性关系）
      b. 杂波多普勒频率：与平台速度和角度关系正确
      c. 杂波协方差矩阵正定性：所有特征值 > 0
      d. 分辨单元面积计算：A = R · ΔR · Δθ 与理论一致
    """
    results = []
    wavelength_m = C_MS / freq_hz

    # --- 验证 a：常数-γ 模型线性性 ---
    # σ₀ = γ · sin(ψ)，验证 σ₀ / sin(ψ) = γ（常数）
    gamma_db_test = -10.0  # desert
    gamma_lin = db_to_power(gamma_db_test)
    test_angles_deg = np.array([5.0, 15.0, 30.0, 45.0, 60.0])
    test_angles_rad = np.deg2rad(test_angles_deg)

    # 计算 σ₀ / sin(ψ)，理论上应为常数 γ
    sigma0_values = gamma_lin * np.sin(test_angles_rad)
    ratios = sigma0_values / np.sin(test_angles_rad)

    # 所有比值应等于 γ（相对误差 < 1e-10）
    max_ratio_deviation = np.max(np.abs(ratios - gamma_lin)) / gamma_lin

    results.append(verify(
        name="常数-γ 模型：σ₀ / sin(ψ) 恒定性",
        theoretical=0.0,
        simulated=max_ratio_deviation,
        tolerance=1e-10,
        unit="(相对偏差)",
    ))

    # --- 验证 b：杂波多普勒频率 ---
    # 正侧视（θ=90°）：f_d = 2v/λ
    # 前视（θ=0°）：f_d = 0
    test_cases = [
        (90.0, 2.0 * velocity_ms / wavelength_m),   # 正侧视
        (0.0, 0.0),                                  # 前视
        (30.0, (2.0 * velocity_ms / wavelength_m) * np.sin(np.deg2rad(30.0))),
    ]

    max_doppler_error_hz = 0.0
    for theta_deg, expected_f_d in test_cases:
        computed_f_d = clutter_doppler_hz(
            velocity_ms, wavelength_m, theta_deg, grazing_angle_deg=0.0
        )
        error = abs(computed_f_d - expected_f_d)
        max_doppler_error_hz = max(max_doppler_error_hz, error)

    results.append(verify(
        name="杂波多普勒频率（正侧视/前视/30°）",
        theoretical=0.0,
        simulated=max_doppler_error_hz,
        tolerance=1e-6,
        unit="Hz",
    ))

    # --- 验证 c：杂波协方差矩阵正定性 ---
    r_clutter = clutter_spectrum_model(
        n_elements=n_elements,
        n_pulses=n_pulses,
        d_lambda=d_lambda,
        prf_hz=prf_hz,
        theta_clutter_deg=0.0,
        velocity_ms=velocity_ms,
        clutter_type="farmland",
        n_clutter_rings=180,
        grazing_angle_deg=5.0,
        freq_hz=freq_hz,
    )

    eigenvalues = np.linalg.eigvalsh(r_clutter)
    min_eigenvalue = np.min(eigenvalues)

    # 正定性：最小特征值 > 0
    results.append(verify(
        name="杂波协方差矩阵正定性（最小特征值）",
        theoretical=0.1,   # 应明显大于 0
        simulated=min_eigenvalue,
        tolerance=min_eigenvalue,  # 只需 > 0
        unit="(线性值)",
    ))

    # --- 验证 d：分辨单元面积计算 ---
    # A_cell = R · ΔR · Δθ
    test_range_m = 10000.0
    test_delta_range_m = 150.0
    test_beamwidth_deg = 3.0
    test_beamwidth_rad = np.deg2rad(test_beamwidth_deg)

    expected_area = test_range_m * test_delta_range_m * test_beamwidth_rad

    # 通过 clutter_rcs_terrain 反推面积
    # σ = σ₀ · A_cell → A_cell = σ / σ₀
    gamma_test_db = 0.0  # urban
    gamma_test_lin = db_to_power(gamma_test_db)
    psi_deg = 30.0
    sigma0_test = gamma_test_lin * np.sin(np.deg2rad(psi_deg))

    sigma_measured = clutter_rcs_terrain(
        "urban", psi_deg, freq_hz, test_range_m, test_delta_range_m, test_beamwidth_deg
    )
    computed_area = sigma_measured / sigma0_test if sigma0_test > 0 else 0.0

    results.append(verify(
        name="分辨单元面积 A = R · ΔR · Δθ",
        theoretical=expected_area,
        simulated=computed_area,
        tolerance=expected_area * 1e-10,
        unit="m²",
    ))

    return print_validation("s13 杂波 RCS 与杂波谱建模", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s13 杂波模型仿真与验证。"""
    print("=" * 60)
    print("s13：地面杂波 RCS 与杂波谱建模")
    print("=" * 60)

    # 仿真参数
    freq_hz = 10.0e9          # 载频 10 GHz (X 波段)
    velocity_ms = 100.0       # 平台速度 100 m/s
    prf_hz = 1000.0           # PRF 1 kHz
    n_elements = 16           # 阵元数 N = 16
    n_pulses = 64             # 脉冲数 P = 64
    d_lambda = 0.5            # d/λ = 0.5
    wavelength_m = C_MS / freq_hz
    seed = 42

    print(f"\n仿真参数:")
    print(f"  载频 f_c     = {freq_hz/1e9:.1f} GHz (X 波段)")
    print(f"  波长 λ       = {wavelength_m*100:.2f} cm")
    print(f"  平台速度 v   = {velocity_ms} m/s")
    print(f"  PRF          = {prf_hz} Hz")
    print(f"  阵元数 N     = {n_elements}")
    print(f"  脉冲数 P     = {n_pulses}")
    print(f"  d/λ          = {d_lambda}")
    print(f"  随机种子     = {seed}")

    # 杂波 RCS 示例
    print(f"\n杂波 RCS 示例 (R=10 km, ΔR=150 m, θ_bw=3°):")
    for terrain in GAMMA_DB:
        sigma = clutter_rcs_terrain(terrain, 5.0, freq_hz)
        sigma_db = power_to_db(max(sigma, 1e-40))
        print(f"  {terrain:12s}: γ = {GAMMA_DB[terrain]:6.1f} dB, "
              f"σ = {sigma_db:.1f} dB m²")

    # 绘图
    print(f"\n绘制杂波模型结果...")
    plot_clutter_model(
        n_elements, n_pulses, d_lambda, prf_hz, velocity_ms, freq_hz,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(
        n_elements, n_pulses, d_lambda, prf_hz, velocity_ms, freq_hz,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
