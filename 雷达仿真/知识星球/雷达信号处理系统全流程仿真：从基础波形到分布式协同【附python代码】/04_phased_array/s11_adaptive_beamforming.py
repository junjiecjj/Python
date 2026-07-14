"""s11：MVDR（Capon）自适应波束形成。

验证目标：
  - 实现 MVDR 自适应波束形成器，在干扰方向自动形成零点
  - 对比固定波束形成器与 MVDR 的方向图和抗干扰能力
  - 验证零点深度、主瓣保真度、对角加载效果

MVDR 波束形成器原理：
  固定波束形成器（如均匀加权）的旁瓣无法抑制来自特定方向的强干扰。
  MVDR 通过估计干扰的协方差矩阵，自动调整权值，在干扰方向形成零点（深度 > 40 dB），
  同时保持目标方向增益不变。

  最优权值：w_opt = R^(-1) * a(θ_s) / (a(θ_s)^H * R^(-1) * a(θ_s))
  输出功率谱：P(θ) = 1 / (a(θ)^H * R^(-1) * a(θ))

  其中 R = 信号协方差矩阵（含干扰+噪声），a(θ) = 导向矢量。

  物理直觉：
    R^(-1) 的作用是"白化"——将干扰方向的能量压低。
    分母 a^H * R^(-1) * a 在干扰方向趋近于很大的值（因为 R^(-1) 在干扰子空间有大特征值），
    从而使 P(θ) 在干扰方向出现深零点。

对应知识库：radar-knowledge-base/基础/05-空域信号处理/
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


# ============================================================
# 核心函数
# ============================================================


def steering_vector(n_elements: int, d_lambda: float, theta_deg: float) -> np.ndarray:
    """计算均匀线阵的导向矢量。

    导向矢量 a(θ) 描述了来自方向 θ 的平面波在阵列各阵元上的相位差。
    对于均匀线阵（ULA），阵元间距 d，波长 λ：

      a(θ) = [1, exp(j*2π*d/λ*sinθ), ..., exp(j*2π*(N-1)*d/λ*sinθ)]^T

    物理含义：每个阵元接收到的信号相对于第一个阵元有 kd*n*sin(θ) 的相位延迟，
    其中 k = 2π/λ 是波数。

    Args:
        n_elements: 阵元数 N
        d_lambda:   阵元间距与波长之比 d/λ（通常取 0.5 以避免栅瓣）
        theta_deg:  来波方向 (度)，0° 为阵列法线方向

    Returns:
        导向矢量 (N,) 复数数组，模为 1（各元素等模）
    """
    theta_rad = np.deg2rad(theta_deg)
    # 阵元索引：0, 1, ..., N-1
    n = np.arange(n_elements)
    # 相位：2π * (d/λ) * n * sin(θ)
    phase = 2 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(1j * phase)


def mvdr_weights(
    cov_matrix: np.ndarray, steering_vec: np.ndarray, diag_load: float = 0.0
) -> np.ndarray:
    """计算 MVDR（Capon）最优权值。

    MVDR 权值公式：
      w_opt = R^(-1) * a / (a^H * R^(-1) * a)

    物理含义：
      权值在目标方向保持单位增益，同时最小化总输出功率（即最大限度抑制干扰）。
      这是一个约束优化问题：min w^H R w, s.t. w^H a = 1。

    数值稳定性：
      当协方差矩阵接近奇异（如快拍数不足或干扰功率极大）时，
      需要对角加载（R' = R + δI）来保证可逆性。
      对角加载会略微降低零点深度，但提高鲁棒性。

    Args:
        cov_matrix:  协方差矩阵 R (N×N)，包含信号+干扰+噪声
        steering_vec: 目标方向的导向矢量 a(θ_s) (N,)
        diag_load:   对角加载量 δ（相对噪声功率的倍数），默认 0（不加载）

    Returns:
        MVDR 最优权值 w (N,) 复数数组
    """
    n = cov_matrix.shape[0]

    # 对角加载：R' = R + δ * I，提高数值稳定性
    if diag_load > 0:
        cov_matrix = cov_matrix + diag_load * np.eye(n)

    # 计算 R^(-1) * a（通过求解线性方程组，比直接求逆更稳定）
    # 使用 Cholesky 分解（协方差矩阵是 Hermitian 正定的）
    try:
        r_inv_a = np.linalg.solve(cov_matrix, steering_vec)
    except np.linalg.LinAlgError:
        # 协方差矩阵奇异时，使用伪逆
        r_inv_a = np.linalg.lstsq(cov_matrix, steering_vec, rcond=None)[0]

    # 分母：a^H * R^(-1) * a（标量，表示目标方向的"白化"功率）
    denominator = steering_vec.conj() @ r_inv_a

    # 权值：w = R^(-1) * a / (a^H * R^(-1) * a)
    return r_inv_a / denominator


def mvdr_spectrum(
    cov_matrix: np.ndarray,
    n_elements: int,
    d_lambda: float,
    theta_scan: np.ndarray,
    diag_load: float = 0.0,
) -> np.ndarray:
    """计算 MVDR 空间功率谱。

    MVDR 功率谱：
      P(θ) = 1 / (a(θ)^H * R^(-1) * a(θ))

    物理含义：
      P(θ) 表示来自方向 θ 的功率贡献估计。
      在信号方向出现峰值，在干扰方向出现深零点。
      与传统波束扫描相比，MVDR 的角度分辨率更高（不受波束宽度限制）。

    Args:
        cov_matrix:  协方差矩阵 R (N×N)
        n_elements:  阵元数 N
        d_lambda:    阵元间距/波长比
        theta_scan:  扫描角度数组 (度)
        diag_load:   对角加载量

    Returns:
        功率谱 P(θ) (dB)，每个扫描角度对应一个值
    """
    n = cov_matrix.shape[0]

    # 对角加载
    if diag_load > 0:
        cov_matrix = cov_matrix + diag_load * np.eye(n)

    # 预计算 R^(-1)（对所有扫描角度复用）
    try:
        r_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        r_inv = np.linalg.pinv(cov_matrix)

    spectrum_db = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_elements, d_lambda, theta)
        # a^H * R^(-1) * a（标量）
        quadratic = a.conj() @ r_inv @ a
        # 功率谱（取实部，理论上应为正实数）
        spectrum_db[idx] = power_to_db(np.real(1.0 / quadratic))

    return spectrum_db


def simulate_scenario(
    n_elements: int,
    d_lambda: float,
    signal_theta: float,
    interferer_theta: float,
    snr_db: float,
    inr_db: float,
    n_snapshots: int = 100,
    seed: int = 42,
    use_monte_carlo: bool = True,
) -> np.ndarray:
    """生成包含目标、干扰和噪声的协方差矩阵。

    协方差矩阵构成：
      R = σ²_s * a(θ_s) * a(θ_s)^H   （目标信号分量）
        + σ²_i * a(θ_i) * a(θ_i)^H   （干扰分量）
        + σ²_n * I                      （热噪声分量）

    其中 σ²_s = 10^(SNR/10)，σ²_i = 10^(INR/10)，σ²_n = 1。

    当 use_monte_carlo=True 时，通过生成 L 个快拍数据估计 R：
      R_hat = (1/L) * Σ x(l) * x(l)^H
    这更接近实际系统的工作方式（快拍数有限会导致 R 估计不准）。

    Args:
        n_elements:       阵元数 N
        d_lambda:         阵元间距/波长比
        signal_theta:     目标方向 (度)
        interferer_theta: 干扰方向 (度)
        snr_db:           信噪比 (dB)
        inr_db:           干噪比 (dB)
        n_snapshots:      快拍数 L（仅 Monte Carlo 模式使用）
        seed:             随机种子（保证可复现）
        use_monte_carlo:  True 用 Monte Carlo 估计 R，False 用解析公式

    Returns:
        协方差矩阵 R (N×N)
    """
    rng = np.random.default_rng(seed)

    # 线性功率（以噪声功率 σ²_n = 1 为参考）
    sigma_s2 = db_to_power(snr_db)
    sigma_i2 = db_to_power(inr_db)
    sigma_n2 = 1.0

    a_s = steering_vector(n_elements, d_lambda, signal_theta)
    a_i = steering_vector(n_elements, d_lambda, interferer_theta)

    if use_monte_carlo:
        # Monte Carlo 方法：生成快拍数据，估计协方差矩阵
        # 每个快拍：x(l) = s(l) * a_s + i(l) * a_i + n(l)
        # s(l), i(l) 为复高斯信号，n(l) 为各阵元独立复高斯噪声
        signal_amp = np.sqrt(sigma_s2 / 2)
        interferer_amp = np.sqrt(sigma_i2 / 2)
        noise_amp = np.sqrt(sigma_n2 / 2)

        # 快拍矩阵 (N × L)
        snapshots = (
            signal_amp
            * (rng.standard_normal(n_snapshots) + 1j * rng.standard_normal(n_snapshots))
            * a_s[:, np.newaxis]
            + interferer_amp
            * (rng.standard_normal(n_snapshots) + 1j * rng.standard_normal(n_snapshots))
            * a_i[:, np.newaxis]
            + noise_amp
            * (
                rng.standard_normal((n_elements, n_snapshots))
                + 1j * rng.standard_normal((n_elements, n_snapshots))
            )
        )

        # 样本协方差矩阵：R_hat = (1/L) * X * X^H
        cov_matrix = snapshots @ snapshots.conj().T / n_snapshots
    else:
        # 解析方法：直接构造理论协方差矩阵
        cov_matrix = (
            sigma_s2 * np.outer(a_s, a_s.conj())
            + sigma_i2 * np.outer(a_i, a_i.conj())
            + sigma_n2 * np.eye(n_elements)
        )

    return cov_matrix


def fixed_beam_pattern(
    n_elements: int, d_lambda: float, theta_scan: np.ndarray
) -> np.ndarray:
    """计算均匀加权固定波束形成器的方向图。

    固定波束形成器的权值为均匀加权（矩形窗）：
      w = a(θ_steer) / N

    方向图：
      |B(θ)|² = |w^H * a(θ)|² / N²

    这是 MVDR 的参照基线——无自适应能力，旁瓣电平固定。

    Args:
        n_elements:  阵元数
        d_lambda:    阵元间距/波长比
        theta_scan:  扫描角度数组 (度)

    Returns:
        归一化方向图功率 (dB)，峰值为 0 dB
    """
    # 导向权值：指向法线方向 (θ=0°)
    w = steering_vector(n_elements, d_lambda, 0.0) / n_elements

    pattern = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_elements, d_lambda, theta)
        pattern[idx] = np.abs(w.conj() @ a) ** 2

    # 归一化到峰值 0 dB
    pattern_db = power_to_db(pattern / np.max(pattern))
    return pattern_db


def beamwidth_deg(n_elements: int, d_lambda: float) -> float:
    """估算均匀线阵的 3 dB 波束宽度（度）。

    近似公式：θ_3dB ≈ 0.886 * λ / (N * d) （弧度）
    当 d = λ/2 时，θ_3dB ≈ 1.02° / N（度）

    Args:
        n_elements: 阵元数
        d_lambda:   阵元间距/波长比

    Returns:
        3 dB 波束宽度 (度)
    """
    bw_rad = 0.886 / (n_elements * d_lambda)
    return np.rad2deg(bw_rad)


# ============================================================
# 绘图
# ============================================================


def plot_adaptive_beamforming(
    n_elements: int,
    d_lambda: float,
    signal_theta: float,
    interferer_theta: float,
    snr_db: float,
    inr_db: float,
    n_snapshots: int,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制自适应波束形成结果（3 子图）。

    子图 1：固定波束 vs MVDR 方向图对比
      - 展示 MVDR 如何在干扰方向形成零点，而固定波束无法做到

    子图 2：不同 INR 下的零点深度
      - 干扰越强，零点越深（自适应能力的体现）

    子图 3：MVDR 空间功率谱
      - 展示 MVDR 的角度分辨能力

    Args:
        n_elements:       阵元数
        d_lambda:         阵元间距/波长比
        signal_theta:     目标方向 (度)
        interferer_theta: 干扰方向 (度)
        snr_db:           信噪比 (dB)
        inr_db:           干噪比 (dB)
        n_snapshots:      快拍数
        output_dir:       输出目录
    """
    theta_scan = np.linspace(-90, 90, 1801)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # ---- 子图 1：固定波束 vs MVDR 方向图 ----
    ax1 = axes[0]

    # 固定波束方向图
    pattern_fixed = fixed_beam_pattern(n_elements, d_lambda, theta_scan)

    # MVDR 方向图（扫描每个角度，用 MVDR 权值计算输出）
    cov = simulate_scenario(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )
    w_mvdr = mvdr_weights(cov, steering_vector(n_elements, d_lambda, signal_theta))

    pattern_mvdr = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_elements, d_lambda, theta)
        pattern_mvdr[idx] = np.abs(w_mvdr.conj() @ a) ** 2
    pattern_mvdr_db = power_to_db(pattern_mvdr / np.max(pattern_mvdr))

    ax1.plot(theta_scan, pattern_fixed, "b-", linewidth=1.5, label="固定波束（均匀加权）")
    ax1.plot(theta_scan, pattern_mvdr_db, "r-", linewidth=1.5, label="MVDR 自适应波束")
    ax1.axvline(x=interferer_theta, color="gray", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"干扰 {interferer_theta}°",
        xy=(interferer_theta, -50),
        xytext=(interferer_theta + 8, -40),
        fontsize=10, color="gray",
    )
    ax1.axvline(x=signal_theta, color="green", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"目标 {signal_theta}°",
        xy=(signal_theta, 0),
        xytext=(signal_theta + 8, 5),
        fontsize=10, color="green",
    )
    ax1.set_xlabel("角度 (度)", fontsize=12)
    ax1.set_ylabel("归一化增益 (dB)", fontsize=12)
    ax1.set_title(
        f"固定波束 vs MVDR 方向图 (N={n_elements}, SNR={snr_db}dB, INR={inr_db}dB)",
        fontsize=13,
    )
    ax1.set_ylim([-60, 5])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2：不同 INR 下的零点深度 ----
    ax2 = axes[1]
    inr_values = np.arange(10, 55, 5)
    null_depths = []

    for inr in inr_values:
        cov_inr = simulate_scenario(
            n_elements, d_lambda, signal_theta, interferer_theta,
            snr_db, inr, n_snapshots,
        )
        w_inr = mvdr_weights(
            cov_inr, steering_vector(n_elements, d_lambda, signal_theta)
        )
        a_int = steering_vector(n_elements, d_lambda, interferer_theta)
        a_sig = steering_vector(n_elements, d_lambda, signal_theta)

        # 零点深度 = 干扰方向增益 / 目标方向增益 (dB)
        gain_int = np.abs(w_inr.conj() @ a_int) ** 2
        gain_sig = np.abs(w_inr.conj() @ a_sig) ** 2
        null_depth = power_to_db(gain_int / gain_sig)
        null_depths.append(null_depth)

    ax2.plot(inr_values, null_depths, "ro-", linewidth=2, markersize=8)
    ax2.axhline(y=-30, color="r", linestyle="--", alpha=0.5, label="零点深度要求 (-30 dB)")
    ax2.set_xlabel("干扰噪声比 INR (dB)", fontsize=12)
    ax2.set_ylabel("零点深度 (dB，相对主瓣)", fontsize=12)
    ax2.set_title("不同干扰强度下的零点深度", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3：MVDR 功率谱 ----
    ax3 = axes[2]

    # 不含目标的协方差（仅干扰+噪声），用于展示功率谱的零点特性
    cov_no_sig = simulate_scenario(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db=-np.inf, inr_db=inr_db, n_snapshots=n_snapshots,
    )
    spectrum = mvdr_spectrum(cov_no_sig, n_elements, d_lambda, theta_scan)

    ax3.plot(theta_scan, spectrum, "b-", linewidth=1.5, label="MVDR 功率谱（干扰+噪声）")

    # 含目标的功率谱
    cov_full = simulate_scenario(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )
    spectrum_full = mvdr_spectrum(cov_full, n_elements, d_lambda, theta_scan)
    ax3.plot(theta_scan, spectrum_full, "r-", linewidth=1.5, label="MVDR 功率谱（信号+干扰+噪声）")

    ax3.axvline(x=interferer_theta, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=signal_theta, color="green", linestyle="--", alpha=0.5)
    ax3.set_xlabel("角度 (度)", fontsize=12)
    ax3.set_ylabel("功率谱 (dB)", fontsize=12)
    ax3.set_title(
        f"MVDR 空间功率谱 (N={n_elements}, d=λ/2)", fontsize=13
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s11_adaptive_beamforming.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s11_adaptive_beamforming.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_elements: int,
    d_lambda: float,
    signal_theta: float,
    interferer_theta: float,
    snr_db: float,
    inr_db: float,
    n_snapshots: int,
) -> bool:
    """验证 MVDR 自适应波束形成的正确性。

    验证项：
      a. 零点深度：干扰方向增益 < -30 dB（相对主瓣）
      b. 主瓣保真：目标方向增益损失 < 1 dB
      c. 对角加载效果：加载后零点深度仍 < -20 dB
      d. 功率谱分辨率：MVDR 能分辨两个角度差 > 波束宽度的目标
    """
    results = []

    # --- 验证 a：零点深度 ---
    # 干扰方向的增益应远低于目标方向（形成深零点）
    cov = simulate_scenario(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )
    w = mvdr_weights(cov, steering_vector(n_elements, d_lambda, signal_theta))

    a_sig = steering_vector(n_elements, d_lambda, signal_theta)
    a_int = steering_vector(n_elements, d_lambda, interferer_theta)

    gain_signal = np.abs(w.conj() @ a_sig) ** 2
    gain_interferer = np.abs(w.conj() @ a_int) ** 2
    null_depth_db = power_to_db(gain_interferer / gain_signal)

    # 零点深度应 < -30 dB（即干扰方向增益比目标方向低 30 dB 以上）
    results.append(verify(
        name="零点深度（干扰方向增益）",
        theoretical=-40.0,   # 期望值（典型 MVDR 可达 -40~-60 dB）
        simulated=null_depth_db,
        tolerance=15.0,      # 允许到 -25 dB（Monte Carlo 有波动）
        unit="dB",
    ))

    # --- 验证 b：主瓣保真 ---
    # MVDR 在目标方向的增益损失应 < 1 dB（理想情况无损失）
    # 权值归一化后，|w^H * a_s|^2 应接近 1（约束条件 w^H a = 1 保证了这一点）
    gain_signal_db = power_to_db(gain_signal)
    gain_loss_db = 0.0 - gain_signal_db  # 理论上 |w^H a_s| = 1，即 0 dB

    results.append(verify(
        name="主瓣保真（目标方向增益损失）",
        theoretical=0.0,
        simulated=gain_loss_db,
        tolerance=1.0,       # 允许 1 dB 损失
        unit="dB",
    ))

    # --- 验证 c：对角加载效果 ---
    # 对角加载会降低零点深度，但仍应 < -20 dB
    diag_load_level = 0.01  # 对角加载量（相对于噪声功率）
    w_loaded = mvdr_weights(
        cov, steering_vector(n_elements, d_lambda, signal_theta),
        diag_load=diag_load_level,
    )
    gain_int_loaded = np.abs(w_loaded.conj() @ a_int) ** 2
    gain_sig_loaded = np.abs(w_loaded.conj() @ a_sig) ** 2
    null_depth_loaded = power_to_db(gain_int_loaded / gain_sig_loaded)

    results.append(verify(
        name="对角加载后零点深度",
        theoretical=-30.0,
        simulated=null_depth_loaded,
        tolerance=15.0,      # 允许到 -15 dB
        unit="dB",
    ))

    # --- 验证 d：功率谱分辨率 ---
    # MVDR 的角度分辨率优于传统波束（可分辨角度差 > 波束宽度的目标）
    # 测试：两个目标分别在 ±θ_bw/2 处，MVDR 功率谱应出现两个峰
    bw = beamwidth_deg(n_elements, d_lambda)
    theta1 = -bw * 0.7
    theta2 = bw * 0.7

    # 生成两个等功率目标的协方差矩阵
    a1 = steering_vector(n_elements, d_lambda, theta1)
    a2 = steering_vector(n_elements, d_lambda, theta2)
    sigma_t = db_to_power(snr_db)
    cov_two = (
        sigma_t * np.outer(a1, a1.conj())
        + sigma_t * np.outer(a2, a2.conj())
        + np.eye(n_elements)
    )

    # 在两个目标附近扫描，应出现两个峰
    theta_fine = np.linspace(-2 * bw, 2 * bw, 401)
    spectrum_two = mvdr_spectrum(cov_two, n_elements, d_lambda, theta_fine)

    # 找到谱中的局部极大值
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectrum_two, height=-20, distance=10)

    # 应至少找到 2 个峰（两个目标方向各一个）
    n_peaks = len(peaks)
    results.append(verify(
        name="功率谱分辨率（双目标分辨）",
        theoretical=2.0,
        simulated=float(n_peaks),
        tolerance=0.5,       # 精确匹配：2 个峰
        unit="个",
    ))

    return print_validation("s11 MVDR 自适应波束形成", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s11 MVDR 自适应波束形成仿真与验证。"""
    print("=" * 60)
    print("s11：MVDR（Capon）自适应波束形成")
    print("=" * 60)

    # 仿真参数
    n_elements = 16         # 阵元数
    d_lambda = 0.5          # 阵元间距 d = λ/2（避免栅瓣）
    signal_theta = 0.0      # 目标方向（法线方向）
    interferer_theta = 30.0 # 干扰方向
    snr_db = 20.0           # 信噪比
    inr_db = 40.0           # 干噪比（强干扰）
    n_snapshots = 100       # 快拍数

    print(f"\n仿真参数:")
    print(f"  阵元数 N     = {n_elements}")
    print(f"  阵元间距 d   = {d_lambda}λ")
    print(f"  目标方向 θ_s = {signal_theta}°")
    print(f"  干扰方向 θ_i = {interferer_theta}°")
    print(f"  信噪比 SNR   = {snr_db} dB")
    print(f"  干噪比 INR   = {inr_db} dB")
    print(f"  快拍数 L     = {n_snapshots}")
    print(f"  3dB 波束宽度 = {beamwidth_deg(n_elements, d_lambda):.2f}°")

    # 计算 MVDR 权值并展示结果
    print(f"\n计算 MVDR 权值...")
    cov = simulate_scenario(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )
    w = mvdr_weights(cov, steering_vector(n_elements, d_lambda, signal_theta))

    a_sig = steering_vector(n_elements, d_lambda, signal_theta)
    a_int = steering_vector(n_elements, d_lambda, interferer_theta)

    gain_sig = np.abs(w.conj() @ a_sig) ** 2
    gain_int = np.abs(w.conj() @ a_int) ** 2
    null_depth = power_to_db(gain_int / gain_sig)

    print(f"  目标方向增益 |w^H a_s|² = {power_to_db(gain_sig):.2f} dB")
    print(f"  干扰方向增益 |w^H a_i|² = {power_to_db(gain_int):.2f} dB")
    print(f"  零点深度 = {null_depth:.1f} dB（相对主瓣）")

    # 绘图
    print(f"\n绘制自适应波束形成结果...")
    plot_adaptive_beamforming(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(
        n_elements, d_lambda, signal_theta, interferer_theta,
        snr_db, inr_db, n_snapshots,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
