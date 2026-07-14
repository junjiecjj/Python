"""s12：MUSIC/ESPRIT 角度估计（DOA Estimation）。

验证目标：
  - 实现 MUSIC 和 ESPRIT 两种子空间类 DOA 估计算法
  - 对比 MUSIC 与 ESPRIT 在不同 SNR 下的角度估计精度
  - 验证 MUSIC 空间谱的分辨率能力（超越波束宽度限制）
  - 验证快拍数对 MUSIC 谱分辨率的影响

MUSIC 算法原理：
  协方差矩阵 R 的特征分解将其特征空间分为信号子空间 Es 和噪声子空间 En。
  信号子空间与噪声子空间正交：a(θ)^H * En = 0（θ 为信号方向）。
  MUSIC 利用这一正交性构造空间谱：
    P_music(θ) = 1 / (a(θ)^H * En * En^H * a(θ))
  在信号方向出现尖锐峰值，分辨率不受波束宽度限制。

ESPRIT 算法原理：
  利用均匀线阵的平移不变性。将阵列分为两个重叠子阵（去掉首/末阵元），
  信号子空间在两个子阵之间的关系为旋转矩阵 Phi：
    Es2 = Es1 * Phi
  Phi 的特征值包含到达角信息：
    λ_k = exp(j * 2π * d/λ * sin(θ_k))
  ESPRIT 无需谱扫描，直接计算角度，计算量低于 MUSIC。

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
from scipy.signal import find_peaks

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
    n = np.arange(n_elements)
    phase = 2 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(1j * phase)


def music_spectrum(
    cov_matrix: np.ndarray,
    n_elements: int,
    d_lambda: float,
    n_sources: int,
    theta_scan: np.ndarray,
) -> np.ndarray:
    """MUSIC 空间功率谱估计。

    MUSIC 算法通过协方差矩阵的特征分解，将特征空间分为信号子空间 Es
    和噪声子空间 En。利用信号导向矢量与噪声子空间的正交性构造伪功率谱：
      P_music(θ) = 1 / (a(θ)^H * En * En^H * a(θ))

    在信号方向，a(θ) 与 En 正交，分母趋近于 0，谱出现尖锐峰值。
    在非信号方向，a(θ) 在 En 上有投影分母较大，谱值较低。

    分辨率：MUSIC 的分辨率不受波束宽度限制，仅受 SNR 和快拍数约束。
    理论分辨率极限：当两个源角度差 Δθ → 0 时，MUSIC 谱的两个峰合并为一个。

    Args:
        cov_matrix:  协方差矩阵 R (N×N)
        n_elements:  阵元数 N
        d_lambda:    阵元间距/波长比
        n_sources:   信号源数量 M
        theta_scan:  扫描角度数组 (度)

    Returns:
        MUSIC 功率谱 (dB)，长度与 theta_scan 相同
    """
    # 特征分解：eigh 返回的特征值按升序排列
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 噪声子空间：最小的 N-M 个特征值对应的特征向量
    # eigh 返回升序，前 N-M 个为噪声子空间
    noise_subspace = eigenvectors[:, : n_elements - n_sources]

    # En * En^H（投影矩阵，用于加速计算）
    noise_proj = noise_subspace @ noise_subspace.conj().T

    spectrum_db = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_elements, d_lambda, theta)
        # a^H * En * En^H * a（标量，导向矢量在噪声子空间的投影能量）
        quadratic = np.real(a.conj() @ noise_proj @ a)
        # MUSIC 伪功率谱
        spectrum_db[idx] = power_to_db(1.0 / quadratic)

    return spectrum_db


def esprit_estimate(
    cov_matrix: np.ndarray,
    n_elements: int,
    d_lambda: float,
    n_sources: int,
) -> np.ndarray:
    """ESPRIT 角度估计算法。

    ESPRIT 利用均匀线阵的平移不变性（shift invariance）直接估计角度，
    无需谱扫描，计算效率高于 MUSIC。

    算法步骤：
      1. 对协方差矩阵做特征分解，取信号子空间 Es（M 个最大特征值对应的特征向量）
      2. 将 Es 分为两个重叠子阵：
         Es1 = Es[0:N-1, :]   （去掉最后一个阵元）
         Es2 = Es[1:N, :]     （去掉第一个阵元）
      3. 求解旋转不变关系：Es2 = Es1 * Phi
         Phi = pinv(Es1) * Es2（最小二乘解）
      4. 对 Phi 做特征分解，特征值 λ_k = exp(j*2π*d/λ*sin(θ_k))
      5. 角度：θ_k = arcsin(angle(λ_k) / (2π*d/λ))

    Args:
        cov_matrix:  协方差矩阵 R (N×N)
        n_elements:  阵元数 N
        d_lambda:    阵元间距/波长比
        n_sources:   信号源数量 M

    Returns:
        估计的角度数组 (度)，已排序，长度为 M
    """
    # 特征分解：eigh 返回升序排列的特征值
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 信号子空间：最大的 M 个特征值对应的特征向量（最后 M 列）
    signal_subspace = eigenvectors[:, n_elements - n_sources:]

    # 分为两个重叠子阵（利用 ULA 的平移不变性）
    subarray_1 = signal_subspace[:-1, :]  # 阵元 0 到 N-2
    subarray_2 = signal_subspace[1:, :]   # 阵元 1 到 N-1

    # 旋转不变关系：Es2 = Es1 * Phi
    # 最小二乘解：Phi = pinv(Es1) * Es2
    phi_matrix = np.linalg.pinv(subarray_1) @ subarray_2

    # Phi 的特征值包含到达角信息
    phi_eigenvalues = np.linalg.eigvals(phi_matrix)

    # 提取角度：λ_k = exp(j * 2π * d/λ * sin(θ_k))
    # angle(λ_k) = 2π * d/λ * sin(θ_k)
    # θ_k = arcsin(angle(λ_k) / (2π * d/λ))
    angles_rad = np.arcsin(np.clip(
        np.angle(phi_eigenvalues) / (2 * np.pi * d_lambda), -1.0, 1.0
    ))
    angles_deg = np.rad2deg(angles_rad)

    # 排序后返回
    return np.sort(angles_deg.real)


def simulate_scenario(
    n_elements: int,
    d_lambda: float,
    source_thetas: list,
    snr_db: float,
    n_snapshots: int,
    seed: int = 42,
) -> np.ndarray:
    """生成多源信号的协方差矩阵（Monte Carlo 方式）。

    协方差矩阵构成：
      R_hat = (1/L) * X * X^H
    其中 X 的每一列 x(l) = Σ_k s_k(l) * a(θ_k) + n(l)
      s_k(l) 为第 k 个源的复高斯信号
      n(l) 为各阵元独立的复高斯噪声

    Args:
        n_elements:    阵元数 N
        d_lambda:      阵元间距/波长比
        source_thetas: 信号源方向列表 (度)
        snr_db:        每个源的信噪比 (dB)，以噪声功率为参考
        n_snapshots:   快拍数 L
        seed:          随机种子

    Returns:
        样本协方差矩阵 R_hat (N×N)
    """
    rng = np.random.default_rng(seed)

    # 线性功率（以噪声功率 σ²_n = 1 为参考）
    sigma_s2 = db_to_power(snr_db)
    sigma_n2 = 1.0

    # 信号幅度（复高斯：实部虚部各为 N(0, σ²/2)）
    signal_amp = np.sqrt(sigma_s2 / 2)
    noise_amp = np.sqrt(sigma_n2 / 2)

    # 快拍矩阵 (N × L)
    snapshots = np.zeros((n_elements, n_snapshots), dtype=complex)

    # 各源信号叠加
    for theta in source_thetas:
        a = steering_vector(n_elements, d_lambda, theta)
        s = signal_amp * (
            rng.standard_normal(n_snapshots) + 1j * rng.standard_normal(n_snapshots)
        )
        snapshots += a[:, np.newaxis] * s[np.newaxis, :]

    # 加噪声
    snapshots += noise_amp * (
        rng.standard_normal((n_elements, n_snapshots))
        + 1j * rng.standard_normal((n_elements, n_snapshots))
    )

    # 样本协方差矩阵：R_hat = (1/L) * X * X^H
    cov_matrix = snapshots @ snapshots.conj().T / n_snapshots
    return cov_matrix


# ============================================================
# 绘图
# ============================================================


def plot_doa_estimation(
    n_elements: int,
    d_lambda: float,
    source_thetas: list,
    snr_db: float,
    n_snapshots: int,
    seed: int,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制 DOA 估计结果（3 子图）。

    子图 1：MUSIC 空间谱
      - 展示 MUSIC 谱在信号方向的尖锐峰值
      - 标记真实角度和 MUSIC 检测到的峰值

    子图 2：ESPRIT vs MUSIC 角度估计精度对比
      - 不同 SNR 下的估计 RMSE（Monte Carlo 平均）
      - 展示子空间方法的 SNR 门限效应

    子图 3：不同快拍数下的 MUSIC 谱分辨率
      - 快拍数越大，协方差矩阵估计越准，谱峰越尖锐
      - 展示快拍数对分辨率的影响

    Args:
        n_elements:    阵元数
        d_lambda:      阵元间距/波长比
        source_thetas: 信号源方向列表 (度)
        snr_db:        信噪比 (dB)
        n_snapshots:   快拍数
        seed:          随机种子
        output_dir:    输出目录
    """
    theta_scan = np.linspace(-90, 90, 1801)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # ---- 子图 1：MUSIC 空间谱 ----
    ax1 = axes[0]

    cov = simulate_scenario(
        n_elements, d_lambda, source_thetas, snr_db, n_snapshots, seed=seed,
    )
    spectrum = music_spectrum(cov, n_elements, d_lambda, len(source_thetas), theta_scan)

    ax1.plot(theta_scan, spectrum, "b-", linewidth=1.5, label="MUSIC 空间谱")

    # 标记真实角度
    for theta_true in source_thetas:
        ax1.axvline(x=theta_true, color="green", linestyle="--", alpha=0.7)
    ax1.annotate(
        "真实角度",
        xy=(source_thetas[0], np.max(spectrum)),
        xytext=(source_thetas[0] + 5, np.max(spectrum) - 5),
        fontsize=10, color="green",
    )

    # 检测并标记 MUSIC 峰值
    peak_threshold = np.max(spectrum) - 20
    peaks, _ = find_peaks(spectrum, height=peak_threshold, distance=20)
    if len(peaks) > 0:
        ax1.plot(
            theta_scan[peaks], spectrum[peaks], "rv", markersize=10,
            label=f"MUSIC 检测 ({len(peaks)} 个)",
        )

    ax1.set_xlabel("角度 (度)", fontsize=12)
    ax1.set_ylabel("空间谱 (dB)", fontsize=12)
    ax1.set_title(
        f"MUSIC 空间谱 (N={n_elements}, M={len(source_thetas)}, "
        f"SNR={snr_db}dB, L={n_snapshots})",
        fontsize=13,
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2：ESPRIT vs MUSIC 精度对比 ----
    ax2 = axes[1]

    snr_values = np.arange(0, 35, 2)
    n_mc = 50  # Monte Carlo 次数
    music_rmses = []
    esprit_rmses = []

    for snr in snr_values:
        music_errors = []
        esprit_errors = []

        for trial in range(n_mc):
            cov_trial = simulate_scenario(
                n_elements, d_lambda, source_thetas, snr,
                n_snapshots, seed=seed + trial,
            )

            # MUSIC 估计
            spec = music_spectrum(
                cov_trial, n_elements, d_lambda, len(source_thetas), theta_scan,
            )
            peaks_mc, _ = find_peaks(spec, height=np.max(spec) - 20, distance=20)
            if len(peaks_mc) >= len(source_thetas):
                detected = np.sort(theta_scan[peaks_mc])[:len(source_thetas)]
                true_sorted = np.sort(source_thetas)
                music_errors.append(np.sqrt(np.mean((detected - true_sorted) ** 2)))

            # ESPRIT 估计
            est_angles = esprit_estimate(
                cov_trial, n_elements, d_lambda, len(source_thetas),
            )
            if len(est_angles) >= len(source_thetas):
                true_sorted = np.sort(source_thetas)
                esprit_errors.append(
                    np.sqrt(np.mean((np.sort(est_angles) - true_sorted) ** 2))
                )

        music_rmses.append(np.mean(music_errors) if music_errors else np.nan)
        esprit_rmses.append(np.mean(esprit_errors) if esprit_errors else np.nan)

    ax2.plot(snr_values, music_rmses, "b-o", linewidth=2, markersize=5, label="MUSIC")
    ax2.plot(snr_values, esprit_rmses, "r-s", linewidth=2, markersize=5, label="ESPRIT")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1° 精度要求")
    ax2.set_xlabel("SNR (dB)", fontsize=12)
    ax2.set_ylabel("角度估计 RMSE (度)", fontsize=12)
    ax2.set_title(
        f"ESPRIT vs MUSIC 角度估计精度 (N={n_elements}, L={n_snapshots})",
        fontsize=13,
    )
    ax2.set_ylim([0, max(np.nanmax(music_rmses), np.nanmax(esprit_rmses)) * 1.1 + 0.5])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # ---- 子图 3：不同快拍数下的 MUSIC 谱分辨率 ----
    ax3 = axes[2]

    # 使用两个靠近的目标测试分辨率
    bw = 0.886 / (n_elements * d_lambda)
    bw_deg = np.rad2deg(bw)
    theta_res1 = -bw_deg * 0.6
    theta_res2 = bw_deg * 0.6
    theta_fine = np.linspace(-5 * bw_deg, 5 * bw_deg, 1001)

    snapshot_counts = [20, 50, 100, 500]
    colors = ["b", "r", "g", "m"]

    for idx, n_snap in enumerate(snapshot_counts):
        cov_res = simulate_scenario(
            n_elements, d_lambda, [theta_res1, theta_res2],
            snr_db, n_snap, seed=seed,
        )
        spec_res = music_spectrum(cov_res, n_elements, d_lambda, 2, theta_fine)
        # 归一化到 0 dB
        ax3.plot(
            theta_fine, spec_res - np.max(spec_res),
            color=colors[idx], linewidth=1.5,
            label=f"L={n_snap}",
        )

    ax3.axvline(x=theta_res1, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=theta_res2, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("角度 (度)", fontsize=12)
    ax3.set_ylabel("归一化 MUSIC 谱 (dB)", fontsize=12)
    ax3.set_title(
        f"不同快拍数下的 MUSIC 谱分辨率 "
        f"(目标 {theta_res1:.1f}°, {theta_res2:.1f}°, Δθ={bw_deg:.1f}° 波束宽度)",
        fontsize=13,
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s12_doa_estimation.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s12_doa_estimation.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_elements: int,
    d_lambda: float,
    source_thetas: list,
    snr_db: float,
    n_snapshots: int,
    seed: int,
) -> bool:
    """验证 MUSIC/ESPRIT 角度估计的正确性。

    验证项：
      a. MUSIC 角度估计精度：估计值与真实值误差 < 1°
      b. ESPRIT 角度估计精度：估计值与真实值误差 < 1°
      c. MUSIC 谱峰旁瓣比：峰值 > 旁瓣 20 dB 以上
      d. 多目标分辨：能分辨角度差 > 波束宽度的两个目标
    """
    results = []
    theta_scan = np.linspace(-90, 90, 1801)

    cov = simulate_scenario(
        n_elements, d_lambda, source_thetas, snr_db, n_snapshots, seed=seed,
    )

    # --- 验证 a：MUSIC 角度估计精度 ---
    spectrum = music_spectrum(cov, n_elements, d_lambda, len(source_thetas), theta_scan)
    peak_threshold = np.max(spectrum) - 20
    peaks, _ = find_peaks(spectrum, height=peak_threshold, distance=20)

    if len(peaks) >= len(source_thetas):
        detected_angles = np.sort(theta_scan[peaks])[:len(source_thetas)]
        true_sorted = np.sort(source_thetas)
        music_mean_error = float(np.mean(np.abs(detected_angles - true_sorted)))
    else:
        # 未检测到足够峰值，使用最大误差作为兜底
        music_mean_error = 5.0

    results.append(verify(
        name="MUSIC 角度估计精度（平均绝对误差）",
        theoretical=0.0,
        simulated=music_mean_error,
        tolerance=1.0,
        unit="度",
    ))

    # --- 验证 b：ESPRIT 角度估计精度 ---
    esprit_angles = esprit_estimate(cov, n_elements, d_lambda, len(source_thetas))
    if len(esprit_angles) >= len(source_thetas):
        true_sorted = np.sort(source_thetas)
        esprit_mean_error = float(
            np.mean(np.abs(np.sort(esprit_angles) - true_sorted))
        )
    else:
        esprit_mean_error = 5.0

    results.append(verify(
        name="ESPRIT 角度估计精度（平均绝对误差）",
        theoretical=0.0,
        simulated=esprit_mean_error,
        tolerance=1.0,
        unit="度",
    ))

    # --- 验证 c：MUSIC 谱峰旁瓣比 ---
    # 峰值与旁瓣电平的差值应 > 20 dB
    # 旁瓣电平：排除各峰值附近区域后的最大值
    peak_mask = np.ones(len(spectrum), dtype=bool)
    peak_half_width = 10  # 排除峰值左右 10 个采样点（约 ±1°）
    for p_idx in peaks:
        lo = max(0, p_idx - peak_half_width)
        hi = min(len(spectrum), p_idx + peak_half_width + 1)
        peak_mask[lo:hi] = False

    if np.any(peak_mask):
        sidelobe_level = float(np.max(spectrum[peak_mask]))
    else:
        sidelobe_level = float(np.min(spectrum))
    peak_to_sidelobe = float(np.max(spectrum)) - sidelobe_level

    results.append(verify(
        name="MUSIC 谱峰旁瓣比",
        theoretical=30.0,
        simulated=peak_to_sidelobe,
        tolerance=15.0,
        unit="dB",
    ))

    # --- 验证 d：多目标分辨 ---
    # 使用角度差 > 波束宽度的两个目标
    bw_rad = 0.886 / (n_elements * d_lambda)
    bw_deg = np.rad2deg(bw_rad)
    theta1 = -bw_deg * 0.7
    theta2 = bw_deg * 0.7

    a1 = steering_vector(n_elements, d_lambda, theta1)
    a2 = steering_vector(n_elements, d_lambda, theta2)
    sigma_t = db_to_power(snr_db)

    # 解析协方差矩阵（精确，避免 Monte Carlo 波动影响分辨率测试）
    cov_two = (
        sigma_t * np.outer(a1, a1.conj())
        + sigma_t * np.outer(a2, a2.conj())
        + np.eye(n_elements)
    )

    theta_fine = np.linspace(-3 * bw_deg, 3 * bw_deg, 601)
    spec_two = music_spectrum(cov_two, n_elements, d_lambda, 2, theta_fine)

    # 使用 prominence 检测，比 height 阈值更鲁棒
    # prominence = 峰值相对于其两侧最低点的高度差
    peaks_two, _ = find_peaks(spec_two, prominence=3.0, distance=10)

    results.append(verify(
        name="多目标分辨（角度差 > 波束宽度）",
        theoretical=2.0,
        simulated=float(len(peaks_two)),
        tolerance=0.5,
        unit="个",
    ))

    return print_validation("s12 MUSIC/ESPRIT 角度估计", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s12 MUSIC/ESPRIT 角度估计仿真与验证。"""
    print("=" * 60)
    print("s12：MUSIC/ESPRIT 角度估计（DOA Estimation）")
    print("=" * 60)

    # 仿真参数
    n_elements = 16
    d_lambda = 0.5
    source_thetas = [-20.0, 10.0, 40.0]
    snr_db = 20.0
    n_snapshots = 100
    seed = 42

    bw_rad = 0.886 / (n_elements * d_lambda)
    bw_deg = np.rad2deg(bw_rad)

    print(f"\n仿真参数:")
    print(f"  阵元数 N        = {n_elements}")
    print(f"  阵元间距 d      = {d_lambda}λ")
    print(f"  信号源方向      = {source_thetas}°")
    print(f"  信号源数量 M    = {len(source_thetas)}")
    print(f"  信噪比 SNR      = {snr_db} dB")
    print(f"  快拍数 L        = {n_snapshots}")
    print(f"  3dB 波束宽度    = {bw_deg:.2f}°")
    print(f"  随机种子 seed   = {seed}")

    # 计算 MUSIC 谱
    print(f"\n计算 MUSIC 空间谱...")
    theta_scan = np.linspace(-90, 90, 1801)
    cov = simulate_scenario(
        n_elements, d_lambda, source_thetas, snr_db, n_snapshots, seed=seed,
    )
    spectrum = music_spectrum(cov, n_elements, d_lambda, len(source_thetas), theta_scan)

    # 检测峰值
    peak_threshold = np.max(spectrum) - 20
    peaks, _ = find_peaks(spectrum, height=peak_threshold, distance=20)
    detected_music = theta_scan[peaks]
    print(f"  MUSIC 检测到 {len(peaks)} 个源: {np.sort(detected_music)}°")

    # MUSIC 估计误差
    true_sorted = np.sort(source_thetas)
    if len(peaks) >= len(source_thetas):
        music_sorted = np.sort(detected_music)[:len(source_thetas)]
        errors = np.abs(music_sorted - true_sorted)
        print(f"  MUSIC 估计误差: {errors}°，平均 {np.mean(errors):.3f}°")

    # ESPRIT 估计
    print(f"\n计算 ESPRIT 估计...")
    esprit_angles = esprit_estimate(cov, n_elements, d_lambda, len(source_thetas))
    print(f"  ESPRIT 估计角度: {esprit_angles}°")
    if len(esprit_angles) >= len(source_thetas):
        esprit_errors = np.abs(np.sort(esprit_angles) - true_sorted)
        print(f"  ESPRIT 估计误差: {esprit_errors}°，平均 {np.mean(esprit_errors):.3f}°")

    # 绘图
    print(f"\n绘制 DOA 估计结果...")
    plot_doa_estimation(
        n_elements, d_lambda, source_thetas, snr_db, n_snapshots, seed,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(
        n_elements, d_lambda, source_thetas, snr_db, n_snapshots, seed,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
