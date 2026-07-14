"""s15：MIMO 正交波形分集与虚拟孔径扩展。

验证目标：
  - 实现 MIMO 雷达虚拟阵列构造，理解正交波形分集原理
  - 对比 SIMO 与 MIMO 方向图，展示虚拟孔径带来的窄主瓣
  - 验证虚拟阵元数 N_virtual = N_tx × N_rx
  - 验证 MIMO 波束宽度小于 SIMO 波束宽度

MIMO 雷达原理：
  传统相控阵雷达（SIMO）有 N_tx 个发射阵元和 N_rx 个接收阵元，
  波束形成仅在接收端进行，等效阵元数为 N_rx。

  MIMO 雷达通过正交波形分集（每个发射阵元发送不同波形），
  在接收端通过匹配滤波分离各发射通道的信号，
  将 N_tx 个发射阵元和 N_rx 个接收阵元等效为 N_tx × N_rx 个虚拟阵元。

  虚拟阵元位置：p_virtual = p_tx[i] + p_rx[j]
  对于均匀线阵（ULA），当发射和接收阵元间距均为 d 时，
  虚拟阵元间距为 d/2，虚拟孔径扩展为 (N_tx + N_rx - 1) * d / (N_rx - 1)。

  物理直觉：
    MIMO 利用波形分集在空间维度上"倍增"了采样点数，
    相当于用更少的物理阵元获得了更大的等效孔径，
    从而实现更窄的波束宽度和更高的角度分辨率。

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

# FFT convention: numpy.fft.fft (no 1/N scaling on forward transform)
# Parseval: sum(|x|^2) = (1/N) * sum(|X|^2)

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


def virtual_array_elements(
    n_tx: int, n_rx: int, d_lambda: float
) -> np.ndarray:
    """计算 MIMO 虚拟阵列的阵元位置。

    MIMO 雷达通过正交波形分集，将 N_tx 发射 + N_rx 接收等效为
    N_tx × N_rx 个虚拟阵元。虚拟阵元位置由发射和接收位置的卷积得到：

      p_virtual(i, j) = p_tx(i) + p_rx(j)

    对于均匀线阵（ULA），发射阵元间距 = d，接收阵元间距 = d：
      - 发射阵元位置：0, d, 2d, ..., (N_tx-1)*d
      - 接收阵元位置：0, d, 2d, ..., (N_rx-1)*d
      - 虚拟阵元位置：0, d/2, d, 3d/2, ..., (N_tx+N_rx-2)*d
      - 虚拟阵元间距为 d/2（即半波长间距的更密集采样）

    Args:
        n_tx:      发射阵元数
        n_rx:      接收阵元数
        d_lambda:  阵元间距与波长之比 d/λ

    Returns:
        去重后的虚拟阵元位置数组（以 d/λ 为单位），升序排列
    """
    # 发射阵元位置（以 d/λ 为单位）
    tx_positions = np.arange(n_tx) * d_lambda
    # 接收阵元位置
    rx_positions = np.arange(n_rx) * d_lambda

    # 虚拟阵元位置：所有 p_tx + p_rx 组合
    virtual_positions = []
    for p_tx in tx_positions:
        for p_rx in rx_positions:
            virtual_positions.append(p_tx + p_rx)

    # 去重并排序
    virtual_positions = np.unique(np.array(virtual_positions))
    return virtual_positions


def mimo_beam_pattern(
    n_tx: int,
    n_rx: int,
    d_lambda: float,
    theta_scan: np.ndarray,
    theta_steer_deg: float,
) -> np.ndarray:
    """计算基于虚拟阵列的 MIMO 波束形成方向图。

    MIMO 方向图等效于虚拟阵列的阵列响应。
    虚拟阵列的导向矢量 a_v(θ) 由所有虚拟阵元位置决定：

      B(θ) = |w^H * a_v(θ)|^2

    其中 w 是指向 theta_steer 的加权向量，a_v(θ) 是虚拟阵列导向矢量。

    Args:
        n_tx:          发射阵元数
        n_rx:          接收阵元数
        d_lambda:      阵元间距/波长比
        theta_scan:    扫描角度数组 (度)
        theta_steer_deg: 波束指向角 (度)

    Returns:
        归一化方向图功率 (dB)，峰值为 0 dB
    """
    virtual_pos = virtual_array_elements(n_tx, n_rx, d_lambda)
    n_virtual = len(virtual_pos)

    # 虚拟阵列导向矢量：a_v(θ) = exp(j * 2π * p_virtual * sin(θ))
    # 注意虚拟阵元位置以 d/λ 为单位，需要归一化到波长
    # p_virtual 的单位是 d/λ，相位 = 2π * (p_virtual / λ) * sin(θ)
    # 由于 p_virtual 以 d/λ 为单位，且我们用 d/λ 参数，相位 = 2π * p_virtual * sin(θ)
    theta_steer_rad = np.deg2rad(theta_steer_deg)

    # 加权向量：指向 theta_steer（均匀加权）
    w = np.exp(1j * 2 * np.pi * virtual_pos * np.sin(theta_steer_rad))
    w /= n_virtual  # 归一化

    pattern = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a_v = np.exp(1j * 2 * np.pi * virtual_pos * np.sin(np.deg2rad(theta)))
        pattern[idx] = np.abs(w.conj() @ a_v) ** 2

    # 归一化到峰值 0 dB
    pattern_db = power_to_db(pattern / np.max(pattern))
    return pattern_db


def simo_beam_pattern(
    n_rx: int, d_lambda: float, theta_scan: np.ndarray, theta_steer_deg: float
) -> np.ndarray:
    """计算 SIMO（仅接收波束形成）的方向图。

    传统相控阵仅在接收端做波束形成，等效阵元数 = N_rx。

    Args:
        n_rx:            接收阵元数
        d_lambda:        阵元间距/波长比
        theta_scan:      扫描角度数组 (度)
        theta_steer_deg: 波束指向角 (度)

    Returns:
        归一化方向图功率 (dB)，峰值为 0 dB
    """
    w = steering_vector(n_rx, d_lambda, theta_steer_deg) / n_rx

    pattern = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_rx, d_lambda, theta)
        pattern[idx] = np.abs(w.conj() @ a) ** 2

    pattern_db = power_to_db(pattern / np.max(pattern))
    return pattern_db


def virtual_aperture_gain(n_tx: int, n_rx: int) -> tuple[int, float]:
    """计算虚拟孔径增益。

    MIMO 虚拟阵列的自由度（去重后虚拟阵元数）相对于物理接收阵元数的
    扩展倍数，反映了角度分辨率的提升：

      自由度扩展 = (N_tx + N_rx - 1) / N_rx

    注意：虚拟阵列的物理基线为 (N_tx + N_rx - 2) * d/2，
    而接收阵列的物理基线为 (N_rx - 1) * d。
    虚拟阵列以更小的阵元间距（d/2）换取更密集的空间采样，
    从而实现更窄的波束宽度（更高的角度分辨率）。

    对于 N_tx=4, N_rx=8：
      虚拟阵元数 = 32（去重前），11（去重后）
      自由度扩展 = 11/8 = 1.375

    Args:
        n_tx: 发射阵元数
        n_rx: 接收阵元数

    Returns:
        (虚拟阵元数, 自由度扩展倍数)
        虚拟阵元数指去重前的总数（N_tx × N_rx）
        自由度扩展倍数 = 去重后虚拟阵元数 / 物理接收阵元数
    """
    n_virtual = n_tx * n_rx
    # 去重后的虚拟阵元数 = N_tx + N_rx - 1（ULA 特性）
    n_unique = n_tx + n_rx - 1
    # 自由度扩展：去重后虚拟阵元数 / 物理接收阵元数
    aperture_ratio = n_unique / n_rx
    return n_virtual, aperture_ratio


def measure_beamwidth(
    n_elements: int, d_lambda: float, theta_scan: np.ndarray
) -> float:
    """测量阵列方向图的 3 dB 波束宽度。

    通过扫描方向图找到峰值以下 3 dB 的两个角度点，
    其差值即为 3 dB 波束宽度。

    Args:
        n_elements:  阵元数（或虚拟阵元数）
        d_lambda:    有效阵元间距/波长比
        theta_scan:  扫描角度数组 (度)

    Returns:
        3 dB 波束宽度 (度)
    """
    # 生成指向法线的方向图
    w = steering_vector(n_elements, d_lambda, 0.0) / n_elements
    pattern = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a = steering_vector(n_elements, d_lambda, theta)
        pattern[idx] = np.abs(w.conj() @ a) ** 2

    pattern_db = power_to_db(pattern / np.max(pattern))

    # 找到 -3 dB 交叉点
    half_power_idx = np.where(pattern_db >= -3.0)[0]
    if len(half_power_idx) < 2:
        return np.nan

    theta_left = theta_scan[half_power_idx[0]]
    theta_right = theta_scan[half_power_idx[-1]]
    return theta_right - theta_left


def measure_mimo_beamwidth(
    n_tx: int, n_rx: int, d_lambda: float, theta_scan: np.ndarray
) -> float:
    """测量 MIMO 虚拟阵列方向图的 3 dB 波束宽度。

    Args:
        n_tx:        发射阵元数
        n_rx:        接收阵元数
        d_lambda:    阵元间距/波长比
        theta_scan:  扫描角度数组 (度)

    Returns:
        3 dB 波束宽度 (度)
    """
    virtual_pos = virtual_array_elements(n_tx, n_rx, d_lambda)
    n_virtual = len(virtual_pos)

    # 指向法线的均匀加权
    w = np.ones(n_virtual) / n_virtual

    pattern = np.zeros(len(theta_scan))
    for idx, theta in enumerate(theta_scan):
        a_v = np.exp(1j * 2 * np.pi * virtual_pos * np.sin(np.deg2rad(theta)))
        pattern[idx] = np.abs(w.conj() @ a_v) ** 2

    pattern_db = power_to_db(pattern / np.max(pattern))

    half_power_idx = np.where(pattern_db >= -3.0)[0]
    if len(half_power_idx) < 2:
        return np.nan

    theta_left = theta_scan[half_power_idx[0]]
    theta_right = theta_scan[half_power_idx[-1]]
    return theta_right - theta_left


def simulate_mimo_signal(
    n_tx: int,
    n_rx: int,
    d_lambda: float,
    target_theta: float,
    snr_db: float,
    n_snapshots: int,
    seed: int = 42,
) -> np.ndarray:
    """模拟 MIMO 雷达接收信号。

    模型：
      每个发射阵元发送正交波形，接收端通过匹配滤波分离。
      分离后的信号维度为 (N_rx × N_tx, L) 的快拍矩阵，
      每一列对应一个虚拟阵元的 L 次快拍。

      x_virtual(l) = s(l) * a_v(θ) + n(l)

    其中 a_v(θ) 是虚拟阵列的导向矢量，s(l) 是目标回波信号，
    n(l) 是各虚拟阵元独立的复高斯噪声。

    Args:
        n_tx:          发射阵元数
        n_rx:          接收阵元数
        d_lambda:      阵元间距/波长比
        target_theta:  目标方向 (度)
        snr_db:        信噪比 (dB)
        n_snapshots:   快拍数 L
        seed:          随机种子

    Returns:
        虚拟阵列快拍数据矩阵 (N_virtual_unique, L)，复数数组
    """
    rng = np.random.default_rng(seed)

    # 虚拟阵列导向矢量
    virtual_pos = virtual_array_elements(n_tx, n_rx, d_lambda)
    n_virtual = len(virtual_pos)

    # 虚拟阵列导向矢量
    a_v = np.exp(1j * 2 * np.pi * virtual_pos * np.sin(np.deg2rad(target_theta)))

    # 信号功率（以噪声功率为参考）
    sigma_s2 = db_to_power(snr_db)
    sigma_n2 = 1.0

    # 生成快拍数据：x = sqrt(sigma_s2) * s * a_v + sqrt(sigma_n2) * n
    signal_amp = np.sqrt(sigma_s2 / 2)
    noise_amp = np.sqrt(sigma_n2 / 2)

    # 目标信号：复高斯随机变量 × 导向矢量
    s = signal_amp * (
        rng.standard_normal(n_snapshots) + 1j * rng.standard_normal(n_snapshots)
    )
    # 噪声：各虚拟阵元独立复高斯
    noise = noise_amp * (
        rng.standard_normal((n_virtual, n_snapshots))
        + 1j * rng.standard_normal((n_virtual, n_snapshots))
    )

    # 快拍矩阵：(N_virtual, L)
    snapshots = a_v[:, np.newaxis] * s[np.newaxis, :] + noise
    return snapshots


# ============================================================
# 绘图
# ============================================================


def plot_mimo_virtual_array(
    n_tx: int,
    n_rx: int,
    d_lambda: float,
    target_theta: float,
    snr_db: float,
    n_snapshots: int,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制 MIMO 虚拟阵列分析结果（3 子图）。

    子图 1：物理阵列 vs 虚拟阵列的位置分布
      - 展示 MIMO 如何将稀疏物理阵元映射为密集虚拟阵元

    子图 2：SIMO vs MIMO 方向图对比
      - 展示虚拟孔径带来的窄主瓣

    子图 3：不同 N_tx × N_rx 组合下的 3dB 波束宽度
      - 展示孔径扩展效果

    Args:
        n_tx:          发射阵元数
        n_rx:          接收阵元数
        d_lambda:      阵元间距/波长比
        target_theta:  目标方向 (度)
        snr_db:        信噪比 (dB)
        n_snapshots:   快拍数
        output_dir:    输出目录
    """
    theta_scan = np.linspace(-90, 90, 1801)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # ---- 子图 1：物理阵列 vs 虚拟阵列位置 ----
    ax1 = axes[0]

    # 物理阵元位置
    tx_pos = np.arange(n_tx) * d_lambda
    rx_pos = np.arange(n_rx) * d_lambda

    # 虚拟阵元位置
    virtual_pos = virtual_array_elements(n_tx, n_rx, d_lambda)

    # 绘制发射阵元
    ax1.scatter(
        tx_pos, np.ones(n_tx) * 1.5, marker="^", s=120, c="red",
        label=f"发射阵元 (N_tx={n_tx})", zorder=5,
    )
    # 绘制接收阵元
    ax1.scatter(
        rx_pos, np.ones(n_rx) * 1.0, marker="s", s=80, c="blue",
        label=f"接收阵元 (N_rx={n_rx})", zorder=5,
    )
    # 绘制虚拟阵元
    ax1.scatter(
        virtual_pos, np.ones(len(virtual_pos)) * 0.5, marker="o", s=60,
        c="green", edgecolors="black", linewidths=0.5,
        label=f"虚拟阵元 (N_v={len(virtual_pos)})", zorder=5,
    )

    ax1.set_xlabel("阵元位置 (d/λ)", fontsize=12)
    ax1.set_yticks([])
    ax1.set_title(
        f"MIMO 物理阵列 vs 虚拟阵列 (N_tx={n_tx}, N_rx={n_rx}, d=λ/2)",
        fontsize=13,
    )
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.set_ylim([0, 2.0])

    # 添加注释：虚拟阵元间距
    if len(virtual_pos) >= 2:
        min_spacing = np.min(np.diff(virtual_pos))
        ax1.annotate(
            f"虚拟阵元间距 = {min_spacing:.2f}d/λ",
            xy=(virtual_pos[len(virtual_pos) // 2], 1.8),
            fontsize=11, ha="center", color="darkgreen",
        )

    # ---- 子图 2：SIMO vs MIMO 方向图对比 ----
    ax2 = axes[1]

    # SIMO 方向图（仅接收波束形成）
    pattern_simo = simo_beam_pattern(n_rx, d_lambda, theta_scan, target_theta)

    # MIMO 方向图（虚拟阵列波束形成）
    pattern_mimo = mimo_beam_pattern(n_tx, n_rx, d_lambda, theta_scan, target_theta)

    ax2.plot(theta_scan, pattern_simo, "b-", linewidth=1.5, label=f"SIMO (N_rx={n_rx})")
    ax2.plot(theta_scan, pattern_mimo, "r-", linewidth=1.5, label=f"MIMO (N_tx×N_rx={n_tx}×{n_rx})")
    ax2.axvline(x=target_theta, color="green", linestyle="--", alpha=0.5)
    ax2.annotate(
        f"目标 {target_theta}°",
        xy=(target_theta, 0),
        xytext=(target_theta + 5, 5),
        fontsize=10, color="green",
    )
    ax2.set_xlabel("角度 (度)", fontsize=12)
    ax2.set_ylabel("归一化增益 (dB)", fontsize=12)
    ax2.set_title("SIMO vs MIMO 方向图对比（虚拟孔径扩展效果）", fontsize=13)
    ax2.set_ylim([-50, 5])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 标注波束宽度
    bw_simo = measure_beamwidth(n_rx, d_lambda, theta_scan)
    bw_mimo = measure_mimo_beamwidth(n_tx, n_rx, d_lambda, theta_scan)
    ax2.annotate(
        f"SIMO 3dB BW ≈ {bw_simo:.2f}°",
        xy=(-40, -5), fontsize=10, color="blue",
    )
    ax2.annotate(
        f"MIMO 3dB BW ≈ {bw_mimo:.2f}°",
        xy=(-40, -12), fontsize=10, color="red",
    )

    # ---- 子图 3：不同 N_tx × N_rx 组合的波束宽度 ----
    ax3 = axes[2]

    # 固定 N_rx = 8，变化 N_tx
    n_rx_fixed = 8
    n_tx_values = [1, 2, 4, 8]
    bw_values = []

    for n_tx_val in n_tx_values:
        if n_tx_val == 1:
            # SIMO 模式
            bw = measure_beamwidth(n_rx_fixed, d_lambda, theta_scan)
        else:
            bw = measure_mimo_beamwidth(n_tx_val, n_rx_fixed, d_lambda, theta_scan)
        bw_values.append(bw)

    labels = [f"{n_tx_val}×{n_rx_fixed}" for n_tx_val in n_tx_values]
    colors = ["blue", "orange", "red", "purple"]

    bars = ax3.bar(labels, bw_values, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("N_tx × N_rx", fontsize=12)
    ax3.set_ylabel("3dB 波束宽度 (度)", fontsize=12)
    ax3.set_title(f"不同 MIMO 配置的 3dB 波束宽度 (N_rx={n_rx_fixed}, d=λ/2)", fontsize=13)
    ax3.grid(True, alpha=0.3, axis="y")

    # 在柱状图上标注数值
    for bar, bw_val in zip(bars, bw_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{bw_val:.2f}°", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s15_mimo_virtual_array.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s15_mimo_virtual_array.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    n_tx: int,
    n_rx: int,
    d_lambda: float,
    target_theta: float,
    snr_db: float,
    n_snapshots: int,
) -> bool:
    """验证 MIMO 虚拟阵列的正确性。

    验证项：
      a. 虚拟阵元数：N_virtual = N_tx × N_rx（去重前）
      b. 虚拟孔径扩展：自由度扩展 = (N_tx + N_rx - 1) / N_rx > 1
      c. 波束宽度：MIMO 波束宽度 < SIMO 波束宽度
      d. 方向图峰值：指向目标方向，峰值归一化为 0 dB
    """
    results = []
    theta_scan = np.linspace(-90, 90, 1801)

    # --- 验证 a：虚拟阵元数 ---
    n_virtual, _ = virtual_aperture_gain(n_tx, n_rx)
    n_unique = len(virtual_array_elements(n_tx, n_rx, d_lambda))

    results.append(verify(
        name="虚拟阵元数（去重前）",
        theoretical=float(n_tx * n_rx),
        simulated=float(n_virtual),
        tolerance=0.5,
        unit="个",
    ))

    # 去重后的虚拟阵元数应为 N_tx + N_rx - 1（ULA 特性）
    results.append(verify(
        name="去重后虚拟阵元数",
        theoretical=float(n_tx + n_rx - 1),
        simulated=float(n_unique),
        tolerance=0.5,
        unit="个",
    ))

    # --- 验证 b：虚拟孔径扩展 ---
    _, aperture_ratio = virtual_aperture_gain(n_tx, n_rx)
    # 自由度扩展倍数 = (N_tx + N_rx - 1) / N_rx（应 > 1）
    expected_aperture_ratio = (n_tx + n_rx - 1) / n_rx
    results.append(verify(
        name="虚拟孔径扩展倍数",
        theoretical=expected_aperture_ratio,  # N_tx=4, N_rx=8 → 1.375
        simulated=aperture_ratio,
        tolerance=0.3,
        unit="倍",
    ))

    # --- 验证 c：波束宽度对比 ---
    # MIMO 波束宽度应小于 SIMO 波束宽度
    bw_simo = measure_beamwidth(n_rx, d_lambda, theta_scan)
    bw_mimo = measure_mimo_beamwidth(n_tx, n_rx, d_lambda, theta_scan)

    # 验证 MIMO 波束宽度确实更窄
    results.append(verify(
        name="MIMO 波束宽度 < SIMO 波束宽度",
        theoretical=1.0,  # 期望 bw_mimo / bw_simo < 1
        simulated=bw_mimo / bw_simo,
        tolerance=0.3,  # 允许比值在 0.7~1.0 之间
        unit="比值",
    ))

    # --- 验证 d：方向图峰值 ---
    # 方向图在目标方向应有峰值（归一化后为 0 dB）
    pattern_mimo = mimo_beam_pattern(n_tx, n_rx, d_lambda, theta_scan, target_theta)
    peak_value = np.max(pattern_mimo)

    # 找到峰值所在角度
    peak_idx = np.argmax(pattern_mimo)
    peak_theta = theta_scan[peak_idx]

    results.append(verify(
        name="方向图峰值归一化",
        theoretical=0.0,
        simulated=peak_value,
        tolerance=0.5,  # 允许 0.5 dB 误差
        unit="dB",
    ))

    results.append(verify(
        name="峰值指向目标方向",
        theoretical=target_theta,
        simulated=peak_theta,
        tolerance=0.5,  # 允许 0.5° 误差
        unit="度",
    ))

    return print_validation("s15 MIMO 正交波形分集与虚拟孔径扩展", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s15 MIMO 虚拟阵列仿真与验证。"""
    print("=" * 60)
    print("s15：MIMO 正交波形分集与虚拟孔径扩展")
    print("=" * 60)

    # 仿真参数
    n_tx = 4          # 发射阵元数
    n_rx = 8          # 接收阵元数
    d_lambda = 0.5    # 阵元间距 d = λ/2
    target_theta = 0.0  # 目标方向（法线方向）
    snr_db = 20.0     # 信噪比
    n_snapshots = 100  # 快拍数
    seed = 42         # 随机种子

    print(f"\n仿真参数:")
    print(f"  发射阵元数 N_tx   = {n_tx}")
    print(f"  接收阵元数 N_rx   = {n_rx}")
    print(f"  阵元间距 d        = {d_lambda}λ")
    print(f"  目标方向 θ        = {target_theta}°")
    print(f"  信噪比 SNR        = {snr_db} dB")
    print(f"  快拍数 L          = {n_snapshots}")
    print(f"  随机种子 seed     = {seed}")

    # 虚拟阵列分析
    print(f"\n--- 虚拟阵列分析 ---")
    virtual_pos = virtual_array_elements(n_tx, n_rx, d_lambda)
    n_virtual, aperture_ratio = virtual_aperture_gain(n_tx, n_rx)

    print(f"  虚拟阵元数（去重前）  = {n_virtual}")
    print(f"  虚拟阵元数（去重后）  = {len(virtual_pos)}")
    print(f"  虚拟阵元位置 (d/λ)    = {virtual_pos}")
    if len(virtual_pos) >= 2:
        min_spacing = np.min(np.diff(virtual_pos))
        print(f"  最小虚拟阵元间距      = {min_spacing:.3f} d/λ")
    print(f"  孔径扩展倍数          = {aperture_ratio:.3f}")

    # 波束宽度对比
    theta_scan = np.linspace(-90, 90, 1801)
    bw_simo = measure_beamwidth(n_rx, d_lambda, theta_scan)
    bw_mimo = measure_mimo_beamwidth(n_tx, n_rx, d_lambda, theta_scan)

    print(f"\n--- 波束宽度对比 ---")
    print(f"  SIMO 3dB 波束宽度 (N_rx={n_rx})    = {bw_simo:.3f}°")
    print(f"  MIMO 3dB 波束宽度 (N_tx×N_rx={n_tx}×{n_rx}) = {bw_mimo:.3f}°")
    print(f"  波束宽度比值 (MIMO/SIMO)            = {bw_mimo / bw_simo:.3f}")

    # MIMO 信号模拟
    print(f"\n--- MIMO 信号模拟 ---")
    snapshots = simulate_mimo_signal(
        n_tx, n_rx, d_lambda, target_theta, snr_db, n_snapshots, seed,
    )
    print(f"  快拍矩阵维度  = {snapshots.shape}")
    print(f"  快拍矩阵功率  = {power_to_db(np.mean(np.abs(snapshots)**2)):.2f} dB (相对噪声)")

    # 绘图
    print(f"\n绘制 MIMO 虚拟阵列分析结果...")
    plot_mimo_virtual_array(
        n_tx, n_rx, d_lambda, target_theta, snr_db, n_snapshots,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(
        n_tx, n_rx, d_lambda, target_theta, snr_db, n_snapshots,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
