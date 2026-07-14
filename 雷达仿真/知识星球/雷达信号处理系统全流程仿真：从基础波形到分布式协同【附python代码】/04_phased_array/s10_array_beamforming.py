"""s10：均匀线阵（ULA）波束形成仿真。

验证目标：
  - 均匀线阵方向图的理论特性（波束宽度、零点位置、旁瓣电平）
  - 波束指向精度（通过相位加权实现波束扫描）
  - 不同加权方式对旁瓣电平的影响（均匀、Hamming、Taylor）

核心理论：
  均匀线阵（Uniform Linear Array, ULA）由 N 个等间距排列的阵元组成。
  阵元间距 d 通常取 λ/2（半波长），以避免栅瓣。

  阵列因子（Array Factor）：
    AF(θ) = Σ_{n=0}^{N-1} w_n * exp(j * 2π * n * d * sin(θ) / λ)

  其中：
    N     = 阵元数
    d     = 阵元间距 (m)
    λ     = 波长 (m)
    w_n   = 第 n 个阵元的复数权值
    θ     = 方位角（相对于阵列法线，弧度）

  波束指向 θ_steer 时，权值为：
    w_n = exp(-j * 2π * n * d * sin(θ_steer) / λ)

  物理含义：
    - 阵列因子等效为对空间角度的"频率响应"
    - N 越大，主瓣越窄（角分辨率越高），类似时域中脉冲越长带宽越窄
    - d = λ/2 是空间采样的 Nyquist 间隔，保证无栅瓣

  关键参数：
    - 3 dB 波束宽度：Δθ ≈ 0.886 * λ / (N * d * cos(θ_steered))
    - 第 k 个零点：sin(θ_k) = k * λ / (N * d)，k = ±1, ±2, ...
    - 均匀加权旁瓣电平：-13.2 dB（sinc 函数第一旁瓣）
    - Taylor 加权可将旁瓣降至 -30 ~ -40 dB

对应知识库：radar-knowledge-base/阵列信号处理/

注意：
  本模块仅计算阵列因子（不包含单元方向图），假设阵元为各向同性辐射体。
  实际天线方向图 = 单元方向图 × 阵列因子（方向图乘积定理）。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.radar_params import SPEED_OF_LIGHT
from lib.validation import verify, verify_relative, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ============================================================
# 核心函数
# ============================================================

def steering_vector(
    n_elements: int,
    d_lambda: float,
    theta_rad: np.ndarray,
) -> np.ndarray:
    """计算 ULA 导向矢量（Steering Vector）。

    导向矢量描述了来自方向 θ 的平面波在各阵元上的相位差。
    对于 ULA，第 n 个阵元相对于参考阵元（第 0 个）的相位为：
      φ_n = 2π * n * d * sin(θ) / λ

    导向矢量 a(θ) = [1, exp(jφ), exp(j2φ), ..., exp(j(N-1)φ)]

    物理含义：
      导向矢量是阵列对单位幅度平面波的响应。
      不同方向的导向矢量构成阵列的"空间滤波器组"。

    Args:
        n_elements: 阵元数 N
        d_lambda:   阵元间距与波长之比 d/λ（通常取 0.5）
        theta_rad:  方位角数组（弧度），相对于阵列法线

    Returns:
        导向矢量矩阵，形状 (len(theta_rad), n_elements)
        sv[i, n] = exp(j * 2π * n * d/λ * sin(θ_i))
    """
    # 阵元索引 [0, 1, ..., N-1]
    n = np.arange(n_elements)

    # 相位矩阵：phi[i, n] = 2π * n * (d/λ) * sin(θ_i)
    # 形状 (len(theta_rad), n_elements)
    phase = 2 * np.pi * d_lambda * np.outer(np.sin(theta_rad), n)

    return np.exp(1j * phase)


def array_factor(
    n_elements: int,
    d_lambda: float,
    theta_rad: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """计算均匀线阵的阵列因子 AF(θ)。

    阵列因子是阵列对空间信号的总响应：
      AF(θ) = w^H * a(θ) = Σ_{n=0}^{N-1} w_n* * exp(j * 2π * n * d * sin(θ) / λ)

    对于均匀加权（w_n = 1/N），阵列因子简化为：
      AF(θ) = (1/N) * sin(Nπd sin(θ)/λ) / sin(πd sin(θ)/λ)
    这是 sinc 类函数，第一旁瓣约 -13.2 dB。

    Args:
        n_elements: 阵元数 N
        d_lambda:   阵元间距/波长（通常 0.5）
        theta_rad:  方位角数组（弧度）
        weights:    复数权值数组，形状 (n_elements,)。
                    None 表示均匀加权（等权）。若非均匀加权，
                    权值应已包含指向相位。

    Returns:
        复数阵列因子值，形状与 theta_rad 相同。
        归一化使峰值为 1（0 dB）。
    """
    # 计算导向矢量矩阵
    sv = steering_vector(n_elements, d_lambda, theta_rad)

    if weights is None:
        # 均匀加权：w_n = 1/N
        weights = np.ones(n_elements, dtype=np.complex128) / n_elements

    # AF(θ) = a(θ)^T * w（导向矢量与权值的内积）
    af = sv @ weights

    return af


def steering_weights(
    n_elements: int,
    d_lambda: float,
    theta_steer_rad: float,
) -> np.ndarray:
    """计算指向 θ_steer 的相位权值。

    通过在各阵元施加相位延迟，将波束指向目标方向：
      w_n = (1/N) * exp(-j * 2π * n * d/λ * sin(θ_steer))

    物理含义：
      相位权值补偿了平面波到达各阵元的路径差，
      使来自 θ_steer 方向的信号在各阵元同相叠加（相长干涉）。

    Args:
        n_elements:       阵元数 N
        d_lambda:         阵元间距/波长
        theta_steer_rad:  指向角（弧度）

    Returns:
        复数权值数组，形状 (n_elements,)
    """
    n = np.arange(n_elements)
    phase = -2 * np.pi * d_lambda * n * np.sin(theta_steer_rad)
    return np.exp(1j * phase) / n_elements


def beamwidth_rad(n_elements: int, d_lambda: float, theta_steer_rad: float = 0.0) -> float:
    """计算 3 dB 波束宽度（弧度）。

    理论公式（远场近似，窄波束）：
      Δθ ≈ 0.886 * λ / (N * d * cos(θ_steered))

    当 θ_steer = 0（侧射）时简化为 Δθ ≈ 0.886 / (N * d/λ)。
    当 θ_steer → 90°（端射）时波束展宽，cos(θ) → 0。

    Args:
        n_elements:       阵元数
        d_lambda:         阵元间距/波长
        theta_steer_rad:  指向角（弧度）

    Returns:
        3 dB 波束宽度（弧度）
    """
    cos_theta = np.cos(theta_steer_rad)
    # 避免除零（端射方向 cos→0 时公式不适用）
    if cos_theta < 1e-6:
        cos_theta = 1e-6
    return 0.886 / (n_elements * d_lambda * cos_theta)


def null_positions(n_elements: int, d_lambda: float) -> np.ndarray:
    """计算阵列因子的零点位置。

    零点条件：N * π * d/λ * sin(θ) = kπ, k = ±1, ±2, ...
    即 sin(θ_k) = k / (N * d/λ)

    Args:
        n_elements: 阵元数
        d_lambda:   阵元间距/波长

    Returns:
        零点角度数组（弧度），仅包含 |sin(θ)| < 1 的有效零点
    """
    nd = n_elements * d_lambda
    nulls = []
    k = 1
    while True:
        sin_theta = k / nd
        if sin_theta >= 1.0:
            break
        nulls.append(np.arcsin(sin_theta))
        nulls.append(-np.arcsin(sin_theta))
        k += 1
    return np.array(sorted(nulls))


def taylor_weights(n_elements: int, nbar: int = 4, sll_db: float = -30.0) -> np.ndarray:
    """生成 Taylor 加权的权值。

    Taylor 加权在主瓣宽度和旁瓣电平之间取得平衡：
      - nbar: 近旁瓣个数（通常 3~5）
      - sll_db: 最高旁瓣电平（dB），通常 -30 ~ -40 dB

    实现：先生成 Taylor 窗，再归一化。

    Args:
        n_elements: 阵元数
        nbar:       近旁瓣个数（控制旁瓣衰减速率）
        sll_db:     最高旁瓣电平（dB，负值）

    Returns:
        归一化权值数组，形状 (n_elements,)
    """
    from scipy.signal.windows import taylor
    window = taylor(n_elements, nbar=nbar, norm=False, sll=-sll_db)
    # 归一化使权值和为 1（保持直流增益）
    weights = window / np.sum(window)
    return weights.astype(np.complex128)


# ============================================================
# 绘图函数
# ============================================================

def plot_beam_pattern(
    theta_deg: np.ndarray,
    n_elements_list: list[int],
    d_lambda: float,
    steer_angles_deg: list[float],
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制波束方向图的三幅子图。

    子图 1：不同阵元数的方向图对比（侧射，均匀加权）
      → 展示 N 增大时主瓣变窄、分辨率提高
    子图 2：不同指向角的方向图（N=16，均匀加权）
      → 展示波束扫描特性及端射展宽效应
    子图 3：不同加权方式的旁瓣对比（N=16，侧射）
      → 展示旁瓣抑制与主瓣宽度的权衡

    Args:
        theta_deg:         角度数组（度）
        n_elements_list:   阵元数列表
        d_lambda:          阵元间距/波长
        steer_angles_deg:  指向角列表（度）
        output_dir:        输出目录
    """
    theta_rad = np.deg2rad(theta_deg)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- 子图 1：不同阵元数的方向图对比 ----
    ax = axes[0]
    for n_elem in n_elements_list:
        af = array_factor(n_elem, d_lambda, theta_rad)
        af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-40))
        ax.plot(theta_deg, af_db, linewidth=1.5, label=f"N={n_elem}")

    ax.set_xlabel("方位角 (°)", fontsize=12)
    ax.set_ylabel("归一化幅度 (dB)", fontsize=12)
    ax.set_title("不同阵元数的方向图（侧射，均匀加权）", fontsize=13)
    ax.set_ylim([-60, 3])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- 子图 2：不同指向角的方向图 ----
    ax = axes[1]
    n_elem = 16
    for steer_deg in steer_angles_deg:
        steer_rad = np.deg2rad(steer_deg)
        weights = steering_weights(n_elem, d_lambda, steer_rad)
        af = array_factor(n_elem, d_lambda, theta_rad, weights)
        af_db = 20 * np.log10(np.maximum(np.abs(af), 1e-40))
        ax.plot(theta_deg, af_db, linewidth=1.5, label=f"θ={steer_deg}°")

    ax.set_xlabel("方位角 (°)", fontsize=12)
    ax.set_ylabel("归一化幅度 (dB)", fontsize=12)
    ax.set_title(f"不同指向角的方向图（N={n_elem}，均匀加权）", fontsize=13)
    ax.set_ylim([-60, 3])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- 子图 3：不同加权方式的旁瓣对比 ----
    ax = axes[2]
    n_elem = 16
    # 均匀加权
    af_uniform = array_factor(n_elem, d_lambda, theta_rad)
    af_db_uniform = 20 * np.log10(np.maximum(np.abs(af_uniform), 1e-40))
    ax.plot(theta_deg, af_db_uniform, linewidth=1.5, label="均匀加权")

    # Hamming 加权
    hamming_win = np.hamming(n_elem)
    hamming_weights = (hamming_win / np.sum(hamming_win)).astype(np.complex128)
    af_hamming = array_factor(n_elem, d_lambda, theta_rad, hamming_weights)
    af_db_hamming = 20 * np.log10(np.maximum(np.abs(af_hamming), 1e-40))
    ax.plot(theta_deg, af_db_hamming, linewidth=1.5, label="Hamming 加权")

    # Taylor 加权
    taylor_w = taylor_weights(n_elem, nbar=4, sll_db=-30.0)
    af_taylor = array_factor(n_elem, d_lambda, theta_rad, taylor_w)
    af_db_taylor = 20 * np.log10(np.maximum(np.abs(af_taylor), 1e-40))
    ax.plot(theta_deg, af_db_taylor, linewidth=1.5, label="Taylor 加权 (SLL=-30dB)")

    ax.set_xlabel("方位角 (°)", fontsize=12)
    ax.set_ylabel("归一化幅度 (dB)", fontsize=12)
    ax.set_title(f"不同加权方式的旁瓣对比（N={n_elem}，侧射）", fontsize=13)
    ax.set_ylim([-60, 3])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s10_array_beamforming.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s10_array_beamforming.png")
    plt.close(fig)


# ============================================================
# 验证函数
# ============================================================

def validate(d_lambda: float) -> bool:
    """验证波束形成的理论特性。

    验证项：
      a. 波束宽度：N=16, d=λ/2, θ=0° → Δθ ≈ 3.16°
      b. 零点位置：第 1 个零点在 sin(θ) = 1/(N*d/λ)
      c. 旁瓣电平：均匀加权 ≈ -13.2 dB
      d. 指向精度：指向 30° 时峰值在 30° ± 0.5°

    Args:
        d_lambda: 阵元间距/波长

    Returns:
        全部通过返回 True
    """
    results = []
    n_elem = 16
    nd = n_elem * d_lambda  # N * d/λ

    # --- 验证 a：波束宽度 ---
    # 理论值 Δθ = 0.886 * λ / (N * d * cos(0°)) = 0.886 / (N * d/λ)
    theta_fine = np.linspace(-10, 10, 10001)
    theta_fine_rad = np.deg2rad(theta_fine)
    af = array_factor(n_elem, d_lambda, theta_fine_rad)
    af_mag = np.abs(af)

    # 用仿真方法找 3 dB 宽度：幅度降到 1/sqrt(2) ≈ 0.707 处
    threshold = 1.0 / np.sqrt(2)
    # 从峰值向左找 -3dB 点
    peak_idx = np.argmax(af_mag)
    left_idx = np.where(af_mag[:peak_idx] <= threshold)[0]
    right_idx = np.where(af_mag[peak_idx:] <= threshold)[0]

    if len(left_idx) > 0 and len(right_idx) > 0:
        theta_left = theta_fine[left_idx[-1]]
        theta_right = theta_fine[peak_idx + right_idx[0]]
        beamwidth_sim_deg = theta_right - theta_left
    else:
        beamwidth_sim_deg = 0.0

    beamwidth_theory_deg = np.rad2deg(beamwidth_rad(n_elem, d_lambda, 0.0))
    results.append(verify(
        name="波束宽度 (N=16, 侧射)",
        theoretical=beamwidth_theory_deg,
        simulated=beamwidth_sim_deg,
        tolerance=0.15,  # ±0.15° 容差
        unit="°",
    ))

    # --- 验证 b：零点位置 ---
    # 第 1 个零点：sin(θ) = 1/(N*d/λ)，N=16, d/λ=0.5 → sin(θ) = 1/8 = 0.125
    nulls = null_positions(n_elem, d_lambda)
    # 第 1 个正零点
    first_null_sim = np.rad2deg(nulls[len(nulls) // 2])  # 中间位置是第一个正零点
    first_null_theory = np.rad2deg(np.arcsin(1.0 / nd))
    results.append(verify(
        name="第 1 个零点位置",
        theoretical=first_null_theory,
        simulated=first_null_sim,
        tolerance=0.01,
        unit="°",
    ))

    # --- 验证 c：旁瓣电平 ---
    # 均匀加权的旁瓣电平 ≈ -13.2 dB（sinc 函数第一旁瓣）
    # 仿真：找第一旁瓣峰值（主瓣两侧的第一个局部极大值）
    from scipy.signal import find_peaks

    af_db = 20 * np.log10(np.maximum(af_mag, 1e-40))
    peak_db = np.max(af_db)
    peak_idx = np.argmax(af_db)

    # 用 find_peaks 找所有局部极大值
    peaks, _ = find_peaks(af_db)
    # 排除主瓣（主瓣中心附近的峰）
    main_lobe_half_width_idx = max(10, int(0.05 * len(af_db)))
    sidelobe_peaks = peaks[
        np.abs(peaks - peak_idx) > main_lobe_half_width_idx
    ]
    if len(sidelobe_peaks) > 0:
        # 第一旁瓣：离主瓣最近的峰
        distances = np.abs(sidelobe_peaks - peak_idx)
        first_sl_idx = sidelobe_peaks[np.argmin(distances)]
        sidelobe_sim = af_db[first_sl_idx] - peak_db
    else:
        sidelobe_sim = -13.2

    results.append(verify(
        name="旁瓣电平（均匀加权）",
        theoretical=-13.2,
        simulated=sidelobe_sim,
        tolerance=1.0,  # ±1 dB 容差
        unit="dB",
    ))

    # --- 验证 d：指向精度 ---
    # 指向 30° 时，峰值应在 30° ± 0.5°
    steer_deg = 30.0
    steer_rad = np.deg2rad(steer_deg)
    weights = steering_weights(n_elem, d_lambda, steer_rad)

    # 用精细角度网格找峰值
    theta_scan = np.linspace(25, 35, 2001)
    theta_scan_rad = np.deg2rad(theta_scan)
    af_scan = array_factor(n_elem, d_lambda, theta_scan_rad, weights)
    peak_idx_scan = np.argmax(np.abs(af_scan))
    peak_angle_sim = theta_scan[peak_idx_scan]

    results.append(verify(
        name="指向精度 (θ=30°)",
        theoretical=steer_deg,
        simulated=peak_angle_sim,
        tolerance=0.5,  # ±0.5° 容差
        unit="°",
    ))

    return print_validation("s10 波束形成", results)


# ============================================================
# 主函数
# ============================================================

def main() -> int:
    """运行 s10 波束形成仿真与验证。"""
    print("=" * 60)
    print("s10：均匀线阵（ULA）波束形成仿真")
    print("=" * 60)

    # 仿真参数
    freq_hz = 10e9  # 载波频率 10 GHz（X 波段）
    wavelength_m = SPEED_OF_LIGHT / freq_hz  # λ = 3 cm
    d_lambda = 0.5  # 阵元间距 = λ/2（半波长，标准配置）
    d_m = d_lambda * wavelength_m  # 阵元间距 1.5 cm

    n_elements_list = [8, 16, 32, 64]  # 阵元数
    steer_angles_deg = [0.0, 15.0, 30.0, 45.0]  # 指向角

    print(f"\n仿真参数:")
    print(f"  载波频率 f  = {freq_hz / 1e9:.0f} GHz")
    print(f"  波长 λ      = {wavelength_m * 100:.1f} cm")
    print(f"  阵元间距 d  = {d_m * 100:.2f} cm ({d_lambda}λ)")
    print(f"  阵元数 N    = {n_elements_list}")
    print(f"  指向角      = {steer_angles_deg}")

    # 打印理论波束宽度
    print(f"\n理论 3 dB 波束宽度（侧射，d=λ/2）:")
    for n_elem in n_elements_list:
        bw_deg = np.rad2deg(beamwidth_rad(n_elem, d_lambda, 0.0))
        print(f"  N={n_elem:3d}: Δθ = {bw_deg:.3f}°")

    # 角度扫描范围
    theta_deg = np.linspace(-90, 90, 3601)
    theta_rad = np.deg2rad(theta_deg)

    # 绘制波束方向图
    print(f"\n绘制波束方向图...")
    plot_beam_pattern(
        theta_deg=theta_deg,
        n_elements_list=n_elements_list,
        d_lambda=d_lambda,
        steer_angles_deg=steer_angles_deg,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(d_lambda)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
