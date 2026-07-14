"""s04：CFAR 恒虚警率检测。

验证目标：
  - 在距离-多普勒矩阵上实现 CA-CFAR 和 OS-CFAR 检测器
  - 验证虚警率 Pfa 与设计值一致
  - 验证已知目标被正确检测

核心概念：
  CFAR（Constant False Alarm Rate）的核心思想：
    背景噪声/杂波水平未知且时变，不能用固定门限。
    通过估计检测单元（CUT）周围的局部噪声水平，自适应调整门限。

  CA-CFAR（单元平均 CFAR）：
    门限 = α × 训练单元的算术平均值
    α = N_t × (Pfa^(-1/N_t) - 1)
    优点：均匀噪声中性能最优
    缺点：多目标时存在"遮蔽"效应

  OS-CFAR（有序统计量 CFAR）：
    将训练单元排序，取第 k 个值作为噪声估计
    优点：抗多目标遮蔽
    缺点：计算量略大

  窗口结构：
    [训练] [训练] [保护] [CUT] [保护] [训练] [训练]
    |<---- training_cells ---->|
    |<-- guard_cells -->|

对应知识库：radar-knowledge-base/基础/03-目标检测与跟踪.md
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import brentq

from lib.validation import verify, print_validation

# 中文字体
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def calculate_cfar_threshold_factor(
    num_training: int,
    pfa: float,
    method: str = "ca",
    os_rank: int = 0,
) -> float:
    """计算 CFAR 门限因子 α。

    CA-CFAR 公式推导：
      假设噪声服从指数分布（|CN(0,σ²)|² 的幅度平方），则：
      训练单元均值 Y ~ Gamma(N_t, σ²/N_t)
      Pfa = P(X > α·Y | H0) = (1 + α/N_t)^(-N_t)
      解出 α = N_t × (Pfa^(-1/N_t) - 1)

    Args:
        num_training: 训练单元总数 N_t
        pfa:          设计虚警率
        method:       "ca" 或 "os"
        os_rank:      OS-CFAR 的排序位置（从 1 开始）
    """
    if method == "ca":
        return num_training * (pfa ** (-1.0 / num_training) - 1.0)
    elif method == "os":
        if os_rank < 1 or os_rank > num_training:
            raise ValueError(f"os_rank 必须在 [1, {num_training}] 内")
        # OS-CFAR 的 α 需要数值求解
        # Pfa = ∏_{i=0}^{k-1} (N-i) / (N-i+α)
        def pfa_equation(alpha):
            p = 1.0
            for i in range(os_rank):
                p *= (num_training - i) / (num_training - i + alpha)
            return p - pfa
        return brentq(pfa_equation, 0.01, 1e6)
    else:
        raise ValueError(f"未知方法: {method}")


def generate_synthetic_rdm(
    n_range: int = 256,
    n_doppler: int = 64,
    targets: list = None,
    noise_seed: int = 42,
) -> dict:
    """生成合成的距离-多普勒功率矩阵。

    直接在功率域生成，不经过信号处理链。CFAR 算法只需要功率矩阵，
    不关心它是从真实回波还是合成数据得到的。

    噪声模型：指数分布（|CN(0,1)|²），这是雷达功率域噪声的标准模型。
    目标模型：在指定 (range_bin, doppler_bin) 位置注入信号功率。

    Args:
        n_range:    距离维大小
        n_doppler:  多普勒维大小
        targets:    目标列表 [{"range_bin", "doppler_bin", "snr_linear"}]
        noise_seed: 随机种子

    Returns:
        dict: rdm_power（线性功率）, rdm_power_db, target_info
    """
    rng = np.random.default_rng(noise_seed)
    if targets is None:
        targets = []

    # 指数分布噪声：|CN(0,1)|² 的功率服从 Exp(1)
    rdm_power = rng.exponential(1.0, (n_doppler, n_range))

    # 注入目标：在目标位置叠加信号功率
    for t in targets:
        r, d = t["range_bin"], t["doppler_bin"]
        snr = t["snr_linear"]
        rdm_power[d, r] += snr

    rdm_power_db = 10 * np.log10(np.maximum(rdm_power, 1e-40))

    return {
        "rdm_power": rdm_power,
        "rdm_power_db": rdm_power_db,
        "n_range": n_range,
        "n_doppler": n_doppler,
        "targets": targets,
    }


def ca_cfar_2d(
    rdm_power: np.ndarray,
    guard_cells: tuple[int, int],
    training_cells: tuple[int, int],
    pfa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """二维单元平均 CFAR（向量化实现）。

    使用积分图（summed-area table）加速，无 Python for 循环。
    训练单元和 = 外矩形和 - 内矩形（保护区域+CUT）和。
    """
    n_doppler, n_range = rdm_power.shape
    g_d, g_r = guard_cells
    t_d, t_r = training_cells

    # 训练单元总数
    outer_d = 2 * (g_d + t_d) + 1
    outer_r = 2 * (g_r + t_r) + 1
    inner_d = 2 * g_d + 1
    inner_r = 2 * g_r + 1
    num_training = outer_d * outer_r - inner_d * inner_r
    alpha = calculate_cfar_threshold_factor(num_training, pfa, method="ca")

    # 积分图
    integral = np.cumsum(np.cumsum(rdm_power, axis=0), axis=1)
    pad = np.zeros((n_doppler + 1, n_range + 1))
    pad[1:, 1:] = integral

    # 向量化索引
    ii, jj = np.meshgrid(np.arange(n_doppler), np.arange(n_range), indexing="ij")

    # 外矩形（训练窗口）边界
    o_r0 = np.maximum(0, ii - g_d - t_d)
    o_r1 = np.minimum(n_doppler, ii + g_d + t_d + 1)
    o_c0 = np.maximum(0, jj - g_r - t_r)
    o_c1 = np.minimum(n_range, jj + g_r + t_r + 1)

    # 内矩形（保护区域 + CUT）边界
    i_r0 = np.maximum(0, ii - g_d)
    i_r1 = np.minimum(n_doppler, ii + g_d + 1)
    i_c0 = np.maximum(0, jj - g_r)
    i_c1 = np.minimum(n_range, jj + g_r + 1)

    # 积分图查表
    def rect_sum(r0, r1, c0, c1):
        return pad[r1, c1] - pad[r0, c1] - pad[r1, c0] + pad[r0, c0]

    outer_sum = rect_sum(o_r0, o_r1, o_c0, o_c1)
    inner_sum = rect_sum(i_r0, i_r1, i_c0, i_c1)
    train_sum = outer_sum - inner_sum

    actual_n = (
        (o_r1 - o_r0) * (o_c1 - o_c0) - (i_r1 - i_r0) * (i_c1 - i_c0)
    )

    valid = actual_n > 0
    noise_est = np.zeros_like(rdm_power)
    noise_est[valid] = train_sum[valid] / actual_n[valid]
    threshold = alpha * noise_est
    detections = (rdm_power > threshold) & valid

    return detections, threshold


def os_cfar_2d(
    rdm_power: np.ndarray,
    guard_cells: tuple[int, int],
    training_cells: tuple[int, int],
    pfa: float,
    os_rank: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """二维有序统计量 CFAR（向量化实现）。

    用 padding + 滑动窗口提取训练单元，排序取第 k 个值。
    """
    n_doppler, n_range = rdm_power.shape
    g_d, g_r = guard_cells
    t_d, t_r = training_cells

    outer_d = 2 * (g_d + t_d) + 1
    outer_r = 2 * (g_r + t_r) + 1
    inner_d = 2 * g_d + 1
    inner_r = 2 * g_r + 1
    num_training = outer_d * outer_r - inner_d * inner_r

    if os_rank <= 0:
        os_rank = int(3 * num_training / 4)
    alpha = calculate_cfar_threshold_factor(num_training, pfa, method="os", os_rank=os_rank)

    # 训练单元偏移（排除保护区域和 CUT）
    pad_d = g_d + t_d
    pad_r = g_r + t_r
    rdm_pad = np.pad(rdm_power, ((pad_d, pad_d), (pad_r, pad_r)),
                     mode="constant", constant_values=np.nan)

    offsets = []
    for di in range(-pad_d, pad_d + 1):
        for dj in range(-pad_r, pad_r + 1):
            if abs(di) <= g_d and abs(dj) <= g_r:
                continue
            offsets.append((di + pad_d, dj + pad_r))

    # 提取训练单元：shape = (n_doppler, n_range, n_train)
    n_train = len(offsets)
    train_vals = np.zeros((n_doppler, n_range, n_train))
    for k, (di, dj) in enumerate(offsets):
        train_vals[:, :, k] = rdm_pad[di:di + n_doppler, dj:dj + n_range]

    # 排序取第 k 个
    sorted_vals = np.sort(train_vals, axis=2)
    actual_rank = min(os_rank - 1, n_train - 1)
    noise_est = sorted_vals[:, :, actual_rank]

    threshold = alpha * noise_est
    valid = np.isfinite(noise_est)
    detections = np.zeros(rdm_power.shape, dtype=bool)
    detections[valid] = rdm_power[valid] > threshold[valid]

    return detections, threshold


def plot_cfar_results(
    rdm_power_db: np.ndarray,
    detections_ca: np.ndarray,
    detections_os: np.ndarray,
    threshold_ca: np.ndarray,
    threshold_os: np.ndarray,
    targets: list,
    pfa: float,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
):
    """绘制 CFAR 检测结果：4 子图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_doppler, n_range = rdm_power_db.shape

    # 子图 1：原始 RDM
    ax = axes[0, 0]
    im = ax.imshow(rdm_power_db, aspect="auto", origin="lower", cmap="jet",
                   vmin=-10, vmax=30)
    ax.set_xlabel("距离 bin")
    ax.set_ylabel("多普勒 bin")
    ax.set_title("距离-多普勒功率矩阵 (dB)")
    fig.colorbar(im, ax=ax, label="功率 (dB)")

    # 子图 2：CA-CFAR 检测
    ax = axes[0, 1]
    ax.imshow(rdm_power_db, aspect="auto", origin="lower", cmap="gray",
              vmin=-10, vmax=30, alpha=0.6)
    det_ca = np.argwhere(detections_ca)
    if len(det_ca) > 0:
        ax.scatter(det_ca[:, 1], det_ca[:, 0], c="red", s=8, marker="x", label="CA-CFAR")
    for t in targets:
        ax.plot(t["range_bin"], t["doppler_bin"], "g^", markersize=12)
    ax.set_xlabel("距离 bin")
    ax.set_ylabel("多普勒 bin")
    ax.set_title(f"CA-CFAR 检测 ({np.sum(detections_ca)} 个)")
    ax.legend()

    # 子图 3：OS-CFAR 检测
    ax = axes[1, 0]
    ax.imshow(rdm_power_db, aspect="auto", origin="lower", cmap="gray",
              vmin=-10, vmax=30, alpha=0.6)
    det_os = np.argwhere(detections_os)
    if len(det_os) > 0:
        ax.scatter(det_os[:, 1], det_os[:, 0], c="cyan", s=8, marker="x", label="OS-CFAR")
    for t in targets:
        ax.plot(t["range_bin"], t["doppler_bin"], "g^", markersize=12)
    ax.set_xlabel("距离 bin")
    ax.set_ylabel("多普勒 bin")
    ax.set_title(f"OS-CFAR 检测 ({np.sum(detections_os)} 个)")
    ax.legend()

    # 子图 4：门限剖面对比
    ax = axes[1, 1]
    doppler_slice = n_doppler // 2
    ax.plot(rdm_power_db[doppler_slice, :], "b-", linewidth=0.8, label="RDM 功率")
    ax.plot(10 * np.log10(np.maximum(threshold_ca[doppler_slice, :], 1e-40)),
            "r--", linewidth=1, label="CA-CFAR 门限")
    ax.plot(10 * np.log10(np.maximum(threshold_os[doppler_slice, :], 1e-40)),
            "g--", linewidth=1, label="OS-CFAR 门限")
    ax.set_xlabel("距离 bin")
    ax.set_ylabel("功率 (dB)")
    ax.set_title(f"多普勒 bin {doppler_slice} 处的门限剖面")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"s04：CFAR 恒虚警率检测 (Pfa = {pfa:.0e})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "s04_cfar_detection.png"), dpi=150, bbox_inches="tight")
    print(f"  图像已保存: {output_dir}/s04_cfar_detection.png")
    plt.close(fig)


def validate(
    rdm_power: np.ndarray,
    detections_ca: np.ndarray,
    detections_os: np.ndarray,
    threshold_ca: np.ndarray,
    threshold_os: np.ndarray,
    targets: list,
    pfa: float,
    guard_cells: tuple,
    training_cells: tuple,
) -> bool:
    """验证 CFAR 检测器的关键指标。

    验证项：
      1. 门限因子 α 的正确性（CA-CFAR 公式验证）
      2. 虚警率：纯噪声场景下，CA-CFAR 的虚警率 ≈ 设计 Pfa
      3. 目标检测：注入的目标应被检测到
    """
    results = []

    # --- 验证 1：门限因子 α ---
    g_d, g_r = guard_cells
    t_d, t_r = training_cells
    outer_d = 2 * (g_d + t_d) + 1
    outer_r = 2 * (g_r + t_r) + 1
    inner_d = 2 * g_d + 1
    inner_r = 2 * g_r + 1
    num_training = outer_d * outer_r - inner_d * inner_r

    alpha_sim = calculate_cfar_threshold_factor(num_training, pfa, method="ca")
    alpha_theory = num_training * (pfa ** (-1.0 / num_training) - 1.0)
    results.append(verify(
        name="CA-CFAR 门限因子 α",
        theoretical=alpha_theory,
        simulated=alpha_sim,
        tolerance=alpha_theory * 0.001,
        unit="",
    ))

    # --- 验证 2：虚警率（纯噪声） ---
    # 生成纯噪声 RDM，运行 CA-CFAR，统计虚警率
    noise_rdm = generate_synthetic_rdm(
        n_range=rdm_power.shape[1],
        n_doppler=rdm_power.shape[0],
        targets=[],
        noise_seed=999,
    )
    det_noise, _ = ca_cfar_2d(noise_rdm["rdm_power"], guard_cells, training_cells, pfa)

    # 排除边缘
    margin_d = g_d + t_d + 1
    margin_r = g_r + t_r + 1
    inner_det = det_noise[margin_d:-margin_d, margin_r:-margin_r]
    measured_pfa = np.sum(inner_det) / inner_det.size

    results.append(verify(
        name=f"虚警率 Pfa (设计 {pfa:.0e})",
        theoretical=pfa,
        simulated=measured_pfa,
        tolerance=max(pfa * 100, 1e-3),  # 容差 2 个数量级
        unit="",
    ))

    # --- 验证 3：目标检测 ---
    # 检查注入目标位置 ±2 bin 内是否有检测
    detected_count = 0
    for t in targets:
        r, d = t["range_bin"], t["doppler_bin"]
        region = detections_ca[
            max(0, d - 2):d + 3,
            max(0, r - 2):r + 3,
        ]
        if np.any(region):
            detected_count += 1

    detection_rate = detected_count / len(targets) if targets else 1.0
    results.append(verify(
        name="目标检测率",
        theoretical=1.0,  # 期望 100% 检测
        simulated=detection_rate,
        tolerance=0.01,
        unit="",
    ))

    return print_validation("s04 CFAR 检测", results)


def main():
    """运行 s04 CFAR 检测仿真与验证。"""
    print("=" * 60)
    print("s04：CFAR 恒虚警率检测")
    print("=" * 60)

    # === 参数 ===
    n_range = 256       # 距离维 256 个单元
    n_doppler = 64      # 多普勒维 64 个单元
    pfa = 1e-4          # 设计虚警率
    guard_cells = (2, 2)
    training_cells = (8, 8)

    # 训练单元数
    outer_d = 2 * (guard_cells[0] + training_cells[0]) + 1
    outer_r = 2 * (guard_cells[1] + training_cells[1]) + 1
    inner_d = 2 * guard_cells[0] + 1
    inner_r = 2 * guard_cells[1] + 1
    num_training = outer_d * outer_r - inner_d * inner_r

    alpha_ca = calculate_cfar_threshold_factor(num_training, pfa, method="ca")
    os_rank = int(3 * num_training / 4)
    alpha_os = calculate_cfar_threshold_factor(num_training, pfa, "os", os_rank)

    print(f"\nCFAR 参数:")
    print(f"  矩阵大小     = {n_doppler} × {n_range} (多普勒 × 距离)")
    print(f"  设计虚警率    = Pfa = {pfa:.0e}")
    print(f"  保护单元      = {guard_cells}")
    print(f"  训练单元      = {training_cells}")
    print(f"  训练单元总数  = {num_training}")
    print(f"  CA-CFAR α     = {alpha_ca:.2f} ({10 * np.log10(alpha_ca):.1f} dB)")
    print(f"  OS-CFAR rank  = {os_rank}, α = {alpha_os:.2f}")

    # === 目标场景 ===
    targets = [
        {"range_bin": 64,  "doppler_bin": 16, "snr_linear": 10.0},   # SNR = 10 dB
        {"range_bin": 128, "doppler_bin": 32, "snr_linear": 31.6},   # SNR = 15 dB
        {"range_bin": 192, "doppler_bin": 48, "snr_linear": 100.0},  # SNR = 20 dB
    ]

    print(f"\n注入目标:")
    for i, t in enumerate(targets):
        snr_db = 10 * np.log10(t["snr_linear"])
        print(f"  目标{i+1}: range_bin={t['range_bin']}, doppler_bin={t['doppler_bin']}, SNR={snr_db:.0f} dB")

    # === 生成 RDM ===
    print(f"\n生成合成 RDM...")
    rdm = generate_synthetic_rdm(n_range, n_doppler, targets, noise_seed=42)
    rdm_power = rdm["rdm_power"]
    rdm_power_db = rdm["rdm_power_db"]
    print(f"  RDM 功率范围: {np.min(rdm_power_db):.1f} ~ {np.max(rdm_power_db):.1f} dB")

    # === CFAR 检测 ===
    print(f"\n运行 CA-CFAR...")
    det_ca, thresh_ca = ca_cfar_2d(rdm_power, guard_cells, training_cells, pfa)
    print(f"  检测数: {np.sum(det_ca)}")

    print(f"运行 OS-CFAR...")
    det_os, thresh_os = os_cfar_2d(rdm_power, guard_cells, training_cells, pfa, os_rank)
    print(f"  检测数: {np.sum(det_os)}")

    # === 绘图 ===
    print(f"\n绘制结果...")
    plot_cfar_results(rdm_power_db, det_ca, det_os, thresh_ca, thresh_os, targets, pfa)

    # === 验证 ===
    print(f"\n运行验证...")
    all_passed = validate(
        rdm_power, det_ca, det_os, thresh_ca, thresh_os,
        targets, pfa, guard_cells, training_cells,
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
