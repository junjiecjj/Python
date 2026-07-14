"""s18：多站定位融合（圆-圆交叉定位）。

验证目标：
  - 实现基于测距的多站定位（multilateration），通过最小二乘法求解目标位置
  - 计算几何精度因子（GDOP），量化站-目标几何构型对定位精度的影响
  - Monte Carlo 仿真验证定位 RMSE 逼近 Cramér-Rao 下界（CRLB）
  - 分析站数和测距噪声对定位精度的影响

多站定位原理：
  每个雷达站独立测量到目标的斜距 R_i，由几何关系：
    (x - x_i)^2 + (y - y_i)^2 = R_i^2

  对 N 个站（N >= 3），将非线性方程线性化后用最小二乘求解：
    将第 i 个方程减去第 1 个方程，消去 x^2 + y^2 项，得到线性方程组 Hx = b。
  其中 H 的每行为 [-2(x_i - x_1), -2(y_i - y_1)]，b 的每行为
    (R_1^2 - R_i^2) + (x_i^2 + y_i^2) - (x_1^2 + y_1^2)。

  GDOP 反映几何构型对定位精度的放大效应：
    GDOP = sqrt(trace((H_geo^T H_geo)^{-1}))
  其中 H_geo 为各站到目标的方向余弦矩阵。
  目标位于站网中心时 GDOP 最小（最优几何），远离站网时 GDOP 急剧增大。

  CRLB 给出无偏估计器的理论最小方差：
    var(x_hat) >= (sigma_R^2) * (H_geo^T H_geo)^{-1}
  其中 sigma_R 为测距噪声标准差。

对应知识库：分布式雷达 / 多站定位
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle

from lib.signal_utils import power_to_db, db_to_power
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# FFT 约定：numpy.fft.fft（正变换无 1/N 缩放）
# Parseval: sum(|x|^2) = (1/N) * sum(|X|^2)


# ============================================================
# 核心函数
# ============================================================


def range_measurement(
    true_range_m: np.ndarray,
    noise_std_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """模拟测距测量：真实距离 + 高斯噪声。

    雷达测距精度由信号带宽和 SNR 决定。
    测量模型：R_measured = R_true + n，n ~ N(0, sigma_R^2)

    Args:
        true_range_m: 各站到目标的真实距离 (m)，形状 (n_stations,)
        noise_std_m:  测距噪声标准差 (m)
        rng:          随机数生成器（可复现）

    Returns:
        测量距离 (m)，形状 (n_stations,)
    """
    noise_m = rng.normal(0.0, noise_std_m, size=true_range_m.shape)
    return true_range_m + noise_m


def cramer_rao_lower_bound(
    station_positions_m: np.ndarray,
    target_position_m: np.ndarray,
    range_noise_std_m: float,
) -> np.ndarray:
    """计算位置估计的 Cramér-Rao 下界（CRLB）。

    CRLB 给出无偏估计器方差的理论下限。
    对于独立高斯测距噪声，Fisher 信息矩阵为：
      F = (1/sigma_R^2) * H_geo^T * H_geo
    其中 H_geo 为方向余弦矩阵，每行为 [(x_t - x_i)/R_i, (y_t - y_i)/R_i]。

    CRLB = F^{-1}，对角线元素为各坐标估计方差的下界。

    Args:
        station_positions_m: 站位置 (n_stations, 2)，单位 m
        target_position_m:   目标位置 (2,)，单位 m
        range_noise_std_m:   测距噪声标准差 (m)

    Returns:
        位置估计标准差下界 [sigma_x, sigma_y] (m)
    """
    n_stations = station_positions_m.shape[0]
    diff = target_position_m - station_positions_m  # (n_stations, 2)
    ranges_m = np.linalg.norm(diff, axis=1)  # (n_stations,)

    # 方向余弦矩阵 H_geo: (n_stations, 2)
    h_geo = diff / ranges_m[:, np.newaxis]

    # Fisher 信息矩阵: F = (1/sigma^2) * H_geo^T @ H_geo
    fisher = h_geo.T @ h_geo / range_noise_std_m**2

    # CRLB 协方差矩阵 = F^{-1}
    crlb_cov = np.linalg.inv(fisher)
    return np.sqrt(np.diag(crlb_cov))  # [sigma_x, sigma_y] (m)


def compute_gdop(
    station_positions_m: np.ndarray,
    target_position_m: np.ndarray,
) -> float:
    """计算几何精度因子（GDOP）。

    GDOP = sqrt(trace((H_geo^T H_geo)^{-1}))
    其中 H_geo 为各站到目标的方向余弦矩阵。

    物理含义：
      GDOP 是几何构型对定位误差的放大系数。
      定位 RMSE ≈ sigma_R * GDOP。
      GDOP = 1 为理想情况（不可能达到），实际 GDOP >= 1。
      目标位于站网中心且站间角度均匀分布时 GDOP 最小。

    Args:
        station_positions_m: 站位置 (n_stations, 2)，单位 m
        target_position_m:   目标位置 (2,)，单位 m

    Returns:
        GDOP 值（无量纲，>= 1）
    """
    diff = target_position_m - station_positions_m  # (n_stations, 2)
    ranges_m = np.linalg.norm(diff, axis=1)  # (n_stations,)

    # 方向余弦矩阵: (n_stations, 2)
    h_geo = diff / ranges_m[:, np.newaxis]

    # GDOP = sqrt(trace((H^T H)^{-1}))
    g = h_geo.T @ h_geo  # (2, 2)
    g_inv = np.linalg.inv(g)
    return np.sqrt(np.trace(g_inv))


def multilateration_2d(
    station_positions_m: np.ndarray,
    measured_ranges_m: np.ndarray,
) -> np.ndarray:
    """多站定位：圆-圆交叉，最小二乘法求解目标位置。

    将非线性距离方程线性化：
      (x - x_i)^2 + (y - y_i)^2 = R_i^2

    用第 i 个方程减去第 1 个方程，消去二次项：
      -2(x_i - x_1)*x - 2(y_i - y_1)*y = R_1^2 - R_i^2 + x_i^2 - x_1^2 + y_i^2 - y_1^2

    写成矩阵形式 Hx = b，用最小二乘求解。

    Args:
        station_positions_m: 站位置 (n_stations, 2)，单位 m
        measured_ranges_m:   各站测量距离 (n_stations,)，单位 m

    Returns:
        估计的目标位置 (2,)，单位 m
    """
    n_stations = station_positions_m.shape[0]

    # 以第 1 个站为参考
    x1, y1 = station_positions_m[0]

    # 构造线性方程组 Hx = b
    h_mat = np.zeros((n_stations - 1, 2))
    b_vec = np.zeros(n_stations - 1)

    for i in range(1, n_stations):
        xi, yi = station_positions_m[i]
        h_mat[i - 1, 0] = -2.0 * (xi - x1)
        h_mat[i - 1, 1] = -2.0 * (yi - y1)
        b_vec[i - 1] = (
            measured_ranges_m[0] ** 2
            - measured_ranges_m[i] ** 2
            + xi**2
            - x1**2
            + yi**2
            - y1**2
        )

    # 最小二乘求解: x = (H^T H)^{-1} H^T b
    result = np.linalg.lstsq(h_mat, b_vec, rcond=None)
    return result[0]  # (2,)


def positioning_rmse(
    station_positions_m: np.ndarray,
    true_position_m: np.ndarray,
    range_noise_std_m: float,
    n_trials: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo 仿真计算定位 RMSE 和偏差。

    每次试验：
      1. 计算各站到目标的真实距离
      2. 加高斯噪声模拟测距误差
      3. 用最小二乘法估计目标位置
      4. 记录估计误差

    Args:
        station_positions_m: 站位置 (n_stations, 2)，单位 m
        true_position_m:     目标真实位置 (2,)，单位 m
        range_noise_std_m:   测距噪声标准差 (m)
        n_trials:            Monte Carlo 试验次数
        seed:                随机种子（保证可复现）

    Returns:
        (rmse_m, bias_m)：定位 RMSE (m) 和平均偏差 (m)
    """
    rng = np.random.default_rng(seed)

    # 真实距离
    diff = true_position_m - station_positions_m
    true_ranges_m = np.linalg.norm(diff, axis=1)

    errors_m = np.zeros((n_trials, 2))

    for trial in range(n_trials):
        # 加噪声
        measured_ranges_m = range_measurement(true_ranges_m, range_noise_std_m, rng)

        # 定位
        estimated_pos_m = multilateration_2d(station_positions_m, measured_ranges_m)
        errors_m[trial] = estimated_pos_m - true_position_m

    # RMSE
    rmse_m = np.sqrt(np.mean(np.sum(errors_m**2, axis=1)))

    # 平均偏差
    bias_m = np.sqrt(np.sum(np.mean(errors_m, axis=0) ** 2))

    return rmse_m, bias_m


# ============================================================
# 绘图
# ============================================================


def plot_multi_station_fusion(
    station_positions_m: np.ndarray,
    true_position_m: np.ndarray,
    range_noise_std_m: float,
    n_trials: int,
    seed: int,
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
) -> None:
    """绘制多站定位融合结果（3 子图）。

    子图 1：多站布局和定位结果散点
      - 站位置、真实目标、Monte Carlo 估计散布

    子图 2：GDOP 等高线图
      - 展示不同位置的定位精度分布

    子图 3：定位 RMSE vs 站数/测距噪声（与 CRLB 对比）

    Args:
        station_positions_m: 站位置 (n_stations, 2)，单位 m
        true_position_m:     目标真实位置 (2,)，单位 m
        range_noise_std_m:   测距噪声标准差 (m)
        n_trials:            Monte Carlo 试验次数
        seed:                随机种子
        output_dir:          输出目录
    """
    rng = np.random.default_rng(seed)
    n_stations = station_positions_m.shape[0]

    # Monte Carlo 定位散点
    diff = true_position_m - station_positions_m
    true_ranges_m = np.linalg.norm(diff, axis=1)
    estimated_positions_m = np.zeros((n_trials, 2))
    for trial in range(n_trials):
        measured_m = range_measurement(true_ranges_m, range_noise_std_m, rng)
        estimated_positions_m[trial] = multilateration_2d(
            station_positions_m, measured_m
        )

    # 转换为 km 显示
    station_positions_km = station_positions_m / 1000.0
    true_position_km = true_position_m / 1000.0
    estimated_positions_km = estimated_positions_m / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- 子图 1：多站布局和定位结果 ----
    ax1 = axes[0]
    ax1.scatter(
        station_positions_km[:, 0],
        station_positions_km[:, 1],
        c="blue",
        marker="s",
        s=120,
        zorder=5,
        label=f"雷达站 ({n_stations}站)",
    )
    for i in range(n_stations):
        ax1.annotate(
            f"S{i+1}",
            xy=(station_positions_km[i, 0], station_positions_km[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="blue",
        )

    # 估计位置散点
    ax1.scatter(
        estimated_positions_km[:, 0],
        estimated_positions_km[:, 1],
        c="red",
        alpha=0.15,
        s=8,
        label="估计位置散布",
    )

    # 真实目标
    ax1.scatter(
        [true_position_km[0]],
        [true_position_km[1]],
        c="green",
        marker="*",
        s=200,
        zorder=6,
        label="真实目标",
    )

    # 站网覆盖范围
    station_radius_km = np.mean(
        np.linalg.norm(station_positions_m, axis=1)
    ) / 1000.0
    circle = Circle(
        (0, 0),
        station_radius_km,
        fill=False,
        linestyle="--",
        color="gray",
        alpha=0.5,
    )
    ax1.add_patch(circle)

    ax1.set_xlabel("X (km)", fontsize=12)
    ax1.set_ylabel("Y (km)", fontsize=12)
    ax1.set_title(
        f"多站定位结果 (σ_R={range_noise_std_m:.0f}m, {n_trials}次试验)",
        fontsize=13,
    )
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2：GDOP 等高线图 ----
    ax2 = axes[1]
    station_radius_m = np.mean(np.linalg.norm(station_positions_m, axis=1))
    grid_limit_m = station_radius_m * 1.3
    grid_n = 200
    x_grid_m = np.linspace(-grid_limit_m, grid_limit_m, grid_n)
    y_grid_m = np.linspace(-grid_limit_m, grid_limit_m, grid_n)
    xx_m, yy_m = np.meshgrid(x_grid_m, y_grid_m)

    gdop_map = np.full((grid_n, grid_n), np.nan)
    for i in range(grid_n):
        for j in range(grid_n):
            pos_m = np.array([xx_m[i, j], yy_m[i, j]])
            ranges_m = np.linalg.norm(pos_m - station_positions_m, axis=1)
            if np.all(ranges_m > 100.0):  # 避免站位奇点
                gdop_map[i, j] = compute_gdop(station_positions_m, pos_m)

    # 等高线 levels
    gdop_valid = gdop_map[~np.isnan(gdop_map)]
    gdop_max = min(np.percentile(gdop_valid, 95), 20.0)
    levels = np.linspace(1.0, gdop_max, 15)

    cs = ax2.contourf(
        xx_m / 1000.0,
        yy_m / 1000.0,
        gdop_map,
        levels=levels,
        cmap="RdYlGn_r",
        extend="max",
    )
    plt.colorbar(cs, ax=ax2, label="GDOP")

    # 站位置
    ax2.scatter(
        station_positions_km[:, 0],
        station_positions_km[:, 1],
        c="blue",
        marker="s",
        s=80,
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
        label="雷达站",
    )
    # 目标
    ax2.scatter(
        [true_position_km[0]],
        [true_position_km[1]],
        c="red",
        marker="*",
        s=150,
        zorder=6,
        label="目标",
    )

    ax2.set_xlabel("X (km)", fontsize=12)
    ax2.set_ylabel("Y (km)", fontsize=12)
    ax2.set_title("GDOP 等高线（定位精度分布）", fontsize=13)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_aspect("equal")

    # ---- 子图 3：RMSE vs 站数/测距噪声 ----
    ax3 = axes[2]

    # 左侧 Y 轴：RMSE vs 站数
    station_counts = np.arange(3, 9)  # 3~8 站
    rmse_vs_stations_m = np.zeros(len(station_counts))
    crlb_vs_stations_m = np.zeros(len(station_counts))

    for idx, n_sta in enumerate(station_counts):
        # 均匀分布在圆上
        angles_rad = np.linspace(0, 2 * np.pi, n_sta, endpoint=False)
        positions_m = np.column_stack(
            [
                station_radius_m * np.cos(angles_rad),
                station_radius_m * np.sin(angles_rad),
            ]
        )
        rmse_m, _ = positioning_rmse(
            positions_m,
            true_position_m,
            range_noise_std_m,
            n_trials=n_trials,
            seed=seed,
        )
        rmse_vs_stations_m[idx] = rmse_m
        crlb_vs_stations_m[idx] = np.sqrt(
            np.sum(cramer_rao_lower_bound(positions_m, true_position_m, range_noise_std_m) ** 2)
        )

    color_rmse = "tab:blue"
    color_crlb = "tab:red"
    ax3.set_xlabel("站数", fontsize=12)
    ax3.set_ylabel("定位 RMSE / CRLB (m)", fontsize=12, color=color_rmse)
    ax3.plot(
        station_counts,
        rmse_vs_stations_m,
        "o-",
        color=color_rmse,
        linewidth=2,
        markersize=8,
        label="仿真 RMSE",
    )
    ax3.plot(
        station_counts,
        crlb_vs_stations_m,
        "s--",
        color=color_crlb,
        linewidth=2,
        markersize=8,
        label="CRLB (理论下界)",
    )
    ax3.tick_params(axis="y", labelcolor=color_rmse)
    ax3.set_xticks(station_counts)

    # 右侧 Y 轴：GDOP vs 站数
    ax3b = ax3.twinx()
    gdop_vs_stations = np.zeros(len(station_counts))
    for idx, n_sta in enumerate(station_counts):
        angles_rad = np.linspace(0, 2 * np.pi, n_sta, endpoint=False)
        positions_m = np.column_stack(
            [
                station_radius_m * np.cos(angles_rad),
                station_radius_m * np.sin(angles_rad),
            ]
        )
        gdop_vs_stations[idx] = compute_gdop(positions_m, true_position_m)

    color_gdop = "tab:green"
    ax3b.plot(
        station_counts,
        gdop_vs_stations,
        "^-.",
        color=color_gdop,
        linewidth=2,
        markersize=8,
        label="GDOP",
    )
    ax3b.set_ylabel("GDOP", fontsize=12, color=color_gdop)
    ax3b.tick_params(axis="y", labelcolor=color_gdop)

    # 合并图例
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc="upper right")

    ax3.set_title("定位精度 vs 站数（RMSE 与 CRLB 对比）", fontsize=13)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "s18_multi_station_fusion.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  图像已保存: {output_dir}/s18_multi_station_fusion.png")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(
    station_radius_m: float,
    range_noise_std_m: float,
    n_trials: int,
    seed: int,
) -> bool:
    """验证多站定位融合的正确性。

    验证项：
      a. 定位精度：RMSE / CRLB < 1.5（接近理论最优）
      b. GDOP 值：3 站 GDOP > 4 站 GDOP（站越多精度越高）
      c. 定位偏差：平均偏差 < 0.1 * RMSE（无偏估计）
      d. 多站增益：4 站 RMSE < 2 站 RMSE（多站融合提升精度）
    """
    results = []
    true_position_m = np.array([0.0, 0.0])

    # --- 验证 a：定位精度 vs CRLB ---
    n_stations = 4
    angles_rad = np.linspace(0, 2 * np.pi, n_stations, endpoint=False)
    station_positions_m = np.column_stack(
        [
            station_radius_m * np.cos(angles_rad),
            station_radius_m * np.sin(angles_rad),
        ]
    )

    rmse_m, bias_m = positioning_rmse(
        station_positions_m, true_position_m, range_noise_std_m,
        n_trials=n_trials, seed=seed,
    )
    crlb_m = cramer_rao_lower_bound(
        station_positions_m, true_position_m, range_noise_std_m,
    )
    crlb_total_m = np.sqrt(np.sum(crlb_m**2))

    rmse_to_crlb_ratio = rmse_m / crlb_total_m
    results.append(verify(
        name="定位精度（RMSE/CRLB 比值 < 1.5）",
        theoretical=1.0,          # 理想情况 RMSE = CRLB
        simulated=rmse_to_crlb_ratio,
        tolerance=0.5,            # 允许到 1.5
        unit="",
    ))

    # --- 验证 b：GDOP 随站数递减 ---
    # 3 站
    angles_3 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    stations_3_m = np.column_stack(
        [station_radius_m * np.cos(angles_3), station_radius_m * np.sin(angles_3)]
    )
    gdop_3 = compute_gdop(stations_3_m, true_position_m)

    # 4 站
    angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    stations_4_m = np.column_stack(
        [station_radius_m * np.cos(angles_4), station_radius_m * np.sin(angles_4)]
    )
    gdop_4 = compute_gdop(stations_4_m, true_position_m)

    # 验证 3 站 GDOP > 4 站 GDOP（即 gdop_3 - gdop_4 > 0）
    gdop_diff = gdop_3 - gdop_4
    results.append(verify(
        name="GDOP 递减（3站GDOP > 4站GDOP）",
        theoretical=0.5,          # 期望差值 > 0
        simulated=gdop_diff,
        tolerance=0.5,            # 允许 gdop_diff >= 0
        unit="",
    ))

    # --- 验证 c：定位偏差 ---
    bias_to_rmse_ratio = bias_m / rmse_m if rmse_m > 0 else 0.0
    results.append(verify(
        name="定位偏差（bias/RMSE < 0.1）",
        theoretical=0.0,
        simulated=bias_to_rmse_ratio,
        tolerance=0.1,
        unit="",
    ))

    # --- 验证 d：多站增益（4 站 RMSE < 2 站 RMSE） ---
    # 注意：2 站定位只有 1 个独立方程，无法唯一确定 2D 位置。
    # 用 3 站 vs 6 站来验证多站增益。
    angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    stations_6_m = np.column_stack(
        [station_radius_m * np.cos(angles_6), station_radius_m * np.sin(angles_6)]
    )
    rmse_6_m, _ = positioning_rmse(
        stations_6_m, true_position_m, range_noise_std_m,
        n_trials=n_trials, seed=seed,
    )
    rmse_ratio = rmse_6_m / rmse_m  # 6 站 RMSE / 4 站 RMSE，应 < 1
    results.append(verify(
        name="多站增益（6站RMSE < 4站RMSE）",
        theoretical=0.7,          # 期望比值 < 1（约 0.7~0.85）
        simulated=rmse_ratio,
        tolerance=0.3,            # 允许到 1.0
        unit="",
    ))

    return print_validation("s18 多站定位融合", results)


# ============================================================
# 主函数
# ============================================================


def main() -> int:
    """运行 s18 多站定位融合仿真与验证。"""
    print("=" * 60)
    print("s18：多站定位融合（圆-圆交叉定位）")
    print("=" * 60)

    # 仿真参数
    station_radius_m = 20000.0      # 站分布半径 (m)
    n_stations = 4                   # 站数
    range_noise_std_m = 50.0        # 测距噪声标准差 (m)
    n_trials = 1000                  # Monte Carlo 试验数
    seed = 42                        # 随机种子
    true_position_m = np.array([0.0, 0.0])  # 目标在原点

    # 站位置：均匀分布在圆上
    angles_rad = np.linspace(0, 2 * np.pi, n_stations, endpoint=False)
    station_positions_m = np.column_stack(
        [
            station_radius_m * np.cos(angles_rad),
            station_radius_m * np.sin(angles_rad),
        ]
    )

    print(f"\n仿真参数:")
    print(f"  站数 N              = {n_stations}")
    print(f"  站分布半径          = {station_radius_m/1000:.0f} km")
    print(f"  目标位置            = ({true_position_m[0]:.0f}, {true_position_m[1]:.0f}) m")
    print(f"  测距噪声 σ_R        = {range_noise_std_m:.0f} m")
    print(f"  Monte Carlo 试验数  = {n_trials}")
    print(f"  随机种子            = {seed}")

    # GDOP
    gdop = compute_gdop(station_positions_m, true_position_m)
    print(f"\nGDOP = {gdop:.3f}")

    # CRLB
    crlb_m = cramer_rao_lower_bound(
        station_positions_m, true_position_m, range_noise_std_m,
    )
    crlb_total_m = np.sqrt(np.sum(crlb_m**2))
    print(f"CRLB 理论下界: σ_x={crlb_m[0]:.2f}m, σ_y={crlb_m[1]:.2f}m")
    print(f"CRLB 综合定位精度: {crlb_total_m:.2f}m")

    # Monte Carlo 定位
    print(f"\n运行 Monte Carlo 定位仿真 ({n_trials} 次)...")
    rmse_m, bias_m = positioning_rmse(
        station_positions_m, true_position_m, range_noise_std_m,
        n_trials=n_trials, seed=seed,
    )
    print(f"定位 RMSE   = {rmse_m:.2f} m")
    print(f"定位偏差    = {bias_m:.2f} m")
    print(f"RMSE/CRLB   = {rmse_m/crlb_total_m:.3f}")

    # 绘图
    print(f"\n绘制多站定位融合结果...")
    plot_multi_station_fusion(
        station_positions_m, true_position_m,
        range_noise_std_m, n_trials, seed,
    )

    # 验证
    print(f"\n运行验证...")
    all_passed = validate(station_radius_m, range_noise_std_m, n_trials, seed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
