"""s08：生产者-消费者模式的雷达信号处理流水线框架。

设计模式：
  生产者-消费者（Producer-Consumer）+ 责任链（Chain of Responsibility）
  每个处理环节（Stage）是一个独立的计算单元，接收输入、产出输出。
  多个 Stage 通过 Pipeline 串联，前一个 Stage 的输出自动流入下一个 Stage 的输入。
  使用 "|" 运算符实现流式语法：pipeline = stage1 | stage2 | stage3

核心类：
  - Stage：抽象基类，定义 process(data) 接口
  - Pipeline：管理 Stage 链，记录每步执行时间和中间结果
  - SNRCalculator：计算给定距离的 SNR（封装 s01 的雷达方程）
  - RangeSweeper：扫描距离范围，输出 SNR vs 距离曲线
  - ThresholdDetector：对 SNR 曲线做门限检测，输出探测距离
  - ReportGenerator：生成结果报告文本

验证项：
  1. 单 Stage 正确性：SNRCalculator(50km) 的输出 = 手动计算值
  2. Pipeline 串联：run() 的最终输出 = 手动逐步执行的结果
  3. 执行时间记录：每个 stage 的耗时应 > 0
  4. 并行正确性：并行结果 = 串行结果
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.radar_params import RadarParams, BOLTZMANN, STANDARD_TEMP
from lib.signal_utils import power_to_db, db_to_power
from lib.validation import verify, print_validation

# 中文字体设置
rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ============================================================
# 流水线基础设施
# ============================================================


class Stage(ABC):
    """处理环节的抽象基类。

    每个 Stage 实现一个独立的计算步骤。通过 "|" 运算符可以将
    两个 Stage 串联，形成责任链。

    设计意图：
      - 单一职责：每个 Stage 只做一件事
      - 可组合：通过 | 运算符自由拼装
      - 可观测：Pipeline 会自动记录每个 Stage 的耗时
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理输入数据，返回输出。

        Args:
            data: 输入数据（类型由具体 Stage 定义）

        Returns:
            处理后的输出数据
        """
        ...

    def __or__(self, other: "Stage") -> "Pipeline":
        """支持 stage1 | stage2 语法，返回包含两个 Stage 的 Pipeline。"""
        pipeline = Pipeline()
        # 如果 self 已经是 Pipeline，先把它的 stages 复制过来
        if isinstance(self, Pipeline):
            for s in self._stages:
                pipeline.add_stage(s)
        else:
            pipeline.add_stage(self)
        # 同理处理 other
        if isinstance(other, Pipeline):
            for s in other._stages:
                pipeline.add_stage(s)
        else:
            pipeline.add_stage(other)
        return pipeline


@dataclass
class StageResult:
    """单个 Stage 的执行记录。"""

    name: str
    output: Any
    elapsed_s: float


@dataclass
class PipelineResult:
    """Pipeline 执行的完整记录。"""

    stages: list[StageResult] = field(default_factory=list)
    final_output: Any = None


class Pipeline(Stage):
    """流水线：管理多个 Stage 的串联执行。

    Pipeline 本身也是一个 Stage（组合模式），可以嵌套到更大的 Pipeline 中。
    执行时依次调用每个子 Stage 的 process()，前一个的输出作为后一个的输入。

    特性：
      - 记录每个 Stage 的执行耗时和中间结果
      - 支持 | 运算符继续追加 Stage
    """

    def __init__(self) -> None:
        super().__init__("Pipeline")
        self._stages: list[Stage] = []

    def add_stage(self, stage: Stage) -> "Pipeline":
        """添加一个处理环节。返回 self 以支持链式调用。"""
        self._stages.append(stage)
        return self

    def process(self, data: Any) -> Any:
        """依次执行所有 Stage，返回最终输出。"""
        result = self.run(data)
        return result.final_output

    def run(self, input_data: Any) -> PipelineResult:
        """执行完整流水线，返回包含每步记录的结果。

        Args:
            input_data: 流水线的初始输入

        Returns:
            PipelineResult，包含每个 Stage 的输出和耗时
        """
        result = PipelineResult()
        current = input_data

        for stage in self._stages:
            start = time.perf_counter()
            current = stage.process(current)
            elapsed = time.perf_counter() - start

            result.stages.append(StageResult(
                name=stage.name,
                output=current,
                elapsed_s=elapsed,
            ))

        result.final_output = current
        return result


# ============================================================
# 具体 Stage 实现
# ============================================================


class SNRCalculator(Stage):
    """计算给定距离处的 SNR（线性值）。

    封装 s01 的雷达方程：
      SNR = (Pt * G^2 * lambda^2 * sigma * T) / ((4*pi)^3 * R^4 * k * T0 * F * B)

    输入：distance_m（float，目标距离，单位 m）
    输出：dict，包含 distance_m 和 snr_linear
    """

    def __init__(self, params: RadarParams) -> None:
        super().__init__("SNRCalculator")
        self.params = params

    def process(self, data: Any) -> dict:
        """计算单个距离的 SNR。

        Args:
            data: 目标距离 (m)，float 类型

        Returns:
            dict: {"distance_m": float, "snr_linear": float}
        """
        distance_m = float(data)
        params = self.params

        # 雷达方程：SNR = (Pt * G^2 * lambda^2 * sigma * T) / ((4pi)^3 * R^4 * k * T0 * F * B)
        numerator = (
            params.pt
            * params.gain_linear ** 2
            * params.wavelength_m ** 2
            * params.target_rcs_m2
            * params.pulse_width_s
        )
        denominator = (
            (4 * np.pi) ** 3
            * distance_m ** 4
            * BOLTZMANN * STANDARD_TEMP
            * params.noise_figure_linear
            * params.bandwidth_hz
        )
        snr_linear = numerator / denominator

        return {"distance_m": distance_m, "snr_linear": snr_linear}


class RangeSweeper(Stage):
    """扫描距离范围，输出 SNR vs 距离曲线。

    对指定的距离区间按步长采样，逐点计算 SNR。
    内部可选择串行或并行执行。

    输入：RadarParams（雷达参数）
    输出：dict，包含 ranges_m（距离数组）和 snr_linear（SNR 数组）
    """

    def __init__(
        self,
        range_start_m: float,
        range_end_m: float,
        num_points: int = 200,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        super().__init__("RangeSweeper")
        self.range_start_m = range_start_m
        self.range_end_m = range_end_m
        self.num_points = num_points
        self.parallel = parallel
        self.max_workers = max_workers

    def process(self, data: Any) -> dict:
        """扫描距离范围，计算 SNR 曲线。

        Args:
            data: RadarParams 实例

        Returns:
            dict: {
                "ranges_m": np.ndarray,       # 距离数组 (m)
                "snr_linear": np.ndarray,     # SNR 线性值数组
                "snr_db": np.ndarray,         # SNR dB 数组
                "params": RadarParams,        # 原始参数（传递给下游）
            }
        """
        params: RadarParams = data
        ranges_m = np.linspace(self.range_start_m, self.range_end_m, self.num_points)

        if self.parallel:
            snr_linear = self._compute_parallel(params, ranges_m)
        else:
            snr_linear = self._compute_serial(params, ranges_m)

        snr_db = np.array([power_to_db(s) for s in snr_linear])

        return {
            "ranges_m": ranges_m,
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "params": params,
        }

    def _compute_serial(self, params: RadarParams, ranges_m: np.ndarray) -> np.ndarray:
        """串行计算每个距离的 SNR。"""
        calc = SNRCalculator(params)
        results = [calc.process(r)["snr_linear"] for r in ranges_m]
        return np.array(results)

    def _compute_parallel(
        self, params: RadarParams, ranges_m: np.ndarray
    ) -> np.ndarray:
        """并行计算每个距离的 SNR（ThreadPoolExecutor）。"""
        calc = SNRCalculator(params)
        snr_results = np.empty(len(ranges_m))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务，保留索引以保持顺序
            future_to_idx = {
                executor.submit(calc.process, r): i
                for i, r in enumerate(ranges_m)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                snr_results[idx] = future.result()["snr_linear"]

        return snr_results


class ThresholdDetector(Stage):
    """对 SNR 曲线做门限检测，输出探测距离。

    找到 SNR 曲线与检测门限的交点，该交点对应的距离即为最大探测距离。

    输入：RangeSweeper 的输出 dict
    输出：dict，追加 detection_range_m 和 detected（是否检测到）
    """

    def __init__(self, threshold_db: float = 13.0) -> None:
        super().__init__("ThresholdDetector")
        self.threshold_db = threshold_db

    def process(self, data: Any) -> dict:
        """门限检测。

        Args:
            data: RangeSweeper 的输出 dict

        Returns:
            dict: 在原 dict 基础上追加:
                "threshold_db": float,
                "detection_range_m": float | None,
                "detected": bool,
        """
        result = dict(data)
        result["threshold_db"] = self.threshold_db

        snr_db = data["snr_db"]
        ranges_m = data["ranges_m"]

        # SNR 超过门限的最远距离即为探测距离
        above_mask = snr_db >= self.threshold_db

        if not np.any(above_mask):
            # 全程低于门限，未检测到目标
            result["detection_range_m"] = None
            result["detected"] = False
        else:
            # 取最后一个超过门限的位置（最远探测距离）
            last_above_idx = np.where(above_mask)[0][-1]
            # 线性插值提高精度：在 last_above_idx 和下一个点之间插值
            if last_above_idx < len(ranges_m) - 1:
                idx = last_above_idx
                # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                # 求 y = threshold_db 时的 x
                x1, x2 = ranges_m[idx], ranges_m[idx + 1]
                y1, y2 = snr_db[idx], snr_db[idx + 1]
                if abs(y2 - y1) > 1e-40:
                    detection_range_m = x1 + (self.threshold_db - y1) * (x2 - x1) / (y2 - y1)
                else:
                    detection_range_m = (x1 + x2) / 2.0
            else:
                detection_range_m = ranges_m[last_above_idx]

            result["detection_range_m"] = float(detection_range_m)
            result["detected"] = True

        return result


class ReportGenerator(Stage):
    """生成结果报告文本。

    将流水线的检测结果格式化为可读的文本报告。

    输入：ThresholdDetector 的输出 dict
    输出：str，报告文本
    """

    def __init__(self) -> None:
        super().__init__("ReportGenerator")

    def process(self, data: Any) -> str:
        """生成报告。

        Args:
            data: ThresholdDetector 的输出 dict

        Returns:
            str: 格式化的报告文本
        """
        params: RadarParams = data["params"]
        lines = [
            "=" * 50,
            "雷达探测距离分析报告",
            "=" * 50,
            f"雷达参数:",
            f"  峰值功率    = {params.pt / 1e6:.1f} MW",
            f"  天线增益    = {params.gain_db:.1f} dB",
            f"  载波频率    = {params.freq_hz / 1e9:.1f} GHz",
            f"  信号带宽    = {params.bandwidth_hz / 1e6:.1f} MHz",
            f"  目标 RCS    = {params.target_rcs_m2:.1f} m^2",
            "",
            f"检测门限    = {data['threshold_db']:.1f} dB",
        ]

        if data["detected"]:
            r_km = data["detection_range_m"] / 1e3
            lines.append(f"最大探测距离 = {r_km:.1f} km")
        else:
            lines.append("最大探测距离 = 未检测到（全程低于门限）")

        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# Pipeline 执行流程图（可选）
# ============================================================


def plot_pipeline_flow(pipeline_result: PipelineResult, output_path: str) -> None:
    """绘制 Pipeline 执行流程图。

    每个 Stage 用一个方框表示，方框内标注名称和耗时。
    Stage 之间用箭头连接，表示数据流向。
    """
    num_stages = len(pipeline_result.stages)
    if num_stages == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, num_stages * 2.5), 4))

    box_width = 1.8
    box_height = 1.2
    gap = 0.6
    total_width = num_stages * box_width + (num_stages - 1) * gap
    start_x = -total_width / 2 + box_width / 2
    center_y = 0.5

    for i, stage_result in enumerate(pipeline_result.stages):
        x = start_x + i * (box_width + gap)

        # 绘制方框
        rect = plt.Rectangle(
            (x - box_width / 2, center_y - box_height / 2),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="#2196F3",
            facecolor="#E3F2FD",
            zorder=2,
        )
        ax.add_patch(rect)

        # Stage 名称和耗时
        ax.text(
            x, center_y + 0.15,
            stage_result.name,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            zorder=3,
        )
        ax.text(
            x, center_y - 0.25,
            f"{stage_result.elapsed_s * 1000:.2f} ms",
            ha="center", va="center",
            fontsize=8, color="#666",
            zorder=3,
        )

        # 箭头（指向下一个 Stage）
        if i < num_stages - 1:
            ax.annotate(
                "",
                xy=(x + box_width / 2 + 0.05, center_y),
                xytext=(x + box_width / 2 + gap - 0.05, center_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#444",
                    lw=1.5,
                ),
                zorder=3,
            )

    ax.set_xlim(-total_width / 2 - 0.5, total_width / 2 + 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pipeline 执行流程", fontsize=14, pad=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 验证
# ============================================================


def validate(params: RadarParams) -> bool:
    """验证 Pipeline 框架的正确性。

    验证策略：
      1. 单 Stage 正确性：SNRCalculator(50km) = 手动雷达方程计算
      2. Pipeline 串联：run() 最终输出 = 手动逐步执行
      3. 执行时间记录：每个 Stage 的 elapsed_s > 0
      4. 并行正确性：并行结果 = 串行结果
    """
    results = []

    # --- 验证 1：单 Stage 正确性 ---
    # 手动用雷达方程计算 50 km 处的 SNR
    test_range_m = 50e3  # 50 km
    numerator = (
        params.pt
        * params.gain_linear ** 2
        * params.wavelength_m ** 2
        * params.target_rcs_m2
        * params.pulse_width_s
    )
    denominator = (
        (4 * np.pi) ** 3
        * test_range_m ** 4
        * BOLTZMANN * STANDARD_TEMP
        * params.noise_figure_linear
        * params.bandwidth_hz
    )
    snr_manual = numerator / denominator

    calc = SNRCalculator(params)
    snr_stage = calc.process(test_range_m)["snr_linear"]

    results.append(verify(
        name="单Stage SNRCalculator(50km)",
        theoretical=snr_manual,
        simulated=snr_stage,
        tolerance=snr_manual * 0.001,
        unit="",
    ))

    # --- 验证 2：Pipeline 串联一致性 ---
    pipeline = (
        RangeSweeper(range_start_m=10e3, range_end_m=100e3, num_points=50)
        | ThresholdDetector(threshold_db=13.0)
        | ReportGenerator()
    )
    pipeline_result = pipeline.run(params)

    # 手动逐步执行：先 sweep，再 detect，再 report
    sweep_stage = RangeSweeper(range_start_m=10e3, range_end_m=100e3, num_points=50)
    detect_stage = ThresholdDetector(threshold_db=13.0)
    report_stage = ReportGenerator()

    sweep_out = sweep_stage.process(params)
    detect_out = detect_stage.process(sweep_out)
    report_manual = report_stage.process(detect_out)

    report_pipeline = pipeline_result.final_output
    # 两份报告文本应完全一致
    results.append(verify(
        name="Pipeline串联=手动逐步",
        theoretical=1.0,  # 用 1.0 表示"一致"
        simulated=1.0 if report_pipeline == report_manual else 0.0,
        tolerance=0.01,
        unit="bool",
    ))

    # --- 验证 3：执行时间记录 ---
    all_timed = all(s.elapsed_s > 0 for s in pipeline_result.stages)
    results.append(verify(
        name="所有Stage记录耗时>0",
        theoretical=1.0,
        simulated=1.0 if all_timed else 0.0,
        tolerance=0.01,
        unit="bool",
    ))

    # --- 验证 4：并行正确性 ---
    sweeper_serial = RangeSweeper(
        range_start_m=10e3, range_end_m=100e3, num_points=50, parallel=False,
    )
    sweeper_parallel = RangeSweeper(
        range_start_m=10e3, range_end_m=100e3, num_points=50, parallel=True,
    )

    out_serial = sweeper_serial.process(params)
    out_parallel = sweeper_parallel.process(params)

    max_diff = np.max(np.abs(out_serial["snr_linear"] - out_parallel["snr_linear"]))
    # 并行和串行应完全一致（确定性计算），允许浮点舍入误差
    results.append(verify(
        name="并行结果=串行结果",
        theoretical=0.0,
        simulated=max_diff,
        tolerance=1e-10,
        unit="abs_diff",
    ))

    return print_validation("s08 Pipeline 框架", results)


def main() -> int:
    """运行 s08 Pipeline 框架验证。"""
    print("=" * 60)
    print("s08：生产者-消费者模式的雷达信号处理流水线框架")
    print("=" * 60)

    # 使用与 s01 相同的 S 波段搜索雷达参数
    params = RadarParams(
        pt=2e6,
        gain_db=40.0,
        freq_hz=3e9,
        bandwidth_hz=5e6,
        pulse_width_s=200e-6,
        prf_hz=500,
        noise_figure_db=3.0,
        target_range_m=30e3,
        target_rcs_m2=5.0,
    )

    # 构建 Pipeline：RangeSweeper -> ThresholdDetector -> ReportGenerator
    # SNRCalculator 被 RangeSweeper 内部调用，不直接参与主管线
    pipeline = (
        RangeSweeper(range_start_m=5e3, range_end_m=200e3, num_points=300)
        | ThresholdDetector(threshold_db=13.0)
        | ReportGenerator()
    )

    # 执行流水线
    print("\n执行 Pipeline...")
    result = pipeline.run(params)

    # 打印各 Stage 耗时
    print("\n各 Stage 执行耗时:")
    for sr in result.stages:
        print(f"  [{sr.name}] {sr.elapsed_s * 1000:.3f} ms")

    # 打印报告
    print("\n" + result.final_output)

    # 绘制执行流程图
    output_path = os.path.join(os.path.dirname(__file__), "output", "s08_pipeline.png")
    plot_pipeline_flow(result, output_path)
    print(f"\n流程图已保存: {output_path}")

    # 额外演示：并行 vs 串行性能对比
    print("\n并行 vs 串行性能对比（500 个距离点）:")
    sweeper_s = RangeSweeper(
        range_start_m=5e3, range_end_m=200e3, num_points=500, parallel=False,
    )
    sweeper_p = RangeSweeper(
        range_start_m=5e3, range_end_m=200e3, num_points=500, parallel=True,
    )

    t0 = time.perf_counter()
    out_s = sweeper_s.process(params)
    serial_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_p = sweeper_p.process(params)
    parallel_time = time.perf_counter() - t0

    print(f"  串行: {serial_time * 1000:.1f} ms")
    print(f"  并行: {parallel_time * 1000:.1f} ms")
    print(f"  最大差异: {np.max(np.abs(out_s['snr_linear'] - out_p['snr_linear'])):.2e}")

    # 运行验证
    print("\n运行验证...")
    all_passed = validate(params)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
