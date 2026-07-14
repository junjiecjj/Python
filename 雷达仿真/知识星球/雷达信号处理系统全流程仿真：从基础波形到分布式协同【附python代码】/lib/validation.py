"""验证工具：对比仿真值与理论值。

每个仿真模块的 validate() 调用此模块的函数来自动判定 PASS/FAIL。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ValidationResult:
    """单个验证项的结果。"""
    name: str           # 验证项名称
    theoretical: float  # 理论期望值
    simulated: float    # 仿真计算值
    tolerance: float    # 允许误差
    unit: str = ""      # 单位（用于显示）
    passed: bool = True # 是否通过

    def __post_init__(self):
        self.passed = abs(self.simulated - self.theoretical) <= self.tolerance


def verify(
    name: str,
    theoretical: float,
    simulated: float,
    tolerance: float,
    unit: str = "",
) -> ValidationResult:
    """验证单个数值。

    Args:
        name:         验证项名称（如"最大探测距离"）
        theoretical:  理论计算值
        simulated:    仿真得到的值
        tolerance:    允许的最大绝对误差
        unit:         单位字符串（如"km", "dB"）

    Returns:
        ValidationResult 对象
    """
    result = ValidationResult(
        name=name,
        theoretical=theoretical,
        simulated=simulated,
        tolerance=tolerance,
        unit=unit,
    )
    return result


def verify_relative(
    name: str,
    theoretical: float,
    simulated: float,
    rel_tolerance: float = 0.01,
    unit: str = "",
) -> ValidationResult:
    """基于相对误差验证。

    Args:
        name:          验证项名称
        theoretical:   理论值
        simulated:     仿真值
        rel_tolerance: 允许的相对误差（默认 1%）
        unit:          单位
    """
    if abs(theoretical) < 1e-40:
        tolerance = rel_tolerance  # 理论值接近 0 时用绝对误差
    else:
        tolerance = abs(theoretical * rel_tolerance)

    return verify(name, theoretical, simulated, tolerance, unit)


def print_validation(module_name: str, results: list[ValidationResult]) -> bool:
    """打印验证结果汇总。

    Args:
        module_name: 模块名称（如 "s01 雷达方程"）
        results:     验证结果列表

    Returns:
        全部通过返回 True，否则 False
    """
    print(f"\n=== {module_name} 验证 ===")
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_passed = False
        error = abs(r.simulated - r.theoretical)
        print(
            f"[{status}] {r.name}: "
            f"理论 {r.theoretical:.4f} {r.unit}, "
            f"仿真 {r.simulated:.4f} {r.unit}, "
            f"误差 {error:.4f} {r.unit}"
        )

    if all_passed:
        print("=== 全部通过 ===\n")
    else:
        print("=== 存在未通过项，请检查理论假设和仿真实现 ===\n")

    return all_passed
