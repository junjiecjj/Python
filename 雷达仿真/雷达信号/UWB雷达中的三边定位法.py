#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:59:08 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


def trilaterate(p1, p2, p3, d1, d2, d3):
    """
    TRILATERATE 三边定位算法求解目标坐标

    输入参数:
        p1, p2, p3 - 三个参考点坐标，格式为[x, y]
        d1, d2, d3 - 目标点到三个参考点的距离

    输出参数:
        x, y - 目标点坐标
    """
    # 提取参考点坐标
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 构建线性方程组 A*[x; y] = B
    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x2), 2 * (y3 - y2)]
    ])

    B = np.array([
        d1**2 - d2**2 + x2**2 + y2**2 - x1**2 - y1**2,
        d2**2 - d3**2 + x3**2 + y3**2 - x2**2 - y2**2
    ])

    # 检查矩阵是否奇异
    if matrix_rank(A) < 2:
        raise ValueError('无法求解，参考点配置可能存在问题（可能共线）')

    # 求解线性方程组（使用最小二乘法处理噪声）
    xy = np.linalg.solve(A, B)
    x, y = xy[0], xy[1]

    return x, y

# 示例用法
if __name__ == "__main__":
    # 参考点坐标
    p1 = [0, 0]
    p2 = [10, 0]
    p3 = [5, 10]

    # 目标点实际坐标(5,5)
    true_x = 5
    true_y = 5

    # 计算理论距离并添加少量噪声
    d1 = np.sqrt((true_x - p1[0])**2 + (true_y - p1[1])**2) + 0.01
    d2 = np.sqrt((true_x - p2[0])**2 + (true_y - p2[1])**2) - 0.02
    d3 = np.sqrt((true_x - p3[0])**2 + (true_y - p3[1])**2) + 0.03

    # 求解目标坐标
    try:
        x, y = trilaterate(p1, p2, p3, d1, d2, d3)

        # 显示结果
        print(f'计算坐标: ({x:.2f}, {y:.2f})')
        print(f'实际坐标: ({true_x:.2f}, {true_y:.2f})')
        print(f'定位误差: {np.sqrt((x - true_x)**2 + (y - true_y)**2):.4f}')

        # 可视化
        plt.figure(figsize=(10, 8))
        # plt.hold(True)
        plt.grid(True)
        plt.axis('equal')

        # 绘制参考点
        plt.plot(p1[0], p1[1], 'ro', markersize=8, label='参考点1')
        plt.plot(p2[0], p2[1], 'go', markersize=8, label='参考点2')
        plt.plot(p3[0], p3[1], 'bo', markersize=8, label='参考点3')

        # 绘制实际位置和计算位置
        plt.plot(true_x, true_y, 'ks', markersize=8, label='实际位置')
        plt.plot(x, y, 'mp', markersize=8, label='计算位置')

        # 绘制参考圆（虚线）
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(p1[0] + d1 * np.cos(theta), p1[1] + d1 * np.sin(theta), 'r--')
        plt.plot(p2[0] + d2 * np.cos(theta), p2[1] + d2 * np.sin(theta), 'g--')
        plt.plot(p3[0] + d3 * np.cos(theta), p3[1] + d3 * np.sin(theta), 'b--')

        plt.legend(loc='best')
        plt.title('三边定位结果可视化')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.show()

    except ValueError as e:
        print(f'错误: {e}')
