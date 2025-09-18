#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 17:48:23 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import matplotlib.colors as mcolors
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)


class ProbabilisticShaping:
    def __init__(self, M, snr_dB):
        """
        概率成形初始化
        M: 星座大小 (16, 64, 256)
        snr_dB: 信噪比(dB)
        """
        self.M = M
        self.snr_linear = 10**(snr_dB/10)
        self.constellation = self._create_constellation()
        self.probabilities = self._calculate_mb_distribution()

    def _create_constellation(self):
        """创建归一化QAM星座图"""
        k = int(np.log2(self.M))
        side_len = int(np.sqrt(self.M))

        # 创建星座点
        real_part = np.arange(-side_len+1, side_len, 2)
        imag_part = np.arange(-side_len+1, side_len, 2)

        constellation = []
        for r in real_part:
            for i in imag_part:
                constellation.append(complex(r, i))

        # 归一化星座点
        constellation = np.array(constellation)
        return constellation / np.sqrt(np.mean(np.abs(constellation)**2))

    def _calculate_mb_distribution(self):
        """计算Maxwell-Boltzmann概率分布"""
        energies = np.abs(self.constellation)**2

        # 使用二分法求解lambda参数
        lambda_low, lambda_high = 0.001, 10.0
        target_energy = 1.0  # 归一化目标能量

        for _ in range(50):  # 二分法迭代
            lambda_mid = (lambda_low + lambda_high) / 2
            Z = np.sum(np.exp(-lambda_mid * energies))
            mean_energy = np.sum(energies * np.exp(-lambda_mid * energies)) / Z

            if mean_energy > target_energy:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid

        lambda_opt = (lambda_low + lambda_high) / 2
        probabilities = np.exp(-lambda_opt * energies)
        return probabilities / np.sum(probabilities)

    def generate_shaped_symbols(self, num_symbols):
        """生成概率成形符号"""
        indices = np.random.choice(len(self.constellation), size=num_symbols, p=self.probabilities)
        return self.constellation[indices]

    def generate_uniform_symbols(self, num_symbols):
        """生成均匀分布符号"""
        indices = np.random.randint(0, self.M, num_symbols)
        return self.constellation[indices]

def calculate_ser_theoretical(M, snr_linear, is_shaped=True):
    """计算理论SER，考虑概率成形增益"""
    # 不同调制阶数的理论SER公式
    if M == 16:
        # 16QAM理论SER
        if is_shaped:
            # 概率成形：大约0.5-0.8 dB增益
            effective_snr = snr_linear * 10**(0.7/10)  # 0.7 dB增益
        else:
            effective_snr = snr_linear

        ser = 3 * special.erfc(np.sqrt(effective_snr/10)) - 2.25 * special.erfc(np.sqrt(effective_snr/10))**2

    elif M == 64:
        # 64QAM理论SER
        if is_shaped:
            effective_snr = snr_linear * 10**(0.6/10)  # 0.6 dB增益
        else:
            effective_snr = snr_linear

        ser = (7/2) * special.erfc(np.sqrt(effective_snr/42)) - (49/16) * special.erfc(np.sqrt(effective_snr/42))**2

    elif M == 256:
        # 256QAM理论SER
        if is_shaped:
            effective_snr = snr_linear * 10**(0.5/10)  # 0.5 dB增益
        else:
            effective_snr = snr_linear

        ser = (15/4) * special.erfc(np.sqrt(effective_snr/170)) - (225/64) * special.erfc(np.sqrt(effective_snr/170))**2

    else:
        ser = special.erfc(np.sqrt(snr_linear))

    return ser

def plot_constellation_with_probability_size():
    """绘制用点大小表示概率的星座图"""
    M_values = [16, 64, 256]
    num_samples = 1000

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, M in enumerate(M_values):
        ps = ProbabilisticShaping(M, 10)

        # 概率成形星座图 - 用点大小表示概率
        ax1 = axes[0, i]
        # 生成概率成形符号
        shaped_symbols = ps.generate_shaped_symbols(num_samples)

        # 计算每个星座点的出现次数（代表概率）
        # 由于浮点数精度问题，我们需要进行近似匹配
        shaped_counts = np.zeros(M)
        for symbol in shaped_symbols:
            # 找到最接近的星座点
            distances = np.abs(ps.constellation - symbol)
            closest_idx = np.argmin(distances)
            shaped_counts[closest_idx] += 1

        max_count = shaped_counts.max()

        # 绘制所有星座点，用大小表示概率
        for j, (symbol, count) in enumerate(zip(ps.constellation, shaped_counts)):
            if count > 0:  # 只绘制出现过的点
                point_size = 20 + (count / max_count) * 300  # 基础大小 + 概率缩放
                ax1.scatter(np.real(symbol), np.imag(symbol),
                           s=point_size, alpha=0.7, color='blue',
                           edgecolors='black', linewidth=0.5)

        ax1.set_title(f'Probability Shaped {M}QAM\n(点大小表示出现概率)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('In-phase')
        ax1.set_ylabel('Quadrature')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 均匀分布星座图 - 所有点大小相同
        ax2 = axes[1, i]
        uniform_symbols = ps.generate_uniform_symbols(num_samples)

        ax2.scatter(np.real(uniform_symbols), np.imag(uniform_symbols),
                   s=30, alpha=0.6, color='red',
                   edgecolors='black', linewidth=0.5)

        ax2.set_title(f'Uniform {M}QAM\n(所有点等概率)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('In-phase')
        ax2.set_ylabel('Quadrature')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # 添加统计信息
        entropy = -np.sum(ps.probabilities * np.log2(ps.probabilities + 1e-12))
        avg_power_shaped = np.sum(ps.probabilities * np.abs(ps.constellation)**2)
        avg_power_uniform = np.mean(np.abs(ps.constellation)**2)

        info_text = f'熵: {entropy:.3f} bits/symbol\nPS功率: {avg_power_shaped:.3f}\nUniform功率: {avg_power_uniform:.3f}'
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_ser_comparison():
    """绘制SER性能比较曲线"""
    M_values = [16, 64, 256]
    snr_range = np.arange(5, 21, 1)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    plt.figure(figsize=(12, 8))

    for i, M in enumerate(M_values):
        ser_shaped = []
        ser_uniform = []

        for snr_dB in snr_range:
            snr_linear = 10**(snr_dB/10)

            # 计算SER
            ser_s = calculate_ser_theoretical(M, snr_linear, is_shaped=True)
            ser_u = calculate_ser_theoretical(M, snr_linear, is_shaped=False)

            ser_shaped.append(ser_s)
            ser_uniform.append(ser_u)

        # 绘制概率成形SER
        plt.semilogy(snr_range, ser_shaped,
                    color=colors[i*2],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    label=f'PS {M}QAM')

        # 绘制均匀分布SER
        plt.semilogy(snr_range, ser_uniform,
                    color=colors[i*2+1],
                    marker='s',
                    linestyle='--',
                    linewidth=2,
                    markersize=5,
                    label=f'Uniform {M}QAM')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.title('概率成形 vs 均匀分布 SER 性能比较\n(概率成形提供0.5-0.7dB增益)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-6, 1)
    plt.xlim(5, 20)
    plt.tight_layout()
    plt.show()

def plot_shaping_gain():
    """绘制成形增益曲线"""
    M_values = [16, 64, 256]
    snr_range = np.arange(5, 21, 1)

    plt.figure(figsize=(10, 6))

    for i, M in enumerate(M_values):
        gain_at_ber = []
        target_ser = 1e-4  # 目标SER

        for snr_dB in snr_range:
            snr_linear = 10**(snr_dB/10)

            # 计算在目标SER下的SNR需求
            # 这里简化计算，实际需要迭代求解
            if M == 16:
                gain = 0.7  # 16QAM增益约0.7dB
            elif M == 64:
                gain = 0.6  # 64QAM增益约0.6dB
            else:
                gain = 0.5  # 256QAM增益约0.5dB

            gain_at_ber.append(gain)

        plt.plot(snr_range, gain_at_ber,
                marker='o', linestyle='-', linewidth=2,
                label=f'{M}QAM Shaping Gain')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Shaping Gain (dB)', fontsize=12)
    plt.title('概率成形增益 vs SNR', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 运行演示
if __name__ == "__main__":
    print("开始概率成形演示...")

    # 1. 用点大小表示概率的星座图
    print("生成概率大小表示的星座图...")
    plot_constellation_with_probability_size()

    # 2. SER性能比较
    print("生成SER性能比较曲线...")
    plot_ser_comparison()

    # 3. 成形增益曲线
    print("生成成形增益曲线...")
    plot_shaping_gain()

    print("演示完成！")
