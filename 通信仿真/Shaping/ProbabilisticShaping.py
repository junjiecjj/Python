#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:16:13 2024

@author: jack

https://blog.csdn.net/Stephanie2014/article/details/108269084

https://blog.csdn.net/weixin_47065524/article/details/138282774
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

    def calculate_ber_theoretical(self, is_shaped=True):
        """计算理论BER"""
        # 简化理论BER计算
        if is_shaped:
            # 概率成形的BER（考虑成形增益）
            shaping_gain = 0.5  # dB的成形增益
            effective_snr = self.snr_linear * 10**(shaping_gain/10)
        else:
            # 均匀分布的BER
            effective_snr = self.snr_linear

        # 不同调制阶数的理论BER近似
        if self.M == 16:
            ber = (3/8) * special.erfc(np.sqrt(effective_snr/5))
        elif self.M == 64:
            ber = (7/24) * special.erfc(np.sqrt(effective_snr/21))
        elif self.M == 256:
            ber = (15/64) * special.erfc(np.sqrt(effective_snr/85))
        else:
            ber = 0.5 * special.erfc(np.sqrt(effective_snr))

        return ber

def compare_modulation_performance():
    """比较不同调制方式的性能"""
    M_values = [16, 64, 256]
    snr_range = np.arange(5, 21, 1)  # 5-20 dB
    colors = list(mcolors.TABLEAU_COLORS.values())

    plt.figure(figsize=(12, 8))

    for i, M in enumerate(M_values):
        ber_shaped = []
        ber_uniform = []

        for snr_dB in snr_range:
            # 概率成形性能
            ps = ProbabilisticShaping(M, snr_dB)
            ber_s = ps.calculate_ber_theoretical(is_shaped=True)
            ber_shaped.append(ber_s)

            # 均匀分布性能
            ber_u = ps.calculate_ber_theoretical(is_shaped=False)
            ber_uniform.append(ber_u)

        # 绘制概率成形曲线
        plt.semilogy(snr_range, ber_shaped,
                    color=colors[i*2],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    label=f'PS {M}QAM')

        # 绘制均匀分布曲线
        plt.semilogy(snr_range, ber_uniform,
                    color=colors[i*2+1],
                    marker='s',
                    linestyle='--',
                    linewidth=2,
                    markersize=5,
                    label=f'Uniform {M}QAM')

        # 添加标注显示成形增益
        if M == 16:
            gain_at_ber = find_gain_at_ber(ber_shaped, ber_uniform, snr_range, target_ber=1e-4)
            print(f"16QAM 成形增益在BER=1e-4时: {gain_at_ber:.2f} dB")

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('概率成形 vs 均匀分布性能比较\n(16QAM, 64QAM, 256QAM)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(1e-6, 1)
    plt.xlim(5, 20)
    plt.tight_layout()
    plt.show()

def find_gain_at_ber(ber_shaped, ber_uniform, snr_range, target_ber=1e-4):
    """计算在特定BER下的成形增益"""
    # 找到均匀分布在目标BER处的SNR
    snr_uniform = None
    for i in range(len(ber_uniform)-1):
        if ber_uniform[i] >= target_ber and ber_uniform[i+1] <= target_ber:
            snr_uniform = snr_range[i] + (snr_range[i+1] - snr_range[i]) * \
                         (target_ber - ber_uniform[i]) / (ber_uniform[i+1] - ber_uniform[i])
            break

    # 找到概率成形在目标BER处的SNR
    snr_shaped = None
    for i in range(len(ber_shaped)-1):
        if ber_shaped[i] >= target_ber and ber_shaped[i+1] <= target_ber:
            snr_shaped = snr_range[i] + (snr_range[i+1] - snr_range[i]) * \
                        (target_ber - ber_shaped[i]) / (ber_shaped[i+1] - ber_shaped[i])
            break

    if snr_uniform is not None and snr_shaped is not None:
        return snr_uniform - snr_shaped
    return 0

def plot_constellation_comparison():
    """绘制不同调制方式的星座图对比"""
    M_values = [16, 64, 256]
    num_symbols = 1000

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, M in enumerate(M_values):
        ps = ProbabilisticShaping(M, 10)

        # 生成概率成形符号
        indices = np.random.choice(len(ps.constellation), size=num_symbols, p=ps.probabilities)
        shaped_symbols = ps.constellation[indices]

        # 生成均匀分布符号
        uniform_indices = np.random.randint(0, M, num_symbols)
        uniform_symbols = ps.constellation[uniform_indices]

        # 绘制概率成形星座图
        axes[0, i].scatter(np.real(shaped_symbols), np.imag(shaped_symbols),
                          alpha=0.6, s=10, color='blue')
        axes[0, i].set_title(f'Probability Shaped {M}QAM', fontsize=12)
        axes[0, i].set_xlabel('In-phase')
        axes[0, i].set_ylabel('Quadrature')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')

        # 绘制均匀分布星座图
        axes[1, i].scatter(np.real(uniform_symbols), np.imag(uniform_symbols),
                          alpha=0.6, s=10, color='red')
        axes[1, i].set_title(f'Uniform {M}QAM', fontsize=12)
        axes[1, i].set_xlabel('In-phase')
        axes[1, i].set_ylabel('Quadrature')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axis('equal')

    plt.tight_layout()
    plt.show()

def plot_probability_distribution():
    """绘制概率分布"""
    M_values = [16, 64, 256]

    plt.figure(figsize=(12, 8))

    for i, M in enumerate(M_values):
        ps = ProbabilisticShaping(M, 10)

        # 对概率进行排序以便更好显示
        sorted_probs = np.sort(ps.probabilities)[::-1]
        symbols = np.arange(len(sorted_probs))

        plt.subplot(3, 1, i+1)
        plt.bar(symbols[:20], sorted_probs[:20], alpha=0.7, color='green')
        plt.title(f'{M}QAM 符号概率分布 (前20个最高概率)', fontsize=11)
        plt.xlabel('符号索引')
        plt.ylabel('概率')
        plt.grid(True, alpha=0.3)

        # 显示熵值
        entropy = -np.sum(ps.probabilities * np.log2(ps.probabilities + 1e-12))
        plt.text(0.7, 0.8, f'熵: {entropy:.3f} bits/symbol',
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.show()

# 运行完整的演示
if __name__ == "__main__":
    print("开始概率成形性能比较...")

    # 1. 性能比较曲线
    print("生成性能比较曲线...")
    compare_modulation_performance()

    # 2. 星座图对比
    print("生成星座图对比...")
    plot_constellation_comparison()

    # 3. 概率分布
    print("生成概率分布图...")
    plot_probability_distribution()

    print("演示完成！")
