#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 18:01:44 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import seaborn as sns

# 设置样式
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
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


class CDMASystem:
    def __init__(self, num_users=4, spreading_factor=16):
        self.K = num_users
        self.N = spreading_factor
        self.codes = self.generate_walsh_codes()
        print(f"系统初始化: {self.K}用户, {self.N}扩频因子")

    def generate_walsh_codes(self):
        """生成Walsh正交码"""
        H = np.array([[1]])
        n = int(np.log2(self.N))
        for _ in range(n):
            H = np.block([[H, H], [H, -H]])
        return H # [:self.K, :self.N]

    def correct_theoretical_ser(self, EbN0_dB):
        """正确的理论SER计算"""
        EbN0_linear = 10**(EbN0_dB / 10)

        # 正确的多址干扰模型
        # 每个干扰用户的方差为 (K-1)/N
        total_interference = (self.K - 1) / self.N
        effective_noise = 1.0 / EbN0_linear + total_interference

        return 0.5 * erfc(np.sqrt(1.0 / (2 * effective_noise)))

    def simulate_cdma_correctly(self, EbN0_dB, num_bits=100000):
        """正确的CDMA仿真"""
        # 生成BPSK数据 (±1)
        data = 2 * np.random.randint(0, 2, (self.K, num_bits)) - 1

        EbN0_linear = 10**(EbN0_dB / 10)
        noise_variance = 1.0 / (2 * EbN0_linear)  # 噪声方差

        errors = 0

        for bit_idx in range(num_bits):
            # 发射端：每个用户独立扩频（功率归一化）
            transmitted_signals = np.zeros((self.K, self.N))
            for user in range(self.K):
                # 每个用户的信号功率为1
                transmitted_signals[user] = data[user, bit_idx] * self.codes[user] / np.sqrt(self.N)

            # 合并所有用户信号
            combined_signal = np.sum(transmitted_signals, axis=0)

            # 添加高斯白噪声
            noise = np.sqrt(noise_variance) * np.random.randn(self.N)
            received_signal = combined_signal + noise

            # 接收端：每个用户独立解调
            for user in range(self.K):
                # 相关解扩
                correlation = np.sum(received_signal * self.codes[user]) / np.sqrt(self.N)

                # 判决
                detected_bit = 1 if correlation > 0 else -1

                # 统计错误
                if detected_bit != data[user, bit_idx]:
                    errors += 1

        ser = errors / (self.K * num_bits)
        return ser

    def debug_single_transmission(self, EbN0_dB=10):
        """调试单次传输过程"""
        print("\n调试单次传输:")
        print("=" * 50)

        # 生成测试数据
        test_data = 2 * np.random.randint(0, 2, self.K) - 1
        print(f"发送数据: {test_data}")

        EbN0_linear = 10**(EbN0_dB / 10)
        noise_variance = 1.0 / (2 * EbN0_linear)

        # 发射过程
        transmitted = np.zeros((self.K, self.N))
        for user in range(self.K):
            transmitted[user] = test_data[user] * self.codes[user] / np.sqrt(self.N)

        combined = np.sum(transmitted, axis=0)
        print(f"合并信号功率: {np.mean(combined**2):.4f}")

        # 添加噪声
        noise = np.sqrt(noise_variance) * np.random.randn(self.N)
        received = combined + noise

        # 接收过程
        decoded = np.zeros(self.K)
        for user in range(self.K):
            correlation = np.sum(received * self.codes[user]) / np.sqrt(self.N)
            decoded[user] = 1 if correlation > 0 else -1

        print(f"解码数据: {decoded.astype(int)}")
        print(f"错误数: {np.sum(decoded != test_data)}")

        return np.sum(decoded != test_data) > 0

def run_comprehensive_test():
    """运行全面的测试"""
    print("CDMA系统正确性测试")
    print("=" * 60)

    # 测试参数
    K = 4
    N = 16
    EbN0_dB_range = np.array([0, 2, 4, 6, 8, 10, 12])
    num_bits = 200000

    cdma = CDMASystem(K, N)

    practical_sers = []
    theoretical_sers = []

    print(f"\nEb/N0(dB) | 仿真SER   | 理论SER   | 相对误差(%)")
    print("-" * 45)

    for EbN0_dB in EbN0_dB_range:
        # 理论值
        theory_ser = cdma.correct_theoretical_ser(EbN0_dB)

        # 仿真值
        practical_ser = cdma.simulate_cdma_correctly(EbN0_dB, num_bits)

        # 避免除零错误
        if theory_ser > 0:
            rel_error = abs(practical_ser - theory_ser) / theory_ser * 100
        else:
            rel_error = 0

        practical_sers.append(practical_ser)
        theoretical_sers.append(theory_ser)

        print(f"{EbN0_dB:8.1f} | {practical_ser:.6f} | {theory_ser:.6f} | {rel_error:8.2f}%")

    # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.semilogy(EbN0_dB_range, practical_sers, 'bo-', linewidth=3,
                markersize=8, label='仿真SER', alpha=0.8)
    plt.semilogy(EbN0_dB_range, theoretical_sers, 'r--', linewidth=3,
                label='理论SER', alpha=0.8)

    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('符号错误率 (SER)', fontsize=12)
    plt.title(f'CDMA系统性能 (K={K}, N={N})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加理论公式
    plt.text(0.02, 0.98, r'$P_e = Q\left(\sqrt{\frac{1}{2(\frac{1}{E_b/N_0} + \frac{K-1}{N})}}\right)$', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 统计结果
    valid_errors = []
    for i in range(len(EbN0_dB_range)):
        if theoretical_sers[i] > 0:
            rel_error = abs(practical_sers[i] - theoretical_sers[i]) / theoretical_sers[i] * 100
            valid_errors.append(rel_error)

    if valid_errors:
        avg_error = np.mean(valid_errors)
        max_error = np.max(valid_errors)
        print(f"\n统计结果:")
        print(f"平均相对误差: {avg_error:.2f}%")
        print(f"最大相对误差: {max_error:.2f}%")

    # 调试单次传输
    print(f"\n单次传输调试 (Eb/N0=10dB):")
    has_errors = cdma.debug_single_transmission(10)
    print(f"本次传输是否有错误: {has_errors}")

def verify_orthogonality():
    """验证Walsh码的正交性"""
    print("\n验证Walsh码正交性:")
    print("=" * 40)

    K = 4
    N = 16
    cdma = CDMASystem(K, N)

    for i in range(K):
        for j in range(K):
            dot_product = np.dot(cdma.codes[i], cdma.codes[j])
            if i == j:
                expected = N
            else:
                expected = 0
            print(f"码{i+1}·码{j+1} = {dot_product:3d} (期望: {expected:3d})")

if __name__ == "__main__":
    # 验证正交性
    verify_orthogonality()

    # 运行主测试
    run_comprehensive_test()
