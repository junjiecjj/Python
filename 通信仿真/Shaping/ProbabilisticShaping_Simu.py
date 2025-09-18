import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from tqdm import tqdm

class ProbabilisticShaping:
    def __init__(self, M):
        """
        概率整形初始化
        M: 星座大小 (16, 64, 256)
        """
        self.M = M
        self.m = int(np.log2(M))
        self.constellation, self.labels = self._create_constellation()
        self.probabilities = self._calculate_mb_distribution()

    def _create_constellation(self):
        """创建QAM星座图和标签"""
        side_len = int(np.sqrt(self.M))

        # 创建星座点
        real_part = np.arange(-side_len+1, side_len, 2)
        imag_part = np.arange(-side_len+1, side_len, 2)

        constellation = []
        labels = []

        for i, r in enumerate(real_part):
            for j, im in enumerate(imag_part):
                constellation.append(complex(r, im))
                bin_i = format(i, f'0{int(np.log2(side_len))}b')
                bin_j = format(j, f'0{int(np.log2(side_len))}b')
                labels.append(bin_i + bin_j)

        return np.array(constellation), labels

    def _calculate_mb_distribution(self):
        """计算Maxwell-Boltzmann概率分布"""
        energies = np.abs(self.constellation)**2

        # 使用二分法求解lambda参数，目标平均功率为1
        lambda_low, lambda_high = 0.001, 10.0
        target_power = 1.0

        for _ in range(50):
            lambda_mid = (lambda_low + lambda_high) / 2
            Z = np.sum(np.exp(-lambda_mid * energies))
            mean_power = np.sum(energies * np.exp(-lambda_mid * energies)) / Z

            if mean_power > target_power:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid

        lambda_opt = (lambda_low + lambda_high) / 2
        probabilities = np.exp(-lambda_opt * energies)
        probabilities = probabilities / np.sum(probabilities)

        return probabilities

    def generate_shaped_symbols(self, num_symbols):
        """生成概率整形符号"""
        indices = np.random.choice(len(self.constellation), size=num_symbols, p=self.probabilities)
        symbols = self.constellation[indices]

        # 计算实际平均功率并归一化到单位功率
        actual_power = np.mean(np.abs(symbols)**2)
        symbols = symbols / np.sqrt(actual_power)

        return symbols, indices

    def generate_uniform_symbols(self, num_symbols):
        """生成均匀分布符号"""
        indices = np.random.randint(0, self.M, num_symbols)
        symbols = self.constellation[indices]

        # 归一化到单位功率
        actual_power = np.mean(np.abs(symbols)**2)
        symbols = symbols / np.sqrt(actual_power)

        return symbols, indices

    def add_awgn_noise(self, symbols, snr_linear):
        """添加AWGN噪声"""
        # 符号功率已经是1，噪声功率为1/SNR
        noise_power = 1.0 / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        return symbols + noise

    def demodulate(self, received_symbols):
        """最大似然解调"""
        decisions = []
        for symbol in received_symbols:
            distances = np.abs(self.constellation - symbol)
            decisions.append(np.argmin(distances))
        return np.array(decisions)

def monte_carlo_simulation(M, snr_dB_range, num_symbols=100000):
    """蒙特卡洛仿真"""
    ser_shaped = []
    ser_uniform = []
    ber_shaped = []
    ber_uniform = []

    ps = ProbabilisticShaping(M)

    print(f"概率整形平均功率: {np.sum(ps.probabilities * np.abs(ps.constellation)**2):.4f}")
    print(f"均匀分布平均功率: {np.mean(np.abs(ps.constellation)**2):.4f}")

    for snr_dB in tqdm(snr_dB_range, desc="Running Monte Carlo Simulation"):
        snr_linear = 10**(snr_dB/10)

        # 概率整形仿真
        shaped_symbols, shaped_indices = ps.generate_shaped_symbols(num_symbols)
        shaped_received = ps.add_awgn_noise(shaped_symbols, snr_linear)
        shaped_decisions = ps.demodulate(shaped_received)

        ser_s = np.mean(shaped_decisions != shaped_indices)
        ser_shaped.append(ser_s)

        # 均匀分布仿真
        uniform_symbols, uniform_indices = ps.generate_uniform_symbols(num_symbols)
        uniform_received = ps.add_awgn_noise(uniform_symbols, snr_linear)
        uniform_decisions = ps.demodulate(uniform_received)

        ser_u = np.mean(uniform_decisions != uniform_indices)
        ser_uniform.append(ser_u)

        # BER近似计算
        ber_shaped.append(ser_s / ps.m)
        ber_uniform.append(ser_u / ps.m)

    return ser_shaped, ser_uniform, ber_shaped, ber_uniform

def plot_performance_comparison(M, snr_dB_range, ser_shaped, ser_uniform, ber_shaped, ber_uniform):
    """绘制性能比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # SER比较
    ax1.semilogy(snr_dB_range, ser_shaped, 'bo-', linewidth=2, markersize=6, label='Probability Shaped')
    ax1.semilogy(snr_dB_range, ser_uniform, 'rs--', linewidth=2, markersize=6, label='Uniform')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Symbol Error Rate (SER)')
    ax1.set_title(f'{M}QAM SER Performance Comparison')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend()

    # BER比较
    ax2.semilogy(snr_dB_range, ber_shaped, 'bo-', linewidth=2, markersize=6, label='Probability Shaped')
    ax2.semilogy(snr_dB_range, ber_uniform, 'rs--', linewidth=2, markersize=6, label='Uniform')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Bit Error Rate (BER)')
    ax2.set_title(f'{M}QAM BER Performance Comparison')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_constellation_with_probability(M):
    """绘制带概率大小的星座图"""
    ps = ProbabilisticShaping(M)
    num_symbols = 5000

    shaped_symbols, _ = ps.generate_shaped_symbols(num_symbols)
    uniform_symbols, _ = ps.generate_uniform_symbols(num_symbols)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 概率整形星座图（点大小表示概率）
    for i, point in enumerate(ps.constellation):
        prob = ps.probabilities[i]
        # 找到这个点的所有出现
        occurrences = np.sum(np.abs(shaped_symbols - point) < 1e-3)
        size = 20 + (occurrences / num_symbols) * 1000  # 大小与概率成正比

        ax1.scatter(np.real(point), np.imag(point), s=size, alpha=0.7,
                   color=plt.cm.viridis(prob/ps.probabilities.max()),
                   edgecolors='black', linewidth=0.5)

    ax1.set_title(f'Probability Shaped {M}QAM\n(点大小表示出现概率)')
    ax1.set_xlabel('In-phase')
    ax1.set_ylabel('Quadrature')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 均匀分布星座图
    ax2.scatter(np.real(uniform_symbols), np.imag(uniform_symbols),
               s=20, alpha=0.6, color='red')
    ax2.set_title(f'Uniform {M}QAM\n(所有点等概率)')
    ax2.set_xlabel('In-phase')
    ax2.set_ylabel('Quadrature')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()

def plot_probability_vs_energy(M):
    """绘制概率与能量的关系"""
    ps = ProbabilisticShaping(M)

    energies = np.abs(ps.constellation)**2
    probabilities = ps.probabilities

    plt.figure(figsize=(10, 6))
    plt.scatter(energies, probabilities, s=100, alpha=0.7, c=energies, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.xlabel('Symbol Energy')
    plt.ylabel('Probability')
    plt.title(f'{M}QAM: Probability vs Energy (Maxwell-Boltzmann Distribution)')
    plt.grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(energies, np.log(probabilities+1e-12), 1)
    x_fit = np.linspace(min(energies), max(energies), 100)
    y_fit = np.exp(z[0] * x_fit + z[1])
    plt.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: P ∝ exp({z[0]:.3f}E)')

    plt.legend()
    plt.yscale('log')
    plt.show()

def calculate_shaping_gain(ser_shaped, ser_uniform, snr_dB_range, target_ser=1e-3):
    """计算成形增益"""
    # 线性插值找到目标SER对应的SNR
    def find_snr_at_ser(ser_values, snr_values, target):
        for i in range(len(ser_values)-1):
            if ser_values[i] >= target and ser_values[i+1] <= target:
                # 线性插值
                x1, y1 = snr_values[i], ser_values[i]
                x2, y2 = snr_values[i+1], ser_values[i+1]
                return x1 + (x2 - x1) * (target - y1) / (y2 - y1)
        return None

    snr_shaped = find_snr_at_ser(ser_shaped, snr_dB_range, target_ser)
    snr_uniform = find_snr_at_ser(ser_uniform, snr_dB_range, target_ser)

    if snr_shaped is not None and snr_uniform is not None:
        gain = snr_uniform - snr_shaped
        print(f"Shaping gain at SER={target_ser}: {gain:.3f} dB")
        return gain
    else:
        print("Target SER not reached in simulation range")
        return 0

# 主程序
if __name__ == "__main__":
    # 参数设置
    M = 16  # 调制阶数
    snr_dB_range = np.arange(8, 20, 1)  # SNR范围
    num_symbols = 100000  # 蒙特卡洛仿真符号数

    print("概率整形蒙特卡洛仿真")
    print(f"调制方式: {M}QAM")
    print(f"SNR范围: {snr_dB_range[0]} to {snr_dB_range[-1]} dB")

    # 1. 绘制概率与能量关系
    print("\n1. 绘制概率与能量关系...")
    plot_probability_vs_energy(M)

    # 2. 绘制带概率大小的星座图
    print("2. 绘制星座图对比...")
    plot_constellation_with_probability(M)

    # 3. 运行蒙特卡洛仿真
    print("3. 运行蒙特卡洛仿真...")
    ser_shaped, ser_uniform, ber_shaped, ber_uniform = monte_carlo_simulation(
        M, snr_dB_range, num_symbols)

    # 4. 绘制性能比较
    print("4. 绘制性能比较...")
    plot_performance_comparison(M, snr_dB_range, ser_shaped, ser_uniform, ber_shaped, ber_uniform)

    # 5. 计算成形增益
    print("5. 计算成形增益...")
    gain = calculate_shaping_gain(ser_shaped, ser_uniform, snr_dB_range)

    # 6. 打印详细结果
    print("\n=== 详细仿真结果 ===")
    ps = ProbabilisticShaping(M)
    shaped_power = np.sum(ps.probabilities * np.abs(ps.constellation)**2)
    uniform_power = np.mean(np.abs(ps.constellation)**2)

    print(f"概率整形平均功率: {shaped_power:.4f}")
    print(f"均匀分布平均功率: {uniform_power:.4f}")
    print(f"功率降低: {(1 - shaped_power/uniform_power)*100:.2f}%")
    print(f"实测成形增益: {gain:.3f} dB")

    # 在典型SNR点的性能
    mid_idx = len(snr_dB_range) // 2
    print(f"\n在SNR={snr_dB_range[mid_idx]} dB时:")
    print(f"概率整形 SER: {ser_shaped[mid_idx]:.4f}")
    print(f"均匀分布 SER: {ser_uniform[mid_idx]:.4f}")
    print(f"性能提升: {ser_uniform[mid_idx]/ser_shaped[mid_idx]:.2f}x")
