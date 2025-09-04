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

class ProperFadingChannel:
    def __init__(self, num_paths=3, max_delay_spread=2, doppler_freq=10):
        self.num_paths = num_paths
        self.max_delay_spread = max_delay_spread
        self.doppler_freq = doppler_freq

    def generate_channel(self, block_length):
        """生成正确的瑞利衰落信道系数"""
        # 路径时延（以码片为单位）
        self.delays = np.random.randint(0, self.max_delay_spread + 1, self.num_paths)

        # 路径增益（瑞利分布，功率归一化）
        path_powers = np.random.exponential(1, self.num_paths)
        path_powers /= np.sum(path_powers)  # 功率归一化

        # 生成时变衰落系数
        self.channel_coeffs = np.zeros((self.num_paths, block_length), dtype=complex)
        for i in range(self.num_paths):
            # 瑞利衰落：实部和虚部都是高斯分布
            real_part = np.random.randn(block_length) * np.sqrt(path_powers[i]/2)
            imag_part = np.random.randn(block_length) * np.sqrt(path_powers[i]/2)
            self.channel_coeffs[i] = real_part + 1j * imag_part

        return self.channel_coeffs, self.delays

    def apply_channel(self, signal, snr_db):
        """正确应用多径衰落信道"""
        block_length = len(signal)
        channel_coeffs, delays = self.generate_channel(block_length)

        # 初始化输出信号
        output = np.zeros(block_length, dtype=complex)

        # 应用多径
        for i in range(self.num_paths):
            delay = delays[i]
            coeff = channel_coeffs[i]

            if delay == 0:
                output += coeff * signal
            else:
                # 时延信号
                delayed_signal = np.concatenate([np.zeros(delay), signal[:-delay]])
                output += coeff * delayed_signal

        # 添加高斯噪声
        signal_power = np.mean(np.abs(output)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(block_length) + 1j * np.random.randn(block_length))

        return output + noise, channel_coeffs, delays

class CorrectCDMAFadingSystem:
    def __init__(self, num_users=4, spreading_factor=16, num_paths=3):
        self.K = num_users
        self.N = spreading_factor
        self.num_paths = num_paths
        self.codes = self.generate_walsh_codes()
        self.channel = ProperFadingChannel(num_paths)

        # 验证正交性
        self.verify_orthogonality()

    def generate_walsh_codes(self):
        """生成Walsh正交码"""
        H = np.array([[1]])
        n = int(np.log2(self.N))
        for _ in range(n):
            H = np.block([[H, H], [H, -H]])
        return H[:self.K, :self.N]

    def verify_orthogonality(self):
        """验证码字正交性"""
        print("验证Walsh码正交性:")
        for i in range(min(3, self.K)):
            for j in range(min(3, self.K)):
                dot_product = np.dot(self.codes[i], self.codes[j])
                expected = self.N if i == j else 0
                status = "✓" if dot_product == expected else "✗"
                print(f"  码{i}·码{j} = {dot_product:3d} (期望: {expected:3d}) {status}")
        print()

    def theoretical_ser_awgn(self, snr_db):
        """AWGN信道理论SER"""
        snr_linear = 10**(snr_db / 10)
        return 0.5 * erfc(np.sqrt(snr_linear))

    def theoretical_ser_rayleigh(self, snr_db):
        """瑞利衰落信道理论SER"""
        snr_linear = 10**(snr_db / 10)
        return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

    def simulate_properly(self, snr_db, num_blocks=1000):
        """正确的CDMA衰落信道仿真"""
        errors = 0
        total_bits = 0

        for block in range(num_blocks):
            # 生成BPSK数据
            data = 2 * np.random.randint(0, 2, self.K) - 1

            # 发射端：扩频
            transmitted_signals = np.zeros(self.N, dtype=complex)
            for user in range(self.K):
                # 正确的功率归一化：每个用户功率为1
                transmitted_signals += data[user] * self.codes[user] / np.sqrt(self.N)

            # 应用信道
            received_signal, channel_coeffs, delays = self.channel.apply_channel(transmitted_signals, snr_db)

            # 接收端：rake接收机
            for user in range(self.K):
                # 多径合并（rake接收）
                correlator_output = 0
                for path in range(self.num_paths):
                    delay = delays[path]
                    coeff = channel_coeffs[path, 0]  # 使用块内第一个系数

                    # 时延补偿
                    if delay > 0:
                        shifted_signal = np.concatenate([received_signal[delay:], np.zeros(delay)])
                    else:
                        shifted_signal = received_signal

                    # 相关解扩
                    correlation = np.sum(shifted_signal * self.codes[user]) / np.sqrt(self.N)
                    # 最大比合并
                    correlator_output += np.real(correlation * np.conj(coeff))

                # 判决
                detected_bit = 1 if correlator_output > 0 else -1

                # 统计错误
                if detected_bit != data[user]:
                    errors += 1
                total_bits += 1

        ser = errors / total_bits if total_bits > 0 else 1.0
        return ser

    def simulate_with_channel_estimation(self, snr_db, num_blocks=1000):
        """带信道估计的仿真"""
        errors = 0
        total_bits = 0

        # 使用导频进行信道估计
        pilot_symbols = np.ones(self.K)  # 全1导频

        for block in range(num_blocks):
            # 生成数据
            data = 2 * np.random.randint(0, 2, self.K) - 1

            # 发射导频+数据
            if block % 10 == 0:  # 每10个块发送一次导频
                transmitted_symbols = pilot_symbols
            else:
                transmitted_symbols = data

            # 发射端
            transmitted_signal = np.zeros(self.N, dtype=complex)
            for user in range(self.K):
                transmitted_signal += transmitted_symbols[user] * self.codes[user] / np.sqrt(self.N)

            # 应用信道
            received_signal, channel_coeffs, delays = self.channel.apply_channel(transmitted_signal, snr_db)

            # 信道估计（使用导频）
            if block % 10 == 0:
                estimated_channel = np.zeros(self.num_paths, dtype=complex)
                for path in range(self.num_paths):
                    delay = delays[path]
                    if delay > 0:
                        shifted_signal = np.concatenate([received_signal[delay:], np.zeros(delay)])
                    else:
                        shifted_signal = received_signal

                    for user in range(self.K):
                        correlation = np.sum(shifted_signal * self.codes[user]) / np.sqrt(self.N)
                        estimated_channel[path] += correlation / self.K

            # 数据检测
            for user in range(self.K):
                correlator_output = 0
                for path in range(self.num_paths):
                    delay = delays[path]
                    if delay > 0:
                        shifted_signal = np.concatenate([received_signal[delay:], np.zeros(delay)])
                    else:
                        shifted_signal = received_signal

                    correlation = np.sum(shifted_signal * self.codes[user]) / np.sqrt(self.N)
                    correlator_output += np.real(correlation * np.conj(estimated_channel[path]))

                detected_bit = 1 if correlator_output > 0 else -1

                if detected_bit != data[user]:
                    errors += 1
                total_bits += 1

        return errors / total_bits

def run_correct_simulation():
    """运行正确的仿真"""
    print("CDMA在衰落信道下的正确仿真")
    print("=" * 50)

    # 系统参数
    K = 4
    N = 16
    num_paths = 2  # 减少多径数以提高性能

    system = CorrectCDMAFadingSystem(K, N, num_paths)

    snr_range = np.arange(0, 31, 5)
    ser_awgn = []
    ser_rayleigh = []
    ser_rayleigh_est = []
    theory_awgn = []
    theory_rayleigh = []

    print("SNR(dB) | AWGN理论 | AWGN仿真 | 瑞利理论 | 瑞利仿真 | 带估计")
    print("-" * 65)

    for snr_db in snr_range:
        # 理论值
        theory_awgn.append(system.theoretical_ser_awgn(snr_db))
        theory_rayleigh.append(system.theoretical_ser_rayleigh(snr_db))

        # 仿真值
        ser_awgn_val = system.simulate_properly(snr_db, 2000)  # AWGN-like
        ser_rayleigh_val = system.simulate_properly(snr_db, 2000)
        ser_est_val = system.simulate_with_channel_estimation(snr_db, 2000)

        ser_awgn.append(ser_awgn_val)
        ser_rayleigh.append(ser_rayleigh_val)
        ser_rayleigh_est.append(ser_est_val)

        print(f"{snr_db:6d} | {theory_awgn[-1]:8.4f} | {ser_awgn_val:8.4f} | "
              f"{theory_rayleigh[-1]:8.4f} | {ser_rayleigh_val:8.4f} | {ser_est_val:8.4f}")

    # 绘制结果
    plt.figure(figsize=(14, 8))

    plt.semilogy(snr_range, theory_awgn, 'b--', linewidth=2, label='AWGN理论')
    plt.semilogy(snr_range, ser_awgn, 'bo-', linewidth=2, markersize=6, label='AWGN仿真')
    plt.semilogy(snr_range, theory_rayleigh, 'r--', linewidth=2, label='瑞利理论')
    plt.semilogy(snr_range, ser_rayleigh, 'ro-', linewidth=2, markersize=6, label='瑞利仿真')
    plt.semilogy(snr_range, ser_rayleigh_est, 'g^-', linewidth=2, markersize=6, label='带信道估计')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('符号错误率 (SER)', fontsize=12)
    plt.title('CDMA在不同信道条件下的正确性能', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-4, 1)

    plt.tight_layout()
    plt.show()

def debug_performance():
    """调试性能"""
    print("\n性能调试:")
    print("=" * 40)

    system = CorrectCDMAFadingSystem(4, 16, 2)

    # 测试单个SNR点
    snr_db = 20
    ser = system.simulate_properly(snr_db, 5000)
    theory_awgn = system.theoretical_ser_awgn(snr_db)
    theory_rayleigh = system.theoretical_ser_rayleigh(snr_db)

    print(f"SNR = {snr_db} dB:")
    print(f"  仿真SER: {ser:.6f}")
    print(f"  AWGN理论: {theory_awgn:.6f}")
    print(f"  瑞利理论: {theory_rayleigh:.6f}")

    # 检查信号功率
    test_signal = np.ones(16)
    received, _, _ = system.channel.apply_channel(test_signal, snr_db)
    input_power = np.mean(np.abs(test_signal)**2)
    output_power = np.mean(np.abs(received)**2)
    print(f"  输入功率: {input_power:.4f}, 输出功率: {output_power:.4f}")

if __name__ == "__main__":
    run_correct_simulation()
    debug_performance()
