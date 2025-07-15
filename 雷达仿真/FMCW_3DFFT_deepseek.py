import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 12          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 12          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12         # 设置 y 轴刻度字体大小
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
plt.rcParams['legend.fontsize'] = 12


# 雷达参数设置
c = 3e8  # 光速 (m/s)
fc = 77e9  # 载频 (Hz)
B = 500e6  # 带宽 (Hz)
Tchirp = 50e-6  # 扫频时间 (s)
S = B / Tchirp  # 调频斜率 (Hz/s)

# 天线阵列参数
nRx = 8  # 接收天线数量
d = 0.5 * (c / fc)  # 天线间距 (m)

# 目标参数 (假设单目标)
target_range = 100  # 目标距离 (m)
target_velocity = 10  # 目标径向速度 (m/s)
target_angle = 60  # 目标角度 (度)

# 采样参数
# Fs = 2.5 * B  # 采样率 (Hz)
Ns = 1024 # int(Tchirp * Fs)    # 每个chirp采样点数
Fs = Ns/Tchirp
Nchirps = 128  # 每个帧的chirp数量
pri = Tchirp  # 脉冲重复间隔 (s)

# 模拟接收信号
t = np.linspace(0, Tchirp, Ns)  # 单个chirp的采样时间

# 计算距离延迟和多普勒频移
tau = 2 * target_range / c  # 往返延迟
fd = 2 * target_velocity * fc / c  # 多普勒频移

# 生成发射信号和接收信号
tx_phase = np.pi * S * t**2
rx_signal = np.zeros((nRx, Nchirps, Ns), dtype=np.complex_)

for rx in range(nRx):
    # 计算该天线的相位差
    phase_shift = 2 * np.pi * fc * rx * d * np.sin(np.deg2rad(target_angle)) / c

    for chirp in range(Nchirps):
        # 考虑距离延迟和多普勒引起的相位变化
        t_delay = t - tau  # 距离延迟

        # 多普勒引起的相位变化 (随时间累积)
        doppler_phase = 2 * np.pi * fd * chirp * pri

        # 接收信号模型 - 关键修正：使用差频信号模型
        rx_signal[rx, chirp, :] = np.exp(1j * (2 * np.pi * S * tau * t +       # 差频项
                                             2 * np.pi * fc * tau +            # 固定相位
                                             2 * np.pi * fd * chirp * pri +    # 多普勒相位
                                             phase_shift))                     # 角度相位

# 添加噪声
noise_power = 0.1
rx_signal += np.sqrt(noise_power/2) * (np.random.randn(*rx_signal.shape) + 1j * np.random.randn(*rx_signal.shape))

# 3D FFT处理
# 1. 距离FFT (对每个chirp的采样点做FFT)
range_fft = np.fft.fft(rx_signal, axis = 2)

# 2. 多普勒FFT (对每个距离门的chirp序列做FFT)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)

# 3. 角度FFT (对每个距离-多普勒单元的天线阵列做FFT)
angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, n = 1024, axis = 0), axes = 0)

# 计算各个维度的坐标
# 距离坐标 - 关键修正：使用正确的距离计算公式
range_resolution = c / (2 * B)  # 距离分辨率
max_range = Fs * c / (2 * S)  # 最大不模糊距离
range_bins = np.arange(Ns//2) * range_resolution

# 速度坐标
velocity_resolution = c / (2 * fc * Nchirps * pri)  # 速度分辨率
max_velocity = c / (4 * fc * pri)  # 最大不模糊速度
velocity_bins = np.linspace(-max_velocity, max_velocity, Nchirps, endpoint=False)

# 角度坐标
angle_bins = np.arcsin(np.linspace(-1, 1, 1024)) * 180 / np.pi

# 寻找峰值位置
power_spectrum = np.abs(angle_fft[..., :Ns//2])**2
peak_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)

estimated_range = range_bins[peak_idx[2]]
estimated_velocity = velocity_bins[peak_idx[1]]
estimated_angle = angle_bins[peak_idx[0]]

print("=== 雷达参数 ===")
print(f"载频: {fc/1e9} GHz, 带宽: {B/1e6} MHz, 扫频时间: {Tchirp*1e6} μs")
print(f"调频斜率: {S/1e12:.2f} THz/s")
print(f"采样率: {Fs/1e6} MHz, 采样点数: {Ns}, chirp数: {Nchirps}")
print("\n=== 性能指标 ===")
print(f"距离分辨率: {range_resolution:.2f} m")
print(f"最大不模糊距离: {max_range:.2f} m")
print(f"速度分辨率: {velocity_resolution:.2f} m/s")
print(f"最大不模糊速度: {max_velocity:.2f} m/s")

print("\n=== 估计结果 ===")
print(f"真实参数: 距离={target_range}m, 速度={target_velocity}m/s, 角度={target_angle}度")
print(f"估计参数: 距离={estimated_range:.2f}m, 速度={estimated_velocity:.2f}m/s, 角度={estimated_angle:.2f}度")

# 可视化结果
plt.figure(figsize=(15, 5))

# 距离-多普勒图 (第一个天线)
plt.subplot(131)
plt.imshow(20*np.log10(np.abs(doppler_fft[0, :, :Ns//2].T)), aspect='auto', cmap='jet', extent=[velocity_bins[0], velocity_bins[-1], range_bins[-1], range_bins[0]])
plt.xlabel('速度 (m/s)')
plt.ylabel('距离 (m)')
plt.title('距离-多普勒图')
plt.colorbar(label='强度 (dB)')
plt.plot(target_velocity, target_range, 'wx', markersize=10, label='真实位置')
plt.plot(estimated_velocity, estimated_range, 'ro', markerfacecolor='none', markersize=10, label='估计位置')
plt.legend()

# 角度-距离图 (取多普勒峰值处)
doppler_peak_idx = np.argmax(np.max(np.abs(doppler_fft[:, :, :Ns//2]), axis=(0,2)))
plt.subplot(132)
plt.imshow(20*np.log10(np.abs(angle_fft[:, doppler_peak_idx, :Ns//2].T)), aspect='auto', cmap='jet', extent=[angle_bins[0], angle_bins[-1], range_bins[-1], range_bins[0]])
plt.xlabel('角度 (度)')
plt.ylabel('距离 (m)')
plt.title('角度-距离图')
plt.colorbar(label='强度 (dB)')
plt.plot(target_angle, target_range, 'wx', markersize=10, label='真实位置')
plt.plot(estimated_angle, estimated_range, 'ro', markerfacecolor='none', markersize=10, label='估计位置')
plt.legend()

# 角度-速度图 (取距离峰值处)
range_peak_idx = np.argmax(np.max(np.abs(doppler_fft[:, :, :Ns//2]), axis=(0,1)))
plt.subplot(133)
plt.imshow(20*np.log10(np.abs(angle_fft[:, :, range_peak_idx].T)), aspect='auto', cmap='jet', extent=[angle_bins[0], angle_bins[-1], velocity_bins[-1], velocity_bins[0]])
plt.xlabel('角度 (度)')
plt.ylabel('速度 (m/s)')
plt.title('角度-速度图')
plt.colorbar(label='强度 (dB)')
plt.plot(target_angle, target_velocity, 'wx', markersize=10, label='真实位置')
plt.plot(estimated_angle, estimated_velocity, 'ro', markerfacecolor='none', markersize=10, label='估计位置')
plt.legend()

plt.tight_layout()
plt.show()












