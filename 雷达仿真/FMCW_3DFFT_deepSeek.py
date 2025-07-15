import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# 雷达参数设置
c = 3e8  # 光速 (m/s)
fc = 77e9  # 载频 (Hz)
bw = 500e6  # 带宽 (Hz)
tm = 50e-6  # 扫频时间 (s)
slope = bw / tm  # 调频斜率 (Hz/s)

# 天线阵列参数
num_rx = 8  # 接收天线数量
d = 0.5 * (c / fc)  # 天线间距 (m)

# 目标参数 (两个目标)
targets = [
    {"range": 50, "velocity": 10, "angle": 30},  # 目标1
    {"range": 75, "velocity": -5, "angle": -20}   # 目标2
]

# 采样参数
fs = 2.5 * bw  # 采样率 (Hz)
num_samples = 1024 # int(tm * fs)  # 每个chirp采样点数
num_chirps = 128  # 每个帧的chirp数量
pri = tm  # 脉冲重复间隔 (s)

# 模拟接收信号
t = np.linspace(0, tm, num_samples)  # 单个chirp的采样时间
rx_signal = np.zeros((num_rx, num_chirps, num_samples), dtype=np.complex_)

for target in targets:
    # 计算当前目标的参数
    tau = 2 * target["range"] / c  # 往返延迟
    fd = 2 * target["velocity"] * fc / c  # 多普勒频移

    for rx in range(num_rx):
        # 计算该天线的相位差
        phase_shift = 2 * np.pi * fc * rx * d * np.sin(np.deg2rad(target["angle"])) / c

        for chirp in range(num_chirps):
            # 接收信号模型
            rx_signal[rx, chirp, :] += np.exp(1j * (
                2 * np.pi * slope * tau * t +      # 差频项
                2 * np.pi * fc * tau +            # 固定相位
                2 * np.pi * fd * chirp * pri +    # 多普勒相位
                phase_shift                       # 角度相位
            ))

# 添加噪声
noise_power = 0.1
rx_signal += np.sqrt(noise_power/2) * (np.random.randn(*rx_signal.shape) + 1j * np.random.randn(*rx_signal.shape))

# 3D FFT处理
# 1. 距离FFT (对每个chirp的采样点做FFT)
range_fft = np.fft.fft(rx_signal, axis=2)

# 2. 多普勒FFT (对每个距离门的chirp序列做FFT)
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)

# 3. 角度FFT (对每个距离-多普勒单元的天线阵列做FFT)
angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, n=1024, axis=0), axes=0)

# 计算各个维度的坐标
# 距离坐标
range_resolution = c / (2 * bw)  # 距离分辨率
max_range = fs * c / (2 * slope)  # 最大不模糊距离
range_bins = np.arange(num_samples//2) * range_resolution

# 速度坐标
velocity_resolution = c / (2 * fc * num_chirps * pri)  # 速度分辨率
max_velocity = c / (4 * fc * pri)  # 最大不模糊速度
velocity_bins = np.linspace(-max_velocity, max_velocity, num_chirps, endpoint=False)

# 角度坐标
angle_bins = np.arcsin(np.linspace(-1, 1, 1024)) * 180 / np.pi

# 检测目标
power_spectrum = np.abs(angle_fft[..., :num_samples//2])**2
spectrum_2d = np.sum(power_spectrum, axis=0)  # 压缩角度维度

# 寻找峰值 (距离-多普勒平面)
peaks, _ = find_peaks(spectrum_2d.ravel(), height=np.max(spectrum_2d)*0.3, distance=10)
peak_indices = np.unravel_index(peaks, spectrum_2d.shape)

# 估计目标参数
estimated_targets = []
for i in range(len(peaks)):
    dop_idx, rng_idx = peak_indices[0][i], peak_indices[1][i]

    # 在角度维度寻找峰值
    angle_slice = power_spectrum[:, dop_idx, rng_idx]
    ang_idx = np.argmax(angle_slice)

    estimated_targets.append({
        "range": range_bins[rng_idx],
        "velocity": velocity_bins[dop_idx],
        "angle": angle_bins[ang_idx]
    })

# 打印结果
print("=== 雷达参数 ===")
print(f"距离分辨率: {range_resolution:.2f} m")
print(f"速度分辨率: {velocity_resolution:.2f} m/s")
print(f"角度分辨率: {1.8:.1f}° (理论值)")  # 近似值

print("\n=== 真实目标 ===")
for i, target in enumerate(targets, 1):
    print(f"目标{i}: 距离={target['range']}m, 速度={target['velocity']}m/s, 角度={target['angle']}°")

print("\n=== 检测结果 ===")
for i, target in enumerate(estimated_targets, 1):
    print(f"目标{i}: 距离={target['range']:.1f}m, 速度={target['velocity']:.1f}m/s, 角度={target['angle']:.1f}°")

# 可视化
plt.figure(figsize=(18, 6))

# 距离-多普勒图 (第一个天线)
plt.subplot(131)
plt.imshow(20*np.log10(np.abs(doppler_fft[0, :, :num_samples//2].T)),
           aspect='auto', cmap='jet',
           extent=[velocity_bins[0], velocity_bins[-1], range_bins[-1], range_bins[0]])
plt.xlabel('速度 (m/s)')
plt.ylabel('距离 (m)')
plt.title('距离-多普勒图')
plt.colorbar(label='强度 (dB)')
for target in targets:
    plt.plot(target["velocity"], target["range"], 'wx', markersize=10)
for target in estimated_targets:
    plt.plot(target["velocity"], target["range"], 'ro', markerfacecolor='none', markersize=10)

# 角度-距离图 (多普勒峰值处)
plt.subplot(132)
integrated_angle_range = np.sum(power_spectrum, axis=1)
plt.imshow(20*np.log10(integrated_angle_range.T),
           aspect='auto', cmap='jet',
           extent=[angle_bins[0], angle_bins[-1], range_bins[-1], range_bins[0]])
plt.xlabel('角度 (度)')
plt.ylabel('距离 (m)')
plt.title('角度-距离图')
plt.colorbar(label='强度 (dB)')
for target in targets:
    plt.plot(target["angle"], target["range"], 'wx', markersize=10)
for target in estimated_targets:
    plt.plot(target["angle"], target["range"], 'ro', markerfacecolor='none', markersize=10)

# 角度-速度图 (距离峰值处)
plt.subplot(133)
integrated_angle_vel = np.sum(power_spectrum, axis=2)
plt.imshow(20*np.log10(integrated_angle_vel.T),
           aspect='auto', cmap='jet',
           extent=[angle_bins[0], angle_bins[-1], velocity_bins[-1], velocity_bins[0]])
plt.xlabel('角度 (度)')
plt.ylabel('速度 (m/s)')
plt.title('角度-速度图')
plt.colorbar(label='强度 (dB)')
for target in targets:
    plt.plot(target["angle"], target["velocity"], 'wx', markersize=10)
for target in estimated_targets:
    plt.plot(target["angle"], target["velocity"], 'ro', markerfacecolor='none', markersize=10)

plt.tight_layout()
plt.show()

