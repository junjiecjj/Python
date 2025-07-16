#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 02:02:28 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

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

def freqDomainView(x, Fs, FFTN = None, type = 'double'): # N为偶数
    if FFTN == None:
        FFTN = 2**int(np.ceil(np.log2(x.size)))
    X = scipy.fftpack.fft(x, n = FFTN)
    # 消除相位混乱
    threshold = np.max(np.abs(X)) / 10000
    X[np.abs(X) < threshold] = 0
    # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
    X = X/x.size               # 将频域序列 X 除以序列的长度 N
    if type == 'single':
        Y = X[0 : int(FFTN/2)+1].copy()       # 提取 X 里正频率的部分,N为偶数
        Y[1 : int(FFTN/2)] = 2*Y[1 : int(FFTN/2)].copy()
        f = np.arange(0, int(FFTN/2)+1) * (Fs/FFTN)
        # 计算频域序列 Y 的幅值和相角
        A = np.abs(Y)                     # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    elif type == 'double':
        f = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(FFTN, 1/Fs))
        Y = scipy.fftpack.fftshift(X, )
        # 计算频域序列 Y 的幅值和相角
        A = np.abs(Y)                     # 计算频域序列 Y 的幅值
        Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
        R = np.real(Y)                    # 计算频域序列 Y 的实部
        I = np.imag(Y)                    # 计算频域序列 Y 的虚部
    return f, Y, A, Pha, R, I

#%% 接收方使用混频技术得到差频信号(exp形式)做后续处理，FFT先做距离维再做速度维，直接用FFT2D完成
c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 1024         # ADC采样点数
Nchirp = 256      # chirp数量
lamba = c/f0      # 波长
Fs = Ns/Tc        # = 1/(t[1] - t[0])     # 模拟信号采样频率
nRx = 8           # RX天线通道数
d = lamba/2       # 天线阵列间距

NumRangeFFT = Ns                        # Range FFT Length
NumDopplerFFT = Nchirp                 # Doppler FFT Length
rangeRes = c/(2*B)                        # 距离分辨率
maxRange = rangeRes * Ns                  # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)        # 速度分辨率
maxVel = velRes * Nchirp/2                # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2
angRes = 2/Ns
maxAng = np.arcsin(lamba/(2*d))
print(f"rangeRes = {rangeRes:.4f}, maxRange = {maxRange:.4f}, velRes = {velRes:.4f}, maxVel = {maxVel:.4f} ")

# tarR = [100, ]     # 目标距离
# tarV = [20, ]      # 目标速度
# tarA = [40,  ]     # 目标角度
# sigma = [0.01 ]    # 高斯白噪声标准差
tarR = [100, 200, 300]  # 目标距离
tarV = [-30, 15, 30]    # 目标速度
tarA = [0, 30, 60]    # 目标角度
# sigma = [0.1, 0.1, 0.1]    # 高斯白噪声标准差
sigma = [0.0, 0.0, 0.0]    # 高斯白噪声标准差

# 目标参数 (两个目标)
targets = []
for k in range(len(tarR)):
    Dic = {"range": tarR[k], "velocity": tarV[k], "angle": tarA[k]}
    targets.append(Dic)

# ### 模拟接收信号, 直接获取差频信号. DeepSeek；
# t = np.linspace(0, Tc, Ns)  # 单个chirp的采样时间
# sigReceive = np.zeros((nRx, Nchirp, Ns), dtype = np.complex_)
# for rx in range(nRx):
#     for chirp in range(Nchirp):
#         for k in range(len(tarR)):
#             # 计算该天线的相位差
#             phase_shift = 2 * np.pi * f0 * rx * d * np.sin(np.deg2rad(tarA[k])) / c

#             # 计算当前目标的参数
#             tau = 2 * tarR[k] / c     # 往返延迟
#             fd = 2 * tarV[k] * f0 / c  # 多普勒频移

#             # 接收信号模型
#             sigReceive[rx, chirp, :] += np.exp(1j * (
#                 2 * np.pi * S * tau * t +         # 差频项
#                 2 * np.pi * f0 * tau +            # 固定相位
#                 2 * np.pi * fd * chirp * Tc +     # 多普勒相位
#                 phase_shift                       # 角度相位
#             ))

# ### 模拟接收信号, 直接获取差频信号；干货 | 利用MATLAB实现FMCW雷达的距离多普勒估计:
# t = np.linspace(0, Tc, Ns)  # 单个chirp的采样时间
# sigReceive = np.zeros((nRx, Nchirp, Ns), dtype = np.complex_)
# for rx in range(nRx):
#     for chirp in range(Nchirp):
#         for k in range(len(tarR)):
#             # 计算该天线的相位差
#             phase_shift = 2 * np.pi * f0 * rx * d * np.sin(np.deg2rad(tarA[k])) / c

#             # 计算当前目标的参数
#             R = (tarR[k] + tarV[k] * Tc * chirp)

#             # 接收信号模型
#             sigReceive[rx, chirp, :] += np.exp(1j *  (
#                 2 * np.pi * (2 * B * R/(c * Tc) + 2 * f0 * tarV[k]/c) * t +
#                 2 * np.pi * 2 * f0 * R/c +
#                 phase_shift
#                 ))

#### 模拟接收信号, 收发相乘获取差频信号；
t = np.linspace(0, Tc, Ns)  # 单个chirp的采样时间
ft = f0 * t + S / 2 * t**2
Sx = np.exp(1j * 2 * np.pi * ft)          # 发射信号

Rx = np.zeros((nRx, Nchirp, Ns), dtype = np.complex_)
for rx in range(nRx):
    for chirp in range(Nchirp):
        for k in range(len(tarR)):
            # 计算该天线的相位差
            phase_shift = 2 * np.pi * f0 * rx * d * np.sin(np.deg2rad(tarA[k])) / c

            dtmp = tarR[k] + tarV[k] * (t + chirp * Tc)
            tau = 2 * dtmp / c                                # 运动目标的时延是动态变化的
            fr = f0 * (t + tau) + S / 2 * (t + tau)**2
            noise = (np.random.randn(*Sx.shape) + 1j * np.random.randn(*Sx.shape)) * np.sqrt(sigma[k])
            Rx[rx, chirp, :] += (np.exp(1j * 2 * np.pi * fr + 1j * phase_shift) + noise )
sigReceive = np.conjugate(Sx) * Rx # 混频

range_win = np.hamming(Ns)           # 加海明窗
doppler_win = np.hamming(Nchirp)

##>>>>>>>>>>>>>> MUSIC算法
X = np.zeros((nRx, int(Nchirp * Ns)),dtype = complex)
for nr in range(nRx):
    X[nr,:] = sigReceive[nr,:,:].flatten()
Rxx = X @ X.T.conjugate() / int(Nchirp * Ns)
# 特征值分解
eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
eigvector = eigvector[:, idx]
eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
Un = eigvector[:, len(tarR):nRx]

# Un = eigvector
UnUnH = Un @ Un.T.conjugate()
Thetalst = np.arange(-90, 90.1, 0.5)
angle = np.deg2rad(Thetalst)
Pmusic = np.zeros(angle.size)
for i, ang in enumerate(angle):
    a = np.exp(1j * np.pi * np.arange(nRx) * np.sin(ang)).reshape(-1, 1)
    Pmusic[i] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]

Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
Pmusic = 10 * np.log10(Pmusic)
peaks, _ =  scipy.signal.find_peaks(Pmusic, height = -20, distance = 10)
angle_est = Thetalst[peaks]

colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111,  )
ax1.plot(Thetalst, Pmusic, color = colors[k])
ax1.scatter(angle_est, Pmusic[peaks], s = 50, c = 'blue')
ax1.set_xlabel('Angle in degrees', )
ax1.set_ylabel('dB', )
ax1.set_title("MUSIC Spectrum")
plt.show()
plt.close()

######>>>>>>>>>>>>>>    3维FFT处理
# # 距离FFT
# range_fft = np.zeros((nRx, Nchirp, Ns), dtype = complex)
# for nrx in range(nRx):
#     for l in range(Nchirp):
#         tmp = sigReceive[nrx, l,:] * range_win
#         tmp_fft = np.fft.fft(tmp, Ns)   # 对每个chirp做N点FFT
#         range_fft[nrx, l, :] = tmp_fft
# # 多普勒FFT
# doppler_fft = np.zeros((nRx, Nchirp, Ns), dtype = complex)
# for nrx in range(nRx):
#     for n in range(Ns):
#         tmp = range_fft[nrx, :, n] * doppler_win
#         tmp_fft = np.fft.fft(tmp, Nchirp) # 对rangeFFT结果进行M点FFT
#         tmp_fft = np.fft.fftshift(tmp_fft)
#         doppler_fft[nrx,:,n] = tmp_fft
# sigDopplerFFT = doppler_fft[1, :, :]

# # 角度FFT
# Q = 128
# angle_fft = np.zeros((Q, Nchirp, Ns), dtype = complex)
# for n in range(Ns):
#     for l in range(Nchirp):
#         tmp = doppler_fft[:, l, n]
#         tmp_fft = np.fft.fft(tmp, Q)
#         tmp_fft = np.fft.fftshift(tmp_fft)   # 对2D FFT结果进行nRx点FFT
#         angle_fft[:, l, n] = tmp_fft

##>>>>>>>>>>>>>>  3D FFT处理
# 1. 距离FFT (对每个chirp的采样点做FFT)
sigReceive1 = sigReceive * range_win[None,:]
range_fft = np.fft.fft(sigReceive1, n = NumRangeFFT, axis = 2)

# 2. 多普勒FFT (对每个距离门的chirp序列做FFT)
range_fft = range_fft * doppler_win[:,None]
doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n = NumDopplerFFT, axis = 1), axes = 1)
# sigDopplerFFT = doppler_fft[3, :, :]
sigDopplerFFT = np.sum(np.abs(doppler_fft), axis = 0)
# 3. 角度FFT (对每个距离-多普勒单元的天线阵列做FFT)
Q = 128
angle_fft = np.fft.fftshift(np.fft.fft(doppler_fft, n = Q, axis = 0), axes = 0)

####>>>>>>>>>>>>>>>>>>>>>
x = np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange
# x = np.arange(NumRangeFFT) / NumRangeFFT * maxRange  # 如果使用这行，则注释掉Z = Z[:, 0:int(NumRangeFFT/2)]
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)
# 角度坐标
angle_bins = np.arcsin(np.linspace(-1, 1, Q)) * 180 / np.pi

#>>>>>>>>>>>> ### 距离-多普勒图 3D
Z = np.abs(sigDopplerFFT)/1e5
Z = Z[:, 0:int(NumRangeFFT/2)]

def ind2sub2D(idx, weigh, heigh):
    row = idx // weigh
    col = idx % weigh
    return row, col

Idxs = scipy.signal.find_peaks(Z.flatten(), np.max(Z)*0.7, distance = 10)[0]
idxs = np.unravel_index(Idxs, Z.shape)

Xscatter = x[idxs[1]]  # = rangeRes * idxs[1]
Yscatter = y[idxs[0]]  # = velRes * (idxs[0] - Nchirp/2)
Zscatter = Z[idxs]
print(f'目标距离：{Xscatter} m ', )
print(f'目标速度：{Yscatter} m/s ', )

##>>>>>>>>>>>>>>> 取log
fig = plt.figure(figsize=(12, 6), constrained_layout = True)
ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(X, Y, Z, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.plot_surface(X, Y, 10*np.log10(Z), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.scatter(Xscatter, Yscatter, Zscatter, s = 20, c = 'r', )
ax1.grid(False)
ax1.invert_xaxis()                                    #   x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 18)
ax1.set_ylabel('Velocity(m/s)', fontsize = 18)
ax1.set_zlabel('Amplitude(dB)', fontsize = 18)
ax1.set_title('DopplerFFT', fontsize = 18)
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(122, )
# im = ax2.imshow(np.abs(Z), aspect='auto', cmap='jet', extent=[x[0], x[-1], y[-1], y[0] ])
im = ax2.imshow(20*np.log10(np.abs(Z)), aspect='auto', cmap='jet', extent=[x[0], x[-1], y[-1], y[0] ])
ax2.set_xlabel('距离 (m)')
ax2.set_ylabel('速度 (m/s)')
ax2.set_title('距离-速度图')
cbar = fig.colorbar(im, ax = ax2, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'

plt.show()
plt.close()

##>>>>>>>>>>>> 绝对值
fig = plt.figure(figsize=(12, 6), constrained_layout = True)
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
# ax1.plot_surface(X, Y, 10*np.log10(Z), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.scatter(Xscatter, Yscatter, Zscatter, s = 20, c = 'r', )
ax1.grid(False)
ax1.invert_xaxis()                                    #   x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 18)
ax1.set_ylabel('Velocity(m/s)', fontsize = 18)
ax1.set_zlabel('Amplitude', fontsize = 18)
ax1.set_title('DopplerFFT', fontsize = 18)
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(122, )
im = ax2.imshow(np.abs(Z), aspect='auto', cmap='jet', extent=[x[0], x[-1], y[-1], y[0] ])
# im = ax2.imshow(20*np.log10(np.abs(Z)), aspect='auto', cmap='jet', extent=[x[0], x[-1], y[-1], y[0] ])
ax2.set_xlabel('距离 (m)')
ax2.set_ylabel('速度 (m/s)')
ax2.set_title('距离-速度图')
cbar = fig.colorbar(im, ax = ax2, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'

plt.show()
plt.close()

#####>>>>>>>>>>>>>>>>>>>>>>>> # 检测目标
power_spectrum = np.abs(angle_fft[..., :NumRangeFFT//2])**2
spectrum_2d = np.sum(power_spectrum, axis = 0)  # 压缩角度维度

# 寻找峰值 (距离-多普勒平面)
peaks, _ = scipy.signal.find_peaks(spectrum_2d.ravel(), height = np.max(spectrum_2d)*0.5, distance = 10)
peak_indices = np.unravel_index(peaks, spectrum_2d.shape)

# 估计目标参数
estimated_targets = []
for i in range(len(peaks)):
    dop_idx, rng_idx = peak_indices[0][i], peak_indices[1][i]

    # 在角度维度寻找峰值
    angle_slice = power_spectrum[:, dop_idx, rng_idx]
    ang_idx = np.argmax(angle_slice)

    estimated_targets.append({
        "range": x[rng_idx],
        "velocity": y[dop_idx],
        "angle": angle_bins[ang_idx]
    })

# 打印结果
print("\n=== 真实目标 ===")
for i, target in enumerate(targets, 1):
    print(f"目标{i}: 距离={target['range']}m, 速度={target['velocity']}m/s, 角度={target['angle']}°")

print("\n=== 检测结果 ===")
for i, target in enumerate(estimated_targets, 1):
    print(f"目标{i}: 距离={target['range']:.1f}m, 速度={target['velocity']:.1f}m/s, 角度={target['angle']:.1f}°")

####>>>>>  结果可视化 2D
fig, axs = plt.subplots(1, 3, figsize = (18, 5), constrained_layout = True)

# 距离-多普勒图 (第一个天线)
# plt.subplot(131)
range_vel = np.abs(doppler_fft[3, :, :NumRangeFFT//2]).T  # or Z.T
range_vel = range_vel/range_vel.max()
tmp = 10*np.log10(range_vel)
im = axs[0].imshow(tmp, aspect='auto', cmap='jet', extent=[y[0], y[-1], x[-1], x[0]])
# im = axs[0].imshow(range_vel, aspect='auto', cmap='jet', extent=[y[0], y[-1], x[-1], x[0]])
axs[0].set_xlabel('速度 (m/s)')
axs[0].set_ylabel('距离 (m)')
axs[0].set_title('距离-速度')
cbar = fig.colorbar(im, ax = axs[0], orientation = 'vertical',) # label='强度 (dB)'
# axs[0].colorbar(label='强度 (dB)')
for target in targets:
    axs[0].plot(target["velocity"], target["range"], 'kx', markersize = 10)
for target in estimated_targets:
    axs[0].plot(target["velocity"], target["range"], 'ro', markerfacecolor = 'none', markersize = 10)

# 角度-距离图 (多普勒峰值处)
# plt.subplot(132)
integrated_angle_range = np.sum(power_spectrum, axis = 1).T
integrated_angle_range = integrated_angle_range/integrated_angle_range.max()
tmp = 10*np.log10(integrated_angle_range)
im = axs[1].imshow(tmp, aspect = 'auto', cmap = 'jet', extent = [angle_bins[0], angle_bins[-1], x[-1], x[0]])
# im = axs[1].imshow(integrated_angle_range, aspect = 'auto', cmap = 'jet', extent = [angle_bins[0], angle_bins[-1], x[-1], x[0]])
axs[1].set_xlabel('角度 (度)')
axs[1].set_ylabel('距离 (m)')
axs[1].set_title('距离-角度')
cbar = fig.colorbar(im, ax = axs[1], orientation = 'vertical',) # label='强度 (dB)'
# axs[1].colorbar(label='强度 (dB)')
for target in targets:
    axs[1].plot(target["angle"], target["range"], 'kx', markersize = 10)
for target in estimated_targets:
    axs[1].plot(target["angle"], target["range"], 'ro', markerfacecolor = 'none', markersize = 10)

# 角度-速度图 (距离峰值处)
# plt.subplot(133)
integrated_angle_vel = np.sum(power_spectrum, axis = 2).T
integrated_angle_vel = integrated_angle_vel/integrated_angle_vel.max()
tmp = 10*np.log10(integrated_angle_vel)
im = axs[2].imshow(tmp, aspect = 'auto', cmap = 'jet', extent = [angle_bins[0], angle_bins[-1], y[-1], y[0]])
# im = axs[2].imshow(integrated_angle_vel, aspect = 'auto', cmap = 'jet', extent = [angle_bins[0], angle_bins[-1], y[-1], y[0]])
axs[2].set_xlabel('角度 (度)')
axs[2].set_ylabel('速度 (m/s)')
axs[2].set_title('速度-角度')
cbar = fig.colorbar(im, ax = axs[2], orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'
# axs[2].colorbar(label='强度 (dB)')
for target in targets:
    axs[2].plot(target["angle"], target["velocity"], 'kx', markersize = 10)
for target in estimated_targets:
    axs[2].plot(target["angle"], target["velocity"], 'ro', markerfacecolor = 'none', markersize = 10)

# plt.tight_layout()
plt.show()
plt.close()

####>>>>> 结果可视化 3D
fig = plt.figure(figsize=(18, 5) )

# 距离-多普勒图 (第一个天线)
X, Y = np.meshgrid(x, y)
ax1 = fig.add_subplot(131, projection = '3d')
angle_vel = np.abs(doppler_fft[3, :, :NumRangeFFT//2])  # Z
angle_vel = angle_vel/angle_vel.max()
tmp = 10*np.log10(angle_vel)
# ax1.plot_surface(X, Y, angle_range, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.plot_surface(X, Y, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.invert_xaxis()                                    #   x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('距离 (m)', fontsize = 12)
ax1.set_ylabel('速度 (m/s)', fontsize = 12)
ax1.set_zlabel('Amplitude', fontsize = 12)
ax1.set_title('距离-速度', fontsize = 12)
ax1.view_init(azim = -135, elev = 30)

# 角度-距离图 (多普勒峰值处)
X, Angle_bins = np.meshgrid(x, angle_bins)
ax2 = fig.add_subplot(132, projection = '3d')
integrated_angle_range = np.sum(power_spectrum, axis = 1)
integrated_angle_range = integrated_angle_range/integrated_angle_range.max()
tmp = 10*np.log10(integrated_angle_range)
# ax2.plot_surface(X, Angle_bins, integrated_angle_range, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.plot_surface(X, Angle_bins, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.invert_xaxis()                                    #   x轴反向
ax2.set_proj_type('ortho')
ax2.set_xlabel('距离 (m)', fontsize = 12)
ax2.set_ylabel('角度 (度)', fontsize = 12)
ax2.set_zlabel('Amplitude', fontsize = 12)
ax2.set_title('距离-角度', fontsize = 12)
ax2.view_init(azim = -135, elev = 30)

# 角度-速度图 (距离峰值处)
Angle_bins, Y = np.meshgrid(y, angle_bins )
ax3 = fig.add_subplot(133, projection = '3d')
integrated_angle_vel = np.sum(power_spectrum, axis = 2)
integrated_angle_vel = integrated_angle_vel/integrated_angle_vel.max()
tmp = 10*np.log10(integrated_angle_vel)
# ax3.plot_surface(Angle_bins, Y, integrated_angle_vel, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax3.plot_surface(Angle_bins, Y, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax3.grid(False)
ax3.invert_xaxis()                                    #   x轴反向
ax3.set_proj_type('ortho')
ax3.set_xlabel('速度 (m/s)', fontsize = 12)
ax3.set_ylabel('角度 (度)', fontsize = 12)
ax3.set_zlabel('Amplitude', fontsize = 12)
ax3.set_title('速度-角度', fontsize = 12)
ax3.view_init(azim = -135, elev = 30)

# plt.tight_layout()
plt.show()
plt.close()

## (四) MUSIC 距离-AOA谱, 下面代码无效
# ang_ax = np.arange(-90, 90, 0.5)             # angle axis
# range_az_music = np.zeros((Ns, ang_ax.size), dtype = complex)
# range_fft = np.fft.fft(sigReceive, axis = 2)
# for i in range(Ns):
#     # X = np.sum(range_fft[:,:,i], axis = 1)[:,None]
#     X = range_fft[:,:,i]
#     Rxx = X @ X.T.conjugate() / Ns
#     # for m in range(M):
#     #    A =  np.sum(rangeFFT1[i,m*numChirps:(m+1)*numChirps,:], 0)
#     #    Rxx = Rxx + 1/M * (A[:,None]@A[:,None].conjugate().T)
#     # 特征值分解
#     eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
#     idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
#     eigvector = eigvector[:, idx]
#     eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
#     Un = eigvector[:, 1:]
#     UnUnH = Un@Un.conjugate().T
#     for a, ang in enumerate(ang_ax):
#         at = np.exp(1j * np.pi * np.arange(nRx) * np.sin(ang)).reshape(-1, 1)
#         range_az_music[i,a] = (at.conjugate().T@at)[0,0]/(at.T.conjugate()@UnUnH@at)[0,0]

# colors = plt.cm.jet(np.linspace(0, 1, 3))
# fig = plt.figure(figsize=(8, 4) )
# ax1 = fig.add_subplot(111, )
# im = ax1.imshow(10*np.log10(np.abs(range_az_music)/np.abs(range_az_music).max()), aspect = 'auto', cmap = 'jet', extent = [ang_ax[0], ang_ax[-1], x[-1], x[0]])
# ax1.set_xlabel('Azimuth')
# ax1.set_ylabel('Range (m)')
# ax1.set_title('MUSIC Range-Angle Map')
# cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', label = '强度 (dB)')
# plt.show()
# plt.close()











