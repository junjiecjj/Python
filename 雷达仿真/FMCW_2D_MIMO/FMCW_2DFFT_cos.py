#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:33:48 2025

@author: jack
"""
#%% 接收方使用混频技术得到差频信号做后续处理，FFT先做距离维再做速度维，直接用FFT2D完成
import numpy as np
import matplotlib.pyplot as plt
import scipy
# from mpl_toolkits.mplot3d import Axes3D
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

c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 1024          # ADC采样点数
Nchirp = 256      # chirp数量
lamba = c/f0      # 波长
Fs = Ns/Tc        # = 1/(t[1] - t[0])    # 模拟信号采样频率,此时要注意，本文采用“数字信号的形式”来模拟发射信号与回波信号，实际上是不会通过这种方式实现的，发射信号和回波信号都是模拟信号。因此在本文中，混频之后得到的中频信号只有单一的频谱，即只保留了混频乘法器积化和差公式后的“差频”信号，“和频”信号由于MATLAB采用数字信号的形式导致采样率过低，N/Tchirp=1024/Tchirp=139MHz，对“和频信号”相当于欠采样，因此并不会保留高频信号，只保留目标的最大中频信号（27.27MHz），从而MATLAB自身形成一个低通（或带通）滤波器。
NumRangeFFT = Ns                         # Range FFT Length
NumDopplerFFT = Nchirp                   # Doppler FFT Length
rangeRes = c/(2 * B)                        # 距离分辨率
maxRange = rangeRes * Ns                  # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)        # 速度分辨率
maxVel = velRes * Nchirp/2                # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2

print(f"rangeRes = {rangeRes:.4f}, maxRange = {maxRange:.4f}, velRes = {velRes:.4f}, maxVel = {maxVel:.4f}, ")

tarR = [ 100, 200]     # 目标距离
tarV = [ 10, -20 ]     # 目标速度
sigma = [0.1, 1.1 ]    # 高斯白噪声标准差

tarR = [10, 100, 200, 400]         # 目标距离,注意，上面的maxRange是2*Fs采样率下的最大探测距离，也就是Ns下的最大探测距离，而根据奈奎斯特采样定理，Fs采样率下的最大距离应该是maxRange/2,所以这里的tarR的最大值不能超过maxRange/2，否则会出现很奇怪的现象，可以改变参数试试。
tarV = [0, 10, -20, 40]            # 目标速度
sigma = [0.1, 0.1, 0.1, 0.1]       # 高斯白噪声标准差
# sigma = [1.1, 1.1, 1.1, 1.1]     # 高斯白噪声标准差

t = np.linspace(0, Tc, Ns)
ft = f0 * t + S / 2 * t**2
Sx = np.cos(2 * np.pi * ft)        # 发射信号

fs = 4 * (f0 + B)
ts = 1/fs
t1 = np.arange(0, Tc-ts, ts)
ft1 = f0 * t1 + S / 2 * t1**2
Sx1 = np.cos(2 * np.pi * ft1)          # 发射信号
f, _, A, _, _, _  = freqDomainView(Sx1, fs, type = 'double')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f, A)
axs.set_title("Sx Signal FFT")
axs.set_xlabel("Frequency")
axs.set_ylabel("Amplitude")
axs.set_xlim(0.76e11, 0.78e11)
plt.show()
plt.close()

##>>>>>>>>>> 接收差频信号 FFT
Rx  = np.zeros((Nchirp, Ns), dtype = complex)
N = np.arange(Ns)
for l in range(Nchirp):
    for k in range(len(tarR)):
        d = tarR[k] + tarV[k] * (t + l * Tc)
        tau = 2 * d / c          # 运动目标的时延是动态变化的
        fr = f0 * (t + tau) + S / 2 * (t + tau)**2
        noise = np.random.randn(*Sx.shape) * np.sqrt(sigma[k])
        Rx[l, :] += (np.cos(2 * np.pi * fr) + noise )
##>>>>>>>>>>> 混频
Mix = Sx * Rx
# range_win = np.hamming(Ns)           # 加海明窗
# doppler_win = np.hamming(Nchirp)
# Mix = Mix * range_win
# Mix = Mix * doppler_win[:,None]

f, _, A, _, _, _  = freqDomainView(Mix[0, 0:Ns], Fs, type = 'single')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f*c/(2*S), A)
axs.set_title("Mix Signal FFT")
axs.set_xlabel("Frequency->Distance")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()

##>>>>>>>>>>> 距离-速度二维FFT
x = np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)
# Z = np.abs(scipy.fft.fftshift(np.fft.fft2(Mix.T, (NumRangeFFT, NumDopplerFFT)), axes = 1))
# Z = Z.T
# or
Z = np.abs(scipy.fft.fftshift(np.fft.fft2(Mix , (NumDopplerFFT, NumRangeFFT)), axes = 0))

Z = Z[:, 0:int(NumRangeFFT/2)]
###
def ind2sub2D(idx, weigh, heigh):
    row = idx // weigh
    col = idx % weigh
    return row, col
Idxs = scipy.signal.find_peaks(Z.flatten() , np.max(Z)*0.5, distance=10)[0]
idxs = ind2sub2D(Idxs, Z.shape[1], Z.shape[0])
Xscatter = x[idxs[1]]  # = rangeRes * idxs[1]
Yscatter = y[idxs[0]]  # = velRes * (idxs[0] - Nchirp/2)
Zscatter = Z[idxs]
print(f'目标距离：{Xscatter} m\n', )
print(f'目标速度：{Yscatter} m/s\n', )


# 距离-多普勒图2D (第一个天线)
plt.subplot(111)
plt.imshow(20*np.log10(Z), aspect='auto', cmap='jet', extent=[x[0], x[-1], y[-1], y[0]])
# plt.imshow(20*np.log10(np.abs(doppler_fft[0, :, :Ns//2].T)), aspect='auto', cmap='jet', extent=[velocity_bins[0], velocity_bins[-1], range_bins[-1], range_bins[0]])
plt.xlabel('距离 (m)')
plt.ylabel('速度 (m/s)')
plt.title('距离-多普勒图')
plt.colorbar(label='强度 (dB)')


### 距离-多普勒图 3D
fig = plt.figure(figsize = (10, 20) )
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z/1e4, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.scatter(Xscatter, Yscatter, Zscatter/1e4, s =  20, c = 'r', )
ax1.grid(False)
ax1.invert_xaxis()  #x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 10)
ax1.set_ylabel('Velocity(m/s)', fontsize = 10)
ax1.set_title('DopplerFFT', fontsize = 10)
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(122, projection = '3d' )
ax2.plot_surface(X, Y, Z/1e4, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.set_xlabel('Range(m)', fontsize = 10)
ax2.set_ylabel('Velocity(m/s)', fontsize = 10)
ax2.set_title('Top view', fontsize = 10)
ax2.set_zticks([])
ax2.view_init(azim = 270, elev = 90)

plt.show()
plt.close()

#>>>>>>>>>>> 距离维 FFT
plt.rcParams["font.family"] = "SimSun"
sig_fft = scipy.fft.fft(Mix, NumRangeFFT, axis = 1)/NumRangeFFT
sig_fft = np.abs(sig_fft)
sig_fft = sig_fft[:, 0:int(NumRangeFFT/2)]

# 结果可视化
fig, axs = plt.subplots(1, 1, figsize = (10, 6), constrained_layout = True)
k = 0
axs.plot(np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange, sig_fft[k,:], lw = 2)
axs.set_title(f"第{k}个chirp的FTF结果")
axs.set_xlabel("距离（频率）")
axs.set_ylabel("幅度")
plt.show()
plt.close()

# 距离FFT结果谱矩阵
xx = np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange
yy = np.linspace(-maxVel, maxVel, Nchirp)
X, Y = np.meshgrid(xx, yy)
fig = plt.figure(figsize=(10, 10) )
ax1 = fig.add_subplot(111, projection = '3d')
ax1.plot_surface(X, Y, sig_fft, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', )
ax1.set_ylabel('Velocity(m/s)', )
ax1.set_zlabel('Amplitude', )
ax1.set_title('距离维FFT', )
ax1.view_init(azim = -135, elev = 30)
plt.show()
plt.close()



































