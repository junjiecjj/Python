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
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 12          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 12          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

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
Ns = 1024          # ADC采样点数
Nchirp = 256      # chirp数量
lamba = c/f0      # 波长
Fs = Ns/Tc        # = 1/(t[1] - t[0])     # 模拟信号采样频率
nRx = 4          # RX天线通道数
d = lamba/2      # 天线阵列间距

NumRangeFFT = Ns                           # Range FFT Length
NumDopplerFFT = Nchirp                     # Doppler FFT Length
rangeRes = c/(2*B)                        # 距离分辨率
maxRange = rangeRes * Ns                  # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)        # 速度分辨率
maxVel = velRes * Nchirp/2                # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2
angRes = 2/Ns
maxAng = np.arcsin(lamba/(2*d))
print(f"rangeRes = {rangeRes:.4f}, maxRange = {maxRange:.4f}, velRes = {velRes:.4f}, maxVel = {maxVel:.4f} ")

tarR = [200, ]  # 目标距离
tarV = [ -20, ]   # 目标速度
tarA = [30,]   # 目标角度

# tarR = [100, 200, 300]  # 目标距离
# tarV = [10, -20, 30]   # 目标速度
# tarA = [-30, 30, 60]   # 目标角度
# # sigma = [0.1, 1.1 ]    # 高斯白噪声标准差

# generate receive signal
sigReceiveTmp = np.zeros((Nchirp, Ns), dtype = complex)
N = np.arange(Ns)
for l in range(Nchirp):
    for k in range(len(tarR)):
        sigReceiveTmp[l,:] += np.exp(1j * 2 * np.pi * ((2 * B * (tarR[k] + tarV[k] * Tc * l)/(c * Tc) + 2 * f0 * tarV[k]/c) * (Tc/Ns) * N + 2 * f0 * (tarR[k] + tarV[k] * Tc * l)/c))
sigReceive = np.zeros((Nchirp, Ns, nRx), dtype = complex)
for nrx in range(nRx):
    for ang in tarA:
        sigReceive[:,:,nrx] += sigReceiveTmp * np.exp(1j * 2 * np.pi * f0 * d * nrx * np.sin(ang) / c)

range_win = np.hamming(Ns)           # 加海明窗
doppler_win = np.hamming(Nchirp)
#### 3维FFT处理
# 距离FFT
range_profile = np.zeros((Nchirp, Ns, nRx), dtype = complex)
for nrx in range(nRx):
    for l in range(Nchirp):
        tmp = sigReceive[l,:,nrx] * range_win
        tmp_fft = np.fft.fft(tmp, Ns)   # 对每个chirp做N点FFT
        range_profile[l,:,nrx] = tmp_fft

# 多普勒FFT
speed_profile = np.zeros((Nchirp, Ns, nRx), dtype = complex)
for nrx in range(nRx):
    for n in range(Ns):
        tmp = range_profile[:,n,nrx] * doppler_win
        tmp_fft = np.fft.fft(tmp, Nchirp) # 对rangeFFT结果进行M点FFT
        tmp_fft = np.fft.fftshift(tmp_fft)
        speed_profile[:,n,nrx] = tmp_fft

# 角度FFT
Q = 128
angle_profile = np.zeros((Nchirp, Ns, Q), dtype = complex)
for n in range(Ns):
    for l in range(Nchirp):
        tmp = speed_profile[l,n,:]
        tmp_fft = np.fft.fft(tmp, Q)
        tmp_fft = np.fft.fftshift(tmp_fft)   # 对2D FFT结果进行nRx点FFT
        angle_profile[l,n,:] = tmp_fft

sigDopplerFFT = speed_profile[:,:,1]

x = np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange
# x = np.arange(NumRangeFFT) / NumRangeFFT * maxRange  # 如果使用这行，则注释掉Z = Z[:, 0:int(NumRangeFFT/2)]
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)

Z = np.abs(sigDopplerFFT)
Z = Z[:, 0:int(NumRangeFFT/2)]

def ind2sub2D(idx, weigh, heigh):
    row = idx // weigh
    col = idx % weigh
    return row, col

Idxs = scipy.signal.find_peaks(Z.flatten()/1e5, height = 0.6)[0]
idxs = ind2sub2D(Idxs, Z.shape[1], Z.shape[0])
Xscatter = x[idxs[1]]  # = rangeRes * idxs[1]
Yscatter = y[idxs[0]]  # = velRes * (idxs[0] - Nchirp/2)
Zscatter = Z[idxs]/1e5
print(f'目标距离：{Xscatter} m\n', )
print(f'目标速度：{Yscatter} m/s\n', )

fig = plt.figure(figsize=(10, 20) )
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z/1e5, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.scatter(Xscatter, Yscatter, Zscatter, s = Zscatter*20, c = 'r', )
ax1.grid(False)
ax1.invert_xaxis()  #x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 10)
ax1.set_ylabel('Velocity(m/s)', fontsize = 10)
ax1.set_zlabel('Amplitude', fontsize = 10)
ax1.set_title('DopplerFFT', fontsize = 10)
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(122, projection = '3d' )
ax2.plot_surface(X, Y, Z/1e5, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.set_xlabel('Range(m)', fontsize = 10)
ax2.set_ylabel('Velocity(m/s)', fontsize = 10)
# ax2.set_zlabel('Amplitude', fontsize = 10)
ax2.set_title('Top view', fontsize = 10)
ax2.set_zticks([])
ax2.view_init(azim = 270, elev = 90)

plt.show()
plt.close()

# 计算峰值位置
angle_profile = np.abs(angle_profile)/1e5

fig = plt.figure(figsize = (8, 6) )
ax1 = fig.add_subplot(111,  )
ax1.plot(angle_profile[11,21,:], )
# ax1.grid(False)
ax1.set_xlabel('Angle', fontsize = 10)
ax1.set_ylabel('Amp', fontsize = 10)
ax1.set_title('Angle FFT', fontsize = 10)
plt.show()
plt.close()


Idxs = np.where(angle_profile == angle_profile.max())
Idxs = np.where(angle_profile > 2.5)
row = Idxs[1]
col = Idxs[0]
pag = Idxs[2]

# 计算目标距离、速度、角度
fb = ((row-1) * Fs)/Ns                      # 差拍频率
fd = ((col-Nchirp/2-1)*Fs)/(Ns*Nchirp)      # 多普勒频率
fw = (pag - Q/2 -1)/Q                       # 空间频率
R = c*(fb-fd)/2/S                           # 距离公式
v = lamba*fd/2;                             # 速度公式
theta = np.arcsin(fw*lamba/d)               # 角度公式
angle = theta*180/np.pi

print(f'目标距离：{R} m\n', )
print(f'目标速度：{v} m/s\n', )
print(f'目标角度：{angle}°\n',)






















