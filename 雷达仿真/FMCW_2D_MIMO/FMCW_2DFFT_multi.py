#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:20:08 2025

@author: jack

利用Python实现FMCW雷达的距离多普勒估计(2D-FFT, 距离FFT，速度FFT)
https://blog.csdn.net/caigen0001/article/details/108815569
干货 | 利用MATLAB实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s?__biz=MzI2NzE1MTU3OQ==&mid=2649214285&idx=1&sn=241742b17b557c433ac7f5010758cd0f&chksm=f2905cf9c5e7d5efc16e84cab389ac24c5561a73d27fb57ca4d0bf72004f19af92b013fbd33b&scene=21#wechat_redirect
干货 | 利用Python实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s/X8uYol6cWoWAX6aUeR7S2A

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
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

#%% (一)接收方直接用理论上精确的差频信号(exp形式)做处理，FFT先做距离维再做速度维，手动完成FFT
# 利用Python实现FMCW雷达的距离多普勒估计(2D-FFT, 距离FFT，速度FFT)
# https://blog.csdn.net/caigen0001/article/details/108815569
# 干货 | 利用MATLAB实现FMCW雷达的距离多普勒估计:
# https://mp.weixin.qq.com/s?__biz=MzI2NzE1MTU3OQ==&mid=2649214285&idx=1&sn=241742b17b557c433ac7f5010758cd0f&chksm=f2905cf9c5e7d5efc16e84cab389ac24c5561a73d27fb57ca4d0bf72004f19af92b013fbd33b&scene=21#wechat_redirect
# 干货 | 利用Python实现FMCW雷达的距离多普勒估计:
# https://mp.weixin.qq.com/s/X8uYol6cWoWAX6aUeR7S2A

c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 512          # ADC采样点数
Nchirp = 256      # chirp数量
Fs = Ns/Tc        # = 1/(t[1] - t[0])     # 模拟信号采样频率
lamba = c/f0      # 波长
NumRangeFFT = Ns*2                        # Range FFT Length
NumDopplerFFT = Nchirp*2                  # Doppler FFT Length
rangeRes = c/(2*B)                        # 距离分辨率
maxRange = rangeRes * Ns                  # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)        # 速度分辨率
maxVel = velRes * Nchirp/2                # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2

tarR = [100, 200]  # 目标距离
tarV = [3, -20]    # 目标速度

# generate receive signal
sigReceive  = np.zeros((Nchirp, Ns), dtype = complex)
N = np.arange(Ns)
t = np.linspace(0, Tc, Ns)
for l in range(Nchirp):
    for k in range(len(tarR)):
        ## 1
        # sigReceive[l,:] += np.exp(1j * 2 * np.pi * ((2 * B * (tarR[k] + tarV[k] * Tc * l)/(c * Tc) + 2 * f0 * tarV[k]/c) * (Tc/Ns) * N + 2 * f0 * (tarR[k] + tarV[k] * Tc * l)/c))
        ## 2
        sigReceive[l,:] += np.exp(1j * 2 * np.pi * ((2 * B * (tarR[k] + tarV[k] * Tc * l)/(c * Tc) + 2 * f0 * tarV[k]/c) * t + 2 * f0 * (tarR[k] + tarV[k] * Tc * l)/c))
        ## 3
        # tau = 2*(tarR[k] + tarV[k] * Tc * l) / c
        # sigReceive[l,:] += np.exp(1j * 2 * np.pi * (f0 * tau + S * tau * t - S/2*tau**2))

# Range win processing
sigRangeWin = np.zeros((Nchirp, Ns), dtype = complex)
for l in range(0, Nchirp):
    sigRangeWin[l] = np.multiply(sigReceive[l], np.hamming(Ns).T)

# Range fft processing
sigRangeFFT = np.zeros((Nchirp, NumRangeFFT), dtype = complex)
for l in range(0, Nchirp):
    sigRangeFFT[l] = np.fft.fft(sigRangeWin[l], NumRangeFFT)

# Doppler win processing
sigDopplerWin = np.zeros((Nchirp, NumRangeFFT), dtype = complex)
for n in range(0, Ns):
    sigDopplerWin[:, n] = np.multiply(sigRangeFFT[:, n], np.hamming(Nchirp).T)

# Doppler fft processing
sigDopplerFFT = np.zeros((NumDopplerFFT, NumRangeFFT), dtype = complex)
for n in range(0, Ns):
    sigDopplerFFT[:, n] = np.fft.fftshift(np.fft.fft(sigDopplerWin[:, n], NumDopplerFFT))

x = np.arange(NumRangeFFT) / NumRangeFFT * maxRange
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)
Z = np.abs(sigDopplerFFT)

fig = plt.figure(figsize=(10, 20) )
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z/1e5, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.invert_xaxis()  #x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 10)
ax1.set_ylabel('Velocity(m/s)', fontsize = 10)
ax1.set_zlabel('Amplitude', fontsize = 10)
ax1.set_title('DopplerFFT', fontsize = 10)
ax1.view_init(azim=-135, elev=30)

ax2 = fig.add_subplot(122, projection='3d' )
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

#%% (二) 接收方直接用理论上精确的差频信号(exp形式)做处理，FFT先做距离维再做速度维.
# 直接用FFT2D替换 “干货 | 利用Python实现FMCW雷达的距离多普勒估计”
import numpy as np
import matplotlib.pyplot as plt
import scipy

c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 512          # ADC采样点数
Nchirp = 256      # chirp数量
Fs = Ns/Tc        # = 1/(t[1] - t[0])          # 模拟信号采样频率
lamba = c/f0      # 波长
NumRangeFFT = Ns*2                         # Range FFT Length
NumDopplerFFT = Nchirp*2                   # Doppler FFT Length
rangeRes = c/(2*B)                         # 距离分辨率
maxRange = rangeRes * Ns                   # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)         # 速度分辨率
maxVel = velRes * Nchirp/2                 # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2

tarR = [100, 200]   # 目标距离
tarV = [3, -20]     # 目标速度

# 接收差频信号
sigReceive  = np.zeros((Nchirp, Ns), dtype = complex)
N = np.arange(Ns)
t = np.linspace(0, Tc, Ns)
for l in range(Nchirp):
    for k in range(len(tarR)):
        ## 1
        # sigReceive[l,:] += np.exp(1j * 2 * np.pi * ((2 * B * (tarR[k] + tarV[k] * Tc * l)/(c * Tc) + 2 * f0 * tarV[k]/c) * (Tc/Ns) * N + 2 * f0 * (tarR[k] + tarV[k] * Tc * l)/c))
        ## 2
        sigReceive[l,:] += np.exp(1j * 2 * np.pi * ((2 * B * (tarR[k] + tarV[k] * Tc * l)/(c * Tc) + 2 * f0 * tarV[k]/c) * t + 2 * f0 * (tarR[k] + tarV[k] * Tc * l)/c))
        ## 3
        # tau = 2*(tarR[k] + tarV[k] * Tc * l) / c
        # sigReceive[l,:] += np.exp(1j * 2 * np.pi * (f0 * tau + S * tau * t - S/2*tau**2))

x = np.arange(NumRangeFFT) / NumRangeFFT * maxRange
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)

# Z = np.abs(scipy.fft.fftshift(np.fft.fft2(sigReceive.T, (NumRangeFFT, NumDopplerFFT)), axes = 1))
# Z = Z.T

# or
Z = np.abs(scipy.fft.fftshift(np.fft.fft2(sigReceive, (NumDopplerFFT, NumRangeFFT)), axes = 0))

fig = plt.figure(figsize = (10, 20) )
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z/1e5, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
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

#%%  (三) 接收方使用混频技术得到差频信号(exp形式)做后续处理, FFT先做距离维再做速度维, 直接用FFT2D完成, 重点看这段
import numpy as np
import matplotlib.pyplot as plt
import scipy
# from mpl_toolkits.mplot3d import Axes3D

c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 1024         # ADC采样点数
Nchirp = 256      # chirp数量
lamba = c/f0      # 波长
Fs = Ns/Tc        # = 1/(t[1] - t[0])     # 模拟信号采样频率
NumRangeFFT = Ns                           # Range FFT Length
NumDopplerFFT = Nchirp                     # Doppler FFT Length
rangeRes = c/(2*B)                        # 距离分辨率
maxRange = rangeRes * Ns                  # 雷达最大探测目标的距离, R_max = c*fs/(2*S) = c*Ns/(2S*Tchirp) = C*Ns/(2*B) = rangeRes * Ns
velRes = lamba / (2 * Nchirp * Tc)        # 速度分辨率
maxVel = velRes * Nchirp/2                # 雷达最大检测目标的速度, Vmax = lamba/(4*Tchirp) = lamba/(2*Nchirp*Tchirp) * Nchirp/2 = velRes * Nchirp/2

print(f"rangeRes = {rangeRes:.4f}, maxRange = {maxRange:.4f}, velRes = {velRes:.4f}, maxVel = {maxVel:.4f} ")

tarR = [100, 200]  # 目标距离
tarV = [10, -20]     # 目标速度
sigma = [0.1, 1.1 ]    # 高斯白噪声标准差

tarR = [10, 100, 200, 400]         # 目标距离,注意，上面的maxRange是2*Fs采样率下的最大探测距离，也就是Ns下的最大探测距离，而根据奈奎斯特采样定理，Fs采样率下的最大距离应该是maxRange/2,所以这里的tarR的最大值不能超过maxRange/2，否则会出现很奇怪的现象，可以改变参数试试。
tarV = [0, 10, -20, 40]            # 目标速度
sigma = [0.1, 0.1, 0.1, 0.1]       # 高斯白噪声标准差
# sigma = [1.1, 1.1, 1.1, 1.1]     # 高斯白噪声标准差

f_IFmax = (S*2*maxRange)/c     # 最高中频频率=Fs,
f_IF = (S*2*max(tarR))/c       # 当前中频频率

print(f"Fs = {Fs}, f_IFmax = {f_IFmax}, f_IF = {f_IF}")

t = np.linspace(0, Tc, Ns)
# t = np.arange(0, Tc, 1/Fs)
ft = f0 * t + S / 2 * t**2
Sx = np.exp(1j * 2 * np.pi * ft)          # 发射信号

##>>>>>>>>>>  发射信号FFT
fs = 4*(f0+B)
ts = 1/fs
t1 = np.arange(0, Tc-ts, ts)
ft1 = f0 * t1 + S / 2 * t1**2
Sx1 = np.exp(1j * 2 * np.pi * ft1)        # 发射信号
f, _, A, _, _, _  = freqDomainView(Sx1, fs, type = 'double')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f, A)
axs.set_xlim(0.76e11, 0.78e11)
axs.set_title("Sx Signal FFT")
axs.set_xlabel("Frequency")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()

##>>>>>>>>>> 接收差频信号 FFT
Rx  = np.zeros((Nchirp, Ns), dtype = complex)
N = np.arange(Ns)
for l in range(Nchirp):
    for k in range(len(tarR)):
        d = tarR[k] + tarV[k] * (t + l * Tc)
        # d = tarR[k] + tarV[k] * (l * Tc)
        tau = 2 * d / c                                # 运动目标的时延是动态变化的
        fr = f0 * (t + tau) + S / 2 * (t + tau)**2
        noise = (np.random.randn(*Sx.shape) + 1j * np.random.randn(*Sx.shape)) * np.sqrt(sigma[k])
        Rx[l, :] += (np.exp(1j * 2 * np.pi * fr) + noise )

##>>>>>>>>>>> 混频
Mix = np.conjugate(Sx) * Rx # 混频
f, _, A, _, _, _  = freqDomainView(Mix[1, 0:Ns], Fs, type = 'single')
fig, axs = plt.subplots(1, 1, figsize = (4, 3), constrained_layout = True)
axs.plot(f*c/(2*S), A)
axs.set_title("Mix Signal FFT")
axs.set_xlabel("Frequency->Distance")
axs.set_ylabel("Amplitude")
plt.show()
plt.close()

##>>>>>>>>>>> 距离-速度二维FFT
x = np.arange(int(NumRangeFFT/2)) / NumRangeFFT * maxRange
# x = np.arange(NumRangeFFT) / NumRangeFFT * maxRange  # 如果使用这行，则注释掉Z = Z[:, 0:int(NumRangeFFT/2)]
# y = np.arange((-Nchirp/2)*velRes, (Nchirp/2)*velRes, velRes)
y = np.linspace(-maxVel, maxVel, NumDopplerFFT)
X, Y = np.meshgrid(x, y)

# Z = np.abs(scipy.fft.fftshift(np.fft.fft2(Mix.T, (NumRangeFFT, NumDopplerFFT)), axes = 1))
# Z = Z.T
## or
Z = np.abs(scipy.fft.fftshift(np.fft.fft2(Mix , (NumDopplerFFT, NumRangeFFT)), axes = 0))

Z = Z[:, 0:int(NumRangeFFT/2)]
fig = plt.figure(figsize = (10, 20) )
ax1 = fig.add_subplot(121, projection = '3d')
ax1.plot_surface(X, Y, Z/1e4, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.invert_xaxis()                    # x轴反向
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range(m)', fontsize = 10)
ax1.set_ylabel('Velocity(m/s)', fontsize = 10)
ax1.set_zlabel('Amplitude', fontsize = 10)
ax1.set_title('DopplerFFT', fontsize = 10)
ax1.view_init(azim=-135,    elev = 30)

ax2 = fig.add_subplot(122, projection = '3d' )
ax2.plot_surface(X, Y, Z/1e4, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.set_xlabel('Range(m)', fontsize = 10)
ax2.set_ylabel('Velocity(m/s)', fontsize = 10)
# ax2.set_zlabel('Amplitude', fontsize = 10)
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

#%% 接收方使用混频技术得到差频信号(cos形式)做后续处理，FFT先做距离维再做速度维，直接用FFT2D完成， 重点看这段
import numpy as np
import matplotlib.pyplot as plt
import scipy
# from mpl_toolkits.mplot3d import Axes3D

c = 3e8           # Speed of Light
f0 = 77e9         # Start Frequency
B = 150e6         # 发射信号带宽
Tc = 20e-6        # 扫频时间
S = B/Tc          # 调频斜率
Ns = 512          # ADC采样点数
Nchirp = 256      # chirp数量
lamba = c/f0      # 波长
Fs = Ns/Tc        # = 1/(t[1] - t[0])          # 模拟信号采样频率
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

tarR = [10, 100, 150, 200]         # 目标距离
tarV = [0, 10, -20, 20]            # 目标速度
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
        # d = tarR[k] + tarV[k] * (l * Tc)
        tau = 2 * d / c          # 运动目标的时延是动态变化的
        fr = f0 * (t + tau) + S / 2 * (t + tau)**2
        noise = np.random.randn(*Sx.shape) * np.sqrt(sigma[k])
        Rx[l, :] += (np.cos(2 * np.pi * fr) + noise )
##>>>>>>>>>>> 混频
Mix = Sx * Rx
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
fig = plt.figure(figsize = (10, 20) )
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z/1e4, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
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




#%%



#%%































