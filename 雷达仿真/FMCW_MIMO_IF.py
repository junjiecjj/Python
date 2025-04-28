#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:09:23 2025

@author: jack
% https://blog.csdn.net/qq_35844208/article/details/128547667

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cp

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

#%% Radar parameters
c = 3e8                      # speed of light
BW = 150e6                   # bandwidth 有效
fc = 77e9                    # carrier frequency
numADC = 256                 # of adc samples
numChirps = 128              # of chirps per frame
numCPI = 10                  # 10帧
Tc = 10e-6                   # PRI, 默认不存在空闲时间
PRF = 1/Tc                   #
Fs = numADC/Tc               # sampling frequency
dt = 1/Fs                    # sampling interval
slope = BW/Tc
lamba = c/fc
N = numCPI*numChirps*numADC                       # total of adc samples
t = np.linspace(0, Tc*numCPI*numChirps, N)        # time axis, one frame 等间隔时间/点数
t_onePulse = np.arange(0, dt*numADC, dt)          # 单chirp时间
numTX = 1
numRX = 8                             # 等效后
dR = c/(2*BW)                         # range resol
Rmax = Fs*c/(2*slope)                 # TI's MIMO Radar doc
dV = lamba/(2*numChirps*Tc)           # velocity resol, lambda/(2*framePeriod)
Vmax = lamba/(4*Tc)                   # Max Unamb velocity m/s
d_rx = lamba/2                        # dist. between rxs
d_tx = 4*d_rx                         # dist. between txs
N_Dopp = numChirps                    # length of doppler FFT
N_range = numADC                      # length of range FFT
N_azimuth = numTX*numRX
R = np.arange(0, Rmax, dR)                   # range axis
V = np.linspace(-Vmax, Vmax, numChirps)      # Velocity axis
ang_ax = np.arange(-90, 91)                  # angle axis

#%% 发射天线位置
tx_loc = []
for i in range(numTX):
    tx_loc.append([i*d_tx, 0, 0])
tx_loc = np.array(tx_loc)
## 接收天线位置
rx_loc = []
for i in range(numRX):
   rx_loc.append([tx_loc[-1][0] + d_tx + i*d_rx, 0, 0])
rx_loc = np.array(rx_loc)

#%% 目标位置
R_radial   = [100, 200]
V_radial   = [10, -25]
Ang_radial = [-30, 30]
numTarget  = len(R_radial)
tar_loc = np.zeros((numTarget, t.size, 3))
for k in range(numTarget):
    r_x = np.sin(Ang_radial[k] * np.pi / 180) * R_radial[k]
    r_y = np.cos(Ang_radial[k] * np.pi / 180) * R_radial[k]
    v_x =  np.sin(Ang_radial[k] * np.pi / 180) * V_radial[k]
    v_y =  np.cos(Ang_radial[k] * np.pi / 180) * V_radial[k]
    tar_loc[k, :, 0] = r_x + v_x * t
    tar_loc[k, :, 1] = r_y + v_y * t

#%% TX signal
# 利用收发天线的位置以及目标参数中雷达的位置信息，先求目标到雷达的2-范数（也就是空间中两点的直线距离），然后转化为目标的延迟时间τ，如此以来得到的信号模型精度更高
delays_tar = np.zeros((numTX, numRX, numTarget, N))
for i in range(numTX):
    for j in range(numRX):
        for k in range(numTarget):
            delays_tar[i,j,k,:] = (np.linalg.norm(tar_loc[k] - np.tile(rx_loc[j], (N, 1)), ord = 2, axis = 1) + np.linalg.norm(tar_loc[k] - np.tile(tx_loc[i], (N, 1)), ord = 2, axis = 1))/c

#%% 接收信号模型 Complex signal
# phase = lambda tx, fx: 2*np.pi*(fx*tx+slope/2*tx**2)                               # transmitted
# phase2 = lambda tx, fx, r, v: 2*np.pi*(2*fx*r/c + tx*(2*fx*v/c + 2*slope*r/c))     # downconverted
# phase_t = phase(t_onePulse, fc)
# # 这里，接收信号没有采用发射与接收混频的形式，而是相位直接做差，分别计算两个目标的中频信号相加，此法等效为混频
# mixed = np.zeros((numTX, numRX, N), dtype = complex)
# for i in range(numTX):
#     for j in range(numRX):
#         signal = np.zeros(N, dtype = complex)
#         for k in range(numChirps*numCPI):
#             for u in range(numTarget):
#                 Phase = phase(t_onePulse - delays_tar[i,j,u,:][int((k+1)*numADC)-1], fc)      # received
#                 signal[int(k*numADC):int((k+1)*numADC)] += np.exp(1j*(phase_t - Phase))
#         mixed[i, j] = signal

#%% add noise
# signal_power = np.sum(np.abs(mixed)**2)/mixed.size
# SNRdB = 100
# noise_pow = signal_power / (10.0**(SNRdB/10.0))
# mixed = mixed + np.sqrt(noise_pow/2) * (np.random.randn(*mixed.shape) + 1j * np.random.randn(*mixed.shape))

#%% IF
# mixed1 = mixed.transpose(1, 0, 2).reshape(numRX*numTX, N)
# RDC = mixed1.reshape(numRX*numTX, numChirps*numCPI, numADC).transpose(2, 1, 0)  # radar data cube

#%% 五、速度-距离  FFT-2D
# RDMs = np.zeros((numADC, numChirps, numTX*numRX, numCPI), dtype = complex)
# for i in range(numCPI):
#     RD_frame = RDC[:, int(i*numChirps): int((i+1)*numChirps), :]
#     RDMs[:,:,:,i] = scipy.fft.fftshift(np.fft.fft2(RD_frame, (N_range, N_Dopp), axes=(0, 1)), axes = 1)

#%% Rx * Tx
ft = fc * t_onePulse + slope / 2 * t_onePulse**2
Sx = np.exp(1j * 2 * np.pi * ft)                  # 发射信号

## Rx
RDC = np.zeros((numADC, numChirps, numCPI, numTX*numRX), dtype = complex)   # radar data cubic
RDMs = np.zeros((numADC, numChirps, numCPI, numTX*numRX), dtype = complex)  # range-doppler FFT
Rx = np.zeros((numADC, numChirps, numCPI, numTX*numRX), dtype = complex)
N = numCPI*numChirps*numADC                                                 # total of adc samples
t = np.linspace(0, Tc*numCPI*numChirps, N)                                  # time axis, one frame 等间隔时间/点数
for i in range(numTX):
    for j in range(numRX):
        Annt = j*numTX + i
        for ncpi in range(numCPI):
            for ncip in range(numChirps):
                for k in range(numTarget):
                    Idx = ncpi*(numADC*numChirps) + ncip*numADC              # 运动目标的时延是动态变化的
                    tau = delays_tar[i,j,k,Idx]
                    fr = fc * (t_onePulse + tau) + slope / 2 * (t_onePulse + tau)**2
                    Rx[:, ncip, ncpi, Annt] += np.exp(1j * 2 * np.pi * fr )

## IF
for i in range(numTX):
    for j in range(numRX):
        Annt = j*numTX + i
        for ncpi in range(numCPI):
            for ncip in range(numChirps):
                RDC[:,ncip, ncpi, Annt] = Rx[:,ncip, ncpi, Annt] * np.conjugate(Sx)

for i in range(numCPI):
    RD_frame = RDC[:,:,i,:]
    RDMs[:,:,i,:] = scipy.fft.fftshift(np.fft.fft2(RD_frame, (N_range, N_Dopp), axes=(0, 1)), axes = 1)

#%%  range-Dopples 2D-plot
tmp = np.sum(np.abs(RDMs), axis = (-2, -1))
# tmp = np.abs(RDMs[:,:,0,0])
fig = plt.figure(figsize = (10, 16), constrained_layout = True)
ax1 = fig.add_subplot(211, )
im = ax1.imshow(10*np.log10(tmp/ tmp.max()), aspect='auto', cmap='jet', extent=[V[0], V[-1], R[-1], R[0]])
# im = ax1.imshow(tmp, aspect='auto', cmap='jet', extent=[V[0], V[-1], R[-1], R[0]])
ax1.set_xlabel('速度 (m/s)')
ax1.set_ylabel('距离 (m)')
ax1.set_title('距离-速度图')
cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'

VV, RR = np.meshgrid(V, R)
ax2 = fig.add_subplot(212, projection = '3d' )
ax2.plot_surface(VV, RR, 10*np.log10(tmp/ tmp.max()), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
# ax2.plot_surface(VV, RR, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.set_proj_type('ortho')
ax2.set_xlabel('速度 (m/s)', fontsize = 20)
ax2.set_ylabel('距离 (m)', fontsize = 20)
ax2.set_zlabel('Amplitude', fontsize = 10)
ax2.set_title('距离-速度图', fontsize = 20)
ax2.set_zticks([])
ax2.view_init(azim = -135, elev = 30)

plt.show()
plt.close()

#%% 六、速度-距离 CA-CFAR
def ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET):
    # e.g. numGuard =2, numTrain =2*numGuard, P_fa =1e-5, SNR_OFFSET = -15
    numTrain2D = numTrain*numTrain - numGuard*numGuard
    RDM_mask = np.zeros_like(RDM_dB)
    for r in range(numTrain + numGuard, RDM_mask.shape[0] - (numTrain + numGuard)):
        for d in range(numTrain + numGuard, RDM_mask.shape[1] - (numTrain + numGuard)):
            Pn = (np.sum(RDM_dB[r-(numTrain+numGuard):r+(numTrain+numGuard)+1, d-(numTrain+numGuard):d+(numTrain+numGuard)+1]) - np.sum(RDM_dB[r-numGuard:r+numGuard+1, d-numGuard:d+numGuard+1]))/ numTrain2D                     # noise level
            a = numTrain2D*(P_fa**(-1/numTrain2D)-1)     # scaling factor of T = α*Pn
            threshold = a*Pn
            if (RDM_dB[r, d] > threshold) and (RDM_dB[r, d] > SNR_OFFSET):
                RDM_mask[r, d] = 1
    cfar_ranges, cfar_dopps = np.where(RDM_mask != 0)     # cfar detected range bins
    ## remaining part is for target location estimation
    remove_idx = []
    for i in range(1, cfar_ranges.size):
       if (np.abs(cfar_ranges[i] - cfar_ranges[i-1]) <= 5) and (np.abs(cfar_dopps[i] - cfar_dopps[i-1]) <= 5):
           remove_idx.append(i)
    ranges = np.delete(cfar_ranges, remove_idx)
    vels = np.delete(cfar_dopps, remove_idx)
    K = ranges.size                    # of detected targets
    return RDM_mask, ranges, vels, K

numGuard = 2            # of guard cells
numTrain = numGuard*2   # of training cells
P_fa = 1e-5;            #  desired false alarm rate
SNR_OFFSET = -5;        #  dB
# tmp = np.sum(np.abs(RDMs), axis = (-2, -1))
tmp = np.abs(RDMs[:,:,0,0])
RDM_dB = 10*np.log10(tmp/ tmp.max())
RDM_mask, cfar_ranges, cfar_dopps, K = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)
print(f"K = {K}")
fig = plt.figure(figsize = (10, 16), constrained_layout = True)
ax1 = fig.add_subplot(211, )
im = ax1.imshow(RDM_mask, aspect = 'auto', cmap = 'jet', extent = [V[0], V[-1], R[-1], R[0]])
# ax1.scatter( V[cfar_dopps], R[cfar_ranges], s = 12, c = 'red')
ax1.set_xlabel('速度 (m/s)')
ax1.set_ylabel('距离 (m)')
ax1.set_title('距离-速度图')
cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', label = '强度 (dB)')

VV, RR = np.meshgrid(V, R)
ax2 = fig.add_subplot(212, projection = '3d' )
ax2.plot_surface(VV, RR, RDM_mask, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax2.grid(False)
ax2.set_proj_type('ortho')
ax2.set_xlabel('速度 (m/s)', fontsize = 20)
ax2.set_ylabel('距离 (m)', fontsize = 20)
ax2.set_zlabel('Amplitude', fontsize = 10)
ax2.set_title('距离-速度图', fontsize = 20)
ax2.set_zticks([])
ax2.view_init(azim = -135, elev = 30)

plt.show()
plt.close()

#%% 七、角度估计
## (一) 角度 3D-FFT
rangeFFT = scipy.fft.fft(RDC[:,:,0,:], N_range, axis = 0)
doppFFT = scipy.fft.fftshift(scipy.fft.fft(rangeFFT, N_Dopp, axis = 1), axes  = 1)
angleFFT = scipy.fft.fftshift(scipy.fft.fft(doppFFT, ang_ax.size, axis = 2), axes = 2)
range_az =  np.sum(np.abs(angleFFT), axis = 1)                          # range-azimuth map
tmp = 10*np.log10(np.abs(range_az)/np.abs(range_az).max())
Ang, RR = np.meshgrid(ang_ax, R)

fig = plt.figure(figsize=(10, 16) )
ax1 = fig.add_subplot(211, projection = '3d')
ax1.plot_surface(Ang, RR, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
# ax1.plot_surface(Ang, RR, range_az, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('Azimuth Angle', )
ax1.set_ylabel('Range (m)', )
ax1.set_zlabel('Amplitude', )
ax1.set_title('FFT Range-Angle Map', )
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(212, )
im = ax2.imshow(tmp, aspect='auto', cmap='jet', extent=[ang_ax[0], ang_ax[-1], R[-1], R[0]])
# im = ax2.imshow(range_az, aspect='auto', cmap='jet', extent=[ang_ax[0], ang_ax[-1], R[-1], R[0]])
ax2.set_xlabel('Azimuth Angle')
ax2.set_ylabel('Range (m)')
ax2.set_title('距离-Angle图')
cbar = fig.colorbar(im, ax = ax2, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'
plt.show()
plt.close()

## (一) 2D-FFT
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111,  )
doas = np.zeros((K, ang_ax.size), dtype = complex)       #  direction of arrivals
for k in range(K):
    doas[k, :] = scipy.fft.fftshift(scipy.fft.fft(doppFFT[int(cfar_ranges[k]), int(cfar_dopps[k]), :], ang_ax.size))
    Idxs = scipy.signal.find_peaks(np.abs(doas[k,:]) , np.max(np.abs(doas[k,:]))*0.5, distance = 10)[0]
    ax1.scatter(ang_ax[Idxs], 10*np.log10(np.abs(doas[k,:][Idxs])), s = 50, c = 'blue')
    ax1.plot(ang_ax, 10*np.log10(np.abs(doas[k,:])))
ax1.set_xlabel('Azimuth Angle', )
ax1.set_ylabel('dB', )
plt.show()
plt.close()

## (二) MUSIC算法
d = 0.5;
M = numCPI        # # of snapshots
a1 = np.zeros((numTX*numRX, ang_ax.size), dtype = complex)
tmp = np.arange(numTX*numRX)
for k in range(ang_ax.size):
        a1[:,k] = np.exp(-1j*2*np.pi*d*tmp*np.sin(ang_ax[k]*np.pi/180))
music_spectrum = np.zeros((K, ang_ax.size), dtype = complex)
Nannt = numTX*numRX
for k in range(K):
    Rxx = np.zeros((numTX*numRX, numTX*numRX), dtype = complex)
    for m in range(M):
       A =  RDMs[int(cfar_ranges[k]), int(cfar_dopps[k]), m, :]
       Rxx = Rxx + 1/M * (A[:,None]@A[:,None].conjugate().T)
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, 1:Nannt]
    UnUnH = Un@Un.conjugate().T
    for a in range(ang_ax.size):
        at = a1[:,a].reshape(-1,1)
        music_spectrum[k, a] = (at.conjugate().T@at)[0,0]/(at.T.conjugate()@UnUnH@at)[0,0]
colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111,  )
for k in range(K):
    ax1.plot(ang_ax, 10*np.log10(np.abs(music_spectrum[k,:])), color = colors[k])
    Idxs = scipy.signal.find_peaks(np.abs(music_spectrum[k,:]) , np.max(np.abs(music_spectrum[k,:]))*0.5, distance = 10)[0]
    ax1.scatter(ang_ax[Idxs], 10*np.log10(np.abs(music_spectrum[k,:][Idxs])), s = 50, c = 'blue')
ax1.set_xlabel('Angle in degrees', )
ax1.set_ylabel('dB', )
ax1.set_title("MUSIC Spectrum")
plt.show()
plt.close()

### MUSIC算法: or directly use the radar data cube
X = np.zeros((numTX*numRX, int(numChirps * numADC * numCPI)), dtype = complex)
for nr in range(numTX*numRX):
    X[nr,:] = RDC[:,:,:,nr].flatten()
Rxx = X @ X.T.conjugate() / int(numChirps * numADC * numCPI)
# 特征值分解
eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
eigvector = eigvector[:, idx]
eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
Un = eigvector[:, len(R_radial):numTX*numRX]

# Un = eigvector
UnUnH = Un @ Un.T.conjugate()
Thetalst = np.arange(-90, 90.1, 0.5)
angle = np.deg2rad(Thetalst)
Pmusic = np.zeros(angle.size)
for i, ang in enumerate(angle):
    a = np.exp(1j * np.pi * np.arange(numTX*numRX) * np.sin(ang)).reshape(-1, 1)
    Pmusic[i] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]

Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
Pmusic = 10 * np.log10(Pmusic)
peaks, _ =  scipy.signal.find_peaks(Pmusic, height = -10, distance = 10)
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

## (三) 点云生成
I = music_spectrum[1,:].argmax()
angle1 = ang_ax[I]
I = music_spectrum[0,:].argmax()
angle2 = ang_ax[I]
coor1 = [cfar_ranges[1]*np.sin(angle1*np.pi/180), cfar_ranges[1]*np.cos(angle1*np.pi/180), 0];
coor2 = [cfar_ranges[0]*np.sin(angle2*np.pi/180), cfar_ranges[0]*np.cos(angle2*np.pi/180), 0];
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111, projection = '3d')
ax1.scatter([coor1[0], coor2[0]], [coor1[1], coor2[1]], [coor1[2], coor2[2]])
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('Range (m) X', )
ax1.set_ylabel('Range (m) Y', )
ax1.set_zlabel('Range (m) Z', )
# ax1.set_title('FFT Range-Angle Map', )
ax1.view_init(azim = -135, elev = 30)

plt.show()
plt.close()

## (四) MUSIC 距离-AOA谱
range_az_music = np.zeros((N_range, ang_ax.size), dtype = complex)
rangeFFT1 = scipy.fft.fft(RDC, axis = 0)
for i in range(N_range):
    Rxx = np.zeros((numTX*numRX, numTX*numRX), dtype = complex)
    for m in range(M):
       A =  np.sum(rangeFFT1[i, :, m, :], 0)
       Rxx = Rxx + 1/M * (A[:,None]@A[:,None].conjugate().T)
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, 1:N]
    UnUnH = Un@Un.conjugate().T
    for a in range(ang_ax.size):
        at = a1[:,a].reshape(-1,1)
        range_az_music[i,a] = (at.conjugate().T@at)[0,0]/(at.T.conjugate()@UnUnH@at)[0,0]
colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 4) )
ax1 = fig.add_subplot(111, )
im = ax1.imshow(20*np.log10(np.abs(range_az_music)/np.abs(range_az_music).max()), aspect = 'auto', cmap = 'jet', extent = [ang_ax[0], ang_ax[-1], R[-1], R[0]])
ax1.set_xlabel('Azimuth')
ax1.set_ylabel('Range (m)')
ax1.set_title('MUSIC Range-Angle Map')
cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', label = '强度 (dB)')
plt.show()
plt.close()

## (五)压缩感知
numTheta = ang_ax.size      # divide FOV into fine grid
B = a1                      # steering vector matrix or dictionary, also called basis matrix

colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111, )
for k in range(K):
    A = RDMs[int(cfar_ranges[k]), int(cfar_dopps[k]),1, :][:,None]
    s = cp.Variable((numTheta,1), complex = True)
    objective = cp.Minimize(cp.norm(s, 1))
    constraints = [cp.norm(A - B@s, 2) <= 1,]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print("status:", prob.status)
    S = np.abs(s.value).flatten()
    ax1.plot(ang_ax, np.log10(S), color = colors[k])
    Idxs = scipy.signal.find_peaks(S , np.max(S)*0.5, distance = 10)[0]
    ax1.scatter(ang_ax[Idxs], np.log10(S)[Idxs], s = 50, c = 'blue')
ax1.set_xlabel('Azimuth')
ax1.set_ylabel('dB')
ax1.set_title('Angle Estimation with Compressed Sensing')

plt.show()
plt.close()


#%%

#%%

#%%

#%%































































































































