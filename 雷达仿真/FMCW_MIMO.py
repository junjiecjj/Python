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
numChirps = 256              # of chirps per frame
numCPI = 10                  # 10帧
T = 10e-6                     # PRI, 默认不存在空闲时间
PRF = 1/T                     #
Fs = numADC/T                 # sampling frequency
dt = 1/Fs                     # sampling interval
slope = BW/T
lamba = c/fc
N = numChirps*numADC*numCPI                       # total of adc samples
t = np.linspace(0, T*numChirps*numCPI, N)         # time axis, one frame 等间隔时间/点数
t_onePulse = np.arange(0, dt*numADC, dt)          # 单chirp时间
numTX = 1
numRX = 8                             # 等效后
dR = c/(2*BW)                         # range resol
Rmax = Fs*c/(2*slope)                 # TI's MIMO Radar doc
# Rmax2 = c/2/PRF                     # lecture 2.3
dV = lamba/(2*numChirps*T)            # velocity resol, lambda/(2*framePeriod)
Vmax = lamba/(4*T)                    # Max Unamb velocity m/s
# DFmax = 1/2*PRF                     # = Vmax/(c/f0/2); % Max Unamb Dopp Freq
d_rx = lamba/2                        # dist. between rxs
d_tx = 4*d_rx                         # dist. between txs
N_Dopp = numChirps                    # length of doppler FFT
N_range = numADC                      # length of range FFT
N_azimuth = numTX*numRX
R = np.arange(0, Rmax, dR)                   # range axis
V = np.linspace(-Vmax, Vmax, numChirps)      # Velocity axis
ang_ax = np.arange(-90, 91)                  # angle axis

#%% 目标参数
r1_radial = 50
v1_radial = 10     #  velocity 1
tar1_angle = -10
r1_x = np.sin(tar1_angle * np.pi / 180) * r1_radial
r1_y = np.cos(tar1_angle * np.pi / 180) * r1_radial
v1_x = np.sin(tar1_angle * np.pi / 180) * v1_radial
v1_y = np.cos(tar1_angle * np.pi / 180) * v1_radial
r1 = [r1_x, r1_y, 0]

r2_radial = 100
v2_radial = -15    #  velocity 2
tar2_angle = 10
r2_x = np.sin(tar2_angle * np.pi / 180) * r2_radial
r2_y = np.cos(tar2_angle * np.pi / 180) * r2_radial
v2_x = np.sin(tar2_angle * np.pi / 180) * v2_radial
v2_y = np.cos(tar2_angle * np.pi / 180) * v2_radial
r2 = [r2_x, r2_y, 0]

tar1_loc = np.zeros((t.size,3))
tar1_loc[:,0] = r1[0] + v1_x*t
tar1_loc[:,1] = r1[1] + v1_y*t
tar2_loc = np.zeros((t.size, 3))
tar2_loc[:,0] = r2[0] + v2_x*t
tar2_loc[:,1] = r2[1] + v2_y*t


#%% 发射天线位置
tx_loc = []
for i in range(numTX):
    tx_loc.append([i*d_tx, 0, 0])
tx_loc = np.array(tx_loc)
# 接收天线位置
rx_loc = []
for i in range(numRX):
   rx_loc.append([tx_loc[-1][0] + d_tx + i*d_rx, 0, 0])
rx_loc = np.array(rx_loc)

#%% TX signal
delays_tar1 = np.zeros((numTX, numRX, N))
delays_tar2 = np.zeros((numTX, numRX, N))
for i in range(numTX):
    for j in range(numRX):
        delays_tar1[i,j,:] = (np.linalg.norm(tar1_loc - np.tile(rx_loc[j], (N, 1)), ord = 2, axis = 1) + np.linalg.norm(tar1_loc - np.tile(tx_loc[i], (N, 1)), ord = 2, axis = 1))/c
        delays_tar2[i,j,:] = (np.linalg.norm(tar2_loc - np.tile(rx_loc[j], (N, 1)), ord = 2, axis = 1) + np.linalg.norm(tar2_loc - np.tile(tx_loc[i], (N, 1)), ord = 2, axis = 1))/c

#%% 接收信号模型 Complex signal
phase = lambda tx, fx: 2*np.pi*(fx*tx+slope/2*tx**2)                               # transmitted
phase2 = lambda tx, fx, r, v: 2*np.pi*(2*fx*r/c + tx*(2*fx*v/c + 2*slope*r/c))     # downconverted

# 这里，接收信号没有采用发射与接收混频的形式，而是相位直接做差，分别计算两个目标的中频信号相加，此法等效为混频
mixed = np.zeros((numTX, numRX, N), dtype = complex)
for i in range(numTX):
    for j in range(numRX):
        signal_1 = np.zeros(N, dtype = complex)
        signal_2 = np.zeros(N, dtype = complex)
        for k in range(numChirps*numCPI):
            phase_t = phase(t_onePulse, fc)
            phase_1 = phase(t_onePulse - delays_tar1[i,j][int((k+1)*numADC)-1], fc)      # received
            phase_2 = phase(t_onePulse - delays_tar2[i,j][int((k+1)*numADC)-1], fc)
            signal_1[int(k*numADC):int((k+1)*numADC)] = np.exp(1j*(phase_t - phase_1))
            signal_2[int(k*numADC):int((k+1)*numADC)] = np.exp(1j*(phase_t - phase_2))
        mixed[i, j] = signal_1 + signal_2

# ## add noise
# signal_power = np.sum(np.abs(mixed)**2)/mixed.size
# SNRdB = 10
# noise_pow = signal_power / (10.0**(SNRdB/10.0))
# mixed = mixed + np.sqrt(noise_pow/2) * (np.random.randn(*mixed.shape) + 1j * np.random.randn(*mixed.shape))

mixed1 = mixed.transpose(1, 0, 2).reshape(numRX*numTX, N)
RDC = mixed1.reshape(numRX*numTX, numChirps*numCPI, numADC).transpose(2, 1, 0)
#%%  五、2D-FFT
# RDC = mixed.reshape(numADC, numChirps*numCPI, numRX*numTX)     # radar data cube
RDMs = np.zeros((numADC, numChirps, numTX*numRX, numCPI), dtype = complex)
for i in range(numCPI):
    RD_frame = RDC[:, int(i*numChirps): int((i+1)*numChirps), :]
    RDMs[:,:,:,i] = scipy.fft.fftshift(np.fft.fft2(RD_frame, (N_range, N_Dopp), axes=(0, 1)), axes = 1)
tmp = np.sum(np.abs(RDMs), axis = (-2, -1))

# fig = plt.figure(figsize=(7, 6), constrained_layout = True)
fig = plt.figure(figsize = (10, 16), constrained_layout = True)
ax1 = fig.add_subplot(211, )
im = ax1.imshow(20*np.log10(tmp/ tmp.max()), aspect='auto', cmap='jet', extent=[V[0], V[-1], R[-1], R[0]])
# im = ax2.imshow(20*np.log10(np.abs(RDMs[:,:,0,0])/ np.abs(RDMs[:,:,0,0]).max()), aspect='auto', cmap='jet', extent=[V[0], V[-1], R[-1], R[0]])
ax1.set_xlabel('速度 (m/s)')
ax1.set_ylabel('距离 (m)')
ax1.set_title('距离-速度图')
cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'

VV, RR = np.meshgrid(V, R)
ax2 = fig.add_subplot(212, projection = '3d' )
ax2.plot_surface(VV, RR, 20*np.log10(tmp/ tmp.max()), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
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
    cfar_ranges = np.delete(cfar_ranges, remove_idx)
    cfar_dopps = np.delete(cfar_dopps, remove_idx)
    K = cfar_dopps.size                    # of detected targets
    return RDM_mask, cfar_ranges, cfar_dopps, K

#%% 六、CA-CFAR
numGuard = 2            # of guard cells
numTrain = numGuard*2   # of training cells
P_fa = 1e-5;            #  desired false alarm rate
SNR_OFFSET = -5;        #  dB
RDM_dB = 10*np.log10(np.abs(RDMs[:,:,1,1])/ np.abs(RDMs[:,:,1,1]).max())
RDM_mask, cfar_ranges, cfar_dopps, K = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)

fig = plt.figure(figsize = (10, 16), constrained_layout = True)
ax1 = fig.add_subplot(211, )
im = ax1.imshow(RDM_mask, aspect = 'auto', cmap = 'jet', extent = [V[0], V[-1], R[-1], R[0]])
cfar_ranges, cfar_dopps = np.where(RDM_mask != 0)
cfar_ranges = R[cfar_ranges]
cfar_dopps = V[cfar_dopps]
ax1.scatter(cfar_dopps, cfar_ranges, s = 12, c = 'red')
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
## （一）3D-FFT
rangeFFT = scipy.fft.fft(RDC[:,:numChirps,:], N_range, axis = 0)
angleFFT = scipy.fft.fftshift(scipy.fft.fft(rangeFFT, ang_ax.size, axis = 2), axes = 2)
range_az =  np.sum(angleFFT, axis = 1)            #  range-azimuth map
tmp = 20*np.log10(np.abs(range_az)/np.abs(range_az).max())
Ang, RR = np.meshgrid(ang_ax, R)

fig = plt.figure(figsize=(10, 16) )
ax1 = fig.add_subplot(211, projection = '3d')
ax1.plot_surface(Ang, RR, tmp, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
# ax1.plot_wireframe(X, Y, sig_fft, color = [0.5,0.5,0.5], linewidth = 0.25)
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('Azimuth Angle', )
ax1.set_ylabel('Range (m)', )
ax1.set_zlabel('Amplitude', )
ax1.set_title('FFT Range-Angle Map', )
ax1.view_init(azim = -135, elev = 30)

ax2 = fig.add_subplot(212, )
im = ax2.imshow(tmp, aspect='auto', cmap='jet', extent=[ang_ax[0], ang_ax[-1], R[-1], R[0]])
# im = ax2.imshow(20*np.log10(np.abs(RDMs[:,:,0,0])/ np.abs(RDMs[:,:,0,0]).max()), aspect='auto', cmap='jet', extent=[V[0], V[-1], R[-1], R[0]])
ax2.set_xlabel('Azimuth Angle')
ax2.set_ylabel('Range (m)')
ax2.set_title('距离-Angle图')
cbar = fig.colorbar(im, ax = ax2, orientation = 'vertical', label='强度 (dB)') # label='强度 (dB)'
plt.show()
plt.close()

fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111,  )
doas = np.zeros((K, ang_ax.size), dtype = complex)       #  direction of arrivals
for i in range(K):
    doas[i, :] = scipy.fft.fftshift(scipy.fft.fft(rangeFFT[int(cfar_ranges[i]), int(cfar_dopps[i]), :], ang_ax.size))
    Idxs = scipy.signal.find_peaks(np.abs(doas[i,:]) , np.max(np.abs(doas[i,:]))*0.5, distance = 10)[0]
    ax1.scatter(ang_ax[Idxs], 10*np.log10(np.abs(doas[i,:][Idxs])), s = 50, c = 'blue')
    ax1.plot(ang_ax, 10*np.log10(np.abs(doas[i,:])))
ax1.set_xlabel('Azimuth Angle', )
ax1.set_ylabel('dB', )
plt.show()
plt.close()

#% (二) MUSIC算法
d = 0.5;
M = numCPI        # # of snapshots
a1 = np.zeros((numTX*numRX, ang_ax.size), dtype = complex)
tmp = np.arange(numTX*numRX)
for k in range(ang_ax.size):
        a1[:,k] = np.exp(-1j*2*np.pi*d*tmp*np.sin(ang_ax[k]*np.pi/180))
music_spectrum = np.zeros((K, ang_ax.size), dtype = complex)
N = numTX*numRX
for k in range(K):
    Rxx = np.zeros((numTX*numRX, numTX*numRX), dtype = complex)
    for m in range(M):
       A =  RDMs[int(cfar_ranges[k]), int(cfar_dopps[k]),:, m]
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
        music_spectrum[k, a] = (at.conjugate().T@at)[0,0]/(at.T.conjugate()@UnUnH@at)[0,0]
colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111,  )
for k in range(K):
    ax1.plot(ang_ax, np.log10(np.abs(music_spectrum[k,:])), color = colors[k])
    Idxs = scipy.signal.find_peaks(np.abs(music_spectrum[k,:]) , np.max(np.abs(music_spectrum[k,:]))*0.5, distance = 10)[0]
    ax1.scatter(ang_ax[Idxs], np.log10(np.abs(music_spectrum[k,:][Idxs])), s = 50, c = 'blue')
ax1.set_xlabel('Angle in degrees', )
ax1.set_ylabel('dB', )
ax1.set_title("MUSIC Spectrum")
plt.show()
plt.close()

#% (三) 点云生成
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

#% （四）MUSIC 距离-AOA谱
range_az_music = np.zeros((N_range, ang_ax.size), dtype = complex)
rangeFFT = scipy.fft.fft(RDC, axis = 0)
for i in range(N_range):
    Rxx = np.zeros((numTX*numRX, numTX*numRX), dtype = complex)
    for m in range(M):
       A =  np.sum(rangeFFT[i,m*numChirps:(m+1)*numChirps,:], 0)
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

#%  Angle Estimation - (五)压缩感知
numTheta = ang_ax.size      # divide FOV into fine grid
B = a1                      # steering vector matrix or dictionary, also called basis matrix

colors = plt.cm.jet(np.linspace(0, 1, 3))
fig = plt.figure(figsize=(8, 6) )
ax1 = fig.add_subplot(111, )
for k in range(K):
    A = RDMs[int(cfar_ranges[k]), int(cfar_dopps[k]),:,1][:,None]
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































































































































