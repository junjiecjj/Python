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

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

# %% Radar parameters
c = 3e8                    # speed of light
BW = 150e6                  # bandwidth 有效
fc = 77e9                  # carrier frequency
numADC = 256               # of adc samples
numChirps = 256            # of chirps per frame
numCPI = 10
T = 10e-6                    # PRI，默认不存在空闲时间
PRF = 1/T
Fs = numADC/T                # sampling frequency
dt = 1/Fs                    # sampling interval
slope = BW/T
lamba = c/fc
N = numChirps*numADC*numCPI                       # total of adc samples
t = np.linspace(0, T*numChirps*numCPI, N)         # time axis, one frame 等间隔时间/点数
t_onePulse = np.arange(0, dt*numADC, dt)          # 单chirp时间
numTX = 1
numRX = 8                             # 等效后
dR = c/(2*BW)                         # range resol
Rmax = Fs*c/(2*slope)                 # TI's MIMO Radar doc
# Rmax2 = c/2/PRF                       # lecture 2.3
dV = lamba/(2*numChirps*T)            # velocity resol, lambda/(2*framePeriod)
Vmax = lamba/(4*T)                    # Max Unamb velocity m/s
# DFmax = 1/2*PRF                       # = Vmax/(c/f0/2); % Max Unamb Dopp Freq
d_rx = lamba/2                        # dist. between rxs
d_tx = 4*d_rx                         # dist. between txs
N_Dopp = numChirps                    # length of doppler FFT
N_range = numADC                      # length of range FFT
N_azimuth = numTX*numRX
R = np.arange(0, Rmax, dR)                   # range axis
V = np.linspace(-Vmax, Vmax, numChirps)      # Velocity axis
ang_ax = np.arange(-90, 91)                  # angle axis

# %%目标参数
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
tar1_loc = np.zeros((t.size,3))
tar1_loc[:,0] = r1[0] + v1_x*t
tar1_loc[:,1] = r1[1] + v1_y*t
tar2_loc = np.zeros((t.size, 3))
tar2_loc[:,0] = r2[0] + v2_x*t
tar2_loc[:,1] = r2[1] + v2_y*t

#%% TX signal
delays_tar1 = np.zeros((numTX, numRX, N))
delays_tar2 = np.zeros((numTX, numRX, N))
r1_at_t = np.zeros((numTX, numRX, N))
r2_at_t = np.zeros((numTX, numRX, N))
tar1_angles = np.zeros((numTX, numRX, N))
tar2_angles = np.zeros((numTX, numRX, N))
tar1_velocities = np.zeros((numTX, numRX, N))
tar2_velocities = np.zeros((numTX, numRX, N))
for i in range(numTX):
    for j in range(numRX):
        delays_tar1[i,j,:] = (np.linalg.norm(tar1_loc - np.tile(rx_loc[j], (N, 1)), ord = 2, axis = 1) +  np.linalg.norm(tar1_loc - np.tile(tx_loc[i], (N, 1)), ord = 2, axis = 1))/c
        delays_tar2[i,j,:] = (np.linalg.norm(tar2_loc - np.tile(rx_loc[j], (N, 1)), ord = 2, axis = 1) + np.linalg.norm(tar2_loc - np.tile(tx_loc[i], (N, 1)), ord = 2, axis = 1))/c

#%% 接收信号模型 Complex signal
phase = lambda tx, fx: 2*np.pi*(fx*tx+slope/2*tx**2)                               # transmitted
phase2 = lambda tx, fx, r, v: 2*np.pi*(2*fx*r/c + tx*(2*fx*v/c + 2*slope*r/c))     # downconverted

fr1 = 2*r1[1]*slope/c
fr2 = 2*r2[1]*slope/c
fd1 = 2*v1_radial*fc/c # doppler freq
fd2 = 2*v2_radial*fc/c
f_if1 = fr1 + fd1      # beat or IF freq
f_if2 = fr2 + fd2

mixed = np.zeros((numTX, numRX, N), dtype = complex)
for i in range(numTX):
    for j in range(numRX):
        for k in range(numChirps*numCPI):
            signal_1 = np.zeros(N, dtype = complex)
            signal_2 = np.zeros(N, dtype = complex)
            phase_t = phase(t_onePulse, fc)
            phase_1 = phase(t_onePulse-delays_tar1[i,j][int(k*numADC)], fc)      # received
            phase_2 = phase(t_onePulse-delays_tar2[i,j][int(k*numADC)], fc)
            # signal_t[int(k*numADC):int((k+1)*numADC)] = np.exp(1j*phase_t)
            signal_1[int(k*numADC):int((k+1)*numADC)] = np.exp(1j*(phase_t - phase_1))
            signal_2[int(k*numADC):int((k+1)*numADC)] = np.exp(1j*(phase_t - phase_2))
        mixed[i,j] = signal_1 + signal_2

#%%  五、2D-FFT
RDC = reshape(cat(3,mixed{:}),numADC,numChirps*numCPI,numRX*numTX); % radar data cube
RDMs = zeros(numADC,numChirps,numTX*numRX,numCPI);
for i in range(numCPI):
    RD_frame = RDC(:,(i-1)*numChirps+1:i*numChirps,:);
    RDMs(:,:,:,i) = fftshift(fft2(RD_frame,N_range,N_Dopp),2);

figure(2);
imagesc(V,R,20*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1))))));
colormap(jet(256))

clim = get(gca,'clim');
caxis([clim(1)/2 0])
xlabel('Velocity (m/s)');
ylabel('Range (m)');





#%% 接收天线位置
































