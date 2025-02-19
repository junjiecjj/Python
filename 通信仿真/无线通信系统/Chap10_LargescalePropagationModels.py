#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:32:46 2025

@author: jack
<Wireless communication systems in Matlab> Chap10

"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 20  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%% Program 10.2: Friis model test.m: Friis free space propagation model

def FriisModel(Pt_dBm, Gt_dBi, Gr_dBi, f, d = 20, L = 1, n = 2):
    # Pt_dBm = Transmitted power in dBm
    # Gt_dBi = Gain of the Transmitted antenna in dBi
    # Gr_dBi = Gain of the Receiver antenna in dBi
    # f = frequency of transmitted signal in Hertz
    # d = array of distances at which the loss needs to be calculated
    # L = Other System Losses, No Loss case L=1
    # n = path loss exponent (n=2 for free space)
    # Pr_dBm = Received power in dBm
    # PL_dB  = constant path loss factor (including antenna gains)
    # WiFi (IEEE 802.11n standard) transmission-reception system operating at f = 2.4 GHz or f = 5 GHz band with 0 dBm (1 mW ) output power from the transmitter
    lamba = 3*10**8 / f
    PL_dB = Gt_dBi + Gr_dBi + 20 * np.log10(lamba/(4*np.pi)) - 10 * n * np.log10(d) - 10 * np.log10(L)
    Pr_dBm = Pt_dBm + PL_dB
    return Pr_dBm, PL_dB

#-----------Input section------------------------
Pt_dBm = 0  # Input - Transmitted power in dBm
Gt_dBi = 1  # Gain of the Transmitted antenna in dBi
Gr_dBi = 1 # Gain of the Receiver antenna in dBi
d = 2.0 ** np.array([0,1,2,3,4,5]) # Array of input distances in meters
L = 1       # Other System Losses, No Loss case L=1
n = 2       # Path loss exponent for Free space
f = 2.4e9   #Transmitted signal frequency in Hertz (2.4G WiFi)
#----------------------------------------------------
Pr1_dBm, PL1_dB = FriisModel(Pt_dBm, Gt_dBi, Gr_dBi, f, d, L, n)
f=5e9  # Transmitted signal frequency in Hertz (5G WiFi)
Pr2_dBm, PL2_dB = FriisModel(Pt_dBm, Gt_dBi, Gr_dBi, f, d, L, n)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.semilogx(d, Pr1_dBm, color = 'b', marker = 'o', ms = 12,  label = 'WiFi:f = 2.4GHz', base = 2)
axs.semilogx(d, Pr2_dBm, color = 'orange',  marker = 'o', ms = 12,  label = 'WiFi:f = 5GHz', base = 2)
axs.set_xlabel('Distance (m)',)
axs.set_ylabel('Received power (dBm)',)
# lb = 'X and X, ' + r'$\rho = {:.2f}$'.format(C_hat[0,0])
lb = "Received power using Friis model for WiFi transmission at f=2.4GHz and f=5GHz"
axs.set_title(lb, fontsize = 18)
axs.legend()

my_x_ticks = [1,2,4,8,16,32]
my_y_ticks = np.arange(-40, -95, -5)
axs.set_xticks(my_x_ticks)
axs.set_yticks(my_y_ticks)
# plt.xticks([1,2,4,8,16,32,35])

plt.show()
plt.close()


#%% Program 10.3: logNormalShadowing.m:Function to model Log-normal shadowing

def logNormalShadowing(Pt_dBm, GT_dBi, Gr_dBi, f, d0, d, L, sigma, n):
    lamba = 3*10**8/f
    K = 20 * np.log10(lamba/(4*np.pi)) - 10 * n * np.log10(d0) - 10 * np.log10(L)
    X = sigma * np.random.randn(d.size)
    PL = GT_dBi + Gr_dBi + K - 10 * n * np.log10(d/d0) -X
    Pr_dBm = Pt_dBm + PL

    return PL, Pr_dBm

Pt_dBm = 0  # Input transmitted power in dBm
Gt_dBi = 1  # Gain of the Transmitted antenna in dBi
Gr_dBi = 1  # Gain of the Receiver antenna in dBi
f = 2.4e9   # Transmitted signal frequency in Hertz
d0 = 1      # assume reference distance = 1m
d = 100 * np.arange(1, 100, 0.2) # Array of distances to simulate
L = 1               # Other System Losses, No Loss case L=1
sigma = 2  # Standard deviation of log Normal distribution (in dB)
n = 2      #  path loss exponent

# Log normal shadowing (with shadowing effect)
PL_shadow, Pr_shadow = logNormalShadowing(Pt_dBm,Gt_dBi, Gr_dBi, f, d0, d, L, sigma, n);
# Friis transmission (no shadowing effect)
Pr_Friss, PL_Friss = FriisModel(Pt_dBm, Gt_dBi, Gr_dBi, f, d, L, n)

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.plot(d, Pr_shadow, color = 'b',   label = 'Log normal shadowing', )
axs.plot(d, Pr_Friss, color = 'r',   label = 'Friss model', )
axs.set_xlabel('Distance (m)',)
axs.set_ylabel(r'$P_r$ (dBm)',)
# lb = 'X and X, ' + r'$\rho = {:.2f}$'.format(C_hat[0,0])
lb = "Log Normal Shadowing Model"
axs.set_title(lb, fontsize = 18)
axs.legend()

plt.show()
plt.close()

#%% Program 10.5: twoRayModel.m: Two ray ground reﬂection model simulation
f=900e6    # frequency of transmission (Hz)
R = -1     # reflection coefficient
Pt = 1     # Transmitted power in mW
Glos = 1   # product of tx,rx antenna patterns in LOS direction
Gref = 1   # product of tx,rx antenna patterns in reflection direction
ht = 50    # height of tx antenna (m)
hr = 2     # height of rx antenna (m)
d = np.arange(1, 10**5, 0.1)  #  separation distance between the tx-rx antennas(m)
L = 1                         # no system losses

#  Two ray ground reflection model
d_los = np.sqrt((ht - hr)**2 + d**2)   #  distance along LOS path
d_ref = np.sqrt((ht + hr)**2 + d**2)   # distance along reflected path
lamba = 3*10**8/f        #  wavelength of the propagating wave
phi = 2 * np.pi*(d_ref-d_los)/lamba    #  phase difference between the paths
s = lamba/(4*np.pi) * (np.sqrt(Glos)/d_los + R * np.sqrt(Gref)/d_ref * np.exp(1j*phi))
Pr = Pt * np.abs(s)*2   # received power
Pr_norm = Pr/Pr[0]   #  normalized received power to start from 0 dBm

#  Approximate models in three difference regions
dc = 4*ht*hr/lamba      #  critical distance
d1 = np.arange(1, ht, 0.1)       #  region 1 -- d<=ht
d2 = np.arange(ht, dc,0.1)       #  region 2 -- ht<=d<=dc
d3 = np.arange(dc, 10**5, 0.1)   #  region 3 -- d>=dc

K_fps = Glos * Gref * lamba**2 / ((4 * np.pi)**2 * L)
K_2ray = Glos * Gref * ht**2 * hr**2 / L

Pr1 = Pt*K_fps/(d1**2 + ht**2)  #  received power in region 1
Pr2 = Pt*K_fps/d2**2           #  received power in region 2
Pr3 = Pt*K_2ray/d3**4          #  received power in region 3

##### plot
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)

# x
axs.semilogx(d, 10 * np.log10(Pr), color = 'b',  label = 'Two ray model', base = 10)
axs.semilogx(d1, 10 * np.log10(Pr1), color = 'k', ls = '-.', label = r'Region1:d < $h_t$', base = 10)
axs.semilogx(d2, 10 * np.log10(Pr2), color = 'r', ls = '-.', label = r'Region2:$h_t$ < d < $d_c$', base = 10)
axs.semilogx(d3, 10 * np.log10(Pr3), color = 'g', ls = '-.', label = r'Region3:d > $d_c$', base = 10)
# axs.semilogx(d, Pr2_dBm, color = 'orange',  marker = 'o', ms = 12,  label = 'WiFi:f = 5GHz', base = 2)
axs.set_xlabel(r'$log_{10}(d)$',)
axs.set_ylabel('Normalized Received power (in dB)',)
axs.set_title("Two ray ground reflection model", fontsize = 18)
axs.legend()

# my_x_ticks = [1,2,4,8,16,32]
# my_y_ticks = np.arange(-40, -95, -5)
# axs.set_xticks(my_x_ticks)
# axs.set_yticks(my_y_ticks)
# plt.xticks([1,2,4,8,16,32,35])

plt.show()
plt.close()





























































































































































































































































