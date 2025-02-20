#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:33:04 2025

@author: jack
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
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
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


#%% Program 11.1: scattering function.m: Plot scattering function power delay profile and Doppler spectrum
from sympy import symbols, lambdify, expand, simplify

fc = 1800e6   #  Carrier frequency (Hz)
fm = 200      #  Maximum Doppler shift (Hz)
trms = 30e-6  #  RMS Delay spread (s)
Plocal = 0.2   # local-mean power (W)

f_delta = fm/30                                      # step size for frequency shifts
f = np.arange((fc - fm) + f_delta, (fc+fm) - f_delta, f_delta) # normalized freq shifts
tau = np.arange(0, trms*3+trms/5, trms/5)            # generate range for propagation delays
TAU, F = np.meshgrid(tau, f)                         # all possible combinations of Taus and Fs

#Example Scattering function equation
Z = Plocal/(4 * np.pi * fm * np.sqrt(1 - ((F - fc) / fm)**2))*1/trms * np.exp(-TAU/trms)



### 1
# num = 301; # number of mesh grids
# x_array = np.linspace(-3,3,num)
# y_array = np.linspace(-3,3,num)
# xx,yy = np.meshgrid(x_array,y_array)
from sympy.abc import x, y
x, y = symbols('x  y')
f_xy =  Plocal/ trms / (4 * np.pi * fm * np.sqrt(1 - ((y - fc) / fm)**2))* np.exp(-x/trms)
f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(TAU, F)

### 2
xx1,xx2 = np.meshgrid(np.linspace(-3,3),np.linspace(-3,3))
ff = np.exp(- xx1**2 - xx2**2)




subplot(2,1,1); mesh(TAU,(F-fc)/fm,Z);%Plot the 3D mesh plot
title('Scattering function S(f,\tau)');xlabel('Delay \tau');
ylabel('(f-fc)/fm');zlabel('Received power');

%Project the 3D plot and plot PDP & Doppler Spectrum
subplot(2,2,3); plot(tau,Z(1,:,:));
title('Power Delay Profile');
xlabel('Delay (\tau)'); ylabel('Received power');
subplot(2,2,4); plot((f-fc)/fm,Z(:,1,:));
title('Doppler Power Spectrum');
xlabel('Doppler shift (f-fc)/fm');ylabel('Received power');

#%% Program 11.2: plot fcf.m: Frequency correlation function (FCF) from power delay profile



#%% Program 11.3, 11.4, 11.5, 11.6



#%% Program 11.7: rician pdf.m: Generating Ricean ﬂat-fading samples and plotting its PDF



#%% Program 11.8: doppler psd acf.m: Generating Ricean ﬂat-fading samples and plotting its PDF




#%% Program 11.9: param MEDS.m: Generate parameters for deterministic model using MEDS method




#%% Program 11.10: Rice method.m: Function to simulate deterministic Rice model




#%% Program 11.11: pdp model.m: TDL implementation of specified power delay profile




#%% Program 11.12: freq selective TDL model.m: Simulate frequency selective Rayleigh block fading channel



























































































































































































































































