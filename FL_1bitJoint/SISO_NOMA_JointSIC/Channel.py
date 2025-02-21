#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:21 2024
@author: JunJie Chen
"""



import numpy as np
import scipy
from sklearn.metrics import pairwise_distances
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import commpy

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


def AWGN_mac(K,  frame_len):
    H = np.ones((K, frame_len))
    return H

def BlockFading_mac(K, frame_len):
    H0 = (np.random.randn(K, ) + 1j * np.random.randn(K, ))/np.sqrt(2)
    H = np.expand_dims(H0, -1).repeat(frame_len, axis = -1)
    return H

def FastFading_mac(K,  frame_len):
    H = (np.random.randn(K, frame_len) + 1j * np.random.randn(K, frame_len))/np.sqrt(2)
    return H

def channelConfig(K, r = 100):
    C0 = -30                             # dB
    C0 = 10**(C0/10.0)                   # 参考距离的路损
    d0 = 1

    ## path loss exponents
    alpha_Au = 3

    ## Rician factor
    beta_Au = 3.76   # dB
    beta_Au = 10**(beta_Au/10)

    # sigmaK2 = -60                        # 噪声功率 dBm
    # sigmaK2 = 10**(sigmaK2/10.0)/1000    #
    # P0 = 30                              # 用户功率 dBm
    # P0 = 10**(P0/10.0)/1000

    # Location, Case II
    BS_locate = np.array([[0, 0, 10]])
    # radius = np.random.rand(K, 1) * r
    radius = (np.linspace(0.05, 1, K) * r).reshape(-1, 1)
    angle = np.random.rand(K, 1) * 2 * np.pi
    users_locate_x = radius * np.cos(angle)
    users_locate_y = radius * np.sin(angle)
    users_locate_z = np.ones((K, 1))
    users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

    ## Distance
    d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)

    ## generate path-loss
    PL_Au = C0 * (d_Au/d0)**(-alpha_Au)

    return BS_locate, users_locate, beta_Au, PL_Au

def Large_rician_block(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    hdLos = np.sqrt(1/2) * (np.ones((K,)) + 1j * np.ones((K,)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, ) + 1j * np.random.randn(K, ))
    h_ds = np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos
    h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    H = np.expand_dims(h_d, -1).repeat(frame_len, axis = -1)
    return H

def Large_rician_fast(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    hdLos = np.sqrt(1/2) * (np.ones((frame_len, K)) + 1j * np.ones((frame_len, K)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(frame_len, K) + 1j * np.random.randn(frame_len, K))
    h_ds = np.sqrt(beta_Au/(1 + beta_Au)) * hdLos + np.sqrt(1/(1 + beta_Au)) * hdNLos
    H = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    return H.T

def Large_rayleigh_block(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    # hdLos = np.sqrt(1/2) * (np.ones((K,)) + 1j * np.ones((K,)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, ) + 1j * np.random.randn(K, ))
    # h_ds = np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos
    h_d = hdNLos @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    H = np.expand_dims(h_d, -1).repeat(frame_len, axis = -1)
    return H

def Large_rayleigh_fast(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    # hdLos = np.sqrt(1/2) * (np.ones((frame_len, K)) + 1j * np.ones((frame_len, K)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(frame_len, K) + 1j * np.random.randn(frame_len, K))
    # h_ds = np.sqrt(beta_Au/(1 + beta_Au)) * hdLos + np.sqrt(1/(1 + beta_Au)) * hdNLos
    H = hdNLos @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    return H.T

np.random.seed(42)
frame_len = 100000
B      = 4e6                    # bandwidth, Hz
sigma2 = -60                    # 噪声功率谱密度, dBm/Hz
sigma2 = 10**(sigma2/10.0)/1000 # 噪声功率谱密度, Watts/Hz
N0     = sigma2 * B             # 噪声功率, Watts

pmax = 30                     # 用户发送功率, dBm
pmax = 10**(pmax/10.0)/1000   # Watts
pmax = 0.1                    # Watts

K = 6
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(K, r = 100)

# sigma2 = 1
H = Large_rayleigh_fast(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = N0)
# H = Large_rician_fast(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au)

Hbar = np.mean(np.abs(H), axis = 1) * np.sqrt(N0)/ np.sqrt(PL_Au.flatten())



##### plot rayleigh
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
x = np.arange(0, 3.01, 0.01)  # Ricean rv
k = 0
beta = beta_Au

sigma2 = 1
r = scipy.stats.rayleigh.rvs(loc = 0, scale = np.sqrt(sigma2/2), size = frame_len)
pdf = x/(sigma2/2) * np.exp(-x**2/(2 * sigma2/2))
# pdf = scipy.stats.rayleigh.pdf(x, loc = 0, scale = np.sqrt(sigma2/2))
hk = H[k,:] * np.sqrt(N0) / np.abs(np.sqrt(PL_Au[k, 0]))
mean1 = np.mean(np.abs(hk))
mean2 = np.mean(np.abs(r))
print(f"{mean1}, {mean2}")
axs.hist(np.abs(hk), 100, density = 1, histtype = 'step', color = 'b', lw = 1, label = "Simulation 1")
axs.hist(r, 100, density = 1, histtype = 'step', color = 'r', lw = 1, label = "Simulation 2")
axs.plot(x, pdf,  color = 'k', lw = 1, ls = 'none', marker = 'o', markevery = 10, label = "Theory")

axs.legend(labelspacing = 0.01)
axs.set_xlabel( 'x',)
axs.set_ylabel(r'$f_{\xi}$(x)',)
axs.set_title("PDF of envelope of received signal")
plt.show()
plt.close()


##### plot rician
# K_factors = np.array([0, 3, 7, 12, 20])   # Ricean K factors
# colors = ['b','r','k','g','m']
# Omega = 1                           # Total average power set to unity

# ##### plot
# fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
# x = np.arange(0, 3.01, 0.01)  # Ricean rv
# k = 0
# beta = beta_Au
# z = 2 * x * np.sqrt(beta*(beta + 1)/Omega)   # to generate modified Bessel function
# I0_z = scipy.special.iv(0, z)        # modified Bessel function of first kind, use iv not jv.
# pdf = (2 * x * (beta+1) / Omega) * np.exp(-beta - (beta+1) * x**2 / Omega) * I0_z

# H = H2[k,:] / np.abs(np.sqrt(PL_Au[k, 0]))

# axs.hist(np.abs(H), 100, density = 1, histtype = 'step', color = colors[0], lw = 1, label = f" Simulation")
# axs.plot(x, pdf,  color = colors[0], lw = 1, ls = 'none', marker = 'o', markevery = 10, label = f" Theory")

# axs.legend(labelspacing = 0.01)
# axs.set_xlabel( 'x',)
# axs.set_ylabel(r'$f_{\xi}$(x)',)
# axs.set_title("PDF of envelope of received signal")
# plt.show()
# plt.close()












































































