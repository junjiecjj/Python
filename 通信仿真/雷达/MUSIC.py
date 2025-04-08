#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:07:05 2024

@author: jack

https://blog.csdn.net/qq_35015368/article/details/127971331

https://zhuanlan.zhihu.com/p/613304918
https://blog.csdn.net/jiangwenqixd/article/details/118459087
https://zhuanlan.zhihu.com/p/678205710
https://www.zhihu.com/question/270353751
"""

# import sys
import numpy as np
import scipy
# import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%%%%%%% MUSIC for Uniform Linear Array %%%%%%%%
# https://zhuanlan.zhihu.com/p/613304918
pi = np.pi
derad = pi/180           # 角度->弧度
N = 8                    # 阵元个数
M = 3                    # 信源数目
theta = np.deg2rad([-30, 0, 60])      # 待估计角度
snr = 20                 # 信噪比
K = 512                  # 快拍数

d = np.arange(0, N).reshape(-1, 1)
A = np.exp(-1j * pi * d @ np.sin(theta).reshape(1,-1) )   # 方向矢量

## 构建信号模型%%%%%
S = np.random.randn(M, K)             # 信源信号，入射信号
X = A@S                                # 构造接收信号
SigPow = np.power(np.abs(X), 2).mean()
noise_pwr = SigPow/(10**(snr/10))
noise = np.sqrt(noise_pwr ) *  np.random.randn(*(X.shape))
X1 = X + noise                  # 将白色高斯噪声添加到信号中
# 计算协方差矩阵
Rxx = X1 @ X1.T.conjugate() / K
# 特征值分解
eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
eigvector = eigvector[:, idx]
eigvector = eigvector[:,::-1]                         # 对应特征矢量排序

Un = eigvector[:, M:N]
# Un = eigvector
UnUnH = Un @ Un.T.conjugate()

angle = np.deg2rad(np.arange(-90, 90.1, 0.5))
Pmusic = np.zeros(angle.size)
for i, ang in enumerate(angle):
    a = np.exp(-1j * pi * d * np.sin(ang))
    Pmusic[i] = 1/np.real(a.T.conjugate() @ UnUnH @ a)[0,0]

Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
Pmusic = 10 * np.log10(Pmusic)
peaks, _ =  scipy.signal.find_peaks(Pmusic, threshold = 3)

## 画图
fig, axs = plt.subplots(1, 1, figsize = (10, 8))
axs.plot(np.arange(-90, 90.1, 0.5), Pmusic , color = 'b', linestyle='-', lw = 3, label = "MUSIC", )
Theta = np.arange(-90, 90.1, 0.5)
axs.plot(Theta[peaks], Pmusic[peaks], linestyle='', marker = 'o', color='r', markersize = 12)

# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs.set_xlabel( "DOA/(degree)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
axs.set_ylabel('Normalized Spectrum/(dB)', fontproperties=font2, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator = MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=16, width=3,)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
axs.spines['bottom'].set_linewidth(1.5)    ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(1.5)      ####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(1.5)     ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(1.5)       ####设置上部坐标轴的粗细

out_fig = plt.gcf()
# out_fig.savefig('music.eps' )
plt.show()
plt.close('all')


#%% https://zhuanlan.zhihu.com/p/678205710
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

fs = 1e6
Ts = 1/fs
N  = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N) * Ts
f0 = 0.02 * 1e6
tx = np.exp(2j * np.pi * f0 * t)

# Simulate three omnidirectional antennas in a line with 1/2 wavelength between adjancent ones, receiving a signal that arrives at an angle
d = 0.5
Nr = 3
theta_degrees = 20                  # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
print(a)

a = np.asmatrix(a)
tx = np.asmatrix(tx)

r = a.T @ tx
print(r.shape)

fig, (ax1) = plt.subplots(1, 1, figsize = (7, 3))
ax1.plot(np.asarray(r[0, :]).squeeze().real[0 : 200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
ax1.plot(np.asarray(r[1, :]).squeeze().real[0 : 200])
ax1.plot(np.asarray(r[2, :]).squeeze().real[0 : 200])
ax1.set_ylabel("Samples")
ax1.set_xlabel("Time")
ax1.grid()
ax1.legend(['0','1','2'], loc = 1)
plt.show()

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.1 * n
fig, (ax1) = plt.subplots(1, 1, figsize = (7, 3))
ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # The asarray and squeeze are just annoyances we have to do because we came from a matrix.
ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
ax1.set_ylabel("Samples")
ax1.set_xlabel("Time")
ax1.grid()
ax1.legend(['0','1','2'], loc = 1)
plt.show()

if True:
    f0 = 0.01 * 1e6
    f1 = 0.02 * 1e6
    f2 = 0.03 * 1e6
    Nr = 8 # 8 elements
    theta1 = 30 / 180 * np.pi
    theta2 = 60 / 180 * np.pi
    theta3 = -30 / 180 * np.pi
    a1 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)))
    a2 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)))
    a3 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)))
    # we'll use 3 different frequencies
    r = a1.T @ np.asmatrix(np.exp(2j*np.pi*f0*t)) + a2.T @ np.asmatrix(np.exp(2j*np.pi*f1*t)) + 0.1 * a3.T @ np.asmatrix(np.exp(2j*np.pi*f2*t))
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.04*n

    # MUSIC Algorithm (part that doesn't change with theta_i)
    num_expected_signals = 3    # Try changing this!
    R = r @ r.H / Nr
    Sigma, U = np.linalg.eig(R) # eigenvalue decomposition, U[:,i] is the eigenvector corresponding to the eigenvalue Sigma[i].

    if True:
        fig, (ax1) = plt.subplots(1, 1, figsize = (7, 3))
        ax1.plot(10 * np.log10(np.abs(Sigma)), '.-')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue [dB]')
        plt.show()
        plt.close()

    eig_val_order = np.argsort(np.abs(Sigma)) # find order of magnitude of eigenvalues
    U = U[:, eig_val_order]                   # sort eigenvectors using this order
    V = np.asmatrix(np.zeros((Nr, Nr - num_expected_signals), dtype = np.complex64)) # Noise subspace is the rest of the eigenvalues
    for i in range(Nr - num_expected_signals):
        V[:, i] = U[:, i]

    theta_scan = np.linspace(-1 * np.pi/2, np.pi / 2, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        a = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))) # look familiar?
        a = a.T
        metric = 1 / (a.H @ V @ V.H @ a) # The main MUSIC equation
        metric = np.abs(metric[0, 0])    # take magnitude
        metric = 10*np.log10(metric)     # convert to dB
        results.append(metric)

    results /= np.max(results)      # normalize
    fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
    ax.plot( theta_scan , results)
    # fig, ax = plt.subplots( )
    # ax.plot(np.rad2deg(theta_scan) , results)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)
    plt.show()
    plt.close('all')

## 从上面的结果可以看出，MUSIC算法的是为了测角用的，每个用户的入射角不同，且也可以使用不同的频率，该算法都能准确测角。
#%% https://zhuanlan.zhihu.com/p/613304918
# 导入模块
# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt

# # 生成快拍数据
# def gen_signal(fre, t_0, theta, speed, numbers, space):
#     res = []
#     for i in range(numbers):
#         res.append(np.exp(2j*np.pi*fre*t_0 - 2j*np.pi*fre*i*space*np.cos(theta)/speed))
#     return np.array(res)

# # 生成方向矢量
# def steer_vector(fre, theta, speed, numbers, sapce):
#     alphas = []
#     for i in range(numbers):
#         alphas.append(np.exp(-2j*np.pi*fre*i*space*np.cos(theta)/speed))
#     return np.array(alphas).reshape(-1, 1)

# # Music算法
# def cal_music(fre, speed, numbers, space, signals, method='signal'):
#     R_x = np.matmul(signals, np.conjugate(signals.T)) / signals.shape[1]
#     lamda, u = np.linalg.eig(R_x)
#     u_s = u[:, np.argmax(lamda)].reshape(-1, 1)
#     u_n = np.delete(u, np.argmax(lamda), axis=1)
#     P = []
#     thetas = np.linspace(-np.pi/2, np.pi/2, 180)
#     for _theta in thetas:
#         _alphas = steer_vector(fre, _theta, speed, numbers, space).reshape(-1, 1)
#         if method == 'signal':
#             P_x = 1 / np.matmul(np.matmul(np.conjugate(_alphas).T, np.eye(len(u_s)) - np.matmul(u_s, np.conjugate(u_s.T))), _alphas)
#         elif method == 'noise':
#             P_x = 1 / (np.matmul(np.matmul(np.matmul(np.conjugate(_alphas).T, u_n), np.conjugate(u_n.T)), _alphas))
#         else:
#             print('there is no ' + method)
#             break
#         P.append(P_x)
#     P = np.array(P).flatten()
#     return thetas/np.pi*180, P

# # 初始化数据
# fs = 20000
# # 定义源信号
# fre = 200
# t = np.arange(0, 0.01, 1/fs)
# theta1 = np.pi / 3
# theta2 = 2 * np.pi / 3
# # 传播速度
# speed = 340
# # 阵元数量
# numbers = 32
# # 阵元之间距离
# space = 1
# # 生成模拟快拍数据
# signals = []
# for t_0 in t:
#     signal1 = gen_signal(fre, t_0, theta1, speed, numbers, space)
#     signal2 = gen_signal(fre, t_0, theta2, speed, numbers, space)
#     signal = signal1 + signal2
#     signals.append(signal.tolist())
# signals = np.array(signals)

# # Music算法处理结果
# thetas, P = cal_music(fre, speed, numbers, space, signals, method='noise')
# plt.figure(figsize=(10, 2))
# plt.plot(thetas, abs(P))
# plt.xlim(-90, 90)
# plt.xlabel('degree')
# plt.ylabel('mag')
# plt.show()


#%%


#%%


#%%


#%%
























































































































































