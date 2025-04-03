#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack
匹配滤波（脉冲压缩） matlab代码，亲测可用:
https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6

脉冲压缩原理以及实验代码详解:
https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10

基于Python的FMCW雷达工作原理仿真（附代码）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485426&idx=1&sn=ad1d302e2177b037778ee9e6d405ec33&chksm=c11f0a67f66883717c39bd6deab5a184182dec6192c517967d68b85b25cafe286607cfee1f7d&scene=21#wechat_redirect

雷达仿真 | 调频连续雷达 (FMCW)仿真Python代码
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247489978&idx=1&sn=8ed933fc275af974846a6f3be00f05d8&chksm=c06f60d3542de596d383a4c3322e1e0f33bd6853ad0962f49be14baaff7c1126ed51d0912525&mpshare=1&scene=1&srcid=0323bPMO68rvEFUU8aa5QnQY&sharer_shareinfo=cf22eefea6d212ac1867dcdaee6a8788&sharer_shareinfo_first=cf22eefea6d212ac1867dcdaee6a8788&exportkey=n_ChQIAhIQrzTnyknZZ2pOw7YOXO15HBKfAgIE97dBBAEAAAAAAKc4N%2B%2BIx4AAAAAOpnltbLcz9gKNyK89dVj0bIgzCpeuPa34D1Ov6V3ZVNbFSz830ZSINdOhiMO4Uw3qKUZFF%2FImjJO464ckbuOZkdSe4h1DJcnocX0ZxNrUBDOpDrKjOASUS8g8h3qrKw38eqEqDov7zgh7O9awFsoWefnY9rAKjSSjR2lhrmRH6icJX1x97e90jc%2FWoOgVyyTbCDDG8uDHbot7VmRc572NQq5ztzDZrGerQDeD%2BJ7%2BZrNugOG0ZauOW%2FkfU36c8T7oc3xiHMNI4imMMqMFS7UEPlluvQR%2FQaLpP1%2B9T8dm58YFWYOji4dCBTENOtiiLeOpPF4l71R1NrLA3OBDCfCKsI7%2BGmtxu%2FBD&acctmode=0&pass_ticket=l3Xl3zfrRyJIluhuYJTPnj02ELo%2F%2Fw4SEt9eaw9t0FoT7Ao94AINqNgjZ5nk%2FjXv&wx_header=0#rd

分享 | 调频连续波 (FMCW) 毫米波雷达目标检测Python仿真:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247491272&idx=1&sn=8c816033438a549fdaeb20e51b154896&chksm=c11f135df6689a4bb0528639e9f437c86e941ef816e78f2b9a935f568db8be529f85234cafd5&token=134337482&lang=zh_CN#rd

干货：FMCW雷达系统信号处理建模与仿真（含MATLAB代码）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247484183&idx=1&sn=fbf605f11d510343c5beda9bdc5c32a4&chksm=c11f0e82f66887941da61052bbfeee2fa37227e7a94d8ebb23fc1e9cc88a357f95ff4a10d012&scene=21#wechat_redirect

第4讲 -- 线性调频连续波LFMCW测量原理：测距、测速、测角:
https://zhuanlan.zhihu.com/p/687473210

利用Python实现FMCW雷达的距离多普勒估计(2D-FFT, 距离FFT，速度FFT)
https://blog.csdn.net/caigen0001/article/details/108815569
干货 | 利用MATLAB实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s?__biz=MzI2NzE1MTU3OQ==&mid=2649214285&idx=1&sn=241742b17b557c433ac7f5010758cd0f&chksm=f2905cf9c5e7d5efc16e84cab389ac24c5561a73d27fb57ca4d0bf72004f19af92b013fbd33b&scene=21#wechat_redirect
干货 | 利用Python实现FMCW雷达的距离多普勒估计:
https://mp.weixin.qq.com/s/X8uYol6cWoWAX6aUeR7S2A

雷达初学者必读 | 毫米波雷达信号处理入门教程:
https://blog.csdn.net/qq_35844208/article/details/127122316
毫米波雷达：信号处理:
https://zhuanlan.zhihu.com/p/524371087?utm_source=wechat_session&utm_medium=social&s_r=0
雷达原理 | 用MATLAB信号处理是如何解算目标的距离和速度信息的？
https://zhuanlan.zhihu.com/p/422798513

[解疑][TI]TI毫米波雷达系列（三）：调频连续波雷达回波信号3DFFT处理原理（测距、测速、测角）
https://blog.csdn.net/qq_35605018/article/details/108816709
回波3DFFT处理（测距、测速、测角）
https://blog.csdn.net/nuaahz/article/details/90719605
雷达信号处理之FMCW 3D-FFT原理（附带MATLAB仿真程序）:
https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247485771&idx=1&sn=8e269280b663226160227aec22806c3e&chksm=c11f04def6688dc8e20c2e92bed6bc4547bf107f87b77bfff29f2c66434fdb5702333184ee1d&scene=178&cur_album_id=2442863581802381317#rd

雷达入门课系列文章（1）| 基于MATLAB的雷达信号处理实验教程
https://zhuanlan.zhihu.com/p/567656893

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22         # 设置 y 轴刻度字体大小
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
plt.rcParams['legend.fontsize'] = 22

#%% 干货：FMCW雷达系统信号处理建模与仿真（含MATLAB代码）:
# https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247484183&idx=1&sn=fbf605f11d510343c5beda9bdc5c32a4&chksm=c11f0e82f66887941da61052bbfeee2fa37227e7a94d8ebb23fc1e9cc88a357f95ff4a10d012&scene=21#wechat_redirect

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

# Radar parameters setting
maxR = 200   # 雷达最大探测目标的距离
rangeRes = 1 # 雷达的距离分率
maxV = 70    # 雷达最大检测目标的速度
fc = 77e9    # 雷达工作频率 载频
c = 3e8
R0 = 90  # 目标距离
v0 = 20   # 目标速度

B = c/(2*rangeRes)          # 发射信号带宽150MHz
Tchirp = 5.5*2*maxR/c       # 扫频时间 (x-axis), 5.5 = sweep time should be at least 5 o 6 times the round trip time
endle_time = 6.3e-6         # 空闲时间
slope = B/Tchirp            # 调频斜率
f_IFmax = (slope*2*maxR)/c  # 最高中频频率
f_IF = (slope*2*R0)/c       # 当前中频频率

Nchirp = 128                                  # chirp数量
Ns = 1024                                     # ADC采样点数
vres = (c/fc)/(2*Nchirp*(Tchirp+endle_time))  # 速度分辨率
Fs = Ns/Tchirp   # = 1/(t[1] - t[0])          # 模拟信号采样频率

# Tx波函数参数
t = np.linspace(0, Nchirp*Tchirp, Nchirp*Ns)   # 发射信号和接收信号的采样时间
# angle_freq = fc * t +  slope / 2 * t**2        # 角频率
# freqTx = fc + slope*t                           # 频率

rt = R0 + v0 * t  # 距离更新
td = 2*rt/c       # 延迟时间
Tx = np.cos(2*np.pi*(fc * t +  slope / 2 * t**2))                # 发射信号 实数信号
Rx = np.cos(2*np.pi*(fc * (t - td) +  slope / 2 * (t - td)**2))                # 接收信号 实数信号
freqTx = fc + slope*t                           # 频率
freqRx = fc + slope*t                           # 频率

Mix = Tx * Rx

# 结果可视化
fig, axs = plt.subplots(3, 2, figsize = (12, 12), constrained_layout = True)

# 发射回波信号
axs[0,0].plot(t[0:Ns], Tx[0:Ns])
axs[0,0].set_title("Tx Signal")
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Amplitude")

# 接收回波信号
axs[0,1].plot(t[0:Ns], Rx[0:Ns])
axs[0,1].set_title("Rx Signal")
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Amplitude")

axs[1,0].plot(t[0:Ns], freqTx[0:Ns], label = "Frequency of Tx signal")
axs[1,0].plot(t[0:Ns] + td[0:Ns], freqRx[0:Ns], label = "Frequency of Rx signal")
axs[1,0].set_title("Frequency of Tx/Rx signal")
axs[1,0].set_xlabel("Time")
axs[1,0].set_ylabel("Frequency")
axs[1,0].legend()

axs[2,0].plot(t[0:Ns], Mix[:Ns])
axs[2,0].set_title("IFx Signal")
axs[2,0].set_xlabel("Time")
axs[2,0].set_ylabel("Amplitude")

f, _, A, _, _, _  = freqDomainView(Mix[:Ns], Fs, type = 'double')
axs[2,1].plot(f, A)
axs[2,1].set_title("IFx FFT")
axs[2,1].set_xlabel("Freq/Hz")
axs[2,1].set_ylabel("Amplitude")

plt.show()
plt.close()

Mix_resp = Mix.reshape(Ns, Nchirp)






#%%


#%%


#%%

































