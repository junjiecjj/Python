#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:30:18 2025

@author: jack

https://blog.csdn.net/qq_43485394/article/details/122655901

"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%  https://blog.csdn.net/qq_43485394/article/details/122655901
# from DigiCommPy.signalgen import square_wave

def rectpuls(t, remove, T):
    Ts = t[1] - t[0]
    fs = 1/Ts

    # t = np.arange(-0.5,0.5, 1/fs) # time base
    rect = (t >= -T/2) * (t <= T/2)
    # res = np.zeros(rect.size)
    K = int(remove*fs)
    rect = np.roll(rect, K)

    # t = t + remove
    return rect

# t = np.arange(-5, 5, 0.01)
# remove = 2
# res = rectpuls(t, remove, 2)
# plt.figure(figsize = (10, 5))
# plt.plot(t, res, label = 'Transmitted Pulse')

# plt.show()


### parameters
f0 = 10e9        # 载波
Tp = 10e-6       # 脉冲持续时间
B = 10e6         # 带宽
fs = 100e6       # 采样频率
c = 3e8          # 光速
R0 = 3e3         # 目标距离
k = B/Tp         # 调频斜率

# signal generation
N = 1024*4       #  采样点
n = np.arange(N)
Ts = 1/fs        #  采样间隔
t = n*Ts
f = -fs/2+ n*(fs/N)
tau_0 = 2*R0/c   #  时延
# tau_1 = 2*R1/c;
st = rectpuls(t, Tp/2, Tp) * np.exp(1j * np.pi * k * (t-Tp/2)**2)    #  参考信号
#  回波信号
secho = rectpuls(t, tau_0+Tp/2, Tp) * np.exp(1j * np.pi * k * (t - tau_0 - Tp/2)**2) * np.exp(-1j * 2 * np.pi * f0 * tau_0)

#  =============== 脉冲压缩 ================
Xs = scipy.fft.fft(st,N);        # 本地副本的FFT
Xecho = scipy.fft.fft(secho,N);  # 输入信号的FFT
Y = np.conjugate(Xs)*Xecho;      # 乘法器
Y = scipy.fft.fftshift(Y);
y = scipy.fft.ifft(Y,N);         # IFFT


##### plot
fig, axs = plt.subplots(4, 2, figsize = (12, 16), constrained_layout = True)

axs[0,0].plot(t * 1e6, np.real(st), color = 'b', label = ' ')
axs[0,0].set_xlabel('时间/us',)
axs[0,0].set_ylabel('幅值',)
axs[0,0].set_title("Real Part of Reference Signal" )
# axs[0].legend()

axs[0,1].plot(t * 1e6, np.imag(st), color = 'b', label = ' ')
axs[0,1].set_xlabel('时间/us',)
axs[0,1].set_ylabel('幅值',)
axs[0,1].set_title("Imagine Part of Reference Signal" )
# axs[0].legend()

axs[1,0].plot(t * 1e6, np.real(secho), color = 'b', label = ' ')
axs[1,0].set_xlabel('时间/us',)
axs[1,0].set_ylabel('幅值',)
axs[1,0].set_title("Real Part of Echo Signal" )
# axs[0].legend()

axs[1,1].plot(t * 1e6, np.imag(secho), color = 'b', label = ' ')
axs[1,1].set_xlabel('时间/us',)
axs[1,1].set_ylabel('幅值',)
axs[1,1].set_title("Imagine Part of Echo Signal" )
# axs[0].legend()

##
X1 = scipy.fft.fftshift(Xs)
axs[2,0].plot(f/(1e6), np.abs(X1), color = 'b', label = ' ')
axs[2,0].set_xlabel('Frequency/MHz',)
axs[2,0].set_ylabel('幅值',)
axs[2,0].set_title("Spectral of Reference Signal" )

axs[2,1].plot(f/(1e6), np.abs(scipy.fft.fftshift(Xecho)), color = 'b', label = ' ')
axs[2,1].set_xlabel('Frequency/MHz',)
axs[2,1].set_ylabel('幅值',)
axs[2,1].set_title("Spectral of Echo Signal" )

axs[3,0].plot(f/(1e6), np.abs(Y), color = 'b', label = ' ')
axs[3,0].set_xlabel('Frequency/MHz',)
axs[3,0].set_ylabel('幅值',)
axs[3,0].set_title("Spectral of the Result of Pulse Compression" )

r = t*c/2;
y = np.abs(y)/max(np.abs(y)) + 1e-10;
axs[3,1].plot(r, 20*np.log10(y), color = 'b', label = ' ')
axs[3,1].set_xlabel('Range/m',)
axs[3,1].set_ylabel('幅值',)
axs[3,1].set_title("Result of Pulse Compression" )

R0_est = r[np.argmax(np.abs(y))]
print(f"R0 = {R0}, R0_est = {R0_est}")
plt.show()
plt.close()


#%% https://mp.weixin.qq.com/s?__biz=MzAwMDE1ODE5NA==&mid=2652542571&idx=1&sn=0e0eb494ac7ee19d18227a5e96c2b27e&chksm=80065fae159d3dd84e1d9c3a866f126b4b306ec97b1a427b7ff664c0c311286533a276ab7193&mpshare=1&scene=1&srcid=0329Q8dj1B90QMlepVAj2Um9&sharer_shareinfo=38d19dc84b14ff1c2d3b069947b97c9c&sharer_shareinfo_first=38d19dc84b14ff1c2d3b069947b97c9c&exportkey=n_ChQIAhIQFCYeQ%2B6%2BTwh8yNrYRb5RTBKfAgIE97dBBAEAAAAAAFd1FUcF70gAAAAOpnltbLcz9gKNyK89dVj0FDSmEnzfw8MsNY2waUVVqmm5UxZzyDzF5tbZS7E1FJ8ks%2FFLirUTE1wQ2Xr5RMr0LSsVrqypI%2F2aqly%2Fl4uofOZAPvQQjCb4t1wr1bgr1iGp0%2Fja6EufHwe6%2BOtX8Muca1J8F%2F1mtxqFxdDnfAIGnTm7M%2BC2BumNQg1gfrdTl6iuQghRu9X1fqpoRIHk%2BmYl7dtIDNp40mke%2FmuiC%2Fr9RUITAQQShNsr%2FvVz5QleWdVWLSST1uCtkvEuYdurrGkLJZKHLp9gZyOW95cPiUp8bNB0gtT7SOTvU9UrH8Eedr8sQLQBsqwtiKAVKJjgqUj6RjiH3yJWarkT&acctmode=0&pass_ticket=fh8TkWVQ2FSWTxDQvzOQRMqDWhGDthA7I9lYcXveqOdL%2Bq7ha%2FaWBw%2Fse4F%2BIMDs&wx_header=0#rd





#%% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247485021&idx=3&sn=742ef5748dce8629ebc43d99ad06befe&chksm=ce5103d1e0308bf218466c6e248c75b18e5064424c93d98e916591d1398c5811881ff4131ad9&mpshare=1&scene=1&srcid=0329wtlZc1KNnFCZ6iw1tk7G&sharer_shareinfo=48caf2d5f9cf6976c915c311eee94f2b&sharer_shareinfo_first=48caf2d5f9cf6976c915c311eee94f2b&exportkey=n_ChQIAhIQYIFUhJ1ixZB7LHc8116UpxKfAgIE97dBBAEAAAAAAFs7BFMneDoAAAAOpnltbLcz9gKNyK89dVj08JjHPehNSxSotXXsU001an68bbK6IqjQ60hNFBjrROO1ZNChcAUoUGNBOq%2BD7vVzTk4zhjDQfHgsd36CwGvEP9cuCpcaF0b84K1woLB5BqZlBpBKeciOu%2FhYsfoYtoJR9v241Kspkw9ouDuSLwYzBApbL88wLKd6vgimG5ZaCZq28gVyWgQiuYepcUZBThnU%2BhV%2FjawaczWvNkPrJ0B0EOkq8aIACuOXLWHj3itH3W%2B%2F6W9ebuggdjgZDtiLb4wjDcgPBWYK8ugu3jGyrVnRKNP0r4QHnH%2F9%2F2EQfqdYWFg8z%2FfxBUOJAlVFN7PKRJwY0AgdHdiwFHCY&acctmode=0&pass_ticket=UOuTwL5JezorkCrj%2BTMjx7yzKbpTU8fEVTb6keEK8pXFIgyOFbr4GvLLR%2F4CSq25&wx_header=0#rd





#%% https://mp.weixin.qq.com/s?__biz=MzUxNTY5NzYzMA==&mid=2247553394&idx=1&sn=85255267d109644bbd54a80a8d161d98&chksm=f82261285e85b08e2ed81badfe157aed88a7c5fa599ad64148d3fe4ddfe2e1cdb2b95c395778&mpshare=1&scene=1&srcid=0329S589unC9aTqoMGAP80Ip&sharer_shareinfo=1239f9b97eb0d5376843535656576bc0&sharer_shareinfo_first=1239f9b97eb0d5376843535656576bc0&exportkey=n_ChQIAhIQrXuc3vlmReLg2VrWk3pxgxKfAgIE97dBBAEAAAAAAH%2FJCqibM60AAAAOpnltbLcz9gKNyK89dVj0qU8QDSNbqKY3HMOHwcWHRbw3xUWZ1kd2zoydLPbQBKZRVcpqSq8JQrP14GYQd53PJjdAvDePUFL6Lj3FcUTrOk39woXEQX%2FX8iLqPvs7a54T2BoA79vTgvhnfkKX9FqmBIUm8hNdQPAcqNfAWcwaObXb4bX4gp9RyMEBbuO2cKCGEzoAL5WvBC0n3EVnE4isF7%2B2O3jEOCToSjCZaEO%2BAyGrQM3QPEwQDn%2BXhkLkqOIs1Y9A0QHvcMykYwhW2A7xFWrnc4IOvCqsZclNnKGsld%2BPEhTp8AKKQtM574RxLhfAdBVNVhUX%2BSel%2F5pezmlpgWSV%2FDm1zJ45&acctmode=0&pass_ticket=juUJ8JHuA70tTcAQyaFf2ZDKkTnOdyVFeAMOFBjosljVKpPPqm9P1olPjK8m7M%2Bg&wx_header=0#rd





#%%



#%%




