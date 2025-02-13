#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:12:19 2025

@author: jack
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%% Generating a sinusoid and plotting its power spectrum
A = 1
fc = 100
fs = 3000
nCyl = 3

## generate signal
t = np.arange(0, nCyl/fc, 1/fs)
x = -A * np.sin(2 * np.pi * fc * t)

## Computing total power of the generated sinusoid signal from time domain
L = len(x)
P = np.linalg.norm(x)**2/L
print(P)

## Computing total power of the generated sinusoid signal from frequency domain
FFTN = L
X = np.fft.fftshift(np.fft.fft(x, n = FFTN))
Px = X * X.conjugate() / L**2
f = np.arange(-FFTN/2, FFTN/2) * fs/FFTN

## plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, x, label = 'x')
font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0].set_xlabel('time (s)',fontproperties=font)
axs[0].set_ylabel('x',fontproperties=font)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

legend1 = axs[0].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[0].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

# Px
axs[1].stem(f, Px, label = 'PSD', linefmt = 'r--', markerfmt = 'o', )
font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1].set_xlabel('Fre (Hz)',fontproperties=font)
axs[1].set_ylabel('power',fontproperties=font)
font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

legend1 = axs[1].legend(loc='best',  prop=font1, borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

axs[1].tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6,  )
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号

out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()


#%%
# https://matplotlib.net.cn/stable/gallery/lines_bars_and_markers/xcorr_acorr_demo.html
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(42)

x, y = np.random.randn(2, 100)
# fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
fig, axs = plt.subplots(2, 1, figsize = (8, 6), constrained_layout = True,  sharex=True)
idx_1, xcor_1, _, _ = axs[0].xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
axs[0].grid(True)
axs[0].set_title('Cross-correlation (xcorr)')

idx_2, acor_2, _, _  = axs[1].acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
axs[1].grid(True)
axs[1].set_title('Auto-correlation (acorr)')

plt.show()


# 计算自相关函数
def autocorrelation(x):
    n = len(x)
    variance = x.var()
    mean = x.mean()
    c0 = np.sum((x - mean) ** 2) / n
    result = np.correlate(x - mean, x - mean, mode = 'same')
    result /= c0
    return result

scorr = autocorrelation(x)




#%% 1 | 随机信号分析与应用：从自相关到功率谱密度的探讨
## https://blog.csdn.net/m0_47410750/article/details/127641729
# https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485665&idx=1&sn=657320b37df1a053cb9486888df066a4&chksm=c0433bf3cc988f20e4451ab260daa1fe94019f0a77a5869617e7c4e59e1f6254ad074afe6035&mpshare=1&scene=1&srcid=0211LzimJAebickSiZy5yhvg&sharer_shareinfo=5c628ed946d7332c6507b4fd15aebf24&sharer_shareinfo_first=522f6c581162c8ec98c461f62c781d2d&exportkey=n_ChQIAhIQvGBNrciFIxFVklHBGphjTRKfAgIE97dBBAEAAAAAAJtdKq50wGQAAAAOpnltbLcz9gKNyK89dVj0CEpWnFvD4NPv5QY6sd5ErmN2lb99E%2BiR6fh%2Brtm8MTWHjWqrhzjEFlO0gvvs8XlB13YCMh9%2FZROZpY0OabfqTG0%2BUcTEFghLGexNWsG5ZsNOIvw8vt4RFe3ynmd5dionQthsl9sp69hHRZToLpa0jKhIIC7Hvz1zFbKu6dDVEm%2BUiDP63tLG4eJxGqHqe4NOlg%2BMpe28TQLJp3XAYJ7IzqMjxbCSFMvDamJJFhqFOoeDer0HyLs2bcZcv9IpxahmCe32pRtPFTGQSPLTCj%2BKsmDbb%2BAb1JaOb538GzNnOB%2FtChXaQ4tNgDHWHLCTLWrcp3eWOpAqLyq3&acctmode=0&pass_ticket=OCj6%2BS1NlxkGxuDLzbqGWrpCOER33t6dnQoMRy4Scc9Gda%2Fj26o8746YuacZ8J6r&wx_header=0#rd






























































































































































































































