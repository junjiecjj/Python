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
font = FontProperties(fname = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

#%% BOOK: <Wireless Communication Systems in Matlab> Program 1.20:
## Generating a sinusoid and plotting its power spectrum
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
axs[0].set_xlabel('time (s)',fontproperties=font)
axs[0].set_ylabel('x',fontproperties=font)

legend1 = axs[0].legend(loc='best', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

# Px
axs[1].stem(f, np.abs(Px), label = 'PSD', linefmt = 'r--', markerfmt = 'o', )
axs[1].set_xlabel('Fre (Hz)',)
axs[1].set_ylabel('power',)

legend1 = axs[1].legend(loc='best', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)

out_fig = plt.gcf()
plt.show()
plt.close()


#%% https://matplotlib.net.cn/stable/gallery/lines_bars_and_markers/xcorr_acorr_demo.html
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(42)

fs = 100
f = 2
t = np.arange(0, 1, 1/fs)
x = np.sin(2 * np.pi * f * t)
y = np.random.randn(x.size)
# fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
fig, axs = plt.subplots(2, 1, figsize = (8, 6), constrained_layout = True,  sharex=True)
idx_1, xcor_1, _, _ = axs[0].xcorr(x, y, usevlines = True, maxlags = 50, normed = True, lw = 2)
axs[0].grid(True)
axs[0].set_title('Cross-correlation (xcorr)')

idx_2, acor_2, _, _  = axs[1].acorr(x, usevlines = True, normed = True, maxlags = 50, lw = 2)
axs[1].grid(True)
axs[1].set_title('Self-correlation (acorr)')

plt.show()

# https://blog.51cto.com/u_16213453/12732833
# 计算自相关函数
def autocorrelation(x):
    n = len(x)
    # variance = x.var()
    mean = x.mean()
    c0 = np.sum((x - mean) ** 2) / n
    result = np.correlate(x - mean, x - mean, mode = 'same')
    result /= c0
    return result

scorr = autocorrelation(x)

#%%

import numpy as np
import scipy
rng = np.random.default_rng()

x = rng.standard_normal(1000)
y = np.concatenate([rng.standard_normal(100), x])
correlation = scipy.signal.correlate(x, y, mode = "full")
lags = scipy.signal.correlation_lags(x.size, y.size, mode = "full")
lag = lags[np.argmax(correlation)]

#%% 1 | 随机信号分析与应用：从自相关到功率谱密度的探讨
## https://blog.csdn.net/m0_47410750/article/details/127641729
# https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485665&idx=1&sn=657320b37df1a053cb9486888df066a4&chksm=c0433bf3cc988f20e4451ab260daa1fe94019f0a77a5869617e7c4e59e1f6254ad074afe6035&mpshare=1&scene=1&srcid=0211LzimJAebickSiZy5yhvg&sharer_shareinfo=5c628ed946d7332c6507b4fd15aebf24&sharer_shareinfo_first=522f6c581162c8ec98c461f62c781d2d&exportkey=n_ChQIAhIQvGBNrciFIxFVklHBGphjTRKfAgIE97dBBAEAAAAAAJtdKq50wGQAAAAOpnltbLcz9gKNyK89dVj0CEpWnFvD4NPv5QY6sd5ErmN2lb99E%2BiR6fh%2Brtm8MTWHjWqrhzjEFlO0gvvs8XlB13YCMh9%2FZROZpY0OabfqTG0%2BUcTEFghLGexNWsG5ZsNOIvw8vt4RFe3ynmd5dionQthsl9sp69hHRZToLpa0jKhIIC7Hvz1zFbKu6dDVEm%2BUiDP63tLG4eJxGqHqe4NOlg%2BMpe28TQLJp3XAYJ7IzqMjxbCSFMvDamJJFhqFOoeDer0HyLs2bcZcv9IpxahmCe32pRtPFTGQSPLTCj%2BKsmDbb%2BAb1JaOb538GzNnOB%2FtChXaQ4tNgDHWHLCTLWrcp3eWOpAqLyq3&acctmode=0&pass_ticket=OCj6%2BS1NlxkGxuDLzbqGWrpCOER33t6dnQoMRy4Scc9Gda%2Fj26o8746YuacZ8J6r&wx_header=0#rd

# https://thinkdsp-cn.readthedocs.io/zh-cn/latest/05-autocorrelation.html#id8

# https://blog.csdn.net/m0_47410750/article/details/127641729

# 应用一：信号的周期性检测
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import statsmodels.tsa.api as smt

from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

f0 = 5
T = 1
fs = 1000
t = np.arange(0, T, 1/fs)
x_sin = np.sin(2 * np.pi * f0 * t)
x_rand = np.random.randn(x_sin.size)

##>>>>>>  method 1
acf_sin, lag_sin = xcorr(x_sin, x_sin, normed = True, detrend = 1, maxlags = x_sin.size - 1)
acf_rand, lag_rand = xcorr(x_rand, x_rand, normed = True, detrend = 1, maxlags = x_rand.size - 1)

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(lag_sin/fs, acf_sin, label = 'x')
axs[0].set_title("正弦波信号的自相关函数")
axs[0].set_xlabel('滞后时间 (秒)',)
axs[0].set_ylabel('自相关系数',)

# noise
axs[1].plot(lag_rand/fs, acf_rand, label = 'PSD',  )
axs[1].set_title("随机信号的自相关函数")
axs[1].set_xlabel('滞后时间 (秒)')
axs[1].set_ylabel('自相关系数')

out_fig = plt.gcf()
plt.suptitle("xcorr",)
plt.show()
plt.close()

##>>>>>>  method 2
acf_sin1 = smt.stattools.acf(x_sin, nlags = x_sin.size - 1, )
acf_rand1 = smt.stattools.acf(x_rand, nlags = x_rand.size - 1, )

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(acf_sin1, label = 'x')
axs[0].set_title("正弦波信号的自相关函数")
axs[0].set_xlabel('滞后时间 (秒)',)
axs[0].set_ylabel('自相关系数',)

# noise
axs[1].plot(acf_rand1, label = 'PSD',  )
axs[1].set_title("随机信号的自相关函数")
axs[1].set_xlabel('滞后时间 (秒)')
axs[1].set_ylabel('自相关系数')

out_fig = plt.gcf()
plt.suptitle("smt.stattools.acf",)
plt.show()
plt.close()

##>>>>>>  method 3
acf_sin2 = correlate_maxlag(x_sin, x_sin, x_sin.size - 1, method = 'auto')
acf_rand2 = correlate_maxlag(x_rand, x_rand, x_rand.size - 1, method = 'auto')

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(get_lags(acf_sin2)/fs, acf_sin2, label = 'x')
axs[0].set_title("正弦波信号的自相关函数")
axs[0].set_xlabel('滞后时间 (秒)',)
axs[0].set_ylabel('自相关系数',)

# noise
axs[1].plot(get_lags(acf_rand2)/fs, acf_rand2, label = 'PSD',  )
axs[1].set_title("随机信号的自相关函数")
axs[1].set_xlabel('滞后时间 (秒)')
axs[1].set_ylabel('自相关系数')

out_fig = plt.gcf()
plt.suptitle("correlate_maxlag",)
plt.show()
plt.close()

##>>>>>>  method 4
X_sin = x_sin - np.mean(x_sin)
X_rand = x_rand - np.mean(x_rand)
n = np.sqrt(np.linalg.norm(X_sin)**2 * np.linalg.norm(X_sin)**2)
acf_sin3 = np.correlate(X_sin, X_sin, mode = 'full') / n
n = np.sqrt(np.linalg.norm(X_rand)**2 * np.linalg.norm(X_rand)**2)
acf_rand3 = np.correlate(X_rand, X_rand, mode = 'full') / n

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(acf_sin3, label = 'x')
axs[0].set_title("正弦波信号的自相关函数")
axs[0].set_xlabel('滞后时间 (秒)',)
axs[0].set_ylabel('自相关系数',)

# noise
axs[1].plot(acf_rand3, label = 'PSD',  )
axs[1].set_title("随机信号的自相关函数")
axs[1].set_xlabel('滞后时间 (秒)')
axs[1].set_ylabel('自相关系数')

out_fig = plt.gcf()
plt.suptitle("np.correlate",)
plt.show()
plt.close()


#%% 应用二：信号噪声分离

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import statsmodels.tsa.api as smt

from Xcorrs import xcorr

f0 = 5
T = 1
fs = 1000
t = np.arange(0, T, 1/fs)
x_sin = np.sin(2 * np.pi * f0 * t)
x_rand = np.random.randn(x_sin.size)
Y = x_sin + x_rand;

##>>>>>>  method 1
acf_sin, lag_sin = xcorr(Y, Y, normed = True, detrend = 1, maxlags = x_sin.size - 1)
acf_rand, lag_rand = xcorr(x_rand, x_rand, normed = True, detrend = 1, maxlags = x_rand.size - 1)

##### plot
fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(lag_sin/fs, acf_sin, label = 'x')
axs[0].set_title("总信号 (正弦波 + 噪声) 的自相关函数")
axs[0].set_xlabel('滞后时间 (秒)',)
axs[0].set_ylabel('自相关系数',)

# noise
axs[1].plot(lag_rand/fs, acf_rand, label = 'PSD',  )
axs[1].set_title("纯噪声信号的自相关函数")
axs[1].set_xlabel('滞后时间 (秒)')
axs[1].set_ylabel('自相关系数')

out_fig = plt.gcf()
plt.suptitle("xcorr",)
plt.show()
plt.close()










































































































































































































