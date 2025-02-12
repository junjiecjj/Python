#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:59:54 2022

@author: jack


https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.fft.html
https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
https://vimsky.com/examples/usage/python-numpy.fft.fftshift.html
https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html
https://zhuanlan.zhihu.com/p/559711158
https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html

本函数是根据scipy和numpy中的fft模块实现FFT，且测试：
numpy和scipy中 都有 fft, fftshift，fftfreq, 测试它们的区别；
结果表明numpy和scipy中fft, fftshift, fftfreq三者完全一样;

并验证fft和根据公式自己编写程序的结果是否一样，表明一样;

#================================ Numpy.fft.fftshift ==================================
a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
fftshift(a)
Out[136]: array([6, 7, 8, 9, 1, 2, 3, 4, 5])

b = [1, 2, 3, 4, 5, 6, 7, 8]
fftshift(b)
Out[138]: array([5, 6, 7, 8, 1, 2, 3, 4])

a = [1, 2, 3, ... n]
fftshift(a):
[(n+2)/2,...,n-1,n, 1,2,...,n/2]           n为偶数
[(n+3)/2,...,n-1,n, 1,2,...,(n+1)/2]       n为基数

#================================ Numpy.fft.fft ==================================
接口：
np.fft.fft(x, n=None, axis=- 1, norm=None)
scipy.fftpack.fft(x, n=None, axis=- 1, overwrite_x=False)
两者返回如下：
The returned complex array contains y(0), y(1),..., y(n-1), where
y(k) = (x * exp(-2*pi*sqrt(-1)*k*np.arange(n)/n)).sum().

x = {x[0], x[1], x[2],..., x[N-1]}
y[k] = \sum_{n=0}^{N-1} x[n]e^{-j*2*pi*k*n /N}

Parameters
 x: Array to Fourier transform.

    n, optional,Length of the Fourier transform. If n < x.shape[axis], x is truncated. If n > x.shape[axis], x is zero-padded. The default results in n = x.shape[axis].

    axisint, optional, Axis along which the fft’s are computed; the default is over the last axis (i.e., axis=-1).

    overwrite_xbool, optional. If True, the contents of x can be destroyed; the default is False.

Returns：
zcomplex ndarray with the elements:
    [y(0), y(1),.., y(n/2), y(1-n/2),..., y(-1)]         n为偶数
    [y(0), y(1),.., y((n-1)/2), y(-(n-1)/2),..., y(-1)]  n为奇数
    where:
    y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

1) fft函数返回的fft结果序列的前半部分对应[0, fs/2]是正频率的结果,后半部分对应[ -fs/2, 0]是负频率的结果。
2) 如果要让实信号fft的结果与[-fs/2, fs/2]对应，则要fft后fftshift一下即可，fftshift的操作是将fft结果以fs/2为中心左右互换

#================================ Numpy.fft.fftfreq ==================================
Numpy.fft.fftfreq:
fft.fftfreq(n, d = 1.0)
    返回离散傅里叶变换采样频率。也就是返回与fft返回结果对应的频率值，长度n应该与fft后的序列长度一样，
    返回的浮点数组 f 包含频率 bin 中心，以每单位样本间隔的周期为单位(开头为零)。例如，如果样本间隔以秒为单位，则频率单位为周期/秒.
    给定窗口长度 n 和样本间距 d:
    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   n为偶数，如n=10, f = [0,1,2,3,4,-5,-4,-3,-2,-1]
    f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   n为奇数，如n=9,  f = [0,1,2,3,4,-4,-3,-2,-1]
    参数：
    n：int，窗口长度。
    d：标量，可选,采样间隔(采样率的倒数)。默认为 1。
    返回：
    f：ndarray
    包含样本频率的长度为 n 的数组。

    我们在画频谱图的时候需要对信号做傅里叶变换，x轴对应的是频率范围，
    使用fftfreq的好处就是可以自动生成频率范围，而不用去考虑信号长度是奇数还是偶数的问题。
    处理实际信号的时候这样使用：fftfreq(len(signal),1/samplerate),即 fftfreq(信号长度，1/采样率)。注意这里的信号长度是信号做傅里叶变换之后的原始长度（画半频谱图的时候则取做傅里叶变换之后的原始长度的前一半长度作为y轴）, 得到频率范围后发现频率有负值，这时同样对频率取一半即可。
    采样率是每秒钟采样点的数量，

fftshift和fftfreq在产生正确的半谱图和全谱图时这么配合使用：
(一)在画半谱图时:
    方法1：
    将fft结果序列/N(除以N)，然后取前半部分(前半部分对应[0, fs/2]是正频率的结果),然后x2(乘以2)，作为纵轴, 然后：
    定义序列 Y 对应的频率刻度
    df=Fs/N;                           # 频率间隔
    f=np.arange(0,int(N/2)+1)*df;      # 频率刻度
    产生f作为横轴，画图;

    方法2：方法二是错的，错在f的最后一个应该是 (N/2+1)*df,而实际上是-N/2*df
    将fft结果序列/N(除以N)，然后取前半部分(前半部分对应[0, fs/2]是正频率的结果),然后x2(乘以2)，作为纵轴, 然后：
    f = np.fft.fftfreq(N, d=1/Fs)的前半段作为横轴
    产生f作为横轴，画图;

(二)在画全谱图时:
    方法1：
    将fft结果序列/N(除以N)，然后fftshift后作为纵轴, 然后：
    定义序列 Y 对应的频率刻度
    df=Fs/N;                                  # 频率间隔
    f=np.arange(-int(N/2),int(N/2))*df;       # 频率刻度
    产生f作为横轴，画图;

    方法2：
    将fft结果序列/N(除以N)，然后fftshift后作为纵轴, 然后：
    定义序列 Y 对应的频率刻度
    f = np.fft.fftshift(fftfreq(N,1/Fs))
    产生f作为横轴，画图;

    方法3：
    将fft结果序列/N(除以N)，然后不需要fftshift，直接作为纵轴, 然后：
    f = np.fft.fftfreq(N,1/Fs) 作为横轴
    产生f作为横轴，画图;

"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
# from matplotlib.pyplot import MultipleLocator
import scipy
# from scipy.fftpack import fft,ifft,fftshift,fftfreq

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete Mono.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=20)


fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
fonttX = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttY = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttitle = {'style': 'normal', 'size': 17}
fontt2 = {'style': 'normal', 'size': 19, 'weight': 'bold'}
fontt3 = {'style': 'normal', 'size': 16, }


# FFT变换，慢的版本
def FFT(xx):
     N = len(xx)
     X = np.zeros(N, dtype = complex) # 频域频谱
     # DTF变换
     for k in range(N):
         for n in range(N):
             X[k] = X[k] + xx[n]*np.exp(-1j*2*np.pi*n*k/N)
     return X

# IFFT变换，慢的版本
def IFFT(XX):
     N = len(XX)
     # IDFT变换
     x_p = np.zeros(N, dtype = complex)
     for n in range(N):
          for k in range(N):
               x_p[n] = x_p[n] + 1/N*X[k]*np.exp(1j*2*np.pi*n*k/N)
     return x_p

#%%======================================================
# ===========  定义时域采样信号 cos(x)
#======================================================

# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.cos(2*np.pi*f1*t)

#%%======================================================
# ===========  定义时域采样信号 cos(x + theta)
#======================================================

# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.cos(2*np.pi*f1*t+np.pi/4)

#%% ======================================================
# ===========  定义时域采样信号 sin(x) = cos(pi/2 - x)
# ======================================================
# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.sin(2*np.pi*f1*t )

#%%======================================================
# ## ===========  定义时域采样信号 sin(x + np.pi/4)
# ## ======================================================
# Fs = 10                          # 信号采样频率
# Ts = 1/Fs                        # 采样时间间隔
# N = 100                           # 采样信号的长度
# t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

# f1 = 2                             # 第一个余弦信号的频率
# x =  np.sin(2*np.pi*f1*t + np.pi/4) # = cos(pi/4 - x) = sin(x - pi/4)

#%%======================================================
# ===========  定义时域采样信号 x
#======================================================
# Fs = 100                      # 信号采样频率,Hz
# Ts = 1/Fs                     # 采样时间间隔
# N = 200                       # 采样信号的长度, N为偶数
# #N = 201                        # 采样信号的长度, N为奇数
# t = np.linspace(0, N-1, N)*Ts   # 定义信号采样的时间点 t.

# f1 = 16               # 第一个余弦信号的频率,Hz
# f2 = 45               # 第二个余弦信号的频率,Hz
# x = 4.5 + 2.7*np.cos(2*np.pi*f1*t - np.pi/4) + 8.2*np.cos(2*np.pi*f2*t-np.pi/6)      # 定义时域采样信号 x

#%%======================================================
## ===========  定义时域采样信号 x
##======================================================
# 定义时域采样信号 x
Fs = 1400                     # 信号采样频率
Ts = 1/Fs                     # 采样时间间隔
N = 1400                      # 采样信号的长度
t = np.linspace(0, N-1, N)*Ts    # 定义信号采样的时间点 t

f1 = 200
f2 = 400
f3 = 600
x =  7*np.cos(2*np.pi*f1*t + np.pi/4) + 5*np.cos(2*np.pi*f2*t + np.pi/2) + 3*np.cos(2*np.pi*f3*t + np.pi/3)  + 4.5 # (4.5是直流)

#%%=====================================================
# 对时域采样信号, 执行快速傅里叶变换 FFT
X = scipy.fftpack.fft(x)
# X = FFT(x)  # 或者用自己编写的，与 fft 一致

### IFFT
IX = scipy.fftpack.ifft(X)
# IX = IFFT(X)*N # 自己写的，和 ifft 一样

# 消除相位混乱
X[np.abs(X) < 1e-8] = 0     # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
X = X/N               # 将频域序列 X 除以序列的长度 N

#%%
#==================================================
# 半谱图
#==================================================
# 提取 X 里正频率的部分, 并且将 X 里负频率的部分合并到正频率
if N%2 == 0:
     Y = X[0 : int(N/2)+1].copy()                 # 提取 X 里正频率的部分,N为偶数
     Y[1 : int(N/2)] = 2*Y[1 : int(N/2)].copy()   # 将 X 里负频率的部分合并到正频率,N为偶数
else: # 奇数时下面的有问题
     Y = X[0 : int(N/2)+1].copy()                   # 提取 X 里正频率的部分,N为奇数
     Y[1 : int(N/2)+1] = 2*Y[1:int(N/2)+1].copy()   # 将 X 里负频率的部分合并到正频率,N为奇数

# 计算频域序列 Y 的幅值和相角
A = abs(Y)                        # 计算频域序列 Y 的幅值
Pha = np.angle(Y, deg=1)          # 计算频域序列 Y 的相角 (弧度制)
R = np.real(Y)                    # 计算频域序列 Y 的实部
I = np.imag(Y)                    # 计算频域序列 Y 的虚部

#  定义序列 Y 对应的频率刻度
df = Fs/N                                 # 频率间隔
if N%2==0:
     f = np.arange(0, int(N/2)+1)*df      # 频率刻度,N为偶数
      # f = scipy.fftpack.fftfreq(N, d=1/Fs)[0:int(N/2)+1] # 方法二:错的
else:#奇数时下面的有问题
     f = np.arange(0, int(N/2)+1)*df       # 频率刻度,N为奇数
#%%
#==================================================
# 全谱图
#==================================================

### 方法一，二：将 X 重新排列, 把负频率部分搬移到序列的左边, 把正频率部分搬移到序列的右边
Y1 = scipy.fftpack.fftshift(X, )      # 新的频域序列 Y
#Y1=X1

# 计算频域序列 Y 的幅值和相角
A1 = abs(Y1);                       # 计算频域序列 Y 的幅值
Pha1 = np.angle(Y1,deg=True)        # 计算频域序列 Y 的相角 (弧度制)
R1 = np.real(Y1)                    # 计算频域序列 Y 的实部
I1 = np.imag(Y1)                    # 计算频域序列 Y 的虚部

###  定义序列 Y 对应的频率刻度
df = Fs/N                           # 频率间隔
if N%2 == 0:
    # 方法一
    f1 = np.arange(-int(N/2),int(N/2))*df      # 频率刻度,N为偶数
    #或者如下， 方法二：
    f1 = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(N, 1/Fs))
else:#奇数时下面的有问题
    f1 = np.arange(-int(N/2),int(N/2)+1)*df    # 频率刻度,N为奇数

# ## 方法三
# # 将 X 不重新排列,
# Y1 = X

# # 计算频域序列 Y 的幅值和相角
# A1 = abs(Y1);                       # 计算频域序列 Y 的幅值
# Pha1 = np.angle(Y1,deg=True)        # 计算频域序列 Y 的相角 (弧度制)
# R1 = np.real(Y1)                    # 计算频域序列 Y 的实部
# I1 = np.imag(Y1)                    # 计算频域序列 Y 的虚部

# # 定义序列 Y 对应的频率刻度
# f1 =  scipy.fftpack.fftfreq(N, 1/Fs)    # 频率刻度
#%%
#==================================================
#     频率刻度错位
#==================================================

# X2 = scipy.fftpack.fft(x)

# # 消除相位混乱
# X2[np.abs(X2)<1e-8] = 0        # 将频域序列 X 中, 幅值小于 1e-8 的数值置零

# # 修正频域序列的幅值, 使得 FFT 变换的结果有明确的物理意义
# X2 = X2/N;            # 将频域序列 X 除以序列的长度 N

# 计算频域序列 Y 的幅值和相角
A2 = abs(X);                       # 计算频域序列 Y 的幅值
Pha2 = np.angle(X,deg=True)       # 计算频域序列 Y 的相角 (弧度制)
R2 = np.real(X)                   # 计算频域序列 Y 的实部
I2 = np.imag(X)                   # 计算频域序列 Y 的虚部

df = Fs/N                           # 频率间隔
if N%2 == 0:
    # 方法一
    f2 = np.arange(0, N)*df      # 频率刻度,N为偶数

#====================================== 开始画图 ===============================================
width = 4
high = 3
horvizen = 5
vertical = 3
fig, axs = plt.subplots(vertical, horvizen, figsize=(horvizen*width, vertical*high), constrained_layout=True)
labelsize = 20


#%% 半谱图
#======================================= 0,0 =========================================
axs[0,0].plot(t, x, color='b', linestyle='-', label='原始信号值',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,0].set_xlabel(r'时间(s)', fontproperties=font3)
axs[0,0].set_ylabel(r'原始信号值', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,1 =========================================
axs[0,1].plot(f, A, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[0,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[0,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,2 =========================================
axs[0,2].plot(f, Pha, color='g', linestyle='-', label='相位',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[0,2].set_ylabel(r'相位', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[0,2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,2].get_xticklabels() + axs[0,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,3 =========================================
axs[0,3].plot(f, R, color='cyan', linestyle='-', label='实部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[0,3].set_ylabel(r'实部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[0,3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,3].get_xticklabels() + axs[0,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 0,4 =========================================
axs[0,4].plot(f, I, color='#FF8C00', linestyle='-', label='虚部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[0,4].set_ylabel(r'虚部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[0,4].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[0,4].get_xticklabels() + axs[0,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#%% 全谱图
#======================================= 1,0 =========================================
axs[1,0].plot(t, IX, color='b', linestyle='-', label='恢复的信号值',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,0].set_xlabel(r'时间(s)', fontproperties=font3)
axs[1,0].set_ylabel(r'逆傅里叶变换信号值', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[1,0].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,1 =========================================
axs[1,1].plot(f1, A1, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,2 =========================================
axs[1,2].plot(f1, Pha1, color='g', linestyle='-', label='相位',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,2].set_ylabel(r'相位', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[1,2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,2].get_xticklabels() + axs[1,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#======================================= 1,3 =========================================
axs[1,3].plot(f1, R1, color='cyan', linestyle='-', label='实部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,3].set_ylabel(r'实部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[1,3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,3].get_xticklabels() + axs[1,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号
#======================================= 1,4 =========================================
axs[1,4].plot(f1, I1, color='#FF8C00', linestyle='-', label='虚部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[1,4].set_ylabel(r'虚部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[1,4].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[1,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[1,4].get_xticklabels() + axs[1,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#%% 频率刻度错位
#======================================= 2,0 =========================================


#======================================= 2,1 =========================================
axs[2,1].plot(f2, A2, color='r', linestyle='-', label='幅度',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,1].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[2,1].set_ylabel(r'幅度', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[2,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True, right=True, labelsize=12, width=3,)
labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(12) for label in labels]  # 刻度值字号


#======================================= 2,2 =========================================
axs[2,2].plot(f2, Pha2, color='g', linestyle='-', label='相位',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,2].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[2,2].set_ylabel(r'相位', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[2,2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2,2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,2].get_xticklabels() + axs[2,2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 2,3 =========================================
axs[2,3].plot(f2, R2, color='cyan', linestyle='-', label='实部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,3].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[2,3].set_ylabel(r'实部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[2,3].legend(loc='best', borderaxespad=0,  edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2,3].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,3].get_xticklabels() + axs[2,3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号


#======================================= 2,4 =========================================
axs[2,4].plot(f2, I2, color='#FF8C00', linestyle='-', label='虚部',)

font3 = FontProperties(fname=fontpath+"simsun.ttf", size=12)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2,4].set_xlabel(r'频率(Hz)', fontproperties=font3)
axs[2,4].set_ylabel(r'虚部', fontproperties=font3)
#axs[0,0].set_title('信号值', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 12}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=14)
legend1 = axs[2,4].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[2,4].tick_params(direction='in', axis='both',top=True,right=True, labelsize=labelsize, width=3,)
labels = axs[2,4].get_xticklabels() + axs[2,4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(labelsize) for label in labels]  # 刻度值字号

#================================= super ===============================================
out_fig = plt.gcf()
#out_fig.savefig(filepath2+'hh.eps',  bbox_inches='tight')
plt.show()
plt.close()





















