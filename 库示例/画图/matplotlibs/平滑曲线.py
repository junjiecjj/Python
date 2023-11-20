#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:14:04 2023

@author: jack
"""


import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# (1) Savitzky-Golay 滤波器实现曲线平滑
"""

https://blog.csdn.net/kaever/article/details/105520941


scipy.signal.savgol_filter(x, window_length, polyorder)

    x为要滤波的信号
    window_length即窗口长度
    取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
    polyorder为多项式拟合的阶数。
    它越小，则平滑效果越明显；越大，则更贴近原始曲线。

    对上面的数据用savgol_filter进行滤波，从而平滑化。结果如下。其中w指window_length,k指polyorder

    w=41,k=2的平滑效果最明显。即window_length越大，polyorder越小，则平滑效果越强
    w=21,k=4最接近原曲线。即window_length越小，polyorder越大，则结果越接近原始曲线。

"""




import numpy as np
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter



Size = 100
X = np.linspace(1, Size,Size)
data = np.random.randint(1, Size, Size)

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

axs.plot(X, data, color='b', linestyle='-', label='origin',)
y = savgol_filter(data, 5, 3, mode= 'nearest')
# 可视化图线
axs.plot(X, y, 'r', label = 'savgol')



font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('X',fontproperties=font)
axs.set_ylabel('Y',fontproperties=font)


font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()

# plt.show()




# (2) 插值法对折线进行平滑曲线处理

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

x = np.array([6, 7, 8, 9, 10, 11, 12])
y = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])

x_smooth = np.linspace(x.min(), x.max(), 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
y_smooth = make_interp_spline(x, y)(x_smooth)

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)


axs.plot(x, y, color='b', linestyle='-', label='origin',)
axs.plot(x_smooth, y_smooth,  color='r', linestyle='-', label='interp_spline',)

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('X',fontproperties=font)
axs.set_ylabel('Y',fontproperties=font)

font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()







# 3、使用 scipy.interpolate.interp1d 插值类绘制平滑曲线

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt




x=np.array([1, 2, 3, 4, 5, 6, 7])
y=np.array([100, 50, 25, 12.5, 6.25, 3.125, 1.5625])


cubic_interploation_model=interp1d(x,y,kind="cubic")
xs=np.linspace(1, 7, 500)
ys=cubic_interploation_model(xs)

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)


axs.plot(x, y, color='b', linestyle='-', label='origin',)
axs.plot(xs, ys,  color='r', linestyle='-', label='interp1d',)


font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('X',fontproperties=font)
axs.set_ylabel('Y',fontproperties=font)

plt.title("Spline Curve Using Cubic Interpolation")

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()







# 4 使用 scipy.ndimage.gaussian_filter1d() 高斯核类绘制平滑曲线



import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


x=np.array([1,2,3,4,5,6,7])
y=np.array([100,50,25,12.5,6.25,3.125,1.5625])
y_smoothed = gaussian_filter1d(y, sigma=5)

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)


axs.plot(x, y, color='b', linestyle='-', label='origin',)
axs.plot(x, y_smoothed,  color='r', linestyle='-', label='gaussian_filter1d',)


font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明



font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('X',fontproperties=font)
axs.set_ylabel('Y',fontproperties=font)

plt.title("Spline Curve Using the Gaussian Smoothing")

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()



# 信号去噪
# 原创 FunIO FunIO
## 5  均值滤波
import numpy as np
import matplotlib.pyplot as plt


# 生成示例信号，包含高频噪声
np.random.seed(0)
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(1000)

# 绘制原始信号
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('original')

# 进行均值滤波去除高频噪声
window_size = 20  # 滤波窗口大小
filtered_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# 绘制去噪后的信号
plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal)
plt.title('filtered')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()


## 6  以下是使用 Python 和 NumPy 库进行中值滤波的示例代码：
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

# Create a sample signal with noise
np.random.seed(0)
signal = np.sin(2 * np.pi * 0.01 * np.arange(0, 1000)) + 0.5 * np.random.randn(1000)

# Apply median filtering
window_size = 5  # Window size
smooth_signal = medfilt(signal, kernel_size=window_size)

# Plot the original and smoothed signals
plt.figure(figsize=(10, 4))
plt.plot(signal, label='Original Signal', alpha=0.7)
plt.plot(smooth_signal, label=f'Median Filter (Window Size {window_size})', color='green')
plt.legend()
plt.title('Median Filtering Example')
plt.xlabel('Sample Points')
plt.ylabel('Amplitude')
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
plt.show()














































































































































































































































































































