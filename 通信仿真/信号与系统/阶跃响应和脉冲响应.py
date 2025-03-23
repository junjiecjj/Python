#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:49:34 2025

@author: jack
"""
# https://blog.csdn.net/ouening/article/details/53074839
# https://blog.csdn.net/Yuxin_007/article/details/136665878

import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 20        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [4, 3] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%%
s1 = scipy.signal.lti([3],[1,2,10])    # 以分子分母的最高次幂降序的系数构建传递函数，s1=3/(s^2+2s+10）
s2 = scipy.signal.lti([1],[1,0.4,1])   # s2=1/(s^2+0.4s+1)
s3 = scipy.signal.lti([5],[1,2,5])     # s3=5/(s^2+2s+5)

t1, y1 = scipy.signal.step(s1)         # 计算阶跃输出，y1是Step response of system.
t2, y2 = scipy.signal.step(s2)
t3, y3 = scipy.signal.step(s3)
t11, y11 = scipy.signal.impulse(s1)
t22, y22 = scipy.signal.impulse(s2)
t33, y33 = scipy.signal.impulse(s3)

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = (12, 6), sharex = 'col', sharey = 'row') # 开启subplots模式
ax1.plot(t1, y1, 'r', label = 's1 Step Response',  )
ax1.set_title('s1 Step Response', fontsize = 21)
ax2.plot(t2,y2,'g', label = 's2 Step Response',  )
ax2.set_title('s2 Step Response', fontsize = 21)
ax3.plot(t3,y3,'b', label = 's3 Step Response', )
ax3.set_title('s3 Step Response', fontsize = 21)

ax4.plot(t11, y11, 'm', label = 's1 Impulse Response', )
ax4.set_title('s1 Impulse Response', fontsize = 21)
ax5.plot(t22, y22, 'y', label = 's2 Impulse Response', )
ax5.set_title('s2 Impulse Response', fontsize = 21)
ax6.plot(t33, y33, 'k', label = 's3 Impulse Response', )
ax6.set_title('s3 Impulse Response', fontsize = 21)

plt.show()
plt.close()


def step_plot(s):
    y,t = scipy.signal.step(s)
    f, axs = plt.subplots(1, 1, figsize=(6, 4), )
    axs.plot(y, t,'b', )
    axs.set_title('Step Response',  )
    axs.set_xlabel('Time(seconds)', )
    axs.set_ylabel('Amplitude', )
    plt.show()
    plt.close()
def impulse_plot(s):
    y, t= scipy.signal.impulse(s)
    f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex = 'col',sharey='row')
    axs.plot(y, t,'b', )
    axs.set_title('Impulse Response' )
    axs.set_xlabel('Time(seconds)', )
    axs.set_ylabel('Amplitude', )
    plt.show()
    plt.close()
s = scipy.signal.lti([4],[1,2,10,8])
step_plot(s)
impulse_plot(s)

#%% https://blog.csdn.net/Yuxin_007/article/details/136665919
import numpy as np
# from scipy.signal import lti, step, impulse
import matplotlib.pyplot as plt

# 定义系统参数
wn = 1.0  # 自然频率
zeta = 0.5  # 阻尼比

# 创建LTI系统
# 传递函数为 H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
numerator = [wn**2]
denominator = [1, 2*zeta*wn, wn**2]
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html#scipy.signal.lti
system = scipy.signal.lti(numerator, denominator)

# 模拟阶跃响应
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html#scipy.signal.step
t1, step_response = scipy.signal.step(system)

# 模拟冲击响应
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html#scipy.signal.impulse
t2, impulse_response = scipy.signal.impulse(system)

# 绘制阶跃响应
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t1, step_response)
plt.title('Step Response')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# 绘制冲击响应
plt.subplot(1, 2, 2)
plt.plot(t2, impulse_response)
plt.title('Impulse Response')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
plt.close()

#%% https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html#scipy.signal.step
lti = scipy.signal.lti([1.0], [1.0, 1.0])
t, y = scipy.signal.step(lti)
f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex = 'col',sharey='row')
axs.plot(t, y, 'b', )
axs.set_title('Step response for 1. Order Lowpass' )
axs.set_xlabel('Time(seconds)', )
axs.set_ylabel('Amplitude', )
plt.show()
plt.close()

system = ([1.0], [1.0, 2.0, 1.0])
t, y = scipy.signal.impulse(system)
f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex = 'col',sharey='row')
axs.plot(t, y, 'b', )
axs.set_title('Impulse Response' )
axs.set_xlabel('Time(seconds)', )
axs.set_ylabel('Amplitude', )
plt.show()
plt.close()


#%% https://blog.51cto.com/u_16175450/13134090
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# 定义一阶低通滤波器的传递函数
# H(s) = 1 / (s + 1)
numerator = [3]
denominator = [1,2,10]

# 创建LTI系统
system = scipy.signal.lti(numerator, denominator)

# 定义时间参数
t = np.linspace(0, 10, 500)  # 0到10秒，500个点

# 创建冲击信号（单位脉冲）
impulse = np.zeros_like(t)
impulse[100] = 1  # 在t=0时刻生成单位脉冲

# 计算系统响应
t_out, y_out, _ = scipy.signal.lsim(system, impulse, t)

# 绘制结果
f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex = 'col',sharey='row')
# plt.figure(figsize=(10, 5))
axs.plot(t_out, y_out, label='冲击响应', color='b')
axs.set_title('一阶低通滤波器的冲击响应')
axs.set_xlabel('时间 (秒)')
axs.set_ylabel('输出')
axs.grid()
plt.legend()
axs.axhline(0, color='black',linewidth=0.5, ls='--')
axs.axvline(0, color='black',linewidth=0.5, ls='--')
plt.show()
plt.close()


#%%
import numpy as np
from scipy.signal import bessel, lsim
import matplotlib.pyplot as plt

b, a = bessel(N = 5, Wn = 2*np.pi*12, btype = 'lowpass', analog = True)
t = np.linspace(0, 1.25, 500, endpoint = False)
u = (np.cos(2*np.pi*4*t) + 0.6*np.sin(2*np.pi*40*t) + 0.5*np.cos(2*np.pi*80*t))

tout, yout, xout = lsim((b, a), U = u, T = t)
f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex = 'col',sharey='row')
axs.plot(t, u, 'r', alpha = 0.5, linewidth = 1, label = 'input')
axs.plot(tout, yout, 'k', linewidth = 1.5, label = 'output')
axs.legend(loc = 'best', shadow = True, framealpha = 1)
axs.set_xlabel('t')
plt.grid(alpha = 0.3)
plt.show()


#%%









#%%











#%%













#%%





























