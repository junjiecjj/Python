#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:49:34 2025

@author: jack
"""
# https://blog.csdn.net/ouening/article/details/53074839
# https://blog.csdn.net/Yuxin_007/article/details/136665878

from scipy.signal import lti,step,impulse
import matplotlib.pyplot as plt

s1=lti([3],[1,2,10])    # 以分子分母的最高次幂降序的系数构建传递函数，s1=3/(s^2+2s+10）
s2=lti([1],[1,0.4,1])   # s2=1/(s^2+0.4s+1)
s3=lti([5],[1,2,5])     # s3=5/(s^2+2s+5)

t1,y1=step(s1)         # 计算阶跃输出，y1是Step response of system.
t2,y2=step(s2)
t3,y3=step(s3)
t11,y11=impulse(s1)
t22,y22=impulse(s2)
t33,y33=impulse(s3)

f,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,sharex='col',sharey='row') # 开启subplots模式
ax1.plot(t1,y1,'r',label='s1 Step Response',linewidth=0.5)
ax1.set_title('s1 Step Response',fontsize=9)
ax2.plot(t2,y2,'g',label='s2 Step Response',linewidth=0.5)
ax2.set_title('s2 Step Response',fontsize=9)
ax3.plot(t3,y3,'b',label='s3 Step Response',linewidth=0.5)
ax3.set_title('s3 Step Response',fontsize=9)

ax4.plot(t11,y11,'m',label='s1 Impulse Response',linewidth=0.5)
ax4.set_title('s1 Impulse Response',fontsize=9)
ax5.plot(t22,y22,'y',label='s2 Impulse Response',linewidth=0.5)
ax5.set_title('s2 Impulse Response',fontsize=9)
ax6.plot(t33,y33,'k',label='s3 Impulse Response',linewidth=0.5)
ax6.set_title('s3 Impulse Response',fontsize=9)

##plt.xlabel('Times')
##plt.ylabel('Amplitude')
#plt.legend()
plt.show()



#自定义step_plot()
import numpy as np
# import control as ctl
import matplotlib.pyplot as plt

def step_plot(s):
    y,t = step(s)
    plt.plot(y, t,'b',linewidth=0.6)
    plt.title('Step Response',fontsize=9)
    plt.xlabel('Time(seconds)',fontsize=9)
    plt.ylabel('Amplitude',fontsize=9)
    plt.show()

def impulse_plot(s):
    y,t= impulse(s)
    plt.plot(y, t,'b',linewidth=0.6)
    plt.title('Impulse Response',fontsize=9)
    plt.xlabel('Time(seconds)',fontsize=9)
    plt.ylabel('Amplitude',fontsize=9)
    plt.show()


s = lti([4],[1,2,10,8])
step_plot(s)
impulse_plot(s)



#%% https://blog.51cto.com/u_16175450/13134090
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# 定义一阶低通滤波器的传递函数
# H(s) = 1 / (s + 1)
numerator = [1]
denominator = [1, 1]

# 创建LTI系统
system = lti(numerator, denominator)

# 定义时间参数
t = np.linspace(0, 10, 500)  # 0到10秒，500个点

# 创建冲击信号（单位脉冲）
impulse = np.zeros_like(t)
impulse[0] = 1  # 在t=0时刻生成单位脉冲

# 计算系统响应
t_out, y_out, _ = lsim(system, impulse, t)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(t_out, y_out, label='冲击响应', color='b')
plt.title('一阶低通滤波器的冲击响应')
plt.xlabel('时间 (秒)')
plt.ylabel('输出')
plt.grid()
plt.legend()
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.show()





#%% https://blog.csdn.net/Yuxin_007/article/details/136665919
import numpy as np
from scipy.signal import lti, step, impulse
import matplotlib.pyplot as plt

# 定义系统参数
wn = 1.0  # 自然频率
zeta = 0.5  # 阻尼比

# 创建LTI系统
# 传递函数为 H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
numerator = [wn**2]
denominator = [1, 2*zeta*wn, wn**2]
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lti.html#scipy.signal.lti
system = lti(numerator, denominator)

# 模拟阶跃响应
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html#scipy.signal.step
t1, step_response = step(system)

# 模拟冲击响应
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html#scipy.signal.impulse
t2, impulse_response = impulse(system)

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






























