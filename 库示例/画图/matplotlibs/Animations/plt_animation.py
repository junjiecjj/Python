#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:15:31 2023

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% 例子1: 每一轮 x 不变，清空并一次性更新 y

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

x = np.linspace(0, 2 * np.pi, 5000)
y = np.exp(-x) * np.cos(2 * np.pi * x)
line, = ax.plot(x, y, color="cornflowerblue", lw=3)
ax.set_ylim(-1.1, 1.1)

# 清空当前帧
def init():
    line.set_ydata([np.nan] * len(x))
    return line,

# 更新新一帧的数据
def update(frame):
    line.set_ydata(np.exp(-x) * np.cos(2 * np.pi * x + float(frame)/100))
    return line,

# 调用 FuncAnimation
ani = FuncAnimation(fig
                   ,update
                   ,init_func=init
                   ,frames=200
                   ,interval=2
                   ,blit=True
                   )

ani.save("1.gif", fps=25, writer="imagemagick")

#%% 例子2: 每一轮同时新增 x 和 y 的一个点

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = plt.plot([], [], "r-", animated=True)
x = []
y = []

def init():
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    x.append(frame)
    y.append(np.sin(frame))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig
                   ,update
                   ,frames=np.linspace(-np.pi ,np.pi, 90)
                   ,interval=10
                   ,init_func=init
                   ,blit=True
                   )
ani.save("2.gif", fps=25, writer="imagemagick")

######################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = plt.plot([], [], "ro", mfc = 'none', markersize=8, alpha=0.7, animated=True)
x = []
y = []

def init():
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    x.append(frame)
    y.append(np.sin(frame))
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig
                   ,update
                   ,frames=np.linspace(-np.pi ,np.pi, 90)
                   ,interval=10
                   ,init_func=init
                   ,blit=True
                   )
ani.save("2_1.gif", fps=25, writer="imagemagick")

#%% 例子3: 同时画两条线，每轮新增两个点

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
line1, = ax.plot(x, y1, color='k')
line2, = ax.plot(x, y2, color='b')

def init():
    return line1, line2,

def update(num):
    line1.set_data(x[:num], y1[:num])
    line2.set_data(x[:num], y2[:num])
    return line1, line2

ani = FuncAnimation(fig
                   ,update
                   ,init_func=init
                   ,frames=len(x)
                   ,interval=25
                   ,blit=True
                   )

ani.save("3.gif", fps=25, writer="imagemagick")

#%% 例子4: 同时画两条线，每轮重新画两条线

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm
fontpath = "/usr/share/fonts/truetype/windows/"
font = fm.FontProperties(fname=fontpath+"simsun.ttf",)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("动态图", fontproperties=font)
ax.grid(True)
ax.set_xlabel("X轴", fontproperties=font)
ax.set_ylabel("Y轴", fontproperties=font)
line1, = ax.plot([], [], "b--", linewidth=2.0, label="sin示例")
line2, = ax.plot([], [], "g+", linewidth=2.0, label="cos示例")
ax.legend(loc="upper left", prop=font, shadow=True)

def init():
    line1, = ax.plot([], [], "b--", linewidth=2.0, label="sin示例")
    line2, = ax.plot([], [], "g+", linewidth=2.0, label="cos示例")
    return line1, line2

def update(frame):
    x = np.linspace(-np.pi + 0.1 * frame, np.pi + 0.1 * frame, 256, endpoint=True)
    y_cos, y_sin = np.cos(x), np.sin(x)

    ax.set_xlim(-4 + 0.1 * frame, 4 + 0.1 * frame)
    ax.set_xticks(np.linspace(-4 + 0.1 * frame, 4 + 0.1 * frame, 9, endpoint=True))
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.linspace(-1, 1, 9, endpoint=True))

    line1, = ax.plot(x, y_cos, "b--", linewidth=2.0, label="sin示例")
    line2, = ax.plot(x, y_sin, "g+", linewidth=2.0, label="cos示例")

    return line1, line2

ani = FuncAnimation(fig
                   ,update
                   ,init_func=init
                   ,frames=np.linspace(-5 ,5, 5)
                   ,interval=1000
                   ,blit=True
                   )

ani.save("4.gif", fps=1, writer="imagemagick")


#%%
#初始化画布
fig = plt.figure()
plt.grid(ls='--')

#绘制一条正弦函数曲线
x = np.linspace(0,2*np.pi,100)
y = np.sin(x)

crave_ani = plt.plot(x,y,'red',alpha=0.5)

#绘制曲线上的切点
point_ani = plt.plot(0,0,'r',alpha=0.4,marker='o')[0]

#绘制x、y的坐标标识
xtext_ani = plt.text(5,0.8,'',fontsize=12)
ytext_ani = plt.text(5,0.7,'',fontsize=12)
ktext_ani = plt.text(5,0.6,'',fontsize=12)

#计算切线的函数
def tangent_line(x0, y0, k):
	xs = np.linspace(x0 - 0.5, x0 + 0.5,100)
	ys = y0 + k * (xs - x0)
	return xs, ys

#计算斜率的函数
def slope(x0):
	num_min = np.sin(x0 - 0.05)
	num_max = np.sin(x0 + 0.05)
	k = (num_max - num_min) / 0.1
	return k

#绘制切线
k = slope(x[0])
xs, ys = tangent_line(x[0], y[0], k)
tangent_ani = plt.plot(xs, ys, c='blue', alpha=0.8)[0]

#更新函数
def updata(num):
	k = slope(x[num])
	xs, ys = tangent_line(x[num], y[num], k)
	tangent_ani.set_data(xs, ys)
	point_ani.set_data(x[num],y[num])
	xtext_ani.set_text('x=%.3f'%x[num])
	ytext_ani.set_text('y=%.3f'%y[num])
	ktext_ani.set_text('k=%.3f'%k)
	return [point_ani,xtext_ani,ytext_ani,tangent_ani,k]

ani = animation.FuncAnimation(fig=fig, func=updata, frames=np.arange(0,100), interval=100)
ani.save('sin_x.gif',  fps=25, writer="imagemagick")
plt.show()
















