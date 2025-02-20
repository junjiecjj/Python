#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:33:55 2025

@author: jack
"""
#%% https://geek-docs.com/matplotlib/matplotlib-ask-answer/plt-subplots_z1.html#google_vignette
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x), label='Sine')
ax2.plot(x, np.cos(x), label='Cosine')
ax3.plot(x, np.tan(x), label='Tangent')

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Function Plot - how2matplotlib.com')

plt.tight_layout()
plt.show()

#%% https://cloud.tencent.com/developer/article/2223274
# 非均匀绘图
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6, 6)) #设置画布大小为6英寸*6英寸
ax1 = plt.subplot(221) #表示将画布分为2行2列，索引为1的子区
ax2 = plt.subplot(222) #表示将画布分为2行2列，索引为2的子区
ax3 = plt.subplot(212) #表示将画布分为2行1列，索引为2的子区

plt.show()

#%% https://zhuanlan.zhihu.com/p/200700834
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
# plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


# -----基础数据-----
t = np.linspace(-2*np.pi, 2*np.pi, 50, endpoint=False)

x1 = np.exp(t/3) * np.sin(t/2 + np.pi/5)
x2 = np.exp(t / 2) * np.cos(t + 1)
x3 = np.sin(t)
x4 = np.cos(3*t + np.pi/3)
x5 = np.cos(t)
x6 = np.sin(3*t + np.pi/3)

# -----基础设置-----
fig = plt.figure(figsize=(16, 9), dpi=300)

# 构建2×3的区域，在起点(0, 0)处开始，跨域1行2列的位置区域绘图
ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=2, ) # projection = '3d'
ax1.plot(t, x1, color="b", linestyle="-.", linewidth=1.0, marker="d")
ax1.set_title("图 01", fontsize=20)
ax1.set_xlabel("$x$", fontsize=15)
ax1.set_ylabel("$y$", fontsize=15)
ax1.text(-2, -2, r"$f(x) = \mathrm{e}^{\frac{x}{3}} \mathrm{sin}{\frac{x}{2} + \frac{\pi}{5}}$", fontsize=20)
plt.grid()

# 构建2×3的区域，在起点(0, 2)处开始，跨域2行1列的位置区域绘图
ax2 = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
ax2.plot(t, x2, color="g", linestyle="-", linewidth=1.5, marker="p")
ax2.set_title("图 02", fontsize=20)
ax2.set_xlabel("$x$", fontsize=15)
ax2.set_ylabel("$y$", fontsize=15)
ax2.text(-4, 5, r"$f(x) = \mathrm{e}^{\frac{x}{2}} \mathrm{cos}{x+1}$", fontsize=20)
ax2.grid()

# 构建2×3的区域，在起点(1, 0)处开始，跨域1行1列的位置区域绘图
ax3 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=1)
ax3.plot(t, x3, color="k", linestyle="none", marker="o", markeredgecolor="orange", label=r"$y=\mathrm{sin}{x}$")
ax3.plot(t, x4, color="r", linestyle="-.", linewidth=1.0, label=r"$y=\mathrm{cos}{3x + \frac{\pi}{3}}$")
ax3.set_title("图 03", fontsize=20)
ax3.set_xlabel("$x$", fontsize=15)
ax3.set_ylabel("$y$", fontsize=15)
ax3.legend(fontsize=15)

# 构建2×3的区域，在起点(1, 1)处开始，跨域1行1列的位置区域绘图
ax4 = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
ax4.plot(t, x5, color="c", linestyle="--", linewidth=0.5, marker="h", label=r"$y=\mathrm{cos}{x}$")
ax4.plot(t, x6, color="b", linestyle=":", linewidth=1.0, marker="x", label=r"$y=\mathrm{sin}{3x + \frac{\pi}{3}}$")
ax4.set_title("图 04", fontsize=20)
ax4.set_xlabel("$x$", fontsize=15)
ax4.set_ylabel("$y$", fontsize=15)
ax4.legend(fontsize=15)

# 设置整个图的标题
plt.suptitle("使用subplot2grid绘制不规则画布的图", fontsize=25)

# plt.savefig("不规则画布.png") # 要想在pgf后端下显示图片，就必须使用该句命令，否则报错
plt.show()

#%% https://blog.csdn.net/ccc369639963/article/details/123003431
import matplotlib.pyplot as plt
#使用 colspan指定列，使用rowspan指定行
a1 = plt.subplot2grid((3,3),(0,0),colspan = 2)
a2 = plt.subplot2grid((3,3),(0,2), rowspan = 3)
a3 = plt.subplot2grid((3,3),(1,0),rowspan = 2, colspan = 2)
import numpy as np
x = np.arange(1,10)
a2.plot(x, x*x)
a2.set_title('square')
a1.plot(x, np.exp(x))
a1.set_title('exp')
a3.plot(x, np.log(x))
a3.set_title('log')
plt.tight_layout()
plt.show()













































































































































































































































