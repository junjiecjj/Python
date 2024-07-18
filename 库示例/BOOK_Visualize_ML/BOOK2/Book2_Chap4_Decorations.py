#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:22:49 2024

@author: jack
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch04_01 修改图脊

import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


plt.rcParams


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5



# 产生数据
x_array = np.linspace(-4, 4, 200)
y_array = x_array*np.exp(-x_array**2)


# 图脊设置
fig = plt.figure(figsize = (6,6), tight_layout=True)

#
ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')


ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines[['bottom', 'left']].set_visible(False)
# 也可以采用：
# ax.spines['bottom'].set_color('none')
# ax.spines['left'].set_color('none')
ax.yaxis.set_ticks_position('right')
ax.xaxis.set_ticks_position('top')

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_position(('data',0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# fig.savefig('Figures/修改图脊，第1组.svg', format='svg')


fig = plt.figure(figsize = (6,6), tight_layout=True)

ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_position(('data',0.2))
ax.spines['left'].set_position(('data',2))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_position(('axes',0.5))
# 取值范围为 [0, 1], 0.5 代表中间
ax.spines['left'].set_position(('axes',0.5))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_color('r')
ax.spines['left'].set_color('r')

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.set_xticks(np.arange(-4,5))


ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['right'].set(alpha = 0.2)
ax.spines['top'].set(alpha = 0.2)
ax.set_yticks(np.arange(-0.5,0.6,0.1))

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.spines['right'].set(edgecolor = 'r')
ax.spines['top'].set(edgecolor = 'r')

# fig.savefig('Figures/修改图脊，第2组.svg', format='svg')



fig = plt.figure(figsize = (6,6), tight_layout=True)


ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set(edgecolor = 'r',
                        linestyle = '--', linewidth = 1)
ax.spines['left'].set(edgecolor = 'r',
                      linestyle = '--', linewidth = 1)
ax.tick_params(axis='x', colors='red')
ax.tick_params(axis='y', colors='red')

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.spines[:].set_color('none')


ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.axis('off')

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.spines['left'].set_position(('outward', 10))

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.spines['bottom'].set_position(('outward', 10))
# 三个选择 'outward', 'axes', 'data'

# fig.savefig('Figures/修改图脊，第3组.svg', format='svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 修改网格

import numpy as np
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

x_array = np.linspace(-4, 4, 200)
y_array = x_array*np.exp(-x_array**2)



fig = plt.figure(figsize = (6,6), tight_layout=True)

ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.grid(linestyle='-',
        linewidth='0.5', color='red')

ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-',
        linewidth='0.5', color='red')
ax.grid(which='minor', linestyle=':',
        linewidth='0.5', color='black')

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-',
        linewidth='0.5', color='red')
ax.grid(which='minor', linestyle=':',
        linewidth='0.5', color='black')

ax.tick_params(which='both',
               top='off',
               left='off',
               right='off',
               bottom='off')

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.grid(True)
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)

ax.set_xlim(-4,4);
ax.set_ylim(-0.5, 0.5);
ax.grid(True)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
ax.set_facecolor("#DBEEF8")
plt.show()

# fig.savefig('Figures/背景网格.svg', format='svg')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 修改图轴


# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter, FormatStrFormatter

import os


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


# 产生数据
x_array = np.linspace(-15, 15, 200)
y_array = np.sin(x_array)


# 轴设置
fig = plt.figure(figsize = (6,6), tight_layout=True)

#
ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15);
ax.set_ylim(-1.5, 1.5);

# 设定 x、y 的刻度值
ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.set_xticks([-15, 0, 5, 10, 15]) # 可以用numpy.linspace()
ax.set_yticks([-0.8, -0.4, 0, 0.8])
# 也可以用 plt.xticks(), plt.yticks() 命令


#
ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('Rotated tick labels')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.tick_params(axis ='x', rotation = 45,
               labelcolor = 'r', labelsize = 10)
ax.tick_params(axis ='y', rotation =-45,
               labelcolor = 'r', labelsize = 10)

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('Hide')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('Hide values, show ticks')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('Hide values, show ticks')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.set_xticks(np.linspace(-4*np.pi, 4*np.pi, 5))
ax.set_xticklabels([r'$-4\pi$',r'$-2\pi$',
                    r'$0\pi$',r'$2\pi$', r'$4\pi$'])

# fig.savefig('Figures/修改图轴，第1组.svg', format='svg')



fig = plt.figure(figsize = (6,6), tight_layout=True)

ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('Hide values, show ticks')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.set_xticks(np.linspace(-9/2*np.pi, 9/2*np.pi, 7))
ax.tick_params(axis = 'x',
               direction = 'in',
               color = 'b',
               width = 1,
               length = 10)
ax.tick_params(axis = 'y',
               direction = 'in',
               color = 'b',
               width = 1,
               length = 10)
ax.set_xticklabels([r'$-\frac{9\pi}{2}$',
                    r'$-3\pi$',
                    r'$-\frac{3\pi}{2}$',
                    r'$0\pi$',
                    r'$\frac{3\pi}{2}$',
                    r'$3\pi$',
                    r'$\frac{9\pi}{2}$'])

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('Hide values, show ticks')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.tick_params(axis = 'x',
               direction = 'inout',
               color = 'b',
               width = 1,
               length = 10)

ax.tick_params(axis = 'y',
               direction = 'inout',
               color = 'r', width = 1,
               length = 10)


ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.tick_params(bottom = False,
               top = True,
               left = False,
               right = True)

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.tick_params(bottom = False,
               top = True,
               left = False,
               right = True)

ax.tick_params(labelbottom = False,
               labeltop = True,
               labelleft = False,
               labelright = True)

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
minor_locator = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
minor_locator = AutoMinorLocator(5)
ax.yaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')

# fig.savefig('Figures/修改图轴，第2组.svg', format='svg')


fig = plt.figure(figsize = (6,6), tight_layout=True)

ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
minor_locator = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minor_locator)
ax.tick_params(which="minor", axis="x", direction="inout",
               color = 'r', length = 10, width = 1)

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
minor_locator = AutoMinorLocator(5)
ax.yaxis.set_minor_locator(minor_locator)
ax.tick_params(which="minor", axis="y", direction="inout",
               color = 'r', length = 10, width = 1)

ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.set_yticks([-1, 0, 1])
minor_locator = AutoMinorLocator(2)
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.set_minor_formatter(FormatStrFormatter('%.3f'))

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
from matplotlib.ticker import FixedLocator, FixedFormatter

x_formatter = FixedFormatter(["A", "B", "C"])
y_formatter = FixedFormatter(['Bottom', 'Center', 'Top'])

x_locator = FixedLocator([-8, 0, 8])
y_locator = FixedLocator([-1, 0, 1])
ax.xaxis.set_major_formatter(x_formatter)
ax.yaxis.set_major_formatter(y_formatter)
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array)
# ax.set_title('original')
ax.set_xlim(-15,15); ax.set_ylim(-1.5, 1.5);
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(2))

# fig.savefig('Figures/修改图轴，第3组.svg', format='svg')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 对数坐标



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MultipleLocator
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

X = np.linspace(0.001, 90, 5000)

fig = plt.figure(figsize=(8, 8))

# X-linear Y-linear
# -----------------------------------------------------------------------------
ax1 = plt.subplot(2, 2, 1, xlim=(0.0, 10), ylim=(0.0, 10))
ax1.plot(X, 10 ** X, color="C0")
ax1.plot(X, X, color="C1")
ax1.plot(X, np.log10(X), color="C2")
ax1.set_ylabel("Linear")
ax1.xaxis.set_major_locator(MultipleLocator(2.0))
ax1.xaxis.set_minor_locator(MultipleLocator(0.4))
ax1.yaxis.set_major_locator(MultipleLocator(2.0))
ax1.yaxis.set_minor_locator(MultipleLocator(0.4))
ax1.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax1.grid(True, "major", color="0.65", linewidth=0.75, zorder=-10)
ax1.tick_params(which="both", labelbottom=False, bottom=False)

ax1.text(1.25, 8.50, "$f(x) = 10^x$", color="C0", fontname = 'Roboto')
ax1.text(5.75, 5.00, "$f(x) = x$", color="C1", fontname = 'Roboto')
ax1.text(5.50, 1.50, "$f(x) = log_{10}(x)$", color="C2", fontname = 'Roboto')
ax1.set_title("X linear - Y linear")


# X-log Y-linear
# -----------------------------------------------------------------------------
ax2 = plt.subplot(2, 2, 2, xlim=(0.001, 100), ylim=(0.0, 10), sharey=ax1)
ax2.set_xscale("log")
ax2.tick_params(which="both", labelbottom=False, bottom=False)
ax2.tick_params(which="both", labelleft=False, left=False)
ax2.plot(X, 10 ** X, color="C0")
ax2.plot(X, X, color="C1")
ax2.plot(X, np.log10(X), color="C2")
ax2.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax2.grid(True, "major", color="0.65", linewidth=0.75, zorder=-10)
ax2.set_title("X logarithmic - Y linear")


# X-linear Y-log
# -----------------------------------------------------------------------------
ax3 = plt.subplot(2, 2, 3, xlim=(0.0, 10), ylim=(0.001, 100), sharex=ax1)
ax3.set_yscale("log")
ax3.plot(X, 10 ** X, color="C0")
ax3.plot(X, X, color="C1")
ax3.plot(X, np.log10(X), color="C2")
ax3.set_ylabel("Logarithmic")
ax3.set_xlabel("Linear")
ax3.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax3.grid(True, "major", color="0.65", linewidth=0.75, zorder=-10)
ax3.set_title("X linear - Y logarithmic")

# X-log Y-log
# -----------------------------------------------------------------------------
ax4 = plt.subplot(2, 2, 4, xlim=(0.001, 100), ylim=(0.001, 100), sharex=ax2, sharey=ax3)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.tick_params(which="both", labelleft=False, left=False)
ax4.plot(X, 10 ** X, color="C0")
ax4.plot(X, X, color="C1")
ax4.plot(X, np.log10(X), color="C2")
ax4.set_xlabel("Logarithmic")
ax4.grid(True, "minor", color="0.85", linewidth=0.50, zorder=-20)
ax4.grid(True, "major", color="0.65", linewidth=0.75, zorder=-10)
ax4.set_title("X logarithmic - Y logarithmic")


# Show
# -----------------------------------------------------------------------------
plt.tight_layout()

# fig.savefig('Figures/对数坐标.svg', format='svg')








#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%c 标注

import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

x_array = np.linspace(-4, 4, 200)
y_array = x_array*np.exp(-x_array**2)
y_array_2 = np.exp(-x_array**2)/2


fig = plt.figure(figsize = (8,12), tight_layout=True)

#
ax = fig.add_subplot(3, 2, 1)
ax.plot(x_array, y_array)
ax.set_xlabel('x axis label')
ax.set_ylabel('y axis label')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 2)
ax.plot(x_array, y_array)
ax.set_xlabel('x axis label', loc = 'left')
ax.set_ylabel('y axis label', loc = 'top')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 3)
ax.plot(x_array, y_array)
ax.set_xlabel('x axis label', loc = 'right')
ax.set_ylabel('y axis label', loc = 'bottom')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 4)
ax.plot(x_array, y_array)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel('x axis label')
ax.set_ylabel('y axis label')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 5)
ax.plot(x_array, y_array)
ax.set_title('This is the tile', loc = 'left')
ax.set_xlabel('x axis label')
ax.set_ylabel('y axis label')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
ax.grid(True)

ax = fig.add_subplot(3, 2, 6)
ax.plot(x_array, y_array, label = 'Legend A')
ax.plot(x_array, y_array_2, label = 'Legend B')
ax.set_title('This is the tile \n In case multiple lines', loc = 'left')
ax.set_xlabel('x axis label')
ax.set_ylabel('y axis label')
# ax.set_title('original')
ax.set_xlim(-4,4); ax.set_ylim(-0.5, 0.5);
plt.legend()
ax.grid(True)

# fig.savefig('Figures/标注.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 三维网格面随视角变化



from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


X, Y, Z = axes3d.get_test_data(0.05)
# 生成测试数据



fig = plt.figure(figsize = (12,12),constrained_layout=True)

angle_array = np.linspace(0, 180, 13)
num_grid = len(angle_array)
gspec = fig.add_gridspec(num_grid, num_grid)

nrows, ncols = gspec.get_geometry()

axs = np.array([[fig.add_subplot(gspec[i, j], projection='3d') for j in range(ncols)] for i in range(nrows)])

for i in range(nrows):

    elev = angle_array[i]

    for j in range(ncols):

        azim = angle_array[j]

        axs[i, j].plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        # axs[i, j].quiver(0, 0, 0, u, v, w, length=0.1, normalize=True)

        axs[i, j].set_proj_type('ortho')
        axs[i, j].grid('off')
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_zticks([])
        axs[i, j].set_xlim(X.min(),X.max())
        axs[i, j].set_ylim(Y.min(),Y.max())
        axs[i, j].set_zlim(Z.min(),Z.max())

        axs[i, j].view_init(elev=elev, azim=azim)

# fig.savefig('Figures/子图，三维曲面视角.svg', format='svg')
plt.show()




































































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































