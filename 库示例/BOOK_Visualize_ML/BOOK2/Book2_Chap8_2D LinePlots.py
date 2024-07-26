


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 平面线图颗粒度

# 导入包
import matplotlib.pyplot as plt
import numpy as np

# import os


#>>>>>>>>>>>>>>>>>>>  颗粒度较低
x_array = np.linspace(0, 4*np.pi, 9)
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, '.--', ms = 10, color='#0088FF')
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/平面线图，低颗粒度_1.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>
x_array = np.linspace(0, 4*np.pi, 13)
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, '.--', ms = 10, color='#0088FF')
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/平面线图，低颗粒度_2.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  合理的颗粒度
x_array = np.linspace(0, 4*np.pi, 101)
# 等差数列的公差为 4*pi/100；数列有101个值
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, '.', ms = 10, color='#0088FF')
# 只绘制 marker
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/平面线图，合理颗粒度_1.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
# 两点连线
ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/平面线图，合理颗粒度_2.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  特殊函数需要更高颗粒度
x_array = np.linspace(-0.1, 0.1, 100001)
y_array = np.sin(1/x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')
# 两点连线
ax.set_xlim((x_array.min(),x_array.max()))
ax.set_ylim((-1.2, 1.2))
ax.set_xticks((x_array.min(),0, x_array.max()))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/特殊函数需要更高颗粒度.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>  对数坐标
x_array_log = np.logspace(0, 10, 101)
y_array = np.log(x_array_log)
x_array_log
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array_log, y_array, color='#0088FF')
# 两点连线
ax.set_xlim((x_array_log.min(),x_array_log.max()))
plt.xscale('log')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.grid()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 阶跃
# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 向前填充
x_array_coarse = np.linspace(0, 4*np.pi, 25)
y_array_coarse = np.sin(x_array_coarse)

x_array_fine = np.linspace(0, 4*np.pi, 101)
y_array_fine = np.sin(x_array_fine)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array_fine, y_array_fine, '--', ms = 10, color='#888888')
# plt.step(x_array_coarse, y_array_coarse, '.-', ms = 10, color='#0088FF')
# where = 'pre' 默认
# 也可以用：
plt.plot(x_array_coarse, y_array_coarse, '.-', drawstyle='steps', ms = 10, color='#0088FF')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/阶跃，向前填充.svg', format='svg')



#>>>>>>>>>>>>>>>>>>>  中间填充
x_array_coarse = np.linspace(0, 4*np.pi, 25)
y_array_coarse = np.sin(x_array_coarse)

x_array_fine = np.linspace(0, 4*np.pi, 101)
y_array_fine = np.sin(x_array_fine)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array_fine, y_array_fine, '--', ms = 10, color='#888888')
# plt.step(x_array_coarse, y_array_coarse, '.-', where = 'mid', ms = 10, color='#0088FF')
# 也可以用：
plt.plot(x_array_coarse, y_array_coarse, '.-', drawstyle='steps-mid', ms = 10, color='#0088FF')
# where = 'pre' 默认

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/阶跃，中间填充.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  向后填充
x_array_coarse = np.linspace(0, 4*np.pi, 25)
y_array_coarse = np.sin(x_array_coarse)

x_array_fine = np.linspace(0, 4*np.pi, 101)
y_array_fine = np.sin(x_array_fine)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array_fine, y_array_fine, '--', ms = 10, color='#888888')
plt.step(x_array_coarse, y_array_coarse, '.-', where = 'post', ms = 10, color='#0088FF')
# where = 'pre' 默认
# 也可以用：
# plt.plot(x_array_coarse, y_array_coarse, '.-', drawstyle='steps-post', ms = 10, color='#0088FF')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/阶跃，向后填充.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 火柴梗图¶

# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 绘制数列
num = 20
n_array = np.arange(1, num + 1)
a_0 = 1 # 首项
d   = 2 # 公差

# a_n = the nth term in the sequence
# a_1 = the first term in the sequence
# d   = the common difference between terms
a_n_array = a_0 + (n_array - 1) * d

fig, ax = plt.subplots(figsize=(5,3))

plt.stem(n_array, a_n_array)
# basefmt=" " 可以隐藏红色baseline

ax.set_xlim((n_array.min(),n_array.max()))
ax.set_xticks((1, 5, 10, 15, 20))
# ax.set_ylim((0, 0.3))
ax.set_xlabel('$n$')
ax.set_ylabel('n-th term, $a_n$')

# fig.savefig('Figures/火柴梗图，数列.svg', format='svg')



#>>>>>>>>>>>>>>>>>>>  绘制概率质量函数
from scipy.stats import binom

n = 20
p = 0.4
k_array = np.arange(0, n + 1)
binomial_PMF_array = binom.pmf(k_array,n,p)

fig, ax = plt.subplots(figsize=(5,3))

plt.stem(k_array, binomial_PMF_array)
# basefmt=" " 可以隐藏红色baseline

ax.set_xlim((k_array.min(),k_array.max()))
ax.set_xticks(np.arange(0, n + 1, 5))
ax.set_ylim((0, 0.3))
ax.set_xlabel('$x$')
ax.set_ylabel('PMF, $f_X(x)$')

# fig.savefig('Figures/火柴梗图，概率质量函数.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 参考线

# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

#>>>>>>>>>>>>>>>>>>>  水平参考线
x_array = np.linspace(0, 4*np.pi, 101)
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')

plt.axhline(y = 1, color = 'r', ls = '--')
plt.axhline(y = 0, color = 'r', ls = '--')
plt.axhline(y = -1, color = 'r', ls = '--')
# 或者：
# ax.hlines(y=[-1, 1], xmin=x_array.min(), xmax=x_array.max(),
#           linewidth=1, color='r', ls = '--')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/水平参考线.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>  水平参考线，指定范围
x_array = np.linspace(0, 4*np.pi, 101)
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')

ax.hlines(y=1, xmin=np.pi/2 - 1, xmax=np.pi/2 + 1, linewidth=1, color='r', ls = '--')

ax.hlines(y=-1, xmin=np.pi/2*3 - 1, xmax=np.pi/2*3 + 1, linewidth=1, color='r', ls = '--')

ax.hlines(y=1, xmin=np.pi/2*5 - 1, xmax=np.pi/2*5 + 1, linewidth=1, color='r', ls = '--')

ax.hlines(y=-1, xmin=np.pi/2*7 - 1, xmax=np.pi/2*7 + 1, linewidth=1, color='r', ls = '--')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/水平参考线，指定范围.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  竖直参考线
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')

plt.axvline(x = np.pi, color = 'r', ls = '--')
plt.axvline(x = 2*np.pi, color = 'r', ls = '--')
plt.axvline(x = 3*np.pi, color = 'r', ls = '--')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/竖直参考线.svg', format='svg')




#>>>>>>>>>>>>>>>>>>>  竖直参考线，指定范围
x_array = np.linspace(0, 4*np.pi, 101)
y_array = np.sin(x_array)

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(x_array, y_array, color='#0088FF')

ax.vlines(x = np.pi/2*np.arange(0,9),
          ymin = y_array.min(),
          ymax = y_array.max(),
          color = 'r', ls = '--')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# fig.savefig('Figures/竖直参考线，指定范围.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  斜线
for pos in np.linspace(-5, 5, 11):
    plt.axline((0, pos), slope=0.5, color='k')

plt.ylim([-10, 10])
plt.xlim([0, 10])
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用面具mask

import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

x_array = np.linspace(-3, 3, 501)
y_array = x_array * np.exp(-x_array ** 2)

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array)
plt.axhline(y = 0.2, color = 'r', ls = '--')
plt.axhline(y = -0.2, color = 'r', ls = '--')

plt.xlim(-3,3)
plt.ylim(-0.5,0.5)

# fig.savefig('Figures/使用面具，原函数.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>  删除法
mask = ((y_array > -0.2) & (y_array < 0.2))
x_array_removed = x_array[mask]
y_array_removed = y_array[mask]

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array_removed, y_array_removed)
plt.axhline(y = 0.2, color = 'r', ls = '--')
plt.axhline(y = -0.2, color = 'r', ls = '--')

plt.xlim(-3,3)
plt.ylim(-0.5,0.5)

# fig.savefig('Figures/使用面具，删除法.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  用 NaN 代替
y_array_IN = np.copy(y_array)
y_array_IN[~mask] = np.nan

y_array_OUT = np.copy(y_array)
y_array_OUT[mask] = np.nan

# 也可以用：numpy.ma.masked_where() 函数
# np.ma.masked_where((y_array > -0.2) & (y_array < 0.2), y_array)

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array_IN)
plt.plot(x_array, y_array_OUT, color = 'r')

plt.axhline(y = 0.2, color = 'r', ls = '--')
plt.axhline(y = -0.2, color = 'r', ls = '--')

plt.xlim(-3,3)
plt.ylim(-0.5,0.5)

# fig.savefig('Figures/使用面具，用NaN代替.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 交点

import numpy as np
import matplotlib.pyplot as plt

# 两个函数
x_array = np.linspace(0, 4*np.pi, 2000)
# 需要有颗粒度很高的数列
f1_array = np.sin(x_array)
f2_array = x_array/5 - 1

#>>>>>>>>>>>>>>>>>>>  绘制交点


# 找到正负变号的位置
loc_intersects = np.argwhere(np.diff(np.sign(f1_array - f2_array))).flatten()

isect = zip(x_array[loc_intersects], f1_array[loc_intersects])

fig, ax = plt.subplots(figsize=(5,3))

for xi, yi in isect:
    print(f'({xi}, {yi})')
    # 打印交点位置
    plt.scatter(xi, yi, color="k", s=100, marker = 'x')

plt.plot(x_array, f1_array, label = '$f_1(x)$')
plt.plot(x_array, f2_array, label = '$f_2(x)$')

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
plt.legend()

# fig.savefig('Figures/交点.svg', format='svg')
############################################################

# input array
in_arr = [[ 2, 0, 7], [ 0, 5, 9]]
print ("Input array : ", in_arr)

out_arr = np.argwhere(in_arr)
print ("Output indices of non zero array element: \n", out_arr)
############################################################



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 最大值、最小值

import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


x_array = np.linspace(-10, 10, 1001)
y_array = x_array * np.exp(-x_array ** 2)

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_array, y_array)

n_max = y_array.argmax()
# 最大值位置
plt.plot(x_array[n_max], y_array[n_max],'xr', ms = 10)

n_min = y_array.argmin()
# 最小值位置
plt.plot(x_array[n_min], y_array[n_min],'xr', ms = 10)

plt.xlim(-3,3)
plt.ylim(-0.5,0.5)

# fig.savefig('Figures/极值.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用色谱给一组曲线着色

# 导数包
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# 导入色谱
from scipy.stats import norm
# 导入正态分布
# 参考
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

#>>>>>>>>>>>>>>>>>>>  1. 用for循环
x_array = np.linspace(-6, 6, 200)
sigma_array = np.linspace(0.5,5,10)
# 设定标准差一系列取值

num_lines = len(sigma_array)
# 概率密度曲线条数

colors = cm.RdYlBu(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号

fig, ax = plt.subplots(figsize = (5,4))

for idx, sigma_idx in enumerate(sigma_array):
    pdf_idx = norm.pdf(x_array, scale = sigma_idx)
    legend_idx = '$\sigma$ = ' + str(sigma_idx)
    plt.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)
    # 依次绘制概率密度曲线

plt.legend()
# 增加图例

plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')

# fig.savefig('Figures/用for循环.svg', format='svg')

#>>>>>>>>>>>>>>>>>>>  2. 用LineCollection
PDF_curves = [np.column_stack([x_array, norm.pdf(x_array, scale = sigma_idx)]) for sigma_idx in sigma_array]

fig, ax = plt.subplots(figsize = (5,4))

lc = LineCollection(np.array(PDF_curves), cmap = 'rainbow', array = sigma_array, linewidth = 1)
# LineCollection 可以看成是一系列线段的集合
# 可以用色谱分别渲染每一条线段
# 这样可以得到颜色连续变化的效果
line = ax.add_collection(lc) #add to the subplot
fig.colorbar(line, label = '$\sigma$')
# 添加色谱条

plt.xlim(x_array.min(), x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')
# fig.savefig('Figures/用LineCollection.svg', format='svg')


#>>>>>>>>>>>>>>>>>>>  3. 用set_prop_cycle()

cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, num_lines))

x = np.linspace(0, 2*np.pi)
ys = np.ones((10, 50)) * np.sin(x)
ys = np.array([ys[i,] * np.linspace(1, 0.1, 10)[i] for i in range(10)])

fig, ax = plt.subplots(figsize = (5, 4))
ax.set_prop_cycle(color=colors)
# 设定线图颜色

plt.plot(x_array, np.array([norm.pdf(x_array, scale = sigma_idx) for sigma_idx in sigma_array]).T)

plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')
# fig.savefig('Figures/用set_prop_cycle().svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 分段渲染
# 导入包
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
# import os

# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

x_array = np.linspace(0, 4*np.pi, 1001)
# 等差数列的公差为 4*pi/100；数列有101个值
y_array = np.sin(x_array)

points = np.array([x_array, y_array]).T.reshape(-1, 1, 2) #  (1001, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1) # (1000, 2, 2)

#>>>>>>>>>>>>>> 用y值作为渲染依据
fig, ax = plt.subplots(figsize=(5,3))

norm = plt.Normalize(y_array.min(), y_array.max())
lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
lc.set_array(y_array)
lc.set_linewidth(1)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
# fig.savefig('1.svg')



#>>>>>>>>>>>>>> 用x值作为渲染依据¶
fig, ax = plt.subplots(figsize=(5,3))

norm = plt.Normalize(x_array.min(), x_array.max())
lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
lc.set_array(x_array)
lc.set_linewidth(1)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
# fig.savefig('2.svg')


#>>>>>>>>>>>>>> 用切线斜率 (一阶导数) 作为渲染依据
fig, ax = plt.subplots(figsize=(5,3))
slope_array = np.cos(x_array)

norm = plt.Normalize(slope_array.min(), slope_array.max())
lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
lc.set_array(slope_array)
lc.set_linewidth(1)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
# fig.savefig('3.svg')


#>>>>>>>>>>>>>> 用凸凹性 (二阶导数) 作为渲染依据
fig, ax = plt.subplots(figsize=(5,3))
convex_array = -np.sin(x_array)

norm = plt.Normalize(convex_array.min(), convex_array.max())
lc = LineCollection(segments, cmap='RdYlBu_r', norm=norm)
lc.set_array(convex_array)
lc.set_linewidth(1)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

ax.set_xlim((0,4*np.pi))
ax.set_ylim((-1.2, 1.2))

ax.set_xlabel('x')
ax.set_ylabel('f(x)')
fig.savefig('4.svg')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 绘制网格


import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

colormap = cm.get_cmap("rainbow")

def plot_grid(xmin: float, xmax: float, ymin: float, ymax: float, n_lines: int, line_points: int, map_func,):
    lines = []
    # 水平线
    for y in np.linspace(ymin, ymax, n_lines):
        lines.append([map_func(x, y) for x in np.linspace(xmin, xmax, line_points)])
    # 竖直线
    for x in np.linspace(xmin, xmax, n_lines):
        lines.append([map_func(x, y) for y in np.linspace(ymin, ymax, line_points)])

    # 绘制所有线条
    for i, line in enumerate(lines):
        p = i / (len(lines) - 1)
        xs, ys = zip(*line)
        # 利用颜色映射
        plt.plot(xs, ys, color=colormap(p))

# 各种映射
def identity(x, y):
    return x, y

def rotate_scale(x, y):
    return x + y, x - y

def shear(x, y):
    return x, x + y

def exp(x, y):
    return math.exp(x), math.exp(y)

def complex_sq(x, y):
    c = complex(x, y) ** 2
    return (c.real, c.imag)

def sin_cos(x: float, y: float):
    return x + math.sin(y * 2) * 0.2, y + math.cos(x * 2) * 0.3

def vortex(x: float, y: float):
    dst = (x - 2) ** 2 + (y - 2) ** 2
    ang = math.atan2(y - 2, x - 2)
    return math.cos(ang - dst * 0.1) * dst, math.sin(ang - dst * 0.1) * dst

fig = plt.figure(figsize=(4, 4))

# 原图
ax = fig.add_subplot(111)
plot_grid(0, 5, 0, 5, 20, 20, identity)

ax.axis('off')
# fig.savefig('Figures/原始网格.svg', format='svg')

fig = plt.figure(figsize=(8, 12))

ax = fig.add_subplot(3, 2, 1)
plot_grid(0, 5, 0, 5, 20, 20, rotate_scale)
ax.axis('off')

ax = fig.add_subplot(3, 2, 2)
plot_grid(0, 5, 0, 5, 20, 20, shear)
ax.axis('off')

ax = fig.add_subplot(3, 2, 3)
plot_grid(0, 5, 0, 5, 20, 20, exp)
ax.axis('off')

ax = fig.add_subplot(3, 2, 4)
plot_grid(0, 5, 0, 5, 20, 20, complex_sq)
ax.axis('off')

ax = fig.add_subplot(3, 2, 5)
plot_grid(0, 5, 0, 5, 20, 20, sin_cos)
ax.axis('off')

ax = fig.add_subplot(3, 2, 6)
plot_grid(0, 5, 0, 5, 20, 20, vortex)
ax.axis('off')

# fig.savefig('Figures/线性、非线性变换.svg', format='svg')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用线条绘制等边三角形生成艺术

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

import matplotlib.pyplot as plt
import numpy as np

theta = 15 # rotation angle

t = [0, 0]

r = 1

points_x = [0, np.sqrt(3)/2 * r, -np.sqrt(3)/2 * r, 0]
points_y = [r, -1/2 * r, -1/2 * r, r]

X = np.column_stack([points_x,points_y])

theta = np.deg2rad(theta)

R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

X = X @ R + t
# X


def eq_l_tri(ax, r, theta, t, color = 'b', fill = False):
    points_x = [0, np.sqrt(3)/2 * r, -np.sqrt(3)/2 * r, 0]
    points_y = [r, -1/2 * r, -1/2 * r, r]

    X = np.column_stack([points_x,points_y])
    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
    X = X @ R.T
    ax.plot(X[:,0], X[:,1], color = color)

    if fill:
        plt.fill(X[:,0], X[:,1], color = color, alpha = 0.1)


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
eq_l_tri(ax, r, 10, t)

plt.show()


fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(100)
delta_angle = 2 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    r = 0.05 + i * 0.05
    eq_l_tri(ax, r, deg, (0,0), colors[i])

plt.axis('off')
# fig.savefig('Figures/旋转三角形_A.svg', format='svg')
plt.show()



fig, ax = plt.subplots(figsize = (8,8))
ax.set_aspect('equal')
ax.plot(0, 0,  color='r', marker='o', markersize=10)

range_array = np.arange(100)
delta_angle = 5 # degrees

colors = plt.cm.RdYlBu(np.linspace(0, 1, len(range_array)))

for i in range_array:
    deg = delta_angle * i
    r = 0.05 + i * 0.05
    eq_l_tri(ax, r, deg, (0,0), colors[i])

plt.axis('off')
# fig.savefig('Figures/旋转三角形_B.svg', format='svg')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  用线条创作生成艺术

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


def lerp(P_a, P_b, t_array):

    P_out = [P_a * (1 - t_idx) + P_b * t_idx for t_idx in t_array]
    P_out = np.array(P_out)

    return P_out
# 三角形
P_0 = np.array([0, 0])
P_1 = np.array([1, 0])
P_2 = np.array([0.5, 0.5 * np.sqrt(3)])

t_array = np.linspace(0, 1, 11, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)
P_2_0 = lerp(P_2, P_0, t_array)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

for i in range(num):
    P_0_1_idx = P_0_1[i, :]
    P_1_2_idx = P_1_2[i, :]
    P_2_0_idx = P_2_0[i, :]

    P_array_idx = np.row_stack((P_0_1_idx, P_1_2_idx, P_2_0_idx, P_0_1_idx))

    plt.plot(P_array_idx[:,0], P_array_idx[:,1], color=colors[i], lw = 0.25)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')
# fig.savefig('Figures/等边三角形，贝塞尔序曲_示意.svg', format='svg')



t_array = np.linspace(0, 1, 101, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)
P_2_0 = lerp(P_2, P_0, t_array)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

for i in range(num):
    P_0_1_idx = P_0_1[i, :]
    P_1_2_idx = P_1_2[i, :]
    P_2_0_idx = P_2_0[i, :]

    P_array_idx = np.row_stack((P_0_1_idx, P_1_2_idx, P_2_0_idx, P_0_1_idx))
    plt.plot(P_array_idx[:,0], P_array_idx[:,1], color=colors[i], lw = 0.25)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')
# fig.savefig('Figures/等边三角形，贝塞尔序曲t.svg', format='svg')


# 直角三角形
P_0 = np.array([0, 0])
P_1 = np.array([1, 0])
P_2 = np.array([1, 1])

t_array = np.linspace(0, 1, 101, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)
P_2_0 = lerp(P_2, P_0, t_array)


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

for i in range(num):
    P_0_1_idx = P_0_1[i, :]
    P_1_2_idx = P_1_2[i, :]
    P_2_0_idx = P_2_0[i, :]

    P_array_idx = np.row_stack((P_0_1_idx, P_1_2_idx, P_2_0_idx, P_0_1_idx))

    plt.plot(P_array_idx[:,0], P_array_idx[:,1], color=colors[i], lw = 0.25)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')
# fig.savefig('Figures/直角三角形，贝塞尔序曲t.svg', format='svg')


# 正方形
P_0 = np.array([0, 0])
P_1 = np.array([1, 0])
P_2 = np.array([1, 1])
P_3 = np.array([0, 1])

t_array = np.linspace(0, 1, 101, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)
P_2_3 = lerp(P_2, P_3, t_array)
P_3_0 = lerp(P_3, P_0, t_array)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

for i in range(num):
    P_0_1_idx = P_0_1[i, :]
    P_1_2_idx = P_1_2[i, :]
    P_2_3_idx = P_2_3[i, :]
    P_3_0_idx = P_3_0[i, :]

    P_array_idx = np.row_stack((P_0_1_idx, P_1_2_idx, P_2_3_idx, P_3_0_idx, P_0_1_idx))

    plt.plot(P_array_idx[:,0], P_array_idx[:,1], color=colors[i], lw = 0.25)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')
# fig.savefig('Figures/正方形，贝塞尔序曲t.svg', format='svg')


# 正五边形
angles = np.linspace(18, 360+18, 5, endpoint = False)
angles_radian = angles * np.pi/180
angles

P_0 = np.array([np.cos(angles_radian[0]), np.sin(angles_radian[0])])
P_1 = np.array([np.cos(angles_radian[1]), np.sin(angles_radian[1])])
P_2 = np.array([np.cos(angles_radian[2]), np.sin(angles_radian[2])])
P_3 = np.array([np.cos(angles_radian[3]), np.sin(angles_radian[3])])
P_4 = np.array([np.cos(angles_radian[4]), np.sin(angles_radian[4])])

t_array = np.linspace(0, 1, 101, endpoint = True)

P_0_1 = lerp(P_0, P_1, t_array)
P_1_2 = lerp(P_1, P_2, t_array)
P_2_3 = lerp(P_2, P_3, t_array)
P_3_4 = lerp(P_3, P_4, t_array)
P_4_0 = lerp(P_4, P_0, t_array)



fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

num = len(P_0_1)
colors = plt.cm.rainbow(np.linspace(0,1,num, endpoint = True))

for i in range(num):

    P_0_1_idx = P_0_1[i, :]
    P_1_2_idx = P_1_2[i, :]
    P_2_3_idx = P_2_3[i, :]
    P_3_4_idx = P_3_4[i, :]
    P_4_0_idx = P_4_0[i, :]

    P_array_idx = np.row_stack((P_0_1_idx,
                                P_1_2_idx,
                                P_2_3_idx,
                                P_3_4_idx,
                                P_4_0_idx,
                                P_0_1_idx))

    plt.plot(P_array_idx[:,0],
             P_array_idx[:,1],
             color=colors[i], lw = 0.25)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.axis('off')
# fig.savefig('Figures/正五边方形，贝塞尔序曲t.svg', format='svg')




















































































































































































































































































































































