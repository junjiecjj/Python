

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap13
# 导入包
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib import cm

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 自定义可视化函数

def visualize_2D(array, title, vmax, vmin):
    fig_width  = math.ceil(array.shape[1] * 0.5)
    fig_length = math.ceil(array.shape[0] * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    # 注意，seaborn.heatmap() 目前只能接受2D数组
    # 本书中，一维数组可视化时用圆圈
    # 可视化时，只有二维、三维数组用方块

    sns.heatmap(array,
                vmax = vmax,
                vmin = vmin,
                annot = True,      # 增加注释
                fmt = ".0f",       # 注释数值的格式
                square = True,     # 热图方格为正方形
                cmap = 'RdYlBu_r', # 指定色谱
                linewidths = .5,   # 方格线宽
                cbar = False,      # 不显示色谱条
                yticklabels=False, # 不显示纵轴标签
                xticklabels=False, # 不显示横轴标签
                ax = ax)           # 指定绘制热图的轴

    # fig.savefig('Figures/' + title + '.svg', format='svg')

# 定义绘制一元数组可视化函数
def visualize_1D(array, title):
    fig, ax = plt.subplots()
    colors = cm.RdYlBu_r(np.linspace(0,1,len(array)))

    for idx in range(len(array)):
        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = str(array[idx]), horizontalalignment = 'center', verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # fig.savefig('Figures/' + title + '.svg', format='svg')

a_1D = np.array([-3, -2, -1, 0, 1, 2, 3])
visualize_1D(a_1D, '手动，一维')



# 二维
a_2D = np.array([[-3, -2, -1],
                 [0,  1,  2]])
# a_2D
# array([[-3, -2, -1],
#        [ 0,  1,  2]])
visualize_2D(a_2D, '手动，二维', 3, -3)


# 二维，行向量
a_row_vector = np.array([[-3, -2, -1, 0, 1, 2, 3]])
# 两层中括号
visualize_2D(a_row_vector, '手动，行向量', 3, -3)

# 随机数
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数，服从连续均匀分布
num = 2000
X_uniform = np.random.uniform(low=-3, high=3, size=(num,2))

fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(X_uniform[:,0],  # 散点横轴坐标
           X_uniform[:,1],  # 散点纵轴坐标
           s = 100,         # 散点大小
           marker = '.',    # 散点marker样式
           alpha = 0.5,     # 透明度
           edgecolors = 'w')# 散点边缘颜色

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xticks((-3,0,3))
ax.set_yticks((-3,0,3))
# fig.savefig('Figures/二元连续均匀随机数.svg', format='svg')



# 二元正态分布随机数
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数，服从二元高斯分布
num = 2000

mu    = np.array([0, 0])    # 质心
rho   = 0  # 相关性系数
Sigma = np.array([[1, rho],
                  [rho, 1]])  # 协方差矩阵

X_binormal = np.random.multivariate_normal(mu, Sigma, size=num)

fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(X_binormal[:,0],
           X_binormal[:,1],
           s = 100,
           marker = '.',
           alpha = 0.5,
           edgecolors = 'w')

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xticks((-3,0,3))
ax.set_yticks((-3,0,3))
# fig.savefig('Figures/二元正态分布随机数.svg', format='svg')

# CSV文件导出、导入
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from numpy import genfromtxt

# 导入鸢尾花数据
iris = load_iris()
# 将numpy array存成CSV文件
np.savetxt("Iris_data.csv", iris.data, delimiter=",")

# 将 CSV 文件读入存成numpy array
Iris_Data_array = genfromtxt('Iris_data.csv', delimiter=',')

# 可视化
fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(Iris_Data_array,   # 鸢尾花数据数组
            cmap = 'RdYlBu_r', # 指定色谱
            ax = ax,           # 指定轴
            vmax = 8,          # 色谱最大值
            vmin = 0,          # 色谱最小值
            xticklabels = [],  # 不显示横轴标签
            yticklabels = [],  # 不显示纵轴标签
            cbar = True)       # 显示色谱条



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap14 NumPy索引和切片
# 导入包
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib import cm
# 导入色谱

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


def visualize_2D(array, title, vmax, vmin):

    fig_width  = math.ceil(array.shape[1] * 0.5)
    fig_length = math.ceil(array.shape[0] * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    # 注意，seaborn.heatmap() 目前只能接受2D数组
    # 本书中，一维数组可视化时用圆圈
    # 可视化时，只有二维、三维数组用方块

    sns.heatmap(array,
                vmax = vmax,
                vmin = vmin,
                annot = True,      # 增加注释
                fmt = ".0f",       # 注释数值的格式
                square = True,     # 热图方格为正方形
                cmap = 'RdYlBu_r', # 指定色谱
                linewidths = .5,   # 方格线宽
                cbar = False,      # 不显示色谱条
                yticklabels=False, # 不显示纵轴标签
                xticklabels=False, # 不显示横轴标签
                ax = ax)           # 指定绘制热图的轴

    # fig.savefig('Figures/' + title + '.svg', format='svg')

# 定义绘制一元数组可视化函数

def visualize_1D(array, title):
    fig, ax = plt.subplots()

    colors = cm.RdYlBu_r(np.linspace(0,1,len(array)))

    for idx in range(len(array)):
        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = str(array[idx]),
                horizontalalignment = 'center',
                verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # fig.savefig('Figures/' + title + '.svg', format='svg')






# 一维数组

# 定义一维数组
a_1D_array = np.arange(-5, 5 + 1)
# 整数索引
a_1D_array[-11]

a_1D_array[len(a_1D_array) - 1]



# 行向量、列向量

# 转为列向量
a_1D_array[:, np.newaxis]

visualize_2D(a_1D_array[:, np.newaxis], '列向量', 5, -5)



a_1D_array[:, np.newaxis].ndim


a_1D_array[:, np.newaxis].squeeze()
# 二维到一维


# 转为行向量
a_1D_array[np.newaxis, :]
visualize_2D(a_1D_array[np.newaxis, :], '行向量', 5, -5)

a_1D_array[None, :]
a_1D_array.reshape(1,-1)

a_1D_array[np.newaxis, :] @ a_1D_array[:, np.newaxis]
a_1D_array[:, np.newaxis] @ a_1D_array[np.newaxis, :]

a_1D_array[:, np.newaxis, np.newaxis]
# 三维


# 切片
# 获取数组中的前三个元素
a_1D_array[:3]
a_1D_array[[0, 1, 2]]

# 获取数组中的最后三个元素
a_1D_array[-3:]
a_1D_array[8:]

a_1D_array[len(a_1D_array)-3:]



# 整数索引
# 获取，第一、二 三，和最后一个元素
a_1D_array[[0, 1, 2, -1]]
a_1D_array[np.r_[0:3, -1]]

# numpy.r_ 是一个用于将切片对象转换为一个沿着第一个轴堆叠的 NumPy 数组的函数。
# 它可以在数组创建和索引时使用。
# 它的作用类似于 numpy.concatenate 和 numpy.vstack，
# 但是使用切片对象作为索引来方便快捷地创建数组。


# 布尔索引
a_1D_array[a_1D_array > 1]


a_1D_array[a_1D_array >= 0]

a_1D_array[a_1D_array < 0]
# 大于-3，小于3的元素
a_1D_array[(a_1D_array < 3) & (a_1D_array > -3)]


a_1D_array_copy_2 = np.copy(a_1D_array)
a_1D_array_copy_2[(a_1D_array_copy_2 < 3) & (a_1D_array_copy_2 > -3)] = 100
a_1D_array_copy_2




# 二维数组
# 定义二维数组
A_2D = np.array([[-7,-6,-5,-4,-3],
                 [-2,-1, 0, 1, 2],
                 [ 3, 4, 5, 6, 7]])

visualize_2D(A_2D, '二维数组', 7, -7)


### 取出行
# 取出第一行，结果为一维向量
A_2D[0]

# 取出第一行，结果为二维行向量
A_2D[[0],:]


A_2D[0, np.newaxis]
A_2D[[0, 2]]
A_2D[[0, 2], :]



### 取出列
A_2D[:,0]

A_2D[...,0]

A_2D[:,0, np.newaxis]
A_2D[:,[0]]

A_2D[np.newaxis, :,0]

A_2D[:,[0,2,4]]



# 取出特定行列组合
A_2D[1,2::]

A_2D[np.newaxis, 1,2::]

A_2D[1,2::, np.newaxis]

A_2D[1::,[0,2,4]]

# 二次切片
A_2D[:,[0,2]][[0, 2], :]

# 使用np.ix_()
A_2D[np.ix_([0, 2], [0, 2])]

#### 布尔索引
A_2D > 0

A_2D[A_2D > 0]

A_2D[A_2D > 0, np.newaxis]


A_2D[np.newaxis, A_2D > 0]




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap15 NumPy常见运算

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib import cm
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 自定义函数
def visualize_2D(array, title, vmax, vmin):

    fig_width  = math.ceil(array.shape[1] * 0.5)
    fig_length = math.ceil(array.shape[0] * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    # 注意，seaborn.heatmap() 目前只能接受2D数组
    # 本书中，一维数组可视化时用圆圈
    # 可视化时，只有二维、三维数组用方块

    sns.heatmap(array,
                vmax = vmax,
                vmin = vmin,
                annot = True,      # 增加注释
                fmt = ".2f",       # 注释数值的格式
                square = True,     # 热图方格为正方形
                cmap = 'RdYlBu_r', # 指定色谱
                linewidths = .5,   # 方格线宽
                cbar = False,      # 不显示色谱条
                yticklabels=False, # 不显示纵轴标签
                xticklabels=False, # 不显示横轴标签
                ax = ax)           # 指定绘制热图的轴

    # fig.savefig('Figures/' + title + '.svg', format='svg')

# 定义绘制一元数组可视化函数

def visualize_1D(array, title, vmax, vmin):
    fig, ax = plt.subplots()

    cmap = cm.get_cmap("RdYlBu")

    array_norm = (array - vmin) / (vmax - vmin)

    colors = cmap(array_norm)

    for idx in range(len(array)):

        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = "{:.1f}".format(array[idx]),
                horizontalalignment = 'center',
                verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # fig.savefig('Figures/' + title + '.svg', format='svg')


# 一维
a_1D = np.arange(-2,2+1)
visualize_1D(a_1D, '一维数组', -4, 4)


all_twos = 2*np.ones_like(a_1D)

visualize_1D(all_twos, '一维数组，全2', -4, 4)



# 广播原则
### 一维
a_1D + 2


# 一维数组、列向量运算

visualize_2D(np.array([[1], [2], [3]]), '列向量，二维', 5, -5)

a_1D + np.array([[1], [2], [3]])
visualize_2D(a_1D + np.array([[1], [2], [3]]), '一维数组、列向量运算，加法', 5, -5)



a_1D * np.array([[1], [2], [3]])
visualize_2D(a_1D * np.array([[1], [2], [3]]), '一维数组、列向量运算，乘法', 5, -5)

a_1D ** np.array([[1], [2], [3]])
visualize_2D(a_1D ** np.array([[1], [2], [3]]), '一维数组、列向量运算，乘幂', 5, -5)



# 二维数组和标量
a_2D = np.random.uniform(-1, 1, (4,6))
a_2D + 2



# 二维数组和一维数组
np.broadcast_to(np.linspace(-1,1,6), (4, 6))

visualize_1D(np.linspace(-1,1,6), '一维数组，6个元素', -4, 4)

a_2D + np.linspace(-1,1,6)
visualize_2D(a_2D + np.linspace(-1,1,6), '二维数组和一维数组，加法', 5, -5)


visualize_2D(np.linspace(-1,1,6).reshape(1,-1), '行向量，6个元素', 5, -5)
visualize_2D(a_2D + np.linspace(-1,1,6).reshape(1,-1), '二维数组和行向量，加法', 5, -5)


a_2D * np.linspace(-1,1,6)
visualize_2D(a_2D * np.linspace(-1,1,6), '二维数组和一维数组，乘法', 5, -5)
visualize_2D(a_2D * np.linspace(-1,1,6).reshape(1,-1), '二维数组和行向量，乘法', 5, -5)

# a_2D + np.array([-2, -1, 0, 1])
# error

visualize_2D(np.array([[-2], [-1], [0], [1]]), '列向量，4个元素', 5, -5)
visualize_2D(a_2D + np.array([[-2], [-1], [0], [1]]), '二维数组和列向量，加法', 5, -5)
visualize_2D(a_2D * np.array([[-2], [-1], [0], [1]]), '二维数组和列向量，乘法', 5, -5)




import numpy as np
import matplotlib.pyplot as plt

# 自定义可视化函数
def visualize_fx(x_array, f_array, title, step = False):

    fig, ax = plt.subplots(figsize = (5,5))
    ax.plot([-5,5],[-5,5], c = 'r', ls = '--', lw = 0.5)

    if step:
        ax.step(x_array, f_array)
    else:
        ax.plot(x_array, f_array)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axvline(0, c = 'k')
    ax.axhline(0, c = 'k')
    ax.set_xticks(np.arange(-5, 5+1))
    ax.set_yticks(np.arange(-5, 5+1))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.grid(True)
    ax.set_aspect('equal', adjustable='box')
    # fig.savefig(title + '.svg', format='svg')
x_array = np.linspace(-4,4,81)

# 幂函数，p = 2
x_array = np.linspace(-5,5,1001)
f_array = np.power(x_array, 2)
visualize_fx(x_array, f_array, '幂函数_p=2')

# 幂函数，p = 3
f_array = np.power(x_array, 3)
visualize_fx(x_array, f_array, '幂函数_p=3')

# 绝对值函数
x_array = np.linspace(-5,5,1001)
f_array = np.abs(x_array)
visualize_fx(x_array, f_array, '绝对值函数')


# 正弦函数
f_array = np.sin(x_array)
visualize_fx(x_array, f_array, '正弦函数')



# 反正弦函数
x_array_ = np.copy(x_array)
x_array_[(x_array_ < -1) | (x_array_ > 1)] = np.nan
f_array = np.arcsin(x_array_)
visualize_fx(x_array_, f_array, '反正弦函数')


# 正切函数
f_array = np.tan(x_array)
f_array[:-1][np.diff(f_array) < 0] = np.nan
visualize_fx(x_array, f_array, '正切函数')



# 反正切函数
f_array = np.arctan(x_array)
visualize_fx(x_array, f_array, '反正切函数')



# 余弦函数
f_array = np.cos(x_array)
visualize_fx(x_array, f_array, '余弦函数')




# 反余弦函数
x_array_ = np.copy(x_array)
x_array_[(x_array_ < -1) | (x_array_ > 1)] = np.nan
f_array = np.arccos(x_array_)
visualize_fx(x_array_, f_array, '反余弦函数')


# 双曲正弦函数
f_array = np.sinh(x_array)
visualize_fx(x_array, f_array, '双曲正弦函数')



# 双曲余弦函数
f_array = np.cosh(x_array)
visualize_fx(x_array, f_array, '双曲余弦函数')



# 双曲正切函数
f_array = np.tanh(x_array)
visualize_fx(x_array, f_array, '双曲正切函数')

# 向下取整函数
f_array = np.floor(x_array)
visualize_fx(x_array, f_array, '向下取整函数', True)



# 向上取整函数
f_array = np.ceil(x_array)
visualize_fx(x_array, f_array, '向上取整函数', True)


# 符号函数
f_array = np.sign(x_array)
visualize_fx(x_array, f_array, '符号函数', True)



# 指数函数
f_array = np.exp(x_array)
visualize_fx(x_array, f_array, '指数函数')

# 对数函数
x_array_ = np.copy(x_array)
x_array_[x_array_<=0] = np.nan
f_array = np.log(x_array_)
visualize_fx(x_array_, f_array, '对数函数')


# 统计函数
A_2D = np.random.randint(0,10, size = (4,6))
visualize_2D(A_2D, '二维数组', 9, 0)


from sklearn.datasets import load_iris

iris = load_iris()
iris_data_array = iris.data

import seaborn as sns
fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(iris_data_array,   # 鸢尾花数据数组
            cmap = 'RdYlBu_r', # 指定色谱
            ax = ax,           # 指定轴
            vmax = 8,          # 色谱最大值
            vmin = 0,          # 色谱最小值
            xticklabels = [],  # 不显示横轴标签
            yticklabels = [],  # 不显示纵轴标签
            cbar = True)       # 显示色谱条

# fig.savefig('Figures/鸢尾花数据热图.svg', format='svg')

# 最大值
A_2D.max(axis = 0)
visualize_1D(A_2D.max(axis = 0), '沿axis = 0，最大值', 0, 9)

A_2D.max(axis = 1)
A_2D.max(axis = 1, keepdims = True)

visualize_1D(A_2D.max(axis = 1), '沿axis = 1，最大值', 0, 9)

# 平均值
np.average(iris_data_array)

np.average(iris_data_array, axis = 0)
np.average(iris_data_array, axis = 1)



## 方差
np.var(iris_data_array)
# 注意，NumPy中默认分母为n


# 标准差
np.std(iris_data_array, axis = 0)
# 注意，NumPy中默认分母为n
# array([0.82530129, 0.43441097, 1.75940407, 0.75969263])
# 协方差矩阵
np.cov(iris_data_array.T, ddof = 1)
# 注意转置
# array([[ 0.68569351, -0.042434  ,  1.27431544,  0.51627069],
#        [-0.042434  ,  0.18997942, -0.32965638, -0.12163937],
#        [ 1.27431544, -0.32965638,  3.11627785,  1.2956094 ],
#        [ 0.51627069, -0.12163937,  1.2956094 ,  0.58100626]])
# ​
fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(np.cov(iris_data_array.T, ddof = 1),
            cmap = 'RdYlBu_r', # 指定色谱
            annot = True,      # 注释
            ax = ax,           # 指定轴
            fmt = ".2f",       # 注释数值的格式
            square = True,     # 热图方格为正方形
            xticklabels = [],  # 不显示横轴标签
            yticklabels = [],  # 不显示纵轴标签
            cbar = True)       # 显示色谱条

# fig.savefig('Figures/鸢尾花数据协方差矩阵.svg', format='svg')



# 相关性系数矩阵

np.corrcoef(iris_data_array.T)
# 注意转置

fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(np.corrcoef(iris_data_array.T),
            cmap = 'RdYlBu_r', # 指定色谱
            annot = True,      # 注释
            ax = ax,           # 指定轴
            fmt = ".2f",       # 注释数值的格式
            square = True,     # 热图方格为正方形
            xticklabels = [],  # 不显示横轴标签
            yticklabels = [],  # 不显示纵轴标签
            cbar = True)       # 显示色谱条

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap16


# 导入包
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib import cm
# 导入色谱

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# ### 定义可视化函数

def visualize_2D(array, title, vmax, vmin):

    fig_width  = math.ceil(array.shape[1] * 0.5)
    fig_length = math.ceil(array.shape[0] * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    # 注意，seaborn.heatmap() 目前只能接受2D数组
    # 本书中，一维数组可视化时用圆圈
    # 可视化时，只有二维、三维数组用方块

    sns.heatmap(array,
                vmax = vmax,
                vmin = vmin,
                annot = True,      # 增加注释
                fmt = ".0f",       # 注释数值的格式
                square = True,     # 热图方格为正方形
                cmap = 'RdYlBu_r', # 指定色谱
                linewidths = .5,   # 方格线宽
                cbar = False,      # 不显示色谱条
                yticklabels=False, # 不显示纵轴标签
                xticklabels=False, # 不显示横轴标签
                ax = ax)           # 指定绘制热图的轴

    fig.savefig('Figures/' + title + '.svg', format='svg')

# 定义绘制一元数组可视化函数

def visualize_1D(array, title):
    fig, ax = plt.subplots()

    colors = cm.RdYlBu_r(np.linspace(0,1,len(array)))

    for idx in range(len(array)):

        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = str(array[idx]),
                horizontalalignment = 'center',
                verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.savefig('Figures/' + title + '.svg', format='svg')


# ### 生成1D数组
a_1D_array = np.arange(-7, 7 + 1)
a_1D_array

visualize_1D(a_1D_array, '1D数组')

len(a_1D_array)
# 长度

# 形状
a_1D_array.shape

# 维数
a_1D_array.ndim


# ### 变形为行向量，1行15列
a_row_vector = a_1D_array.reshape(1,-1)
# 有时候，当你想要改变数组的形状，但是不确定新的形状应该是多少时，你可以使用 -1 这个特殊值来指定某一个维度的大小
a_row_vector

np.reshape(a_1D_array, (1,-1))
# 把numpy.reshape() 当成一个函数来用

np.reshape(a_1D_array, (1,15))
# 把numpy.reshape() 当成一个函数来用

visualize_2D(a_row_vector, '行向量', 7, -7)


# 形状
a_row_vector.shape

# 维数
a_row_vector.ndim


a_1D_array.reshape(1,15)

a_1D_array.reshape(-1,15)

a_1D_array.reshape(1,len(a_1D_array))


# ### 改成列向量，15行1列

a_col_vector = a_1D_array.reshape(-1,1)
a_col_vector


visualize_2D(a_col_vector, '列向量', 7, -7)


a_col_vector.shape

a_1D_array.reshape(-1,1)

np.reshape(a_1D_array, (-1,1))
# 把numpy.reshape() 当成一个函数来用

np.reshape(a_1D_array, (15,1))
# 把numpy.reshape() 当成一个函数来用

np.reshape(a_1D_array, (len(a_1D_array),1))


# ### 改成 3 行 5 列，先行后列
# 请大家试着将一维数组写成2行 8列数组，看一下是否报错

A_3_by_5 = a_1D_array.reshape(3, 5)
# 先行后列为默认顺序
A_3_by_5

visualize_2D(A_3_by_5, '矩阵，3 x 5', 7, -7)

a_1D_array.reshape(3, -1)

a_1D_array.reshape(-1, 5)


# ### 改成 3 行 5 列，先列后行

A_3_by_5_col_order = a_1D_array.reshape(3, 5, order = 'F')
A_3_by_5_col_order

visualize_2D(A_3_by_5_col_order, '矩阵，3 x 5, 先列后行', 7, -7)

# ### 改成 5 行 3 列

A_5_by_3 = a_1D_array.reshape(5, 3)
A_5_by_3

a_1D_array.reshape(5, -1)

a_1D_array.reshape(-1, 3)

a_1D_array.reshape(5, int(len(a_1D_array)/5))
# 形状参数必须是整数，不能是float

visualize_2D(A_5_by_3, '矩阵，5 x 3, 先行后列', 7, -7)

A_5_by_3_col_order = a_1D_array.reshape(5, 3, order = 'F')
A_5_by_3_col_order

visualize_2D(A_5_by_3_col_order, '矩阵，5 x 3, 先列后行', 7, -7)

# ### 从 3 * 5 到 5 * 3

A_3_by_5.reshape(5,3)

visualize_2D(A_3_by_5.reshape(5,3), '从 3 X 5 到 5 X 3', 7, -7)

np.reshape(np.reshape(a_1D_array, (3,5)), (5,3))

a_1D_array.reshape(3,5).reshape(5,3)

a_1D_array.reshape(3,-1).reshape(5,-1)


A_3_by_5_col_order.reshape(5,3)


# ### 变成三维 3D

a_1D_array_long = np.arange(-13,13 + 1)
a_1D_array_long

visualize_1D(a_1D_array_long, '1D数组，27元素')

A_3D_3_by_3_by_3 = a_1D_array_long.reshape(3,3,3)
A_3D_3_by_3_by_3

a_1D_array_long.reshape(3,3,-1)

visualize_2D(A_3D_3_by_3_by_3[0,:,:], '3D_第一页', 13, -13)

visualize_2D(A_3D_3_by_3_by_3[1,:,:], '3D_第二页', 13, -13)

visualize_2D(A_3D_3_by_3_by_3[2,:,:], '3D_第三页', 13, -13)


# ### 视图 vs 副本

# 判断是否共享内存？
np.shares_memory(a_1D_array, A_5_by_3)

np.shares_memory(a_1D_array, a_col_vector)

np.shares_memory(a_1D_array, a_row_vector)

# 新形状和原始形状的元素数量相同，返回视图
a = np.array([[1, 2], [3, 4]])
b = a.reshape(4)
print(b)     # [1 2 3 4]
b[0] = 0
print(a)     # [[0 2], [3 4]]


# ### 转置

# 一维数组的转置还是其本身
a_1D_array.T
# 请大家学习使用numpy.swapaxes()

a_row_vector.T

np.transpose(a_row_vector)


a_col_vector.T

np.transpose(a_col_vector)


A_3_by_5_T = A_3_by_5.T
A_3_by_5_T

visualize_2D(A_3_by_5_T, '矩阵，3 x 5, 转置', 7, -7)

np.transpose(A_3_by_5)

# 视图 vs 副本
# 定义一个二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 使用.transpose()方法得到数组的转置
arr_transpose = arr.transpose()

# 使用.T属性得到数组的转置
arr_T = arr.T

# 修改新数组中的元素
arr_transpose[0, 1] = 100
arr_T[2, 0] = 200

# 输出结果
print("原始数组：")
print(arr)
print("使用.transpose()方法得到的数组的转置：")
print(arr_transpose)
print("使用.T属性得到的数组的转置：")
print(arr_T)


# ### 扁平化

A_3_by_5.ravel()
# 需要注意的是，ravel()函数返回的是原始数组的视图（view），而不是其副本（copy）。
# 因此，如果修改新数组中的任何元素，原始数组也会受到影响。
# 如果需要返回一个数组副本，可以使用flatten()函数。

A_3_by_5.ravel().shape


A_3_by_5.ravel().ndim

A_3_by_5.reshape(-1)


A_3_by_5.ravel(order = 'F')


A_3_by_5_flatten = A_3_by_5.flatten()
# A_3_by_5_flatten是A_3_by_5的副本


A_3_by_5_flatten[0] = 1000
A_3_by_5_flatten


A_3_by_5
# A_3_by_5并没有变化

# 视图 vs 副本

# 定义一个多维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 使用flatten()函数将多维数组降为一维数组
new_arr1 = arr.flatten()

# 使用ravel()函数将多维数组降为一维数组
new_arr2 = arr.ravel()

# 修改新数组中的元素
new_arr1[-1] = 1000
new_arr2[0] = 200

# 输出结果
print("原始数组：")
print(arr)
print("使用flatten()函数得到的一维数组：")
print(new_arr1)
print("使用ravel()函数得到的一维数组：")
print(new_arr2)


# ### 旋转
A_3_by_5
np.rot90(A_3_by_5)

visualize_2D(np.rot90(A_3_by_5), '逆时针旋转90度', 7, -7)

A_3_by_5_anti_c_90 = np.rot90(A_3_by_5)

# 视图 vs 副本

# 创建一个 2x2 的二维数组
arr = np.array([[1, 2], [3, 4]])

# 将数组逆时针旋转90度，得到数组的视图
rotated_arr = np.rot90(arr)

# 修改视图的值
rotated_arr[0, 0] = 100

# 打印原数组和修改后的视图
print("原数组：")
print(arr)

print("修改后的视图：")
print(rotated_arr)


# ### 翻转
np.flip(A_3_by_5)
# 沿着所有轴翻转

visualize_2D(np.flip(A_3_by_5), '沿着所有轴翻转', 7, -7)

# 沿着指定的轴翻转
# 沿着指定轴进行翻转
arr3 = np.array([[1, 2], [3, 4], [5, 6]])
flipped_arr3 = np.flip(arr3, axis=0)  # 沿着第一个轴进行翻转
print(flipped_arr3)
# 输出 [[5 6] [3 4] [1 2]]

flipped_arr4 = np.flip(arr3, axis=1)  # 沿着第二个轴进行翻转
print(flipped_arr4)
# 输出 [[2 1] [4 3] [6 5]]

flipped_arr5 = np.flip(arr3, axis=(0, 1))  # 沿着所有轴进行翻转
print(flipped_arr5)
# 输出 [[6 5] [4 3] [2 1]]

np.flipud(A_3_by_5)

np.fliplr(A_3_by_5)


# # NumPy数组规整


# 导入包
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


def visualize_2D(array, title, vmax, vmin):
    fig_width  = math.ceil(array.shape[1] * 0.5)
    fig_length = math.ceil(array.shape[0] * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    # 注意，seaborn.heatmap() 目前只能接受2D数组
    # 本书中，一维数组可视化时用圆圈
    # 可视化时，只有二维、三维数组用方块
    sns.heatmap(array,
                vmax = vmax,
                vmin = vmin,
                annot = True,      # 增加注释
                fmt = ".0f",       # 注释数值的格式
                square = True,     # 热图方格为正方形
                cmap = 'RdYlBu_r', # 指定色谱
                linewidths = .5,   # 方格线宽
                cbar = False,      # 不显示色谱条
                yticklabels=False, # 不显示纵轴标签
                xticklabels=False, # 不显示横轴标签
                ax = ax)           # 指定绘制热图的轴

    fig.savefig('Figures/' + title + '.svg', format='svg')

# 定义绘制一元数组可视化函数

def visualize_1D(array, title, vmax, vmin):
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("RdYlBu_r")
    array_norm = (array - vmin) / (vmax - vmin)
    colors = cmap(array_norm)

    for idx in range(len(array)):

        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = "{:.1f}".format(array[idx]),
                horizontalalignment = 'center',
                verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.savefig('Figures/' + title + '.svg', format='svg')

# ## 堆叠
a1 = np.arange(1, 5 + 1)
visualize_1D(a1, 'a1，一维', 5, -5)

a2 = np.arange(-5, 0)
visualize_1D(a2, 'a2，一维', 5, -5)

# ### 沿行堆叠
np.stack((a1, a2))
# 默认， axis = 0
visualize_2D(np.stack((a1, a2)), '沿行叠加，np.stack', 5, -5)

# 多个数组拼接
np.stack((a1, a2, a1, a2, a1, a2))
np.row_stack((a1, a2))

np.vstack((a1, a2))
np.vstack((a1.reshape(-1,1), a2.reshape(-1,1)))

# 创建一个空数组
arr = np.empty((0, 3), dtype=int)
# 通过for循环向数组中添加元素
for i in range(5):
    row = [i, i+1, i+2]
    arr = np.vstack([arr, row])
print(arr)

# 在这个示例中，首先使用np.empty()函数创建了一个空数组，
# 其形状为(0, 3)，表示该数组有0行3列。
# 接着通过for循环，生成一个包含5行3列的数组，
# 每一行的值都是从i开始的三个连续整数。

# 在for循环中，使用np.vstack()函数将每一行添加到数组中。
# np.vstack()函数可以将多个数组沿着垂直方向堆叠起来，
# 因此可以将每一行当作一个数组，
# 然后将它们依次添加到原来的数组中。


# ### 沿列堆叠
np.stack((a1, a2), axis = 1)
visualize_2D(np.stack((a1, a2), axis = 1), '沿列叠加，np.stack', 5, -5)

np.column_stack((a1, a2))
visualize_2D(np.column_stack((a1, a2)), '沿列叠加，np.stack', 5, -5)

np.hstack((a1, a2))

# numpy.hstack 函数可以用来沿着水平方向将多个数组堆叠在一起，形成一个新的数组
visualize_1D(np.hstack((a1, a2)), 'a1，a2，一维堆叠', 5, -5)
visualize_1D(np.hstack((a2, a1)), 'a1，a2，一维堆叠，反向', 5, -5)
np.hstack((a1.reshape(-1,1), a2.reshape(-1,1)))


# ### 拼接numpy.concatenate()
A_3_by_3 = np.arange(-4,4+1).reshape(3,3)
visualize_2D(A_3_by_3, 'A_3_by_3', 5, -5)


B_3_by_1 = np.array([[4, 0, 4]])
visualize_2D(B_3_by_1, 'B_3_by_1', 5, -5)


# 转置
visualize_2D(B_3_by_1.T, 'B_3_by_1.T', 5, -5)


# #### 沿行
np.concatenate((A_3_by_3, B_3_by_1), axis=0)
visualize_2D(np.concatenate((A_3_by_3, B_3_by_1), axis=0), '拼接，沿行', 5, -5)

np.vstack((A_3_by_3, B_3_by_1))


# #### 沿列
np.concatenate((A_3_by_3, B_3_by_1.T), axis=1)
visualize_2D(np.concatenate((A_3_by_3, B_3_by_1.T), axis=1), '拼接，沿列', 5, -5)


np.hstack((A_3_by_3, B_3_by_1.T))
np.concatenate((A_3_by_3, B_3_by_1), axis=None)


# ### 堆叠结果为三维数组
A = np.arange(1, 24 + 1).reshape(4, -1)
visualize_2D(A, '二维数组A', 24, -24)

B = np.arange(-24, 0).reshape(4, -1)
visualize_2D(B, '二维数组B', 24, -24)
# #### axis = 0，前后堆叠
A_B_0 = np.stack((A, B))
# 默认叠合方向 axis = 0

# 取出第一页A
A_B_0[0, :, :]
A_B_0[0, ...]

visualize_2D(A_B_0[0, ...], '沿深度堆叠，axis = 0，第0页', 12, -12)
visualize_2D(A_B_0[1, ...], '沿深度堆叠，axis = 0，第1页', 12, -12)

# #### axis = 1，上下堆叠
A_B_1 = np.stack((A, B), axis=1)

visualize_2D(A_B_1[0, ...], '沿深度堆叠，axis = 1，第0页', 24, -24)
visualize_2D(A_B_1[1, ...], '沿深度堆叠，axis = 1，第1页', 24, -24)
visualize_2D(A_B_1[2, ...], '沿深度堆叠，axis = 1，第2页', 24, -24)
visualize_2D(A_B_1[3, ...], '沿深度堆叠，axis = 1，第3页', 24, -24)
# 取出A
A_B_1[:, 0, :]


# #### axis = 2，左右堆叠
A_B_2 = np.stack((A, B), axis=2)
# dimension 2
visualize_2D(A_B_2[0, ...], '沿深度堆叠，axis = 2，第0页', 24, -24)
visualize_2D(A_B_2[1, ...], '沿深度堆叠，axis = 2，第1页', 24, -24)
visualize_2D(A_B_2[2, ...], '沿深度堆叠，axis = 2，第2页', 24, -24)
visualize_2D(A_B_2[3, ...], '沿深度堆叠，axis = 2，第3页', 24, -24)

# 取出A
A_B_2[...,0]


# ## 重复
# ### 重复numpy.repeat()

a_1D = np.arange(-2,2+1)
visualize_1D(a_1D, 'a_1D，一维', 3, -3)

np.repeat(a_1D, 2)
visualize_1D(np.repeat(a_1D, 2), 'a_1D，一维，重复两次', 3, -3)
visualize_1D(np.repeat(a_1D, 3), 'a_1D，一维，重复三次', 3, -3)


# ### 瓷砖numpy.tile()
np.tile(a_1D, 2)
visualize_1D(np.tile(a_1D, 2), 'a_1D，一维，瓷砖二次', 3, -3)
visualize_1D(np.tile(a_1D, 3), 'a_1D，一维，瓷砖三次', 3, -3)

np.tile(a_1D, (2,1))
visualize_2D(np.tile(a_1D, (2,1)), 'a_1D，一维，瓷砖(2,1)', 3, -3)

np.tile(a_1D, (2,2))
visualize_2D(np.tile(a_1D, (2,2)), 'a_1D，一维，瓷砖(2,2)', 3, -3)

A = np.arange(-3,3).reshape(2,3)
np.tile(A, 2)
visualize_2D(np.tile(A, 2), 'a_1D，一维，瓷砖(2,2)', 3, -3)

np.tile(A, (2,1))
visualize_2D(np.tile(A, (2,1)), 'a_1D，一维，瓷砖(2,2)', 3, -3)

np.tile(A, (2,2))
visualize_2D(np.tile(A, (2,2)), 'a_1D，一维，瓷砖(2,2)', 3, -3)


# ## 分块矩阵
# ### 合成
A = np.eye(2)
B = np.arange(-4,4+1).reshape(3,3)
M = np.block([[A,                np.zeros((2, 3))],
              [np.zeros((3, 2)), B               ]])

visualize_2D(M, '合成矩阵M', 5, -5)


# ### 切割
# #### 沿指定轴numpy.split()
a_1D = np.arange(-9,9)
visualize_1D(a_1D, 'a_1D，一维', 10, -10)

a_3_splits = np.split(a_1D, 3)

np.split(a_1D, [3, 5, 6, 10, 16])
# 指定切割indexes

A_9_by_9 = np.arange(-40,40+1).reshape(9,9)
A_9_by_9


visualize_2D(A_9_by_9, 'A_9_by_9', 40, -40)

A_3_splits_axis_0 = np.split(A_9_by_9, 3)
A_3_splits_axis_0

A_3_splits_axis_1 = np.split(A_9_by_9, 3, axis = 1)
A_3_splits_axis_1


# #### 沿着水平方向切割
np.hsplit(A_9_by_9, 3)


# #### 沿着竖直方向切割
np.vsplit(A_9_by_9, 3)


# ## 插入、删除
# ### 附加numpy.append()
np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)

# ### 插入
a = np.array([[1, 1], [2, 2], [3, 3]])
np.insert(a, [1], [[1],[2],[3]], axis=1)

np.insert(a, 1, [1, 2, 3], axis=1)


# ### 删除
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
np.delete(arr, 1, 0)































































































































































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap17
























#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chap18







































































































































































































































































































































































































































































































































































































































































































































