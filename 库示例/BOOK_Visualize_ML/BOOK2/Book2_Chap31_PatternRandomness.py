







# 用热图展示无理数的小数位
from mpmath import mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 圆周率
mp.dps = 2**10 + 1
digits = str(mp.pi)[2:]

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((2**5, 2**5))

fig, ax = plt.subplots()

ax = sns.heatmap(digits_matrix, vmin=0, vmax=9,
                 cmap="rainbow",
                 yticklabels=False,
                 xticklabels=False)

ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)

## 常数e
# mp.dps = 2**10 + 1
digits = str(mp.e)[2:]

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((2**5, 2**5))


fig, ax = plt.subplots()

ax = sns.heatmap(digits_matrix, vmin=0, vmax=9,
                 cmap="rainbow",
                 yticklabels=False,
                 xticklabels=False)

ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)

## 根号2
# mp.dps = 2**10 + 1
digits = str(mp.sqrt(2))[2:]

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((2**5, 2**5))
fig, ax = plt.subplots()

ax = sns.heatmap(digits_matrix, vmin=0, vmax=9,
                 cmap="rainbow",
                 yticklabels=False,
                 xticklabels=False)

ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)


## 黄金分割比
# mp.dps = 2**10 + 1
digits = str((1 + mp.sqrt(5))/2)[2:]

digits_list = [int(x) for x in digits]

digits_array  = np.array(digits_list)
digits_matrix = digits_array.reshape((2**5, 2**5))
fig, ax = plt.subplots()

ax = sns.heatmap(digits_matrix, vmin=0, vmax=9,
                 cmap="rainbow",
                 yticklabels=False,
                 xticklabels=False)

ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False)



#%% 从散点到概率密度

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


# 从seaborn中导入鸢尾花样本数据
iris_sns = sns.load_dataset("iris")
fig, ax = plt.subplots()
ax = sns.kdeplot( data=iris_sns, x="sepal_length", y="sepal_width", fill=True, cmap = 'RdYlBu_r', n_levels = 20)

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签

ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度

ax.axis('scaled')
# 设定横纵轴尺度1:1
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)

ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围


### 增加概率密度
import scipy.stats as st

XX,YY = np.meshgrid(np.linspace(4, 8, 100),
                    np.linspace(1, 5, 100))
positions = np.vstack([XX.ravel(), YY.ravel()])
samples = iris_sns[['sepal_length','sepal_width']].to_numpy()
kernel = st.gaussian_kde(samples.T)

PDF_xy = np.reshape(kernel(positions).T, XX.shape)
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")
plt.contour(XX,YY,PDF_xy,levels = 15, cmap = 'RdYlBu_r')
# 利用 seaborn.scatterplot() 绘制散点图
# x对应横轴特征，鸢尾花数据帧列名 "sepal_length"
# y对应纵轴特征，鸢尾花数据帧列名 "sepal_width"
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度
ax.axis('scaled')
# 设定横纵轴尺度1:1
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围


### 用散点颜色代表概率密度
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = kernel(samples.T), palette = 'RdYlBu_r')
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度
ax.axis('scaled')
# 设定横纵轴尺度1:1
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围

### 使用自己定义的色谱
blues_cmap = sns.light_palette('#0091FE', as_cmap=True)
# 函数第一个输入为Hex色号
# 类似函数，seaborn.dark_palette()

fig, ax = plt.subplots()
ax = sns.kdeplot(data=iris_sns, x="sepal_length", y="sepal_width", fill=True, cmap = blues_cmap, n_levels = 15)
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度
ax.axis('scaled')
# 设定横纵轴尺度1:1
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围


fig, ax = plt.subplots()
ax = sns.kdeplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "species", fill=False, n_levels = 10)
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签

ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度
ax.axis('scaled')
# 设定横纵轴尺度1:1
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)

ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围

#%% 满足二元高斯分布的随机数
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import multivariate_normal

## 生成数据
np.random.seed(2)
mu_X = 0
mu_Y = 0
MU = [mu_X, mu_Y]
sigma_X = 1
sigma_Y = 1
num = 400
X_grid = np.linspace(-3,3,200)
Y_grid = np.linspace(-3,3,200)

XX, YY = np.meshgrid(X_grid, Y_grid)
XXYY = np.dstack((XX, YY))

## 可视化
rho_array = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
fig = plt.figure(figsize = (6,9))
for idx in range(6):
    rho = rho_array[idx]
    # covariance
    SIGMA = [[sigma_X**2, sigma_X*sigma_Y*rho],
             [sigma_X*sigma_Y*rho, sigma_Y**2]]
    bi_norm = multivariate_normal(MU, SIGMA)
    pdf_fine = bi_norm.pdf(XXYY)
    X, Y = multi_norm(MU, SIGMA, num).T
    center_X = np.mean(X)
    center_Y = np.mean(Y)

    ax = plt.subplot(3,2,idx + 1)
    # plot center of data
    plt.plot(X,Y,'.', color = '#223C6C', alpha = 1, markersize = 5)

    levels = np.linspace(-pdf_fine.max() * 0.2, pdf_fine.max() * 1.1, 20)
    ax.contourf(XX, YY, pdf_fine, levels = levels, cmap = 'RdYlBu_r')
    ax.contour(XX, YY, pdf_fine, levels = levels, colors = 'w')
    ax.axvline(x = 0, color = 'k', linestyle = '--')
    ax.axhline(y = 0, color = 'k', linestyle = '--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim((-3,3))
    ax.set_ylim((-3,3))

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')

#%% # 用色谱分段渲染一条曲线
# 导入包
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

### 1. 定义函数
def colorline(x, y, cmap):

    norm=plt.Normalize(0.0, 1.0)
    # 归一化函数，将数据线性归一化在 [0, 1] 区间
    segments = make_segments(x, y)
    # make_segments 自定义函数，将一条线打散成一系列线段

    lc = LineCollection(segments, array = np.linspace(0.0, 1.0, len(x)),
                              cmap=cmap, norm=norm,
                              linewidth=1, alpha=1)
    # LineCollection 可以看成是一系列线段的集合
    # 可以用色谱分别渲染每一条线段
    # 这样可以得到颜色连续变化的效果


    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    # 将一条线打散成一系列线段
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


### 2. 生成平面随机轨迹
N_steps = 10000;
# 随机轨迹的步数

delta_x = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))
delta_y = np.random.normal(loc=0.0, scale=1.0, size=(N_steps,1))
# 生成满足正态分布的随机数

disp_x = np.cumsum(delta_x, axis = 0);
disp_y = np.cumsum(delta_y, axis = 0);
# 用累加生成平面轨迹

disp_x = np.vstack(([0],disp_x))
disp_y = np.vstack(([0],disp_y))
# 给轨迹添加起点 (0, 0)

### 3. 可视化一条曲线
fig, ax = plt.subplots(figsize = (6,6))
plt.style.use('dark_background')
# 使用黑色背景

colorline(disp_x, disp_y, cmap='rainbow_r')
# 调用自定义函数 colorline

plt.plot(disp_x[0],disp_y[0],'wx', markersize = 12)
plt.plot(disp_x[-1],disp_y[-1],'wx', markersize = 12)
# 绘制起点、终点

plt.xticks([])
plt.yticks([])
# fig.savefig('Figures/可视化一条曲线.svg', format='svg')


### 4. 生成多条轨迹
N_steps = 200;
# 步数

N_paths = 50;
# 轨迹数量

sigma = 1
delta_t = 0.2
# 随机过程的参数
# 请参考《数据有道》第8章

delta_X = np.random.normal(loc=0.0, scale=sigma*np.sqrt(delta_t), size=(N_steps,N_paths))
# 生成服从正态分布随机数

t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t
# 时间戳数据

X = np.cumsum(delta_X, axis = 0)
# 用累加生成沿时间多条轨迹轨迹

X_0 = np.zeros((1,N_paths))
X = np.vstack((X_0,X))
# 给轨迹添加起点


fig, ax = plt.subplots(figsize = (6,6))
plt.style.use('dark_background')

for idx in range(X.shape[1]):
    # 分别绘制50条轨迹

    y_idx = X[:,idx]

    colorline(t_n, y_idx, cmap='rainbow')
    # 每条轨迹分段着色

ax.set_xlim([0,N_steps*delta_t])
ax.set_ylim([-20,20])
ax.set_yticks([])
ax.set_xticks([])
plt.show()
# fig.savefig('Figures/可视化多条曲线.svg', format='svg')


#%% # 随机行走的趋势

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


N_steps = 100;
# number of steps
N_paths = 100;
# number of paths
std = 1
mu  = -0.2 # 0, 0.2
delta_t = 1

delta_X = np.random.normal(loc=0.0, scale=std*np.sqrt(delta_t), size=(N_steps,N_paths)) + mu * delta_t
t_n = np.linspace(0,N_steps,N_steps+1,endpoint = True)*delta_t

X = np.cumsum(delta_X, axis = 0);
X_0 = np.zeros((1,N_paths))
X = np.vstack((X_0,X))

rows = 1
cols = 2


fig, ax = plt.subplots(figsize=(10,8))

num_layers = 3

num_lines = 10

for idx in range(num_layers):
    err_up = mu * t_n + (idx + 1)*std*np.sqrt(t_n)
    err_down = mu * t_n - (idx + 1)*std*np.sqrt(t_n)

    plt.fill_between(t_n, err_up, err_down, alpha=0.2, color='#008DF6', edgecolor = None)

ax.plot(t_n, X, lw=0.25,color = '#223C6C')
ax.plot(t_n, np.mean(X,axis = 1),color = 'r')
ax.set_xlim([0,N_steps])
ax.set_ylim([np.floor(X.min()/5 - 1)*5,np.ceil(X.max()/5 + 1)*5])
ax.set_yticks([])
ax.set_xticks([])
ax.axis('off')










#%%

























