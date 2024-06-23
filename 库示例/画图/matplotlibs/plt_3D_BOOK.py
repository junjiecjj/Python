



#==========================================================================================================
##########################################  3D Scatter Plot, 三维散点 ######################################
#==========================================================================================================

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")

# 处理数据
# Load the iris data
iris_sns = sns.load_dataset("iris")

x1 = iris_sns['sepal_length']
x2 = iris_sns['sepal_width']
x3 = iris_sns['petal_length']
x4 = iris_sns['petal_width']

labels = iris_sns['species'].copy()

labels[labels == 'setosa']     = 1
labels[labels == 'versicolor'] = 2
labels[labels == 'virginica']  = 3

rainbow = plt.get_cmap("rainbow")

## 1
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()


## 2: 投影，沿z
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, zdir = 'z', zs = 1)
# 投影在 z = 1平面上

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()

## 3: 投影，沿y
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x3, zdir = 'y', zs = 5)
# 投影在 y = 5 平面上
ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/投影，沿y.svg', format='svg')
plt.show()


## 4: 投影，沿x
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x2, x3, zdir = 'x', zs = 8)
# 投影在 x = 8 平面上

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/投影，沿x.svg', format='svg')
plt.show()


## 5:利用散点大小展示第四个特征
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3, s = x4*20)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/利用散点大小展示第四个特征.svg', format='svg')
plt.show()

## 6: 利用颜色展示分类标签
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter_h = ax.scatter(x1, x2, x3,
                       c = labels,
                       cmap=rainbow)

classes = ['Setosa', 'Versicolor', 'Virginica']

plt.legend(handles=scatter_h.legend_elements()[0], labels=classes)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/利用颜色展示分类标签.svg', format='svg')
plt.show()


## 7: 颜色分类 + 散点大小
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3,
           s = x4*20,
           c = labels,
           cmap=rainbow)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/颜色分类 + 大小.svg', format='svg')
plt.show()

## 8: 利用色谱展示第四维特征
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter_plot = ax.scatter(x1, x2, x3,
                          c = x4,
                          cmap=rainbow)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
fig.colorbar(scatter_plot, ax = ax, shrink = 0.5, aspect = 10)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/利用色谱展示第四维特征.svg', format='svg')
plt.show()

## 9: 用标记类型展示特征
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1[x4>1],  x2[x4>1],  x3[x4>1],
           marker='o')
ax.scatter(x1[x4<=1], x2[x4<=1], x3[x4<=1],
           marker='x')

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('利用marker shape展示特征.svg', format='svg')
plt.show()


#%% 可视化三元概率分布
# 导入包
from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

def visualize_multinomial(p_array, num = 20):
    x1_array = np.arange(num + 1)
    x2_array = np.arange(num + 1)

    xx1, xx2 = np.meshgrid(x1_array, x2_array)

    xx3 = num - xx1 - xx2
    xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

    PMF_ff = multinomial.pmf(x = np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, n = num, p = p_array)
    PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)
    PMF_ff = np.reshape(PMF_ff, xx1.shape)


    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")

    scatter_plot = ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(),
                 s = 50,
                 marker = '.',
                 alpha = 1,
                 c = PMF_ff.ravel(),
                 cmap = 'RdYlBu_r')

    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xticks([0,num])
    ax.set_yticks([0,num])
    ax.set_zticks([0,num])

    ax.set_xlim(0, num)
    ax.set_ylim(0, num)
    ax.set_zlim3d(0, num)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    # fig.colorbar(scatter_plot, ax = ax, shrink = 0.5, aspect = 10)
    title = '_'.join(str(round(p_i,2)) for p_i in p_array)
    title = 'p_array_' + title
    ax.set_title(title)

    # fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()


p_array = [1/3, 1/3, 1/3]
visualize_multinomial(p_array)

p_array = [0.2, 0.2, 0.6]
visualize_multinomial(p_array)


p_array = [0.2, 0.6, 0.2]
visualize_multinomial(p_array)



#%% scatter3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 自定义马哈距离、高斯分布密度函数
def Mahal_d(Mu, Sigma, x):
    # 计算马哈距离

    # 中心化，mu为质心
    x_demeaned = x - Mu

    # 协方差矩阵求逆
    inv_covmat = np.linalg.inv(Sigma)

    # 计算马氏距离平方
    mahal_sq = x_demeaned @ inv_covmat @ x_demeaned.T
    print(mahal_sq.shape)

    # 仅保留对角线元素
    mahal_sq = np.diag(mahal_sq)

    # 对角线元素开平方，得到马氏距离
    mahal_d = np.sqrt(mahal_sq)

    return mahal_d

def Mahal_d_2_pdf(d, Sigma):
    # 将马氏距离转化为概率密度

    # 计算第一个缩放因子，和协方差行列式有关
    scale_1 = np.sqrt(np.linalg.det(Sigma))

    # 计算第二个缩放因子，和高斯函数有关
    scale_2 = (2*np.pi)**(3/2)

    # 高斯函数，马氏距离转为亲近度
    gaussian = np.exp(-d**2/2)

    # 完成缩放，得到概率密度值
    pdf = gaussian/scale_1/scale_2

    return pdf

# 产生网格数据、概率密度
x1 = np.linspace(-2,2,31)
x2 = np.linspace(-2,2,31)
x3 = np.linspace(-2,2,31)

xxx1, xxx2, xxx3 = np.meshgrid(x1,x2,x3)

Mu = np.array([[0, 0, 0]])

Sigma = np.array([[1, 0.6, -0.4],
                  [0.6, 1.5, 1],
                  [-0.4, 1, 2]])

x_array = np.vstack([xxx1.ravel(), xxx2.ravel(), xxx3.ravel()]).T

# 首先计算马氏距离
d_array = Mahal_d(Mu, Sigma, x_array)
d_array = d_array.reshape(xxx1.shape)

# 将马氏距离转化成概率密度PDF
pdf_zz = Mahal_d_2_pdf(d_array, Sigma)

xmin, xmax = xxx1.min(), xxx1.max()
ymin, ymax = xxx2.min(), xxx2.max()
zmin, zmax = xxx3.min(), xxx3.max()

normalize = mpl.colors.Normalize(vmin=0, vmax=0.1)

# 沿x3
fig = plt.figure(figsize=(6, 36))

for fig_idx, x3_slice_idx in enumerate(np.arange(0, len(x3), 5)):

    ax = fig.add_subplot(len(np.arange(0, len(x3), 5)), 1, fig_idx + 1, projection='3d')

    ax.scatter(xxx1[:, :, x3_slice_idx].ravel(),
               xxx2[:, :, x3_slice_idx].ravel(),
               xxx3[:, :, x3_slice_idx].ravel(),
               c=pdf_zz[:, :, x3_slice_idx].ravel(),
               cmap='turbo',
               norm=normalize, s=4)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    ax.set_xticks([-2,0,2])
    ax.set_yticks([-2,0,2])
    ax.set_zticks([-2,0,2])
    ax.view_init(azim=-120, elev=30)
    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

# fig.savefig('Figures/高斯分布_along_x3.svg', format='svg')
plt.show()

### 沿x2
fig = plt.figure(figsize=(6, 36))

for fig_idx,x2_slice_idx in enumerate(np.arange(0,len(x2),5)):

    ax = fig.add_subplot(len(np.arange(0,len(x2),5)), 1, fig_idx + 1, projection='3d')

    ax.scatter(xxx1[:, x2_slice_idx, :].ravel(),
               xxx2[:, x2_slice_idx, :].ravel(),
               xxx3[:, x2_slice_idx, :].ravel(),
               c=pdf_zz[:, x2_slice_idx, :].ravel(),
               cmap='turbo',
               norm=normalize, s=4)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    ax.set_xticks([-2,0,2])
    ax.set_yticks([-2,0,2])
    ax.set_zticks([-2,0,2])
    ax.view_init(azim=-120, elev=30)
    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

# fig.savefig('Figures/高斯分布_along_x2.svg', format='svg')
plt.show()


### 沿x1
fig = plt.figure(figsize=(6, 36))

for fig_idx,x1_slice_idx in enumerate(np.arange(0,len(x1),5)):

    ax = fig.add_subplot(len(np.arange(0,len(x2),5)), 1, fig_idx + 1, projection='3d')

    ax.scatter(xxx1[x1_slice_idx, :, :].ravel(),
               xxx2[x1_slice_idx, :, :].ravel(),
               xxx3[x1_slice_idx, :, :].ravel(),
               c=pdf_zz[x1_slice_idx, :, :].ravel(),
               cmap='turbo',
               norm=normalize, s=4)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    ax.set_xticks([-2,0,2])
    ax.set_yticks([-2,0,2])
    ax.set_zticks([-2,0,2])
    ax.view_init(azim=-120, elev=30)
    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

# fig.savefig('Figures/高斯分布_along_x1.svg', format='svg')
plt.show()


#%% Dirichlet分布概率密度


import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 自定义可视化函数
def visualize_Dirichlet(alpha_array, num = 50):

    x1_ = np.linspace(0,1,num + 1)
    x2_ = np.linspace(0,1,num + 1)

    xx1_, xx2_ = np.meshgrid(x1_, x2_)

    xx3_ = 1.0 - xx1_ - xx2_
    xx3_ = np.where(xx3_ > 0.0005, xx3_, np.nan)

    rv = st.dirichlet(alpha_array)

    PDF_ff_ = rv.pdf(np.array(([xx1_.ravel(), xx2_.ravel(), xx3_.ravel()])))
    PDF_ff_ = np.reshape(PDF_ff_, xx1_.shape)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")

    # Creating plot
    PDF_ff_ = np.nan_to_num(PDF_ff_)
    ax.scatter3D(xx1_.ravel(),
                 xx2_.ravel(),
                 xx3_.ravel(),
                 c=PDF_ff_.ravel(),
                 alpha = 1,
                 marker='.',
                 cmap = 'RdYlBu_r')

    ax.set_proj_type('ortho')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.set_xticks(np.linspace(0,1,6))
    # ax.set_yticks(np.linspace(0,1,6))
    # ax.set_zticks(np.linspace(0,1,6))

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])

    ax.set_xlim(x1_.min(), x1_.max())
    ax.set_ylim(x2_.min(), x2_.max())
    ax.set_zlim3d([0,1])
    # ax.view_init(azim=20, elev=20)
    ax.view_init(azim=30, elev=30)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')

    ax.set_box_aspect(aspect = (1,1,1))

    ax.grid()
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()

alpha_array = [1, 2, 2]
visualize_Dirichlet(alpha_array)


alpha_array = [2, 1, 2]
visualize_Dirichlet(alpha_array)

alpha_array = [2, 2, 1]
visualize_Dirichlet(alpha_array)


alpha_array = [4, 2, 1]
visualize_Dirichlet(alpha_array)

alpha_array = [1, 1, 1]
visualize_Dirichlet(alpha_array)

alpha_array = [2, 2, 2]
visualize_Dirichlet(alpha_array)

alpha_array = [4, 4, 4]
visualize_Dirichlet(alpha_array)

#%% Dirichlet分布随机数


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import dirichlet

# 确定随机数种子，保证结果可复刻
np.random.seed(0)


def visualize_sample(alpha_array):

    samples = np.random.dirichlet(alpha_array, size=500)

    # 计算Dirichlet概率密度值
    pdf_values = dirichlet.pdf(samples.T, alpha_array)

    # 创建三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图，并用颜色映射表示概率密度值
    ax.scatter(samples[:, 0],
               samples[:, 1],
               samples[:, 2],
               s = 3.8,
               c=pdf_values,
               cmap='RdYlBu_r')

    ax.plot([0,1],[1,0],[0,0],c='k',ls = '--')
    ax.plot([0,1],[0,0],[1,0],c='k',ls = '--')
    ax.plot([0,0],[0,1],[1,0],c='k',ls = '--')

    ax.set_proj_type('ortho')
    ax.view_init(azim=30, elev=30)
    ax.set_box_aspect([1,1,1])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_xticklabels([])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zticks([0,1])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.grid(c = '0.88')

    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title

    # fig.savefig(title + '.svg', format='svg')
    plt.show()

alpha_array = [1, 2, 2]
visualize_sample(alpha_array)

alpha_array = [2, 1, 2]
visualize_sample(alpha_array)

alpha_array = [2, 2, 1]
visualize_sample(alpha_array)


alpha_array = [4, 4, 4]
visualize_sample(alpha_array)


alpha_array = [8, 8, 8]
visualize_sample(alpha_array)

#==========================================================================================================
##########################################  3D Line Plot, 三维线图 ######################################
#==========================================================================================================

#%%

## 三维线图
# 导入包
import numpy as np
import matplotlib.pyplot as plt


# 创建数据
# 弧度数组
theta = np.linspace(-24 * np.pi, 24 * np.pi, 1000)
z = np.linspace(-2, 2, 1000)
r = z**2 + 1
# 参数方程
x = r * np.sin(theta)
y = r * np.cos(theta)

# 可视化线图
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
# 绘制三维线图
ax.plot(x, y, z)

ax.set_proj_type('ortho')
ax.grid(False)
# 修改视角
# ax.view_init(elev=90, azim=-90)
# ax.view_init(elev=0, azim=-90)
# ax.view_init(elev=0, azim=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(False)
plt.show()


# 创建数据
z = np.linspace(0, 2, 1000)
r = z
x = r * np.sin(theta)
y = r * np.cos(theta)
# 可视化线图
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot(x, y, z)

ax.set_proj_type('ortho')

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(False)
plt.show()

#%% 一元高斯分布概率密度

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# 导入色谱

# 自定义一元高斯概率密度函数
def gaussian_1D(x_array, mu, sigma):

    z = (x_array - mu)/sigma

    factor = 1/sigma/np.sqrt(2*np.pi)

    PDF_array = factor * np.exp(-z**2/2)

    return PDF_array



# 随  𝜇 变化
# 创建数据
x_array = np.linspace(-8,8,121)
mu_array = np.arange(-4,4 + 1)
num_lines = len(mu_array)
# 概率密度曲线条数

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

colors = cm.rainbow(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号

for mu_idx, color_idx in zip(mu_array, colors):

    # 可以使用：

    # ax.plot(x_array, gaussian_1D(x_array, mu_idx, 1),
    #         zs = mu_idx,
    #         zdir = 'y',
    #         color = color_idx)

    # 也可以：
    ax.plot(x_array, # x 坐标
            x_array*0 + mu_idx, # y 坐标
            gaussian_1D(x_array, mu_idx, 1), # z 坐标
            color = color_idx)

ax.set(xlim=[x_array.min(), x_array.max()],
       ylim=[mu_array.min(), mu_array.max()])

ax.set_xticks([-8,0,8])
ax.set_yticks([-4,0,4])
ax.set_zticks([0, 0.5])
ax.view_init(azim=-145, elev=15)
ax.set_proj_type('ortho')
ax.set_xlabel('$x$')
ax.set_ylabel(r'$\mu$')
ax.set_zlabel('$f_X(x)$')
ax.set_box_aspect((1, 1, 1))
ax.grid(False)
plt.show()


# 随  𝜎 变化
# 产生数据
x_array = np.linspace(-8,8,121)
sigma_array = np.linspace(1, 5, 9)
num_lines = len(sigma_array)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# 概率密度曲线条数

colors = cm.rainbow(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号

for sigma_idx,color_idx in zip(sigma_array,colors):

    # 可以使用：
    # ax.plot(x_array, gaussian_1D(x_array, 0, sigma_idx),
    #         zs = sigma_idx, zdir = 'y',
    #         color = color_idx)

    # 也可以：
    ax.plot(x_array, # x 坐标
            x_array*0 + sigma_idx, # y 坐标
            gaussian_1D(x_array, 0, sigma_idx), # z 坐标
            color = color_idx)

ax.set(xlim=[x_array.min(), x_array.max()],
       ylim=[sigma_array.min(), sigma_array.max()])

ax.set_xticks([-8,0,8])
ax.set_yticks([sigma_array.min(),sigma_array.max()])
ax.set_zticks([0, 0.5])
ax.view_init(azim=-145, elev=15)
ax.set_proj_type('ortho')
ax.set_xlabel('$x$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel('$f_X(x)$')
ax.set_box_aspect((1, 1, 1))
ax.grid(False)
plt.show()

#%% 投影
# 导入包
import numpy as np
import matplotlib.pyplot as plt



# 产生网格数据
grid = np.linspace(-3,3)

xx1,xx2 = np.meshgrid(np.linspace(-3,3),np.linspace(-3,3))
ff = np.exp(- xx1**2 - xx2**2)
# 高斯函数


# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 绘制曲面
ax.plot_wireframe(xx1, xx2, ff,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

# 绘制两条曲线
ax.plot(grid, # y坐标
        np.sqrt(np.pi) * np.exp(-grid**2), # z坐标
        zs=3, zdir='x') # x坐标值固定为3
ax.plot(grid, # x坐标
        np.sqrt(np.pi) * np.exp(-grid**2), # z坐标
        zs=3, zdir='y') # y坐标值固定为3

ax.view_init(azim=-120, elev=30)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(0,2)
ax.set_xticks((-2, 0, 2))
ax.set_yticks((-2, 0, 2))
ax.set_zticks((0, 1, 2))
ax.grid(False)
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect((1, 1, 1))
plt.show()


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 绘制曲面
ax.plot_wireframe(xx1, xx2, ff,
                  color = [0.3,0.3,0.3],
                  linewidth = 0.25)

# 绘制两条曲线
ax.plot(grid, np.sqrt(np.pi) * np.exp(-grid**2), zs=-3, zdir='x')
ax.plot(grid, np.sqrt(np.pi) * np.exp(-grid**2), zs=-3, zdir='y')

ax.view_init(azim=-120, elev=30)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_zlim(0,2)
ax.set_xticks((-2, 0, 2))
ax.set_yticks((-2, 0, 2))
ax.set_zticks((0, 1, 2))
ax.grid(False)
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect((1, 1, 1))
# plt.show()


#%% 火柴梗图
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial



# 沿 z
p_array = [0.15, 0.35, 0.5]
num = 15

x1_array = np.arange(num + 1)
x2_array = np.arange(num + 1)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

xx3 = num - xx1 - xx2
xx3 = np.where(xx3 >= 0.0, xx3, np.nan)

PMF_ff = multinomial.pmf(x=np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T,
                         n=num, p=p_array)

PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)

PMF_ff = np.reshape(PMF_ff, xx1.shape)


fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection="3d")

ax.stem(xx1.ravel(), xx2.ravel(), PMF_ff.ravel(), basefmt=" ")

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xticks([0,num])
ax.set_yticks([0,num])
ax.set_zticks([0,0.06])

ax.set_xlim(0, num)
ax.set_ylim(0, num)
ax.set_zlim(0, 0.06)
ax.view_init(azim=30, elev=30)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel(r'$f_{X_1,X_2}(x_1,x_2)$')
ax.set_box_aspect(aspect = (1,1,1))

ax.grid()
# fig.colorbar(scatter_plot, ax = ax, shrink = 0.5, aspect = 10)
title = '_'.join(str(round(p_i,2)) for p_i in p_array)
title = 'p_array_' + title
plt.show()


#%% 单位正方体的 12 条边
from matplotlib import pyplot as plt
import numpy as np

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")

# 八个点的坐标
A = [1, 1, 1]

B = [1, 0, 1]
C = [1, 1, 0]
D = [0, 1, 1]

E = [1, 0, 0]
F = [0, 1, 0]
G = [0, 0, 1]

O = [0, 0, 0]
Labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'O']
Data = np.row_stack((A,B,C,D,E,F,G,O))


# 可视化散点
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Data[:,0],Data[:,1],Data[:,2],
          alpha = 1,
          s = 40)

for label_idx, [x, y, z] in zip(Labels, Data):
    label = label_idx + ': (%d, %d, %d)' % (x, y, z)
    ax.text(x, y, z, label)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xticks((0,1))
ax.set_yticks((0,1))
ax.set_zticks((0,1))
ax.view_init(azim=30, elev=30)
ax.set_box_aspect((1, 1, 1))
ax.set_proj_type('ortho')
# fig.savefig('Figures/可视化散点.svg', format='svg')
plt.show()

# 12条参考线
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Data[:,0],Data[:,1],Data[:,2],
          alpha = 1,
          s = 40)

for label_idx, [x, y, z] in zip(Labels, Data):
    label = label_idx + ': (%d, %d, %d)' % (x, y, z)
    ax.text(x, y, z, label)

# 绘制 AB、AC、AD
ax.plot([A[0], B[0]],
        [A[1], B[1]],
        [A[2], B[2]])

ax.plot([A[0], C[0]],
        [A[1], C[1]],
        [A[2], C[2]])

ax.plot([A[0], D[0]],
        [A[1], D[1]],
        [A[2], D[2]])

# 绘制 OE、OF、OG

ax.plot([O[0], E[0]],
        [O[1], E[1]],
        [O[2], E[2]])

ax.plot([O[0], F[0]],
        [O[1], F[1]],
        [O[2], F[2]])

ax.plot([O[0], G[0]],
        [O[1], G[1]],
        [O[2], G[2]])

# 绘制 OE、OF、OG

ax.plot([O[0], E[0]],
        [O[1], E[1]],
        [O[2], E[2]])

ax.plot([O[0], F[0]],
        [O[1], F[1]],
        [O[2], F[2]])

ax.plot([O[0], G[0]],
        [O[1], G[1]],
        [O[2], G[2]])

# 绘制 BE、CE

ax.plot([B[0], E[0]],
        [B[1], E[1]],
        [B[2], E[2]])

ax.plot([C[0], E[0]],
        [C[1], E[1]],
        [C[2], E[2]])

# 绘制 CF、DF
ax.plot([C[0], F[0]],
        [C[1], F[1]],
        [C[2], F[2]])

ax.plot([D[0], F[0]],
        [D[1], F[1]],
        [D[2], F[2]])

# 绘制 GB、GD
ax.plot([B[0], G[0]],
        [B[1], G[1]],
        [B[2], G[2]])

ax.plot([D[0], G[0]],
        [D[1], G[1]],
        [D[2], G[2]])

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xticks((0,1))
ax.set_yticks((0,1))
ax.set_zticks((0,1))
ax.view_init(azim=30, elev=30)
ax.set_box_aspect((1, 1, 1))
ax.set_proj_type('ortho')
# fig.savefig('Figures/12条参考线.svg', format='svg')
plt.show()




#%% 可视化偏导数
import numpy as np
from sympy import lambdify, diff, exp, latex, simplify
from sympy.abc import x, y
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


# 创建数据
num = 301 # number of mesh grids
x_array = np.linspace(-3, 3, num)
y_array = np.linspace(-3, 3, num)

xx, yy = np.meshgrid(x_array,y_array)
# 二元函数
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x, y], f_xy)
f_xy_zz = f_xy_fcn(xx, yy)

################## 沿x方向切线
x_s = np.linspace(-2.4,2.4,9)
y_s = np.linspace(-2.4,2.4,9)
xx_s, yy_s = np.meshgrid(x_s, y_s)

# 符号偏导
df_dx = f_xy.diff(x)
df_dx_fcn = lambdify([x,y], df_dx)
# 定义函数绘制沿x方向切线
def plot_d_x_tangent(x_t, y_t, df_dx_fcn, f_xy_fcn, color, ax):
    # 计算切线斜率 (偏导数)
    k = df_dx_fcn(x_t, y_t)
    # 小彩灯z轴位置，切点坐标 (x_t,y_t,z_t)
    z_t = f_xy_fcn(x_t, y_t)
    # 切线x轴数组
    x_array = np.linspace(x_t-0.6, x_t + 0.6, 10)
    # 切线函数
    z_array = k*(x_array - x_t) + z_t
    # 绘制切线
    ax.plot(x_array, x_array*0 + y_t, z_array, color = color, lw = 1)
    # 绘制小彩灯 (切点)
    ax.plot(x_t, y_t, z_t, color = color, marker = '.', markersize = 10)
    return


fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10,10))

ax.plot_wireframe(xx, yy, f_xy_zz, color = [0.5,0.5,0.5],  rstride=15,
                  cstride=0, ## 沿x方向
                  linewidth = 2)

colors = plt.cm.rainbow(np.linspace(0,1,len(xx_s.ravel())))

for i in np.linspace(0, len(xx_s.ravel())-1, len(xx_s.ravel())):
    i = int(i)
    x_t = xx_s.ravel()[i]
    y_t = yy_s.ravel()[i]
    color = colors[i,:]
    plot_d_x_tangent(x_t, y_t, df_dx_fcn, f_xy_fcn, color, ax)
ax.set_proj_type('ortho')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
# ax.view_init(azim=-90, elev=0)​
plt.tight_layout()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel(r'$f(x,y)$')

# plt.savefig('1.svg')
plt.show()

############### 沿y方向切线
df_dy = f_xy.diff(y)
df_dy_fcn = lambdify([x,y],df_dy)
# 定义函数绘制沿y方向切线
def plot_d_y_tangent(x_t, y_t, df_dy_fcn, f_xy_fcn, color, ax):
    k = df_dy_fcn(x_t, y_t)
    z_t = f_xy_fcn(x_t, y_t)

    y_array = np.linspace(y_t-0.6,y_t+0.6, 10)
    z_array = k*(y_array - y_t) + z_t

    ax.plot(y_array*0 + x_t,y_array, z_array, color = color, lw = 0.2)
    # partial x1, tangent line

    ax.plot(x_t, y_t, z_t, color = color,
              marker = '.', markersize = 5)
    # tangent point
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10,10))

ax.plot_wireframe(xx,yy, f_xy_zz,
                  color = [0.5,0.5,0.5],
                  rstride=0, cstride=15,
                  linewidth = 0.25)

colors = plt.cm.rainbow(np.linspace(0,1,len(yy_s.ravel())))
for i in np.linspace(0,len(yy_s.ravel())-1,len(yy_s.ravel())):
    i = int(i)
    x_t = xx_s.ravel()[i]
    y_t = yy_s.ravel()[i]

    color = colors[i,:]

    plot_d_y_tangent(x_t, y_t, df_dy_fcn, f_xy_fcn, color, ax)

ax.set_proj_type('ortho')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
# ax.view_init(azim=0, elev=0)​
plt.tight_layout()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()

## 3
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10,10))

ax.plot_wireframe(xx,yy, f_xy_zz, color = [0.5,0.5,0.5], rstride=0, cstride=15, linewidth = 0.25)
ax.set_proj_type('ortho')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.view_init(azim=-135, elev=30)
# ax.view_init(azim=-90, elev=0)

plt.tight_layout()
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()

#==========================================================================================================
##########################################  3D Mesh Surface, 网格曲面 ######################################
#==========================================================================================================

#%% 绘制网格曲面
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
from matplotlib import cm
# 导入色谱模块

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")

# 1. 定义函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2)\
    - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2)\
    - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数


# 2. 网格函数
def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

# 3. 展示网格面，网格粗糙
xx, yy = mesh(num = 11)
zz = xx * 0

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, zz,
                  color = [0.5,0.5,0.5],
                  linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/展示网格面，网格粗糙.svg', format='svg')
plt.show()


# 4. 绘制函数网格曲面，网格粗糙
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.5,0.5,0.5],
                  linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/绘制函数网格曲面，网格粗糙.svg', format='svg')
plt.show()

# 5. 展示网格面，网格过密
xx, yy = mesh(num = 101)
zz = xx * 0

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, zz,
                  color = [0.8,0.8,0.8],
                  rstride=1, cstride=1,
                  linewidth = 0.25)

# ax.plot_wireframe(xx,yy, zz,
#                   color = 'k',
#                   rstride=5, cstride=5,
#                   linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('')
ax.set_zticks([])
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/展示网格面，网格过密.svg', format='svg')
plt.show()

# 6. 绘制函数网格曲面，网格过密
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.5,0.5,0.5],
                  rstride=1, cstride=1,
                  linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/绘制函数网格曲面，网格过密.svg', format='svg')
plt.show()


# 7. 增大步幅
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = '#0070C0',
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/增大步幅.svg', format='svg')
plt.show()



# 8. 仅绘制沿x方向曲线
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = '#0070C0',
                  rstride=5, cstride=0,
                  linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/仅绘制沿x方向曲线.svg', format='svg')
plt.show()




# 10. 特别强调特定曲线
# 请大家试着绘制一条 x = 1曲线

x_array = np.linspace(-3,3,100)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = '0.5',
                  rstride=5, cstride=5,
                  linewidth = 0.25)

y_level = 0 + np.zeros_like(x_array)
ax.plot(x_array, y_level, f_xy_fcn(x_array, y_level), c = 'r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/特定曲线_y = 0.svg', format='svg')
plt.show()



x_array = np.linspace(-2,3,100)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = '0.5',
                  rstride=5, cstride=5,
                  linewidth = 0.25)

y_array = 1 - x_array
# x + y = 1
ax.plot(x_array, y_array, f_xy_fcn(x_array, y_array), c = 'r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/特定曲线_x + y = 1.svg', format='svg')
plt.show()


# 11. 绘制网格化散点
xx_scatter, yy_scatter = mesh(num = 21)

ff_scatter = f_xy_fcn(xx_scatter,yy_scatter)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.6,0.6,0.6],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.scatter(xx_scatter.ravel(),yy_scatter.ravel(),ff_scatter,c = ff_scatter,s = 10,cmap = 'RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/增加网格散点.svg', format='svg')
plt.show()



# 12. 绘制不规则散点
xx_scatter, yy_scatter = mesh(num = 21)

ff_scatter = f_xy_fcn(xx_scatter,yy_scatter)


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.6,0.6,0.6],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

x_rand = np.random.rand(500) * 6 - 3
y_rand = np.random.rand(500) * 6 - 3
f_rand = f_xy_fcn(x_rand,y_rand)

ax.scatter(x_rand,y_rand,f_rand,c = f_rand,s = 10,cmap = 'RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/不规则散点.svg', format='svg')
plt.show()





#%% 将第四维数据映射到三维网格曲面
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy.abc import x, y
from sympy import lambdify, diff, exp, latex

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")


def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy


xx, yy = mesh(num = 201)
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
f_xy_zz = f_xy_fcn(xx, yy)



#########################  1. 一般曲面 f(x,y)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

surf = ax.plot_surface(xx,yy,f_xy_zz,
                       cmap='turbo',
                       linewidth=1, # 线宽
                       shade=False) # 删除阴影
ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zticks([])
fig.colorbar(surf, shrink=0.8, aspect=20)
ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-135, elev=60)
plt.tight_layout()
ax.grid(False)
# fig.savefig('Figures/一般曲面.svg', format='svg')
plt.show()

######################### 2
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
colors = cm.turbo(norm_plt(f_xy_zz))

surf = ax.plot_surface(xx,yy,f_xy_zz,
                       facecolors=colors,
                       linewidth=1, # 线宽
                       shade=False) # 删除阴影
surf.set_facecolor((0,0,0,0))
ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zticks([])
# m = cm.ScalarMappable(cmap=cm.turbo)
# m.set_array(f_xy_zz)
# plt.colorbar(m)
ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-135, elev=60)
plt.tight_layout()
ax.grid(False)
# fig.savefig('Figures/一般曲面.svg', format='svg')
plt.show()

#########################  2. 将第四维数据 V(x,y) 投影到三维曲面 f(x,y)
V = np.sin(xx) * np.sin(yy)
# V(x,y)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

norm_plt = plt.Normalize(V.min(), V.max())
colors = cm.turbo(norm_plt(V))

surf = ax.plot_surface(xx, yy, f_xy_zz,
                       facecolors=colors,
                       linewidth=1, shade=False)

surf.set_facecolor((0,0,0,0))

ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zticks([])

ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-135, elev=60)
plt.tight_layout()
ax.grid(False)
# fig.savefig('Figures/将第四维数据投影到三维曲面.svg', format='svg')
plt.show()


########################  3. 调换第三 f(x,y)、四维 V(x,y)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
colors = cm.turbo(norm_plt(f_xy_zz))

surf = ax.plot_surface(xx,yy, V,
                       facecolors=colors,
                       linewidth=1, shade=False)

surf.set_facecolor((0,0,0,0))

ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zticks([])

ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-135, elev=60)
plt.tight_layout()
ax.grid(False)
# fig.savefig('Figures/调换第三、四维.svg', format='svg')
plt.show()



#%% 绘制填充平面,  平行于不同平面的剖面

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 绘制xy平行面，网格
s_fine = np.linspace(0, 10, 11)
xx, yy = np.meshgrid(s_fine,s_fine)
# 生成网格数据

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 导入3D轴

zz = np.zeros_like(xx) + 1
# numpy.zeros_like(xx) 构造一个形状和 xx 一致的全 0 矩阵

ax.plot_surface(xx, yy, zz, color = 'b', alpha = 0.1)
# 绘制网格曲面，透明度为 0.1

ax.plot_wireframe(xx, yy, zz)
ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xy平行面.svg', format='svg')
plt.show()


# 2. 绘制xy平行面，无网格
s_coarse = np.linspace(0, 10, 2)
xx, yy = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

zz = np.zeros_like(xx) + 1
ax.plot_surface(xx, yy, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx, yy, np.zeros_like(xx) + 1)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xy平行面，无网格.svg', format='svg')
plt.show()

# 3. 绘制xy平行面，若干平行平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for z_idx in np.arange(10 + 1):
    zz = np.zeros_like(xx) + z_idx
    ax.plot_surface(xx, yy, zz, color = 'b', alpha = 0.1)
    ax.plot_wireframe(xx, yy, zz, linewidth = 0.25)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xy平行面，若干平行平面.svg', format='svg')
plt.show()



# 4. 绘制xz平行面，网格
s_fine = np.linspace(0, 10, 11)
xx, zz = np.meshgrid(s_fine, s_fine)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, xx*0 + 1, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx, xx*0 + 1, zz)
ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xz平行面，网格.svg', format='svg')
plt.show()


# 5. 绘制xz平行面，无网格
xx, zz = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xx, xx*0 + 1, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx, xx*0 + 1, zz, color = 'b')

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xz平行面，无网格.svg', format='svg')
plt.show()

# 6. 绘制xz平行面，若干平行平面
xx, zz = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for y_idx in np.arange(10):
    ax.plot_surface(xx, xx*0 + y_idx, zz, color = 'b', alpha = 0.1)
    ax.plot_wireframe(xx, xx*0 + y_idx, zz, color = 'b', linewidth = 0.25)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/xz平行面，若干平行平面.svg', format='svg')
plt.show()



# 7. 绘制yz平行面，网格
yy, zz = np.meshgrid(s_fine,s_fine)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(yy*0 + 1, yy, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(yy*0 + 1, yy, zz)
ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/yz平行面，网格.svg', format='svg')
plt.show()



# 8. 绘制yz平行面，无网格

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(yy*0 + 1, yy, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(yy*0 + 1, yy, zz)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/yz平行面，无网格.svg', format='svg')
plt.show()

# 9. 绘制yz平行面，若干平行平面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x_idx in np.arange(10):
    ax.plot_surface(yy*0 + x_idx, yy, zz, color = 'b', alpha = 0.1)
    ax.plot_wireframe(yy*0 + x_idx, yy, zz, color = 'b', linewidth = 0.25)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
# fig.savefig('Figures/yz平行面，若干平行平面.svg', format='svg')
plt.show()

# 10. 垂直于 xy 平面
s_coarse = np.linspace(0, 10, 2)
yy, zz = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(10 - yy, yy, zz, color = 'b', alpha = 0.1)
ax.plot_wireframe(10 - yy, yy, zz)

ax.set_proj_type('ortho')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.grid(False)
plt.show()


#%% 可视化剖面线
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
import os

from matplotlib import cm
# 导入色谱模块

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2)\
    - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2)\
    - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数

def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

# 2. 剖面线，平行于xy
xx, yy = mesh(num = 121)
ff = f_xy_fcn(xx,yy)
z_level = 2
# 指定 z 轴高度

xx_, yy_ = np.meshgrid(np.linspace(-3, 3, 2),np.linspace(-3, 3, 2))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 绘制剖面
zz_ = np.zeros_like(xx_) + z_level
ax.plot_surface(xx_, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, yy_, zz_, color = 'b',
                  lw = 0.2)

# 绘制网格曲面
ax.plot_wireframe(xx,yy, ff,
                  color = [0.6, 0.6, 0.6],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

# 绘制指定一条剖面线
ax.contour(xx, yy, ff,
           levels = [z_level],
           colors = 'r',
           linewidths = 1)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/剖面线，平行于xy.svg', format='svg')
plt.show()


# 3. 剖面线，平行于 xz
y_level = 0
xx_, zz_ = np.meshgrid(np.linspace(-3, 3, 2),np.linspace(-8, 8, 2))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 绘制剖面
ax.plot_surface(xx_, xx_*0 + y_level, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, xx_*0 + y_level, zz_, color = 'b',
                  lw = 0.2)

# 绘制曲面网格
ax.plot_wireframe(xx,yy, ff,
                  color = [0.6, 0.6, 0.6],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

# 绘制指定一条剖面线
x_array = np.linspace(-3,3,101)
y_array = x_array*0 + y_level
ax.plot(x_array, y_array, f_xy_fcn(x_array,y_array),
        color = 'r', lw = 1)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/剖面线，平行于xz.svg', format='svg')
plt.show()


# 4. 剖面线，平行于 yz
x_level = 0
yy_, zz_ = np.meshgrid(np.linspace(-3, 3, 2),np.linspace(-8, 8, 2))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_surface(yy_*0 + x_level, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(yy_*0 + x_level, yy_, zz_, color = 'b',
                  lw = 0.2)

ax.plot_wireframe(xx,yy, ff,
                  color = [0.6, 0.6, 0.6],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

y_array = np.linspace(-3,3,101)

# 绘制指定一条剖面线
x_array = y_array*0 + x_level
ax.plot(x_array, y_array, f_xy_fcn(x_array,y_array),
        color = 'r', lw = 1)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8, 8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/剖面线，yz.svg', format='svg')
plt.show()


#%%  三维线图的平面填充, 填充曲线下方剖面
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
# 导入符号变量
import os

from matplotlib import cm
# 导入色谱模块

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy


# 1. 二元高斯分布
xx1, xx2 = mesh(num = 101)
points = np.dstack((xx1, xx2))
# 将 xx1和xx2 在深度方向拼接，得到代表 (x1, x2) 坐标的数组

bivariate_normal = multivariate_normal([0, 0],
                                      [[1, -0.6],
                                       [-0.6, 1]])

PDF_ff = bivariate_normal.pdf(points)
# 二元高斯分布概率密度函数值

# 2. 指定 x1 具体值
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1, xx2, PDF_ff,
                  color = [0.5,0.5,0.5],
                  rstride=0, cstride=2,
                  linewidth = 0.25)

x1 = np.linspace(-3,3,101)
x2 = np.linspace(-3,3,101)
x1_loc_array = np.arange(0,len(x1),10)
facecolors = cm.rainbow(np.linspace(0, 1, len(x1_loc_array)))

for idx in range(len(x1_loc_array)):
    x_loc = x1_loc_array[idx]
    x_idx = x1[x_loc]
    x_i_array = x2*0 + x_idx
    z_array = PDF_ff[:,x_loc]

    ax.plot(x_i_array, x2, z_array, color=facecolors[idx,:],
            linewidth = 1.5)

    ax.add_collection3d(plt.fill_between(x2, 0*z_array, z_array,
                                         color=facecolors[idx,:],
                                         alpha=0.2),
                        # 给定填充对象
                        zs=x_idx, # 指定位置
                        zdir='x') # 指定方向


ax.set_proj_type('ortho')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Joint PDF, $f_{X_1,X_2}(x_1,x_2)$')
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_zticks([0, 0.05, 0.1, 0.15, 0.2])
ax.set_zlim3d([0,0.2])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/指定 x1 具体值.svg', format='svg')
plt.show()


# 3. 指定 x2 具体值
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx1, xx2, PDF_ff,
                  color = [0.5,0.5,0.5],
                  rstride=2, cstride=0,
                  linewidth = 0.25)

x2_loc_array = np.arange(0,len(x1),10)
facecolors = cm.rainbow(np.linspace(0, 1, len(x2_loc_array)))

for idx in range(len(x2_loc_array)):
    x_loc = x2_loc_array[idx]
    x_idx = x2[x_loc]
    x_i_array = x1*0 + x_idx
    z_array = PDF_ff[x_loc,:]

    ax.plot(x1, x_i_array, z_array, color=facecolors[idx,:],
            linewidth = 1.5)

    ax.add_collection3d(plt.fill_between(x1, 0*z_array, z_array,
                                         color=facecolors[idx,:],
                                         alpha=0.2),
                        zs=x_idx, zdir='y')

ax.set_proj_type('ortho')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Joint PDF, $f_{X_1,X_2}(x_1,x_2)$')
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_zticks([0, 0.05, 0.1, 0.15, 0.2])
ax.set_zlim3d([0,0.2])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/指定 x2 具体值.svg', format='svg')
plt.show()




#%% 圆形薄膜振荡模式
import numpy as np
from scipy.special import jn, jn_zeros
import matplotlib.pyplot as plt
mmax = 5

def displacement(n, m, r, theta, mmax = 5):

    """
    鼓膜在极坐标系下的位移，
    其中
    n表示模式的正整数，
    m表示Bessel函数的阶数，
    r表示径向坐标，
    theta表示角坐标
    mmax 表示Bessel函数的最大阶数
    """

    # 计算Bessel函数Jn的零点，并选择其中第m个零点，将其赋值给变量k
    k = jn_zeros(n, mmax+1)[m]
    #  返回计算得到的鼓膜位移，该位移是正弦函数和Bessel函数的乘积
    return np.sin(n*theta) * jn(n, r*k)

# 极坐标
r = np.linspace(0, 1, 1001)
theta = np.linspace(0, 2 * np.pi, 1001)

# 极坐标转化为直角坐标，也可以用meshgrid()
xx = np.array([rr*np.cos(theta) for rr in r])
yy = np.array([rr*np.sin(theta) for rr in r])



def visualize(n,m,title):

    zz = np.array([displacement(n, m, rr, theta) for rr in r])

    fig = plt.figure(figsize = (8,4))
    ax = fig.add_subplot(121, projection='3d')

    surf = ax.plot_wireframe(xx,yy,zz,
                             cstride = 50,
                             rstride = 50,
                             colors = '0.8',
                             linewidth=0.25)
    ax.contour(xx,yy,zz,
               cmap='RdYlBu_r',
               levels = 15,
               linewidths=1)

    ax.set_proj_type('ortho')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f(x,y)$')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zz.min()*5,zz.max()*5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(azim=15, elev=45)
    # ax.view_init(azim=-135, elev=60)
    plt.tight_layout()
    ax.grid(False)
    ax.axis('off')

    ax = fig.add_subplot(122)

    ax.contourf(xx,yy,zz,
               cmap='RdYlBu_r',
               levels = 15)
    ax.contour(xx,yy,zz,
               colors = 'w',
               levels = 15,
               linewidths=0.25)

    ax.plot(np.cos(theta),np.sin(theta),'k')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')
    # fig.savefig(title + '.svg')
    plt.show()

visualize(4,0,'4,0')


#==========================================================================================================
##########################################  3D Contours, 三维等高线 ######################################
#==========================================================================================================


































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































