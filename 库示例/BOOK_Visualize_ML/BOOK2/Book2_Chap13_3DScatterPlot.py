



#==========================================================================================================
##########################################  3D Scatter Plot, 三维散点 ######################################
#==========================================================================================================

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

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
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
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
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
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
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
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
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
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
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()

## 6: 利用颜色展示分类标签
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter_h = ax.scatter(x1, x2, x3, c = labels, cmap=rainbow)

classes = ['Setosa', 'Versicolor', 'Virginica']

plt.legend(handles=scatter_h.legend_elements()[0], labels=classes)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.show()


## 7: 颜色分类 + 散点大小
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, x3, s = x4*20, c = labels, cmap=rainbow)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.show()

## 8: 利用色谱展示第四维特征
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter_plot = ax.scatter(x1, x2, x3, c = x4, cmap=rainbow)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
fig.colorbar(scatter_plot, ax = ax, shrink = 0.5, aspect = 10)
ax.set_box_aspect([1,1,1])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

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
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.show()


#%% 用三维散点可视化多项分布
# 导入包
from scipy.stats import multinomial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    ax = plt.axes(projection = "3d")
    scatter_plot = ax.scatter3D(xx1.ravel(), xx2.ravel(), xx3.ravel(), s = 50, marker = '.', alpha = 1, c = PMF_ff.ravel(), cmap = 'RdYlBu_r')

    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

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
    plt.show()

p_array = [1/3, 1/3, 1/3]
visualize_multinomial(p_array)

p_array = [0.2, 0.2, 0.6]
visualize_multinomial(p_array)

p_array = [0.2, 0.6, 0.2]
visualize_multinomial(p_array)
plt.close('all')

#%%  用三维散点切片可视化高斯分布
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
x1 = np.linspace(-2, 2, 31)
x2 = np.linspace(-2, 2, 31)
x3 = np.linspace(-2, 2, 31)
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
               c = pdf_zz[:, :, x3_slice_idx].ravel(),
               cmap = 'turbo',
               norm = normalize, s = 4)

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
plt.show()

### 沿x1
fig = plt.figure(figsize=(6, 36))
for fig_idx,x1_slice_idx in enumerate(np.arange(0,len(x1),5)):
    ax = fig.add_subplot(len(np.arange(0,len(x2),5)), 1, fig_idx + 1, projection='3d')
    ax.scatter(xxx1[x1_slice_idx, :, :].ravel(),
               xxx2[x1_slice_idx, :, :].ravel(),
               xxx3[x1_slice_idx, :, :].ravel(),
               c = pdf_zz[x1_slice_idx, :, :].ravel(),
               cmap  ='turbo',
               norm = normalize, s = 4)

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
plt.show()

#%% Dirichlet分布概率密度
import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
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
    # fig.savefig('Figures/' + title + '.svg', format='svg')
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
    ax.view_init(azim = 30, elev = 30)
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticklabels([])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.grid(c = '0.88')

    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    ax.set_title(title)
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



















