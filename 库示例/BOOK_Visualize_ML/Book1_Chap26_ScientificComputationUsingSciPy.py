




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 计算、可视化成对距离
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib as mpl
import seaborn as sns
import string
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 产生随机数
num = 26
np.random.seed(0)
data = np.random.randint(10 + 1, size=(num, 2))
labels = list(string.ascii_uppercase)

cmap = mpl.cm.get_cmap('RdYlBu_r')

fig, ax = plt.subplots(figsize = (10, 10))
# 绘制成对线段
for i, d in enumerate(itertools.combinations(data, 2)):
    d_idx = euclidean(d[0],d[1])
    plt.plot([d[0][0],d[1][0]], [d[0][1],d[1][1]], color = cmap(d_idx/np.sqrt(2)/10), lw = 1)
ax.scatter(data[:,0],data[:,1], marker = 'x',color = 'k',s = 50,zorder=100)
# 添加标签
for i, txt in enumerate(labels):
    ax.annotate(txt,(data[i,0] + 0.2, data[i,1] + 0.2))

ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_xticks(np.arange(11))
ax.set_yticks(np.arange(11))
plt.xlabel('x'); plt.ylabel('y')
ax.grid(ls='--',lw=0.25,color=[0.5,0.5,0.5])
ax.set_aspect('equal', adjustable='box')
# fig.savefig('Figures/成对距离连线.svg', format='svg')



# 计算成对距离矩阵
pairwise_distances = distance_matrix(data, data) #  (26, 26)
fig, ax = plt.subplots()
sns.heatmap(pairwise_distances, cmap = 'RdYlBu_r', square = True, xticklabels = labels,yticklabels = labels, ax = ax)
# fig.savefig('Figures/成对距离矩阵热图.svg', format='svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化成对距离矩阵下三角部分 (不含主对角线元素)
# 可视化成对距离矩阵下三角部分 (不含主对角线元素)
# 产生随机数
num = 26
np.random.seed(0)
data = np.random.randint(10 + 1, size=(num, 2))
labels = list(string.ascii_uppercase)

# 计算成对距离矩阵
pairwise_ds = distance_matrix(data, data)
# 产生蒙皮/面具
# numpy.triu() 函数的"triu"代表"triangle upper"，它是"numpy"库中的函数，用于获取矩阵的上三角部分 (包括对角线)，而将下三角部分设置为 0。
mask = np.triu(np.ones_like(pairwise_ds, dtype=bool))

fig, ax = plt.subplots()
sns.heatmap(pairwise_ds, mask = mask, cmap = 'RdYlBu_r', square = True, xticklabels = labels, yticklabels = labels, ax = ax)
# fig.savefig('下三角.svg', format='svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 比较六种插值方法
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 生成随数据
np.random.seed(8)
x = np.linspace(0, 10, 10)
y = np.random.rand(10) * 10
x_fine = np.linspace(0, 10, 1001)

# 创建一个图形对象，包含六个子图
fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex = 'col', sharey = 'row')
axes = axes.flatten()

# 六种插值方法
methods = ['linear','quadratic','cubic', 'previous','next','nearest']

for i, method in enumerate(methods):
    # 创建 interp1d 对象
    f = interp1d(x, y, kind=method)

    # 生成插值后的新数据点
    y_fine = f(x_fine)

    # 绘制子图
    axes[i].plot(x, y, 'o', label='Data', markeredgewidth=1.5, markeredgecolor = 'w', zorder = 100)
    axes[i].plot(x_fine,y_fine,label='Interpolated')
    axes[i].set_title(f'Method: {method}')
    axes[i].legend()
    axes[i].set_xlim(0, 10)
    axes[i].set_ylim(0, 10)
    axes[i].set_aspect('equal', adjustable='box')
plt.tight_layout()
# fig.savefig('不同插值方法.svg', format='svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化一元高斯分布概率密度函数

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm


x_array = np.linspace(-6, 6, 200)
mu_array = np.linspace(-4, 4, 9)
# 设定均值一系列取值

colors = cm.RdYlBu(np.linspace(0,1,len(mu_array)))

# 均值对一元高斯分布PDF影响
fig, ax = plt.subplots(figsize = (5,4))
for idx, mu_idx in enumerate(mu_array):
    pdf_idx = norm.pdf(x_array,scale = 1,loc = mu_idx)
    legend_idx = '$\mu$ = ' + str(mu_idx)
    plt.plot(x_array, pdf_idx,  color=colors[idx], label = legend_idx)

plt.legend(ncol=3)
ax.set_xlim(x_array.min(),x_array.max())
ax.set_ylim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('PDF, $f_X(x)$')

# 设定标准差一系列取值
sigma_array = np.linspace(0.5,5,10)
colors = cm.RdYlBu(np.linspace(0,1,len(sigma_array)))

# 标准差对一元高斯分布PDF影响
fig, ax = plt.subplots(figsize = (5,4))
for idx, sigma_idx in enumerate(sigma_array):
    pdf_idx = norm.pdf(x_array, scale = sigma_idx)
    legend_idx = '$\sigma$ = ' + str(sigma_idx)
    plt.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)

plt.legend()
ax.set_xlim(x_array.min(),x_array.max())
ax.set_ylim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('PDF, $f_X(x)$')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化二元高斯分布PDF
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

rho_array = [-0.9, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 0.9]
sigma_X = 1; sigma_Y = 1 # 标准差
mu_X = 0;    mu_Y = 0    # 期望
width = 4
X = np.linspace(-width,width,321)
Y = np.linspace(-width,width,321)
XX, YY = np.meshgrid(X, Y)
XXYY = np.dstack((XX, YY))

# 曲面
fig = plt.figure(figsize = (8,8))
for idx, rho_idx in enumerate(rho_array):
    # 质心
    mu    = [mu_X, mu_Y]
    # 协方差
    Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho_idx],
            [sigma_X*sigma_Y*rho_idx, sigma_Y**2]]
    # 二元高斯分布
    bi_norm = multivariate_normal(mu, Sigma)

    # 利用 bi_norm 的 pdf()方法生成概率密度函数值 fX,Y(x,y)。这个方法的输入为生成的三维数组xxyy，代表一组网格坐标点。
    f_X_Y_joint = bi_norm.pdf(XXYY) # (321, 321)

    ax = fig.add_subplot(3,3, idx+1,projection='3d')
    ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=10, cstride=10, color = [0.3,0.3,0.3], linewidth = 0.25)

    ax.contour(XX,YY, f_X_Y_joint,15, cmap = 'RdYlBu_r')
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_zlabel('$f_{X,Y}(x,y)$')
    ax.view_init(azim=-120, elev=30)
    ax.set_proj_type('ortho')

    ax.set_xlim(-width, width); ax.set_ylim(-width, width)
    ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
    # ax.axis('off')

plt.tight_layout()
# fig.savefig('二元高斯分布，曲面.svg', format='svg')
plt.show()

# 平面填充等高线
fig = plt.figure(figsize = (8,8))
for idx, rho_idx in enumerate(rho_array):
    mu = [mu_X, mu_Y]
    Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho_idx], [sigma_X*sigma_Y*rho_idx, sigma_Y**2]]
    bi_norm = multivariate_normal(mu, Sigma)
    f_X_Y_joint = bi_norm.pdf(XXYY)

    ax = fig.add_subplot(3,3,idx+1)
    ax.contourf(XX, YY, f_X_Y_joint,
    levels = 12, cmap='RdYlBu_r')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.axis('off')

plt.tight_layout()
fig.savefig('二元高斯分布，等高线.svg', format='svg')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















































































































































































































































































































































































