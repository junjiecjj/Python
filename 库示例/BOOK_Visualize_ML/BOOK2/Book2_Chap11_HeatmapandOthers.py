

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 热图

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2

iris = load_iris()
# 从sklearn导入鸢尾花数据

X = iris.data
y = iris.target
# X
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

# 可视化鸢尾花四个量化特征数据
fig, ax = plt.subplots(figsize = (3,4))
ax = sns.heatmap(X_df, # 数据
                 cmap='RdYlBu_r', # 色谱
                 xticklabels=list(X_df.columns), # 横轴标签
                 yticklabels=False, # 关闭纵轴标签
                 cbar_kws={"orientation": "vertical"}, # 色谱条行为
                 vmin=0, vmax=8) # 色谱最小、最大值
plt.title('X')

# 聚类热图
ax = sns.clustermap(X_df, cmap='RdYlBu_r', xticklabels=list(X_df.columns), yticklabels=False, figsize = (6,12), vmin=0, vmax=8)

# 计算格拉姆矩阵
G = X.T @ X
print(G.max())
print(G.min())
fig,axs = plt.subplots(1,5,figsize = (8,3), gridspec_kw={'width_ratios': [3, 0.5, 3, 0.5, 3]})

plt.sca(axs[0])
ax = sns.heatmap(G, cmap = 'RdYlBu_r', vmax = 5000, vmin = 0, annot = True, fmt=".0f", cbar_kws = {'orientation':'horizontal'}, yticklabels=False, square = 'equal')
plt.title('$G$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(X.T, cmap = 'RdYlBu_r', vmax = 0, vmin = 8, cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels = False, annot=False)
plt.title('$X^T$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(X, cmap = 'RdYlBu_r', vmax = 0, vmin = 8, cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels=False, annot=False)
plt.title('$X$')

#>>>>>>>>>>>>>>>>> 上三角
mask_tri = np.zeros_like(G)
mask_tri[np.triu_indices_from(mask_tri)] = True
mask_tri = mask_tri.astype(dtype=bool)

fig, axs = plt.subplots(1, 2)

sns.heatmap(G, ax = axs[0], vmax = 5000, vmin = 0, annot = True, fmt=".0f", cmap = 'RdYlBu_r', square = True, mask = ~mask_tri, cbar = False, linecolor = [0.5, 0.5, 0.5])

sns.heatmap(G, ax = axs[1], vmax = 5000, vmin = 0, cmap = 'RdYlBu_r', annot = True, fmt=".0f", square = True, mask = mask_tri, cbar = False, linecolor = [0.5, 0.5, 0.5])

#>>>>>>>>>>>>>>>>> Cholesky分解
L = np.linalg.cholesky(G)
L
fig,axs = plt.subplots(1,5,figsize = (8,3), gridspec_kw={'width_ratios': [3, 0.5, 3, 0.5, 3]})

plt.sca(axs[0])
ax = sns.heatmap(G, cmap = 'RdYlBu_r', vmax = 5000, vmin = 0, cbar_kws = {'orientation':'horizontal'}, yticklabels=False, annot=True, fmt=".0f", square = 'equal')
plt.title('$G$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(L, cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels = False, annot=True)
plt.title('$L$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(L.T, cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels=False, annot=True)
plt.title('$L^T$')

#>>>>>>>>>>>>>>>>> 特征值分解
Lambdas, V = np.linalg.eig(G)
print(Lambdas)
print(V)

fig,axs = plt.subplots(1,7,figsize = (18,5), gridspec_kw={'width_ratios': [3, 0.5, 3, 0.5, 3, 0.5, 3]})

plt.sca(axs[0])
ax = sns.heatmap(G, cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, annot=True, fmt=".0f", yticklabels=False, xticklabels=False, square = 'equal')
plt.title('$G$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V, cmap = 'RdYlBu_r', vmax = 1, vmin = -1, cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels = False, square = 'equal')
plt.title('$V$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(np.diag(Lambdas), cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels = False, annot=True, fmt=".0f", square = 'equal')
plt.title('$\Lambda$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V.T, cmap = 'RdYlBu_r', vmax = 1, vmin = -1, cbar_kws = {'orientation':'horizontal'}, yticklabels=False, xticklabels=False, square = 'equal')
plt.title('$V^T$')

#>>>>>>>>>>>>>>>>> 奇异值分解
U,S,VT = np.linalg.svd(X, full_matrices = False)
V = VT.T
fig,axs = plt.subplots(1,7,figsize = (18,5), gridspec_kw={'width_ratios': [3, 0.5, 3, 0.5, 3, 0.5, 3]})

plt.sca(axs[0])
ax = sns.heatmap(X, cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, vmin = 0, vmax = 8, yticklabels=False, xticklabels = False)
plt.title('$X$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(U, cmap = 'RdYlBu_r', vmax = 1, vmin = -1, cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels=False)
plt.title('$U$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(np.diag(S), cmap = 'RdYlBu_r', cbar_kws = {'orientation':'horizontal'}, xticklabels = False, yticklabels=False, annot=True, square = 'equal')
plt.title('$S$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V.T, cmap = 'RdYlBu_r', vmax = 1, vmin = -1, cbar_kws = {'orientation':'horizontal'}, yticklabels=False, xticklabels=False, square = 'equal')
plt.title('$V^T$')

# fig.savefig('Figures/奇异值分解，SVD.svg', format='svg')
# 统计
#>>>>>>>>>>>>>>>>> 协方差矩阵
Sigma = X_df.cov()
Corr  = X_df.corr()
fig, ax = plt.subplots(figsize = (4,4))

sns.heatmap(Sigma, ax = ax, annot = True, fmt=".2f",  cmap = 'RdYlBu_r', square = True, cbar_kws = {'orientation':'vertical'}, linecolor = [0.5, 0.5, 0.5])

fig, ax = plt.subplots(figsize = (4,4))

sns.heatmap(Corr, ax = ax, cmap = 'RdYlBu_r', annot = True, fmt=".2f", square = True, cbar_kws = {'orientation':'vertical'}, linecolor = [0.5, 0.5, 0.5])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 伪彩色网格图

import matplotlib.pyplot as plt
import numpy as np

# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))

z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()


fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
# fig.savefig('Figures/比较_pcolormesh.svg', format='svg')


fig, ax = plt.subplots()
c = ax.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% kNN分类

import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
y = iris.target

x_min, x_max = 4, 8
y_min, y_max = 2, 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))

num_neighbors = [2,5]
fig = plt.figure(figsize=(6, 9))
for idx, n_neighbors in enumerate(num_neighbors):
    ax = fig.add_subplot(2, 1, idx + 1)
    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.axis('tight')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 伪彩色网格图可视化几何变换

import numpy as np
import matplotlib.pyplot as plt

delta = 1/10

x_array = np.linspace(-1 + delta/2,1 - delta/2,int(2/delta))
y_array = np.linspace(-1 + delta/2,1 - delta/2,int(2/delta))

xx, yy = np.meshgrid(x_array, y_array)
zz = xx + yy
complex_zz = xx + yy*1j

colors = plt.cm.rainbow(np.linspace(0, 1, len(xx.flatten())))

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
ax.pcolormesh(xx, yy, zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
plt.show()

fig = plt.figure(figsize=(8, 12))
ax = fig.add_subplot(3, 2, 1)
ax.pcolormesh(np.real(complex_zz*np.exp(np.pi/6*1j)), np.imag(complex_zz*np.exp(np.pi/6*1j)), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# plt.show()

ax = fig.add_subplot(3, 2, 2)
ax.pcolormesh(np.real(2*complex_zz*np.exp(np.pi/3*1j)), np.imag(2*complex_zz*np.exp(np.pi/3*1j)), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# plt.show()

ax = fig.add_subplot(3, 2, 3)
ax.pcolormesh(np.real(np.exp(complex_zz)), np.imag(np.exp(complex_zz)), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# plt.show()

ax = fig.add_subplot(3, 2, 4)
ax.pcolormesh(np.real(complex_zz**3), np.imag(complex_zz**3), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# plt.show()

ax = fig.add_subplot(3, 2, 5)
ax.pcolormesh(np.real(1/complex_zz), np.imag(1/complex_zz), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
# plt.show()

ax = fig.add_subplot(3, 2, 6)
ax.pcolormesh(np.real(complex_zz - 1/complex_zz), np.imag(complex_zz - 1/complex_zz), zz*0 + np.nan, edgecolors = colors, shading='auto')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 非矢量图片
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 导入 matplotlib 库中用于处理图像的 image 模块，并将其命名为 mpimg
import numpy as np


# 导入照
img = mpimg.imread('iris_photo.jpg')
print(img)

# 展示图片
fig, ax = plt.subplots(figsize=(4,8))
imgplot = plt.imshow(img)

# 展示部分像素
fig, ax = plt.subplots(figsize=(4,8))
imgplot = plt.imshow(img[400:400 + 10,400:400 + 10,:])

# 分析图片
fig, axs = plt.subplots(figsize=(6,3), ncols = 1, nrows = 3, sharex=True)

all_red = img[:,:,0].ravel()
axs[0].hist(all_red, bins=256, range=(0, 256), color = 'r')

axs[0].yaxis.set_ticks([])
axs[0].set_xlim(0,255)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)

all_green = img[:,:,1].ravel()
axs[1].hist(all_green, bins=256, range=(0, 256), color = 'g')

axs[1].yaxis.set_ticks([])
axs[1].set_xlim(0,255)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)

all_blue = img[:,:,2].ravel()
axs[2].hist(all_blue, bins=256, range=(0, 256), color = 'b')

axs[2].yaxis.set_ticks([])
axs[2].set_xlim(0,255)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)

# 绘制红绿蓝三个不同通道
fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (4,12))

for i, subplot in zip(range(3), axs):
    temp = np.zeros(img.shape, dtype='uint8')
    temp[:,:,i] = img[:,:,i]
    subplot.imshow(temp)
    # subplot.set_axis_off()

# 只保留两色
fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (4,12))

for i, subplot in zip(range(3), axs):
    zeros = np.zeros(img.shape, dtype='uint8')
    temp  = np.copy(img)
    temp[:,:,i] = zeros[:,:,i]
    subplot.imshow(temp)
    # subplot.set_axis_off()

# 使用色谱
fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (4,12))
axs[0].imshow(img[:, :, 0], cmap = 'RdYlBu_r')
axs[1].imshow(img[:, :, 0], cmap = 'Greys_r')
axs[2].imshow(img[:, :, 0], cmap = 'Reds_r')
# fig.savefig('Figures/鸢尾花照片，红色通道，使用色谱.svg', format='svg')

fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (4,12))
axs[0].imshow(img[:, :, 1], cmap = 'RdYlBu_r')
axs[1].imshow(img[:, :, 1], cmap = 'Greys_r')
axs[2].imshow(img[:, :, 1], cmap = 'Greens_r')


fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (4,12))
axs[0].imshow(img[:, :, 2], cmap = 'RdYlBu_r')
axs[1].imshow(img[:, :, 2], cmap = 'Greys_r')
axs[2].imshow(img[:, :, 2], cmap = 'Blues_r')

## 灰度
from skimage import color
from skimage import io

# DOWNSAMPLE = 4
# R = img[::DOWNSAMPLE, ::DOWNSAMPLE, 0]
# G = img[::DOWNSAMPLE, ::DOWNSAMPLE, 1]
# B = img[::DOWNSAMPLE, ::DOWNSAMPLE, 2]
# image_blackwhite = 0.2989 * R + 0.5870 * G + 0.1140 * B

X = color.rgb2gray(io.imread('iris_photo.jpg'))
X.shape
fig, axs = plt.subplots(figsize=(4,8))
plt.imshow(X, cmap='gray')
# fig.savefig('Figures/鸢尾花照片，灰度.svg', format='svg')

# 替换部分色块
X_copy = np.copy(X)
X_copy[0:500, 0:500] = 1
fig, axs = plt.subplots(figsize=(4,8))
plt.imshow(X_copy, cmap='gray')
# fig.savefig('Figures/鸢尾花照片，灰度，替换.svg', format='svg')

# 降低像素
DOWNSAMPLE = 200
image_downsized = img[::DOWNSAMPLE, ::DOWNSAMPLE, :]
fig, ax = plt.subplots(figsize=(4,8))
imgplot = plt.imshow(image_downsized)
# fig.savefig('Figures/鸢尾花照片，低像素.svg', format='svg')

# 插值，平滑
methods = ['none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']
fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(6,8), subplot_kw={'xticks': [], 'yticks': []})
for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(image_downsized, interpolation=interp_method)
    ax.set_title(str(interp_method))
plt.tight_layout()
# fig.savefig('Figures/鸢尾花照片，低像素，插值.svg', format='svg')
# 仿射变换

import matplotlib.transforms as mtransforms
def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none', origin='lower', extent=[-2, 4, -3, 2], clip_on=True)
    trans_data = transform + ax.transData
    im.set_transform(trans_data)
    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--", transform = trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)

# prepare image and figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# image rotation
do_plot(ax1, img, mtransforms.Affine2D().rotate_deg(30))
# image skew
do_plot(ax2, img, mtransforms.Affine2D().skew_deg(30, 15))
# scale and reflection
do_plot(ax3, img, mtransforms.Affine2D().scale(-1, .5))
# everything and a translation
do_plot(ax4, img, mtransforms.Affine2D().rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






























































































































































































































































































































































































































































































































































































































































































































































































































































































