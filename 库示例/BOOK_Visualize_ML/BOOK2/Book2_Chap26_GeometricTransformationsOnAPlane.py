

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_01 扁平化文字对象的仿射变换

import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import PathPatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
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
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

#
text = 'Fuck you'
text_path = TextPath((0, 0), text, size=10)

fig = plt.figure(figsize=(8, 12))

# 原图
ax = fig.add_subplot(3, 2, 1)
trans = Affine2D()
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

# 平移
ax = fig.add_subplot(3, 2, 2)
trans = Affine2D().translate(50,50)
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

# 缩放
ax = fig.add_subplot(3, 2, 3)
trans = Affine2D().scale(2,5)
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

# 缩放、旋转、平移
ax = fig.add_subplot(3, 2, 4)
trans = Affine2D().scale(2,2).rotate(45).translate(20,20)
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

# 缩放、沿y剪切
ax = fig.add_subplot(3, 2, 5)
trans = Affine2D().scale(2,2).skew_deg(0,20)
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)

# 缩放、沿x剪切
ax = fig.add_subplot(3, 2, 6)
trans = Affine2D().scale(2,2).skew_deg(50, 0)
p = PathPatch(trans.transform_path(text_path))
ax.add_patch(p)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 100), ax.set_ylim(0, 100)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_02 平面网格的仿射变换
# Bk_2_Ch12_13:平面仿射变换, 线性变换
# Bk_2_Ch26_02 平面网格的仿射变换
# BK_2_Ch08_10, 非线性变换,绘制网格
# BK_2_Ch10_05 线性、非线性几何变换
# 导入包
import numpy as np
import matplotlib.pyplot as plt

# 产生网格数据
x1 = np.arange(-5, 5 + 1, step=1) # (11,)
x2 = np.arange(-5, 5 + 1, step=1)

XX1, XX2 = np.meshgrid(x1,x2) # (11, 11)
X = np.column_stack((XX1.ravel(), XX2.ravel())) #  (121, 2)

# 自定义可视化函数
def visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name):
    colors = np.arange(len(XX1.ravel()))

    fig, ax = plt.subplots(figsize = (5,5), constrained_layout = True)
    # 绘制原始网格
    plt.plot(XX1, XX2, color = [0.8,0.8,0.8], lw = 0.25)
    plt.plot(XX1.T, XX2.T, color = [0.8,0.8,0.8], lw = 0.25)
    # plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'plasma', zorder=1e3)

    #绘制几何变换后 的网格
    plt.plot(ZZ1,ZZ2,color = '#0070C0', lw = 0.25)
    plt.plot(ZZ1.T, ZZ2.T,color = '#0070C0', lw = 0.25)
    plt.scatter(ZZ1.ravel(), ZZ2.ravel(), c = colors, s = 10, cmap = 'rainbow', zorder=1e3)

    plt.axis('scaled')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.axhline(y = 0, color = 'k')
    ax.axvline(x = 0, color = 'k')
    plt.xticks([])
    plt.yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

#>>>>>>>>>>>> 原始网格
colors = np.arange(len(XX1.ravel()))

fig, ax = plt.subplots(figsize = (5,5))
# 绘制原始网格
plt.plot(XX1, XX2,color = '#0070C0', lw = 0.25)
plt.plot(XX1.T, XX2.T,color = '#0070C0', lw = 0.25)
plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 10, cmap = 'rainbow', zorder=1e3)

plt.axis('scaled')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#>>>>>>>>>>>> 平移
t = np.array([[4.5, -1.5]])
Z = X + t;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '平移'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 旋转
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
Z = X@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '逆时针旋转30度'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

# 先平移，再旋转
Z = (X + t)@R.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
fig_name = '先平移，再逆时针旋转30度'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

# 先旋转，再平移
Z = X@R.T + t;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先逆时针旋转30度，再平移'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 等比例放大
S = np.array([[2, 0],
              [0, 2]])

Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例放大'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 等比例缩小
S = np.array([[0.8, 0],
              [0,   0.8]])
Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '等比例缩小'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 非等比例缩放
S = np.array([[2, 0],
              [0, 1.5]])
Z = X@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '非等比例缩放'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 先缩放，再旋转
Z = X@S@R;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先缩放，再旋转'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 先旋转，再放大
Z = X@R@S;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再缩放'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 横轴镜像
M = np.array([[1, 0],
              [0, -1]])
# 绕原点，逆时针旋转30
theta = 30/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2, 0],
              [0, 1.5]])

# 先旋转，再放大，最后横轴镜像
Z = X@R@S@M;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再放大，最后横轴镜像'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 纵轴镜像
M = np.array([[-1, 0],
              [0, 1]])

# 先旋转，再放大，最后纵轴镜像
Z = X@R@S@M;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '先旋转，再放大，最后纵轴镜像'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向横轴投影
P = np.array([[1, 0],
              [0, 0]])
Z = X@P;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向横轴投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向纵轴投影
P = np.array([[0, 0],
              [0, 1]])
Z = X@P;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向纵轴投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 向特定过原点直线投影
theta = 30/180 * np.pi
# 过原点，和横轴夹角30度直线

v = np.array([[np.cos(theta)],
              [np.sin(theta)]])
Z = X@v@v.T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '向特定过原点直线投影'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 沿横轴剪切
T = np.array([[1, 0],
              [1.5, 1]])
Z = X@T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿横轴剪切'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)

#>>>>>>>>>>>> 沿纵轴剪切
T = np.array([[1, 1.5],
              [0, 1]])
Z = X@T;
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig_name = '沿纵轴剪切'
visualize_transform(XX1, XX2, ZZ1, ZZ2, fig_name)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_03 正圆散点的几何变换
# 导入包
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os

# 产生数据
num_points = 37;
theta_array = np.linspace(0, 2*np.pi, num_points).reshape((-1, 1))
# 单位圆参数方程，散点
X_circle = np.column_stack((np.cos(theta_array), np.sin(theta_array)))
# 单位圆散点，利用色谱逐一渲染
colors = plt.cm.rainbow(np.linspace(0, 1, len(X_circle)))
# 单位圆参数方程，连线
theta_array_fine = np.linspace(0, 2*np.pi, 500).reshape((-1, 1))
X_circle_fine = np.column_stack((np.cos(theta_array_fine), np.sin(theta_array_fine)))
# 1/4正方形顶点，首尾相连
X_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
# 完整正方形顶点，首尾相连
X_square_big = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]])
# 圆心坐标
center_array = X_circle*0
# Chol分解
A = np.array([[1.25, -0.75], [-0.75,1.25]])
SIGMA = A.T @ A
L = np.linalg.cholesky(SIGMA)
R = L.T

# 可视化
fig, axs = plt.subplots(figsize = (15, 25), nrows = 5, ncols = 3, constrained_layout = True)
for theta, ax_idx in zip(np.linspace(0, np.pi, 15, endpoint=False), axs.ravel()):
    # 定义旋转矩阵
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # 只旋转
    X_square_R_rotated = X_square @ U
    X_square_big_R_rotated = X_square_big @ U
    # 先旋转，再剪切
    X_R = X_circle @ U @ R
    X_circle_fine_R = X_circle_fine @ U @ R
    X_square_R = X_square @ U @ R
    X_square_big_R = X_square_big @ U @R
    # # 绘制单位圆
    ax_idx.plot(X_circle_fine[:,0], X_circle_fine[:,1], c = [0.8,0.8,0.8], lw = 0.2)
    # # 绘制单位圆上的“小彩灯”
    ax_idx.scatter(X_circle[:,0], X_circle[:,1], s = 28, marker = '.', c = colors, zorder=1e3)
    # # 绘制大小两个正方形
    ax_idx.plot(X_square[:,0], X_square[:,1], c='k', linewidth = 1)
    ax_idx.plot(X_square_big[:,0], X_square_big[:,1], c='k')
    # 绘制大小两个正方形，剪切 > 旋转
    ax_idx.plot(X_square_R[:,0], X_square_R[:,1], c='k', linewidth = 1)
    ax_idx.plot(X_square_big_R[:,0], X_square_big_R[:,1], c='k')
    # 绘制大小两个正方形，旋转
    ax_idx.plot(X_square_R_rotated[:,0], X_square_R_rotated[:,1], c='k', linewidth = 1)
    ax_idx.plot(X_square_big_R_rotated[:,0], X_square_big_R_rotated[:,1], c='k')
    # 绘制椭圆
    ax_idx.plot(X_circle_fine_R[:,0], X_circle_fine_R[:,1], c=[0.8,0.8,0.8], lw = 0.2)
    # 绘制两点连线，可视化散点运动轨迹
    ax_idx.plot(([i for (i,j) in X_circle], [i for (i,j) in X_R]), ([j for (i,j) in X_circle], [j for (i,j) in X_R]),c=[0.8,0.8,0.8], lw = 0.2)
    # 绘制椭圆上的“小彩灯”
    ax_idx.scatter(X_R[:,0],X_R[:,1], s = 28, marker = '.', c = colors, zorder=1e3)
    # 绘制水平、竖直线
    ax_idx.axvline(x = 0, c = 'k', lw = 0.2)
    ax_idx.axhline(y = 0, c = 'k', lw = 0.2)

    # 装饰
    ax_idx.axis('scaled')
    ax_idx.set_xbound(lower = -2, upper = 2.)
    ax_idx.set_ybound(lower = -2., upper = 2.)
    ax_idx.set_xticks([])
    ax_idx.set_yticks([])
    ax_idx.spines['top'].set_visible(False)
    ax_idx.spines['right'].set_visible(False)
    ax_idx.spines['bottom'].set_visible(False)
    ax_idx.spines['left'].set_visible(False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_04 平面点线投影

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.covariance import EmpiricalCovariance

# 自定义函数
def normal_pdf_1d(x, mu,sigma):
    # 一元高斯分布PDF函数
    scaling = 1/sigma/np.sqrt(2*np.pi)
    z = (x - mu)/sigma
    pdf = scaling*np.exp(-z**2/2)
    return pdf

def draw_vector(vector, RBG):
    # 绘制箭头图
    array = np.array([[0, 0, vector[0], vector[1]]], dtype=object)
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=0.5,color = RBG)
# 导入数据
# iris_sns = sns.load_dataset("iris")
iris = load_iris()
# 处理数据
# 将数据帧转化为numpy数组
X = iris.data
# 取出鸢尾花数据集前两列
X = np.array(X[:,:4])
x1 = X[:,:2]
# 产生网格化数据
xx_maha, yy_maha = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400),)
# 合并​
zz_maha = np.c_[xx_maha.ravel(), yy_maha.ravel()] #  (160000, 2)
# 计算协方差矩阵
emp_cov_Xc = EmpiricalCovariance().fit(x1)

# 计算马氏距离平方​
mahal_sq_Xc = emp_cov_Xc.mahalanobis(zz_maha) # (160000,)
mahal_sq_Xc = mahal_sq_Xc.reshape(xx_maha.shape) # (400, 400)
# 开平方得到马氏距离
mahal_d_Xc = np.sqrt(mahal_sq_Xc) # mahal_d_Xc

# 可视化
theta_array = np.linspace(0, 90, 7)
# 不同的投影角度

for theta_deg in theta_array:
    print('====================')
    print('theta = ' + str(theta_deg))
    theta = theta_deg*np.pi/180
    R1 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    v1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    # 通过原点直线对应的单位切向量
    # 一次投影，得到投影点在v1向量上坐标
    z1_1D = x1@v1
    # 计算投影点的统计量：均值、标准差
    mu1 = z1_1D.mean()
    std1 = z1_1D.std()

    # 二次投影，得到投影点在xy平面的坐标
    proj = v1@v1.T
    z1_2D = x1@proj

    print('std1 = ' + str(std1))
    x1_array = np.linspace(mu1 - 4*std1, mu1 + 4*std1, 100)

    pdf1_array = normal_pdf_1d(x1_array, mu1, std1)
    PDF1 = np.column_stack((x1_array, -pdf1_array))
    # 旋转一元高斯PDF曲线
    PDF1_v1 = PDF1@R1

    fig, ax = plt.subplots(figsize = (10,10), constrained_layout = True)
    # 绘制马氏距离等高线
    plt.contourf(xx_maha, yy_maha, mahal_d_Xc, cmap='Blues_r', levels = np.linspace(0,4,21))
    plt.contour(xx_maha, yy_maha, mahal_d_Xc, colors='k', levels = [1,2,3])
    # 绘制样本数据散点
    sns.scatterplot(x=x1[:,0], y=x1[:,1], zorder = 1e3, ax = ax)
    # 绘制投影点
    plt.plot(z1_2D[:,0],z1_2D[:,1], marker = 'x', markersize = 8, color = 'b')
    # 绘制原始数据、投影点之间的连线
    plt.plot(([i for (i,j) in z1_2D], [i for (i,j) in x1]), ([j for (i,j) in z1_2D], [j for (i,j) in x1]),c=[0.6,0.6,0.6])
    # 绘制过原点的投影直线
    plt.plot((-10, 10), (-10*np.tan(theta), 10*np.tan(theta)),c = 'k')
    # 投影一元高斯PDF曲线
    plt.plot(PDF1_v1[:,0], PDF1_v1[:,1], color = 'b')
    # 绘制质心、投影质心
    plt.plot(x1[:,0].mean(), x1[:,1].mean(), marker = 'x', color = 'r', mew = 4, markersize = 18)
    plt.plot(z1_2D[:,0].mean(), z1_2D[:,1].mean(), marker = 'x', color = 'r', mew = 4, markersize = 18)

    # 绘制箭头
    draw_vector(v1, 'b')
    ax.axvline(x = 0, c = 'k')
    ax.axhline(y = 0, c = 'k')
    ax.axis('scaled')
    ax.set_xbound(lower = 0, upper = 10)
    ax.set_ybound(lower = 0, upper = 8)
    # ax.xaxis.set_ticks([])
    # ax.yaxis.set_ticks([])

    ax.set_xlim([x1[:,0].min() - 6, x1[:,0].max() + 3])
    ax.set_ylim([x1[:,1].min() - 3, x1[:,1].max() + 3])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch26_05 二维数据分别朝16个不同方向投影
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from matplotlib.pyplot import cm

# 自定义函数
def normal_pdf_1d(x, mu,sigma):
    # 一元高斯分布PDF
    scaling = 1/sigma/np.sqrt(2*np.pi)
    z = (x - mu)/sigma
    pdf = scaling*np.exp(-z**2/2)
    return pdf
xx_maha, yy_maha = np.meshgrid( np.linspace(-20, 20, 400), np.linspace(-20, 20, 400),) # (400, 400)
zz_maha = np.c_[xx_maha.ravel(), yy_maha.ravel()] # (160000, 2)

# 创建数据
cov = np.array([[1, 0.86], [0.86, 1]])
x1 = np.random.multivariate_normal([0, 0], cov, size = 200) # (118, 2)
emp_cov_Xc = EmpiricalCovariance().fit(x1) #
print(f"emp_cov_Xc.covariance_ = {emp_cov_Xc.covariance_}")
mahal_sq_Xc = emp_cov_Xc.mahalanobis(zz_maha) # (160000, )
mahal_sq_Xc = mahal_sq_Xc.reshape(xx_maha.shape) # (400, 400)
mahal_d_Xc = np.sqrt(mahal_sq_Xc)
central = x1.mean(axis = 0).reshape(-1, 1)

# 可视化
theta_array = np.linspace(0, 360, num = 12, endpoint= False)
# theta_array = [30, ]
# 定义16个不同投影角度
colors = cm.hsv(np.linspace(0, 1, len(theta_array)))
fig, ax = plt.subplots(figsize = (10, 10), constrained_layout = True)
# 绘制质心
plt.plot(x1[:,0].mean(), x1[:,1].mean(), marker = 'x', color = 'k', markersize = 18)
# theta_array = [135, ]
for idx, theta in enumerate(theta_array):
    color_idx = colors[idx]
    theta = theta*np.pi/180
    # 投影方向，用列向量表示
    v1 = np.array([[np.cos(theta)], [np.sin(theta)]])
    proj = v1@v1.T
    # 距离
    R = 18
    alpha = theta + np.pi/2
    p0 = R*np.array([[np.cos(alpha)], [np.sin(alpha)]]) + central
    # 朝不过原点的直线投影结果，二维平面
    z1_2D = x1 @ proj + p0.T @ (np.eye(2) - proj) # (118, 2) + (1, 2)
    # 投影数据，一维数组
    z1_1D = x1@v1 # (118, 1)
    # 计算投影数据均值和标准差
    mu1 = z1_1D.mean()
    std1 = z1_1D.std()
    x1_array = np.linspace(mu1 - 4*std1, mu1 + 4*std1,100)
    # 乘4放大PDF曲线高度
    pdf1_array = normal_pdf_1d(x1_array, mu1, std1)*4
    PDF1 = np.column_stack((x1_array, pdf1_array))
    # 旋转矩阵
    R1 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # 一元高斯曲线的几何变换
    PDF1_v1 = PDF1@R1 + p0.T @ (np.eye(2) - proj)
    # 绘制投影结果
    plt.plot(z1_2D[:,0], z1_2D[:,1], marker = '.', markersize = 6, color = color_idx, markeredgecolor = 'w')
    plt.plot(([i for (i,j) in z1_2D], [i for (i,j) in x1]), ([j for (i,j) in z1_2D], [j for (i,j) in x1]), c=color_idx, lw = 0.1)
    plt.plot(PDF1_v1[:,0], PDF1_v1[:,1], color = color_idx)
    # 绘制投影质心
    plt.plot(z1_2D[:,0].mean(),z1_2D[:,1].mean(), marker = 'x', color = color_idx, markersize = 8)
# 绘制马氏距离等高线,zorder用来控制绘图顺序，其值越大，画上去越晚，线条的叠加就是在上面的
plt.contour(xx_maha, yy_maha, mahal_d_Xc, colors='k', levels=np.arange(1,5), zorder = 1000)
# 绘制散点数据
plt.scatter(x1[:,0],x1[:,1], alpha = 0.4, c = 'k', zorder = 1000)
ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
# ax.set_xbound(lower = -20, upper = 20)
# ax.set_ybound(lower = -20, upper = 20)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3D 点面投影

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
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


def project_points_to_plane(points, plane_coeffs):
    """
    将点集投影到任意平面
    :param points: 点集，形状为 (n, 3) 的 numpy 数组
    :param plane_coeffs: 平面系数 [A, B, C, D]，对应 Ax + By + Cz + D = 0
    :return: 投影点集，形状为 (n, 3)
    """
    A, B, C, D = plane_coeffs
    n = np.array([A, B, C])
    denominator = A**2 + B**2 + C**2  # 分母

    # 计算投影点
    t = -(A * points[:,0] + B * points[:,1] + C * points[:,2] + D) / denominator
    projected_points = points + t[:, np.newaxis] * n

    return projected_points

# 定义一般平面方程：Ax + By + Cz + D = 0
plane_coeffs = [0, 0, 1, -4]
plane_coeffs = [1, -1, 1, -4] # A=1, B=2, C=-3, D=4

A, B, C, D = plane_coeffs
n = np.array([A, B, C])
# 生成随机点集
np.random.seed(42)
points = np.random.randn(20, 3) + 10  # 20个随机点，范围约±5

# 计算投影点
projected_points = project_points_to_plane(points, plane_coeffs)

# 验证投影点是否在平面上
print(f"projected_points - points = {projected_points - points}")
print(f"{projected_points @ np.array(plane_coeffs[:-1])[:,None]+ plane_coeffs[-1]}")
# 验证正交性
print("验证投影向量与法向量正交性:")

if C != 0:
    x = 1
    y = 1
    z = (-D-A*x-B*y)/C
pointInPlane = np.array([x, y,z])
for i in range(len(points)):
    proj_vec = projected_points[i] - pointInPlane  # 投影向量（从原点到投影点）
    dot = proj_vec @ n
    print(f"点{i}: 点积 = {dot:.2e} (应接近0)")

# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho') # #  正交投影模式
ax.view_init(azim = -135, elev = 30)
ax.set_aspect('equal', adjustable='box')
# 绘制原始点（红色）
ax.scatter(points[:,0], points[:,1], points[:,2], c='r', s=50, label='原始点')

# 绘制投影点（蓝色）
ax.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], c='b', s=50, label='投影点')

# 绘制平面
xlim, ylim = ax.get_xlim(), ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 20), np.linspace(ylim[0], ylim[1], 20))
zz = (-plane_coeffs[0] * xx - plane_coeffs[1] * yy - plane_coeffs[3]) / plane_coeffs[2]
ax.plot_surface(xx, yy, zz, alpha=0.4, color='b', label=f'平面: {A}x{'+'+str(B) if B>=0 else B}y{'+'+str(C) if C>=0 else C}z{'+'+str(D) if D>=0 else D} = 0')

p0x = np.mean(xlim)
p0y = np.mean(ylim)
p0z = (-plane_coeffs[0] * p0x - plane_coeffs[1] * p0y - plane_coeffs[3]) / plane_coeffs[2]
d = 5
n = n/np.linalg.norm(n)
p1x = p0x + d*n[0]; p1y = p0y + d*n[1]; p1z = p0z + d*n[2]
pointInPlane = np.array([p0x, p0y, p0z])
# 绘制向量 c，投影到xy
ax.quiver(*pointInPlane,  *n, length=3, color='k', label='法向量', arrow_length_ratio=0.1)
ax.plot([p0x, p1x], [p0y, p1y], [p0z, p1z], 'g--', linewidth = 2, label='法向量',)
# 绘制投影线
for i in range(len(points)):
    ax.plot([points[i,0], projected_points[i,0]], [points[i,1], projected_points[i,1]], [points[i,2], projected_points[i,2]], 'k--', linewidth=0.8, alpha=0.3)

# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(False)
# 设置等比例坐标轴
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.legend()
plt.title('三维点到一般平面的投影', pad=20)
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3D 空间点集到空间直线的投影

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_points_to_line(points, Q, v):
    """
    将点集投影到直线
    :param points: 点集，形状为 (n, 3) 的 numpy 数组
    :param Q: 直线上的参考点，形状为 (3,)
    :param v: 直线的方向向量，形状为 (3,)
    :return: 投影点集，形状为 (n, 3)
    """
    v = v / np.linalg.norm(v)  # 单位化方向向量
    t = np.dot(points - Q, v)   # 计算所有点的参数 t
    return Q + t[:, np.newaxis] * v

# 示例数据
Q = np.array([1, 2, 0])       # 直线上的点
v = np.array([1, -1, 1])       # 方向向量
points = np.random.randn(4, 3) * 10  # 随机生成10个点

# 计算投影点
projected_points = project_points_to_line(points, Q, v)

# 验证正交性
for i in range(len(points)):
    print(f"点 {i} 投影正交性:",
          np.dot(points[i] - projected_points[i], v))  # 应接近0

# 可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制直线
xlim, ylim = ax.get_xlim(), ax.get_ylim()
t_vals = np.linspace(-5, 5, 100)
line_points = Q + t_vals[:, np.newaxis] * v
ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 'b-', linewidth=2, label='直线 L')

# 绘制原始点（红色）和投影点（绿色）
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=80, label='原始点')
ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], c='g', s=80, label='投影点')

# 绘制投影线（黑色虚线）
for i in range(len(points)):
    ax.plot([points[i, 0], projected_points[i, 0]], [points[i, 1], projected_points[i, 1]], [points[i, 2], projected_points[i, 2]], 'k--', alpha=0.5)

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(False)
# 设置等比例坐标轴
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('空间点集到直线的投影', pad=20)
plt.tight_layout()
plt.show()






















































































































































































































































































































































































































































































































































