
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
for sigma_idx,color_idx in zip(sigma_array, colors):
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

xx1, xx2 = np.meshgrid(np.linspace(-3,3), np.linspace(-3,3))
ff = np.exp(- xx1**2 - xx2**2)
# 高斯函数

# 可视化
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# 绘制曲面
ax.plot_wireframe(xx1, xx2, ff, color = [0.3,0.3,0.3], linewidth = 0.25)

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
ax.plot_wireframe(xx1, xx2, ff, color = [0.3,0.3,0.3], linewidth = 0.25)

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

PMF_ff = multinomial.pmf(x=np.array(([xx1.ravel(), xx2.ravel(), xx3.ravel()])).T, n=num, p=p_array)

PMF_ff = np.where(PMF_ff > 0.0, PMF_ff, np.nan)

PMF_ff = np.reshape(PMF_ff, xx1.shape)


fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection="3d")

ax.stem(xx1.ravel(), xx2.ravel(), PMF_ff.ravel(), basefmt=" ")

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

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

ax.scatter(Data[:,0],Data[:,1],Data[:,2], alpha = 1, s = 40)

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
plt.show()

# 12条参考线
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(Data[:,0],Data[:,1],Data[:,2], alpha = 1, s = 40)

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

fig, ax = plt.subplots(subplot_kw = {'projection': '3d'}, figsize = (10,10))
ax.plot_wireframe(xx, yy, f_xy_zz, color = [0.5,0.5,0.5],  rstride = 15, cstride = 0, linewidth = 2)
colors = plt.cm.rainbow(np.linspace(0, 1, len(xx_s.ravel())))
for i in np.linspace(0, len(xx_s.ravel()) - 1, len(xx_s.ravel())):
    i = int(i)
    x_t = xx_s.ravel()[i]
    y_t = yy_s.ravel()[i]
    color = colors[i, :]
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

    ax.plot(x_t, y_t, z_t, color = color,  marker = '.', markersize = 5)
    # tangent point
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10,10))

ax.plot_wireframe(xx, yy, f_xy_zz, color = [0.5,0.5,0.5], rstride=0, cstride=15, linewidth = 0.25)

colors = plt.cm.rainbow(np.linspace(0, 1, len(yy_s.ravel())))
for i in np.linspace(0, len(yy_s.ravel())-1, len(yy_s.ravel())):
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









