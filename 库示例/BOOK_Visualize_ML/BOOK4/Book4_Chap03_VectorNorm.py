


#%%   Bk4_Ch3_01.py

import matplotlib.pyplot as plt
import numpy as np

p_values = [0.05, 0.2, 0.5, 1, 1.5, 2, 4, 8, np.inf]

x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)

fig, axes = plt.subplots(ncols=3,nrows=3,
                         figsize=(12, 12))

for p, ax in zip(p_values, axes.flat):

    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)

    # plot contour of Lp
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')

    # plot contour of Lp = 1
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # decorations

    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')

plt.show()


#%%  Bk4_Ch3_02.py

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

u = [0,0,4, 3]
v = [0,0,-2,4]
u_bis = [4,3,v[2],v[3]]
w = [0,0,2,7]

fig, ax = plt.subplots()

plt.quiver([u[0], u_bis[0], w[0]],
           [u[1], u_bis[1], w[1]],
           [u[2], u_bis[2], w[2]],
           [u[3], u_bis[3], w[3]],
           angles='xy', scale_units='xy',
           scale=1, color=sns.color_palette())

plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.text(3, 1, r'$||\vec{u}||_2$',
         color=sns.color_palette()[0], size=12,
         ha='center',va='center')

plt.text(3, 6, r'$||\vec{v}||_2$',
         color=sns.color_palette()[1], size=12,
         ha='center',va='center')

plt.text(0, 4, r'$||\vec{u}+\vec{v}||_2$',
         color=sns.color_palette()[2], size=12,
         ha='center',va='center')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xticks(np.arange(-2,8 + 1))
ax.set_yticks(np.arange(-2,8 + 1))
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# reference: Essential Math for Data Science



#%% # 平面Lp范数等高线  # Bk4_Ch3_01.py  Bk2_Ch25

## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)

## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz

## 可视化
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(6, 9))
for p, ax in zip(p_values, axes.flat):
    # 计算范数
    zz = Lp_norm(p)
    # 绘制等高线
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # 绘制Lp norm = 1的等高线
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # 装饰
    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')


#%% # 平面Lp范数等高线
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)

## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz

## 可视化
# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 18), projection = '3d')
fig = plt.figure(figsize=(12, 18), constrained_layout = True)
for i, p in enumerate(p_values):
    ax = fig.add_subplot(3, 2, i+1, projection='3d')
    # 计算范数
    zz = Lp_norm(p)

    ## 4 plot_wireframe() 绘制网格曲面 + 三维等高线
    ax.plot_wireframe(xx1, xx2, zz, color = [0.5,0.5,0.5], linewidth = 0.25)

    # 三维等高线
    # colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
    # 三维等高线
    colorbar = ax.contour(xx1, xx2, zz, 20,  cmap = 'hsv')
    # fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

    # 二维等高线
    ax.contour(xx1, xx2, zz, zdir = 'z', offset= zz.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

    fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
    ax.set_proj_type('ortho')

    # 3D坐标区的背景设置为白色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel(r'$\it{x_1}$')
    ax.set_ylabel(r'$\it{x_2}$')
    ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
    ax.set_title(f"p = {p}")
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())

    ax.view_init(azim=-135, elev=30)
    ax.grid(False)
plt.show()











































