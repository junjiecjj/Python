

"""
关于plot_surface使用，

(一)，ax.plot_surface(xx, yy, f_xy_zz, cmap='turbo', linewidth=1, shade=False) # 删除阴影
    此时画出的表面图是全曲面的着色的，这种用法不多。
(二) 第二种是去掉全曲面，只对线条着色
    norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
    colors = cm.turbo(norm_plt(f_xy_zz))
    surf = ax.plot_surface(xx, yy, f_xy_zz, facecolors = colors, linewidth = 1, shade = False) # 删除阴影
    # or
    V = np.sin(xx) * np.sin(yy)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
    norm_plt = plt.Normalize(V.min(), V.max())
    colors = cm.turbo(norm_plt(V))
    surf = ax.plot_surface(xx, yy,  f_xy_zz,  facecolors=colors,  linewidth=1, shade=False)
    surf.set_facecolor((0,0,0,0))
    f_xy_zz是根据z = f(x,y)计算出的z值，这个值一般是在3D中使用的Z值，也就是Z轴的高度。
    此时会利用f_xx_yy或者V的函数值对曲面的颜色进行控制，
    如果使用f_xy_zz，则Z轴的高度就是对应函数值，等高线将会是平行的，
    如果使用V值，则是“将第四维数据 V(x,y) 投影到三维曲面 f(x,y)”，也就是曲线的形状是(x,y,f_xy_zz)，但是颜色不是f_xy_zz控制，而是第四维V控制，这时候如果想画等高线，得使用
    for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):的方式，见Book2_chap29,Book2_chap32。
"""


#%%==========================================================================================================
##########################################  3D Mesh Surface, 网格曲面 ######################################
#==========================================================================================================

#%% 绘制网格曲面 BK_2_Ch15_01
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
from matplotlib import cm
# 导入色谱模块

# 1. 定义函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数

# 2. 网格函数
def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3, 3, num)
    y_array = np.linspace(-3, 3, num)
    xx,yy = np.meshgrid(x_array, y_array)

    return xx, yy

# 3. 展示网格面，网格粗糙
xx, yy = mesh(num = 11)
zz = xx * 0

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, zz, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

# 4. 绘制函数网格曲面，网格粗糙
ff = f_xy_fcn(xx,yy)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx,yy, ff, color = [0.5,0.5,0.5], linewidth = 0.25)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

# 5. 展示网格面，网格过密
xx, yy = mesh(num = 101)
zz = xx * 0

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_wireframe(xx,yy, zz, color = [0.8,0.8,0.8], rstride=1, cstride=1, linewidth = 0.25)

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
plt.show()

# 6. 绘制函数网格曲面，网格过密
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_wireframe(xx,yy, ff, color = [0.5,0.5,0.5], rstride=1, cstride=1, linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/绘制函数网格曲面，网格过密.svg', format='svg')
plt.show()


# 7. 增大步幅
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_wireframe(xx, yy, ff, color = '#0070C0', rstride = 5, cstride = 5, linewidth = 0.25)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/增大步幅.svg', format='svg')
plt.show()

# 8. 仅绘制沿x方向曲线
ff = f_xy_fcn(xx,yy)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, ff, color = '#0070C0', rstride=5, cstride=0, linewidth = 0.25)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/仅绘制沿x方向曲线.svg', format='svg')
plt.show()


# 10. 特别强调特定曲线
# 请大家试着绘制一条 x = 1曲线
x_array = np.linspace(-3,3,100)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, ff, color = '0.5', rstride=5, cstride=5, linewidth = 0.25)

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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, ff, color = '0.5', rstride=5, cstride=5, linewidth = 0.25)
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
ff_scatter = f_xy_fcn(xx_scatter, yy_scatter)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, ff, color = [0.6,0.6,0.6], rstride=5, cstride=5, linewidth = 0.25)
ax.scatter(xx_scatter.ravel(), yy_scatter.ravel(), ff_scatter, c = ff_scatter, s = 10, cmap = 'RdYlBu_r')
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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)

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


#%% 将第四维数据映射到三维网格曲面, Bk_2_Ch15_02
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy.abc import x, y
from sympy import lambdify, diff, exp, latex

def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)
    return xx, yy

xx, yy = mesh(num = 201)
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数
f_xy_zz = f_xy_fcn(xx, yy)

#########################  1. 一般曲面 f(x,y)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

surf = ax.plot_surface(xx, yy, f_xy_zz, cmap='turbo', linewidth=1, shade=False) # 删除阴影
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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
colors = cm.turbo(norm_plt(f_xy_zz))

surf = ax.plot_surface(xx, yy, f_xy_zz, facecolors = colors, linewidth = 1, shade = False) # 删除阴影
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

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

norm_plt = plt.Normalize(V.min(), V.max())
colors = cm.turbo(norm_plt(V))

surf = ax.plot_surface(xx, yy, f_xy_zz,  facecolors=colors,  linewidth=1, shade=False)
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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
colors = cm.turbo(norm_plt(f_xy_zz))

surf = ax.plot_surface(xx, yy, V,  facecolors=colors,  linewidth=1, shade=False)
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


#%% Bk_2_Ch15_03 # 可视化 Dirichlet 分布

import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from matplotlib import cm
### 1. 定义可视化函数

def visualize(alpha_array, mesh_only = True):
    # 生成网格化数据
    theta_1_array = theta_2_array = np.linspace(0,1,201)
    tt1, tt2 = np.meshgrid(theta_1_array, theta_2_array)
    tt3 = 1 - tt1 - tt2
    # Points = np.column_stack([tt1.ravel(), tt2.ravel(), tt3.ravel()])
    # Points_filtered = Points[Points[:,2]>=0, :]
    # pdf = dirichlet.pdf(Points_filtered.T, alpha_array)
    PDF_array = []
    for t_1_idx, t_2_idx in zip(tt1.ravel(), tt2.ravel()):
        t_3_idx = 1 - t_1_idx - t_2_idx
        if t_3_idx < 0.005:
            PDF_idx = np.nan
        else:
            PDF_idx = dirichlet.pdf([t_1_idx, t_2_idx, t_3_idx], alpha_array)
        PDF_array.append(PDF_idx)
    PDF_FF = np.reshape(PDF_array, tt1.shape)
    # 可视化
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    norm_plt = plt.Normalize(np.nanmin(PDF_FF), np.nanmax(PDF_FF))
    colors = cm.RdYlBu_r(norm_plt(PDF_FF))
    surf = ax.plot_surface(tt1, tt2, tt3, facecolors = colors, linewidth = 0.25, shade = False, cstride = 10, rstride = 10)
    if mesh_only:
        surf.set_facecolor((0,0,0,0))
    ax.view_init(azim=30, elev=30)
    ax.set_proj_type('ortho')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_zlim((0,1))
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_zlabel(r'$\theta_3$')
    ax.set_xticks((0,1))
    ax.set_yticks((0,1))
    ax.set_zticks((0,1))
    ax.grid(False)
    title = '_'.join(str(v) for v in alpha_array)
    title = 'alphas_' + title
    if mesh_only:
        title = title + '_mesh_only'
    # fig.savefig('Figures/' + title + '.svg', format='svg')
alpha_array = [1, 2, 2]
visualize(alpha_array, True)

alpha_array = [1, 2, 2]
visualize(alpha_array, False)

alpha_array = [2, 1, 2]
visualize(alpha_array, True)

alpha_array = [2, 2, 1]
visualize(alpha_array, True)

alpha_array = [2, 1, 2]
visualize(alpha_array, False)

alpha_array = [2, 2, 1]
visualize(alpha_array, False)

alpha_array = [2, 2, 2]
visualize(alpha_array, True)

alpha_array = [2, 2, 2]
visualize(alpha_array, False)


#%% 绘制填充平面,  BK_2_Ch15_04  平行于不同平面的剖面
# 导入包
import numpy as np
import matplotlib.pyplot as plt
import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 1. 绘制xy平行面，网格
s_fine = np.linspace(0, 10, 11)
xx, yy = np.meshgrid(s_fine, s_fine)
# 生成网格数据
fig = plt.figure( figsize = (8,8))
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
plt.show()


# 2. 绘制xy平行面，无网格
s_coarse = np.linspace(0, 10, 2) # 重点在这行导致无网格
xx, yy = np.meshgrid(s_coarse,s_coarse)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )
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
plt.show()

# 3. 绘制xy平行面，若干平行平面
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d' )

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

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d',  )
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


# 5. 绘制xz平行面，无网格, 重点在这行导致无网格
xx, zz = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )

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

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )

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
yy, zz = np.meshgrid(s_fine, s_fine)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )
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
yy, zz = np.meshgrid(s_coarse,s_coarse)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )

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
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )

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

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d', )

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


#%% 可视化剖面线 BK_2_Ch15_05
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

# 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

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

xx_, yy_ = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-3, 3, 2))
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
# 绘制剖面
zz_ = np.zeros_like(xx_) + z_level
ax.plot_surface(xx_, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, yy_, zz_, color = 'b', lw = 0.2)

# 绘制网格曲面
ax.plot_wireframe(xx, yy, ff, color = [0.6, 0.6, 0.6], rstride = 5, cstride = 5, linewidth = 0.25)

# 绘制指定一条剖面线
ax.contour(xx, yy, ff,
           levels = [z_level],
           colors = 'r',
           linewidths = 1)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

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

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

# 绘制剖面
ax.plot_surface(xx_, xx_*0 + y_level, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(xx_, xx_*0 + y_level, zz_, color = 'b', lw = 0.2)

# 绘制曲面网格
ax.plot_wireframe(xx,yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)

# 绘制指定一条剖面线
x_array = np.linspace(-3,3,101)
y_array = x_array*0 + y_level
ax.plot(x_array, y_array, f_xy_fcn(x_array,y_array), color = 'r', lw = 1)

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

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_surface(yy_*0 + x_level, yy_, zz_, color = 'b', alpha = 0.1)
ax.plot_wireframe(yy_*0 + x_level, yy_, zz_, color = 'b', lw = 0.2)

ax.plot_wireframe(xx,yy, ff, color = [0.6, 0.6, 0.6], rstride=5, cstride=5, linewidth = 0.25)

y_array = np.linspace(-3,3,101)

# 绘制指定一条剖面线
x_array = y_array*0 + x_level
ax.plot(x_array, y_array, f_xy_fcn(x_array,y_array), color = 'r', lw = 1)

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


#%%  三维线图的平面填充, 填充曲线下方剖面 BK_2_Ch15_06
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
# 导入符号变量

from matplotlib import cm
# 导入色谱模块

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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_wireframe(xx1, xx2, PDF_ff, color = [0.5,0.5,0.5], rstride=0, cstride=2, linewidth = 0.25)

x1 = np.linspace(-3,3,101)
x2 = np.linspace(-3,3,101)
x1_loc_array = np.arange(0, len(x1), 10)
facecolors = cm.rainbow(np.linspace(0, 1, len(x1_loc_array)))

for idx in range(len(x1_loc_array)):
    x_loc = x1_loc_array[idx]
    x_idx = x1[x_loc]
    x_i_array = x2*0 + x_idx
    z_array = PDF_ff[:,x_loc]
    ax.plot(x_i_array, x2, z_array, color=facecolors[idx,:], linewidth = 1.5)
    ax.add_collection3d(plt.fill_between(x2, 0*z_array, z_array, color=facecolors[idx,:], alpha=0.2), # 给定填充对象
                        zs = x_idx, # 指定位置
                        zdir = 'x') # 指定方向
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
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx1, xx2, PDF_ff, color = [0.5,0.5,0.5], rstride=2, cstride=0, linewidth = 0.25)
x2_loc_array = np.arange(0,len(x1),10)
facecolors = cm.rainbow(np.linspace(0, 1, len(x2_loc_array)))
for idx in range(len(x2_loc_array)):
    x_loc = x2_loc_array[idx]
    x_idx = x2[x_loc]
    x_i_array = x1*0 + x_idx
    z_array = PDF_ff[x_loc,:]
    ax.plot(x1, x_i_array, z_array, color=facecolors[idx,:], linewidth = 1.5)
    ax.add_collection3d(plt.fill_between(x1, 0*z_array, z_array, color = facecolors[idx,:], alpha=0.2), zs=x_idx, zdir='y')

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


#%% 圆形薄膜振荡模式 BK_2_Ch15_07
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

def visualize(n = 4, m = 0, title = '4,0'):
    zz = np.array([displacement(n, m, rr, theta) for rr in r])
    fig = plt.figure( figsize = (8,8))
    ax = fig.add_subplot(121, projection='3d',  )
    surf = ax.plot_wireframe(xx, yy, zz, cstride = 50, rstride = 50, colors = '0.8', linewidth=0.25)
    ax.contour(xx, yy, zz, cmap='RdYlBu_r', levels = 15, linewidths=1)
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
    ax.contourf(xx, yy, zz, cmap = 'RdYlBu_r', levels = 15)
    ax.contour(xx, yy, zz, colors = 'w', levels = 15, linewidths = 0.25)

    ax.plot(np.cos(theta),np.sin(theta),'k')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.axis('off')
    plt.show()
visualize(4, 0, '4,0')

#%%=======================
# 1 表面图（Surface plots）
#======================
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
# 1. 定义函数¶
num = 301; # number of mesh grids
x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx, yy = np.meshgrid(x_array, y_array)

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx, yy)

## 1 用plot_surface() 绘制二元函数曲面
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho')
#  正交投影模式

surf = ax.plot_surface(xx, yy, ff, cmap = cm.RdYlBu, linewidth = 0, antialiased = False)
# 使用 RdYlBu 色谱
# 请大家试着调用其他色谱

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
# 设定横纵轴标签

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())
# 设定横、纵轴取值范围

ax.view_init(azim=-135, elev=30)
# 设定观察视角

ax.grid(False)
# 删除网格

fig.colorbar(surf, shrink=0.5, aspect=20)
plt.show()

## 2 翻转色谱
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho')
surf = ax.plot_surface(xx,yy,ff, cmap='RdYlBu_r', linewidth=0, antialiased=False)

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)

ax.grid(False)

fig.colorbar(surf, shrink=0.5, aspect=20)
plt.show()

## 3 只保留网格线, 同样使用 plot_surface()，不同的是只保留彩色网格(use facecolors)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(ff.min(), ff.max())
colors = cm.RdYlBu_r(norm_plt(ff))
# colors = cm.Blues_r(norm_plt(ff))

surf = ax.plot_surface(xx,yy,ff, facecolors = colors,
                       rstride = 5,
                       cstride = 5,
                       linewidth = 1, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 三维等高线
colorbar = ax.contour(xx,yy, ff, 20,  cmap = 'hsv')
# colorbar = ax.contour3D(xx,yy, ff, 20,  cmap = 'hsv')
# fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

# 二维等高线
# ax.contour(xx, yy, ff, zdir='z', offset= ff.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

ax.set_proj_type('ortho')
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

# 设置X、Y、Z面的背景是白色
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())
ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

## 4 plot_wireframe() 绘制网格曲面 + 三维等高线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot_wireframe(xx,yy, ff, color = [0.5,0.5,0.5], linewidth = 0.25)

# 三维等高线
# colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
# 三维等高线
colorbar = ax.contour(xx,yy, ff, 20,  cmap = 'hsv')
# fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

# 二维等高线
ax.contour(xx, yy, ff, zdir='z', offset= ff.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
ax.set_proj_type('ortho')

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

# 5. 绘制网格化散点
num = 70 # number of mesh grids
x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx,yy = np.meshgrid(x_array,y_array)

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx, yy)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

ax.plot_wireframe(xx,yy, ff,
                  color = [0.6,0.6,0.6],
                  rstride = 5, cstride=5,
                  linewidth = 0.25)
# ax.scatter(xx, yy, ff, c = ff, s = 10, cmap = 'RdYlBu_r')
ax.scatter(xx, yy, ff, c = 'blue', s = 1,  )

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/增加网格散点.svg', format='svg')
plt.show()


#6  用冷暖色表示函数的不同高度取值
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

surf = ax.plot_surface(xx,yy, ff,
                cmap=cm.RdYlBu_r,
                rstride=2, cstride=2,
                linewidth = 0.25,
                edgecolors = [0.5,0.5,0.5],
                shade = False) # 删除阴影 shade = False
# surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.view_init(azim=-135, elev=30)
plt.tight_layout()
ax.grid(False)
plt.show()
plt.close('all')































