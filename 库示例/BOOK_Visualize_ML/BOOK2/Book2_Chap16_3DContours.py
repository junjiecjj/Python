

#==========================================================================================================
##########################################  3D Contours, 三维等高线 ######################################
#==========================================================================================================

#%% 沿z方向空间等高线 Bk_2_Ch16_01
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量

from matplotlib import cm
# 导入色谱模块

# 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)

def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy
xx, yy = mesh(num = 101)

ff = f_xy_fcn(xx,yy)

# 2. 空间等高线，z方向
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
ax.contour(xx, yy, ff, levels = 20, cmap='RdYlBu_r')
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，z方向.svg', format='svg')
plt.show()

# 3. 空间等高线，z = 8
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# 绘制单色网格曲面
ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
# 绘制三维等高线
ax.contour(xx, yy, ff,
           zdir='z', offset=8,
           levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，z = 8.svg', format='svg')
plt.show()

# 4. 空间等高线，z = 0
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.contour(xx, yy, ff, zdir='z', offset=0, levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，z = 0.svg', format='svg')
plt.show()


# 5. 空间等高线，z = -8
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.contour(xx, yy, ff, zdir='z', offset=-8, levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，z = -8.svg', format='svg')
plt.show()

# 6. 空间填充等高线，z方向
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
# contour是线条，contourf是全彩色，这个是等高线的关键区别
# ax.contourf(xx, yy, ff, levels = 20, cmap='RdYlBu_r')
ax.contour(xx, yy, ff, levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间填充等高线，z方向.svg', format='svg')
plt.show()

# 7. 空间填充等高线，z = 8
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.contourf(xx, yy, ff, zdir='z', offset=8, levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间填充等高线，z = 8.svg', format='svg')
plt.show()


# 8. 空间填充等高线，z = 0
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
ax.contourf(xx, yy, ff, zdir='z', offset=0, levels = 20, cmap='RdYlBu_r' )
# ax.contour(xx, yy, ff, zdir='z', offset=0, levels = 20, cmap='RdYlBu_r')
# ax.contour(xx, yy, ff, zdir='z', offset=0, levels = 20, colors = 'k')
ax.contour(xx, yy, ff, zdir='z', offset=-8, levels = 10, colors = 'k',)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间填充等高线，z = 0.svg', format='svg')
plt.show()

# 9. 空间填充等高线，z = -8
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)

ax.contour(xx, yy, ff, zdir='z', offset=-8, levels = 10, colors = 'k',)
ax.contour(xx, yy, ff, zdir='z', offset=-8, levels = 20, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间填充等高线，z = -8.svg', format='svg')
plt.show()

#%% 沿x、y方向空间等高线 Bk_2_Ch16_02
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量

from matplotlib import cm
# 导入色谱模块

# 自定义函数
def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)
    return xx, yy

# 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy = 3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)
f_xy_fcn = lambdify([x, y], f_xy)
# 将符号函数表达式转换为Python函数
xx, yy = mesh(num = 121)
ff = f_xy_fcn(xx, yy)

# 2. 空间等高线，x方向
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# ax.plot_wireframe(xx,yy, ff,
#                   color = [0.8, 0.8, 0.8],
#                   rstride=2, cstride=2,
#                   linewidth = 0.75)
level_array = np.linspace(-3, 3, 30)
ax.contour(xx, yy, ff,
           zdir='x',
           levels = level_array,
           linewidths = 1,
           linestyles = '--',
           cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，x方向.svg', format='svg')
plt.show()

# 绘制剖面线
yy_, zz_ = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-8, 8, 2))
fig = plt.figure(figsize = (12, 20))
level_array = np.arange(-2.25, 2.25, 0.3)
for idx,level_idx in enumerate(level_array,1):
    ax = fig.add_subplot(5, 3, idx, projection = '3d')
    # 绘制剖面
    ax.plot_surface(yy_*0 + level_idx, yy_, zz_, color = 'b', alpha = 0.1)
    ax.plot_wireframe(yy_*0 + level_idx, yy_, zz_, color = 'b', lw = 0.2)
    ax.plot_wireframe(xx,yy, ff, color = [0.8, 0.8, 0.8], rstride=5, cstride=5,  linewidth = 0.25)
    ax.contour(xx, yy, ff, zdir='x', levels = [level_idx], linewidths = 1, linestyles = '--',)
    ax.set_proj_type('ortho')
    # 另外一种设定正交投影的方式
    ax.set_xlabel(r'$\it{x}$')
    ax.set_ylabel(r'$\it{y}$')
    ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(-8,8)
    ax.view_init(azim=-120, elev=30)
    ax.grid(False)
plt.show()

# 3. 空间等高线，x = 3
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff, color = [0.8, 0.8, 0.8], rstride=5, cstride=5, linewidth = 0.25)
ax.contour(xx, yy, ff, zdir='x', offset=3, levels = level_array, cmap='rainbow', linewidths = 1, linestyles = '--',)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，x = 3.svg', format='svg')
plt.show()

# 4. 空间等高线，x = 0
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff, color = [0.8, 0.8, 0.8], rstride=5, cstride=5, linewidth = 0.25)
# ax.contour(xx, yy, ff, zdir='x',  levels = level_array, cmap='rainbow', linewidths = 1, linestyles = '--',)
ax.contour(xx, yy, ff, zdir='x', offset=0, levels = level_array, cmap='rainbow', linewidths = 1, linestyles = '--',)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，x = 0.svg', format='svg')
plt.show()

# 5. 空间等高线，x = -3
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff, color = [0.8, 0.8, 0.8], rstride = 5, cstride = 5, linewidth = 0.25)
ax.contour(xx, yy, ff, zdir='x', offset=-3, levels = level_array, cmap='rainbow', linewidths = 1, linestyles = '--',)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，x = -3.svg', format='svg')
plt.show()

# 6. 空间等高线，y方向
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
level_array = np.linspace(-3,3,30)
ax.contour(xx, yy, ff, zdir='y', levels = level_array, cmap='rainbow', linewidths = 1, linestyles = '--',)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，y方向.svg', format='svg')
plt.show()

xx_, zz_ = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-8, 8, 2))
fig = plt.figure(figsize = (18, 8), constrained_layout=True)
level_array = np.arange(-2.25,2.25,0.3)
for idx, level_idx in enumerate(level_array,1):
    ax = fig.add_subplot(5, 3, idx, projection = '3d', )
    # 绘制剖面
    ax.plot_surface(xx_, xx_*0 + level_idx, zz_, color = 'b', alpha = 0.1)
    ax.plot_wireframe(xx_, xx_*0 + level_idx, zz_, color = 'red', lw = 0.2)

    ax.plot_wireframe(xx,yy, ff, color = [0.8, 0.8, 0.8], rstride=5, cstride=5, linewidth = 0.25)
    ax.contour(xx, yy, ff, zdir='y', levels = [level_idx], linewidths = 1, linestyles = '--',)
    ax.set_proj_type('ortho')
    # 另外一种设定正交投影的方式

    ax.set_xlabel(r'$\it{x}$')
    ax.set_ylabel(r'$\it{y}$')
    ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax.set_zlim(-8,8)
    ax.view_init(azim=-120, elev=30)
    ax.grid(False)
plt.show()
# fig.savefig('2.svg')

# 7. 空间等高线，y = 3
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
ax.contour(xx, yy, ff, zdir='y', offset=3, levels = level_array, cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，y = 3.svg', format='svg')
plt.show()

# 8. 空间等高线，y = 0
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx,yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
ax.contour(xx, yy, ff, zdir='y', offset=0, levels = level_array, cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，y = 0.svg', format='svg')
plt.show()

# 9. 空间等高线，y = -3
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(xx, yy, ff,
                  color = [0.8, 0.8, 0.8],
                  rstride=5, cstride=5,
                  linewidth = 0.25)
ax.contour(xx, yy, ff, zdir='y', offset=-3, levels = level_array, cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
ax.view_init(azim=-120, elev=30)
ax.grid(False)
# fig.savefig('Figures/空间等高线，y = -3.svg', format='svg')
plt.show()

#%% 沿x、y方向空间等高线在平面上投影 Bk_2_Ch16_03
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
from matplotlib import cm
# 导入色谱模块

def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)
    return xx, yy

# 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
xx, yy = mesh(num = 121)
ff = f_xy_fcn(xx, yy)

# 2. 在xz平面投影
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

level_array = np.linspace(-3,3,61)
ax.contour(xx, yy, ff, zdir='y', levels = level_array, cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
# ax.view_init(azim=-90, elev=0)
ax.grid(False)
plt.show()

x1_array  = np.linspace(-3, 3, 200)
x2_slices = np.linspace(-3,3, 5)
num_lines = len(x2_slices)
colors = cm.rainbow(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号

fig, ax = plt.subplots(figsize = (5,4))

for idx, x2_idx in enumerate(x2_slices):
    ff_idx = f_xy_fcn(x1_array,x1_array*0 + x2_idx)
    legend_idx = '$x_2$ = ' + str(x2_idx)
    plt.plot(x1_array, ff_idx, color=colors[idx], label = legend_idx)
    # 依次绘制概率密度曲线
# plt.show()
plt.legend()
# 增加图例

plt.xlim(x1_array.min(),x1_array.max())
# plt.ylim(-8,8)
plt.xlabel('$x$')
plt.ylabel('$f(x,y)$')
# fig.savefig('Figures/在xz平面投影_for循环.svg', format='svg')
plt.show()

# 3. 在yz平面投影
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
level_array = np.linspace(-3,3,61)
ax.contour(xx, yy, ff, zdir='x', levels = level_array, cmap='rainbow')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(-8,8)
# ax.view_init(azim=0, elev=0)
ax.grid(False)
# fig.savefig('Figures/在yz平面投影.svg', format='svg')
plt.show()

########
x2_array  = np.linspace(-3, 3, 200)
x1_slices = np.linspace(-3,3,6*10 + 1)

num_lines = len(x1_slices)
colors = cm.rainbow(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号
fig, ax = plt.subplots(figsize = (5,4))
for idx, x1_idx in enumerate(x1_slices):
    ff_idx = f_xy_fcn(x2_array*0 + x1_idx,x2_array)
    legend_idx = '$x_1$ = ' + str(x1_idx)
    plt.plot(x2_array, ff_idx, color=colors[idx], label = legend_idx)

# plt.legend()
# 增加图例
plt.xlim(x2_array.min(),x2_array.max())
# plt.ylim(-8,8)
plt.xlabel('$y$')
plt.ylabel('$f(x,y)$')

plt.show()

#%% 利用极坐标产生等高线坐标 Bk_2_Ch16_04
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex, simplify
from sympy import symbols
# 导入符号变量

from matplotlib import cm
# 导入色谱模块

def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-1.2,1.2,num)
    y_array = np.linspace(-1.2,1.2,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

# 1. 定义二次型
x1, x2 = symbols('x1 x2')
# 自定义函数计算二次型函数值
def quadratic(Q, xx1, xx2):
    x = np.array([[x1],
                  [x2]])

    # 二次型，符号
    f_x1x2 = x.T @ Q @ x

    f_x1x2_fcn = lambdify([x1, x2], f_x1x2[0][0])
    # 将符号函数表达式转换为Python函数

    ff = f_x1x2_fcn(xx1, xx2)
    # 计算二元函数函数值

    return ff, simplify(f_x1x2[0][0])

# 2. 自定义可视化函数
def visualize(Q, title):

    xx1, xx2 = mesh(num = 201)
    ff,f_x1x2 = quadratic(Q, xx1, xx2)

    ### 单位圆坐标
    theta_array = np.linspace(0, 2*np.pi, 100)
    x1_circle = np.cos(theta_array)
    x2_circle = np.sin(theta_array)

    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(xx1, xx2, ff, 15, cmap='RdYlBu_r')
    ax.plot(x1_circle, x2_circle, color = 'k')
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(xx1, xx2, ff,
                      color = [0.5,0.5,0.5],
                      rstride=10, cstride=10,
                      linewidth = 0.25)
    ax.contour(xx1, xx2, ff,cmap = 'RdYlBu_r', levels = 15)
    f_circle, _ = quadratic(Q, x1_circle, x2_circle)
    ax.plot(x1_circle, x2_circle, f_circle, color = 'k')

    ax.set_proj_type('ortho')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([])
    ax.view_init(azim=-120, elev=30)
    # ax.view_init(azim=-135, elev=60)
    plt.tight_layout()
    ax.grid(False)

    # fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()
    return f_x1x2

# 3. 开口朝上正椭圆面
Q = np.array([[4,0],
              [0,1]])
f_x1x2 = visualize(Q, '开口朝上正椭圆面')
# f_x1x2

# 4. 开口朝上旋转椭圆面
Q = np.array([[2,-1],
              [-1,2]])
f_x1x2 = visualize(Q, '开口朝上旋转椭圆面')

# 5. 开口朝下正椭圆面
Q = np.array([[-4,0],
              [0,-1]])
f_x1x2 = visualize(Q, '开口朝下正椭圆面')

# 6. 开口朝下旋转椭圆面
Q = np.array([[-2,-1],
              [-1,-2]])
f_x1x2 = visualize(Q, '开口朝下旋转椭圆面')

# 7. 旋转山谷
Q = np.array([[1,-1],
              [-1,1]])
f_x1x2 = visualize(Q, '旋转山谷')

# 8. 旋转山脊
Q = np.array([[-1,1],
              [1,-1]])
f_x1x2 = visualize(Q, '旋转山脊')

# 9. 双曲面
Q = np.array([[1,0],
                [0,-1]])
f_x1x2 = visualize(Q, '双曲面')

# 10. 旋转双曲面
Q = np.array([[0,1],
              [1,0]])
f_x1x2 = visualize(Q, '旋转双曲面')

#%% 提取等高线坐标 Bk_2_Ch16_05
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex, symbols
x1, x2 = symbols('x1 x2')
# 导入符号变量

from matplotlib import cm

def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

###########  1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_x1x2 =  3*(1-x1)**2*exp(-(x1**2) - (x2+1)**2) - 10*(x1/5 - x1**3 - x2**5)*exp(-x1**2-x2**2)  - 1/3*exp(-(x1+1)**2 - x2**2)

f_x1x2_fcn = lambdify([x1,x2],f_x1x2)
# 将符号函数表达式转换为Python函数
xx1, xx2 = mesh(num = 201)
ff = f_x1x2_fcn(xx1, xx2)

# 2. 计算  𝑓(𝑥1,𝑥2) 对  𝑥1 一阶偏导
df_dx1 = f_x1x2.diff(x1)
df_dx1_fcn = lambdify([x1,x2],df_dx1)
df_dx1_zz = df_dx1_fcn(xx1,xx2)

###########  3. 定位  ∂𝑓(𝑥1,𝑥2)/∂𝑥1=0
fig, ax = plt.subplots()

colorbar = ax.contourf(xx1, xx2, df_dx1_zz, 20, cmap='turbo')
ax.contour(xx1, xx2, df_dx1_zz, levels = [0], colors = 'k')
# 黑色线代表偏导为 0

fig.colorbar(colorbar, ax=ax)
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.gca().set_aspect('equal', adjustable='box')
# fig.savefig('Figures/对x1偏导.svg', format='svg')
plt.show()

###########  4. 将  ∂𝑓(𝑥1,𝑥2)/∂𝑥1=0 映射到  𝑓(𝑥1,𝑥2) 曲面上
fig, ax = plt.subplots()

colorbar = ax.contourf(xx1, xx2, ff, 20, cmap='RdYlBu_r')
ax.contour(xx1, xx2, df_dx1_zz, levels = [0], colors = 'k')

fig.colorbar(colorbar, ax=ax)
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.gca().set_aspect('equal', adjustable='box')
# fig.savefig('Figures/对x1偏导为0映射到f(x1,x2).svg', format='svg')
plt.show()


###########  5. 计算  𝑓(𝑥1,𝑥2) 对 𝑥2 一阶偏导
df_dx2 = f_x1x2.diff(x2)
df_dx2_fcn = lambdify([x1,x2],df_dx2)
df_dx2_zz = df_dx2_fcn(xx1,xx2)

###########  6. 定位  ∂𝑓(𝑥1,𝑥2)∂𝑥2=0
fig, ax = plt.subplots()

colorbar = ax.contourf(xx1, xx2, df_dx2_zz, 20, cmap='turbo')
ax.contour(xx1, xx2, df_dx2_zz, levels = [0], colors = 'k')
# 黑色线代表偏导为 0

fig.colorbar(colorbar, ax=ax)
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.gca().set_aspect('equal', adjustable='box')
# fig.savefig('Figures/对x2偏导.svg', format='svg')
plt.show()


###########  7. 将  ∂𝑓(𝑥1,𝑥2)∂𝑥2=0 映射到  𝑓(𝑥1,𝑥2) 曲面上
fig, ax = plt.subplots()

colorbar = ax.contourf(xx1, xx2, ff, 20, cmap='RdYlBu_r')
ax.contour(xx1, xx2, df_dx2_zz, levels = [0], colors = 'k')

fig.colorbar(colorbar, ax=ax)
ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.gca().set_aspect('equal', adjustable='box')
# fig.savefig('Figures/对x2偏导为0映射到f(x1,x2).svg', format='svg')
plt.show()


###########  提取等高线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
CS_x = ax.contour(xx1, xx2, df_dx2_zz, levels = [0])
ax.cla()
ax.plot_wireframe(xx1, xx2, ff, color = [0.5,0.5,0.5], rstride=5, cstride=5, linewidth = 0.25)
colorbar = ax.contour(xx1, xx2, ff, 20, cmap = 'RdYlBu_r')

# 在 for 循环中，分别提取等高线数值
for i in range(0,len(CS_x.allsegs[0])):
    contour_points_x_y = CS_x.allsegs[0][i]
    # 计算黑色等高线对应的 f(x1,x2) 值
    contour_points_z = f_x1x2_fcn(contour_points_x_y[:,0],  contour_points_x_y[:,1])
    # 绘制映射结果
    ax.plot(contour_points_x_y[:,0],
            contour_points_x_y[:,1],
            contour_points_z,
            color = 'k',
            linewidth = 1)
ax.set_proj_type('ortho')

ax.set_xlim(xx1.min(), xx1.max())
ax.set_ylim(xx2.min(), xx2.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$f(x_1,x_2)$')

ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-135, elev=60)
plt.tight_layout()
ax.grid(False)
# fig.savefig('Figures/对x2偏导为0映射到f(x1,x2)，三维曲面.svg', format='svg')
plt.show()


#%% Bk_2_Ch16_06 # 绘制交线
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y

from matplotlib import cm
# 导入色谱模块
def mesh(num = 101):
    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

### 1. 定义符号函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)
f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
xx, yy = mesh(num = 401)
f_xy_zz = f_xy_fcn(xx, yy)

### 2. 曲面和平面交线
# x + y + 1 = 0
linear_eq = xx + yy + 1
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
gap = f_xy_zz - linear_eq
# 两个曲面之差
CS_x = ax.contour(xx,yy, gap, levels = [0], colors = '#339933')
# 两个曲面之差为0处，即交线位置
ax.cla()
# 清空图片

norm_plt = plt.Normalize(f_xy_zz.min(), f_xy_zz.max())
colors = cm.RdYlBu_r(norm_plt(f_xy_zz))

surf = ax.plot_surface(xx,yy,f_xy_zz, facecolors=colors, rstride=5, cstride=5, linewidth=0.25, shade=False) # 删除阴影
surf.set_facecolor((0,0,0,0))
ax.plot_wireframe(xx,yy, linear_eq, color = 'k', rstride=5, cstride=5, linewidth = 0.25)
# 分段绘制交线
for i in range(0,len(CS_x.allsegs[0])):
    contour_points_x_y = CS_x.allsegs[0][i]
    contour_points_z = f_xy_fcn(contour_points_x_y[:,0], contour_points_x_y[:,1])
    ax.plot3D(contour_points_x_y[:,0], contour_points_x_y[:,1], contour_points_z, color = 'k', linewidth = 1)
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/曲面和平面交线_3D.svg', format='svg')


fig, ax = plt.subplots()
colorbar = ax.contourf(xx,yy, f_xy_zz, 20, cmap='RdYlBu_r')
ax.contour(xx,yy, gap, levels = [0], colors = 'k')
# fig.colorbar(colorbar, ax=ax)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

plt.gca().set_aspect('equal', adjustable='box')
# fig.savefig('Figures/曲面和平面交线_平面.svg', format='svg')


#%% # 可视化三元高斯分布 Bk_2_Ch16_07, 参见切豆腐:Bk_2_Ch16_07, Bk2_Ch21_02, BK_2_Ch25_04
# 导入包
import matplotlib.pyplot as plt
import numpy as np

### 1. 自定义高斯分布概率密度函数,参见Book5, 第一章(test.6)多元正态分布的概率密度函数 PDF.
def Mahal_d_2_pdf(d, Sigma):
    # 将马氏距离转化为概率密度
    scale_1 = np.sqrt(np.linalg.det(Sigma))
    scale_2 = (2*np.pi)**(3/2)
    pdf = np.exp(-d**2/2)/scale_1/scale_2
    return pdf

def Mahal_d(Mu, Sigma, x):
    # 计算马哈距离
    x_demeaned = x - Mu
    # 中心化
    inv_covmat = np.linalg.inv(Sigma)
    # 矩阵逆
    left = np.dot(x_demeaned, inv_covmat)
    mahal = np.dot(left, x_demeaned.T)
    return np.sqrt(mahal).diagonal()

### 2. 生成数据
x1 = np.linspace(-2,2,30)
x2 = np.linspace(-2,2,30)
x3 = np.linspace(-2,2,30)

xxx1,xxx2,xxx3 = np.meshgrid(x1,x2,x3)
Mu = np.array([[0, 0, 0]])
Sigma = np.array([[1, 0.6, -0.4],
                  [0.6, 1.5, 1],
                  [-0.4, 1, 2]])
x_array = np.vstack([xxx1.ravel(),xxx2.ravel(), xxx3.ravel()]).T
# 首先计算马氏距离
d_array = Mahal_d(Mu, Sigma, x_array)
d_array = d_array.reshape(xxx1.shape)

# 将马氏距离转化成概率密度PDF
pdf_zz = Mahal_d_2_pdf(d_array, Sigma)

# 设定统一等高线分层
# levels = np.linspace(0,pdf_zz.max(),20)
levels_PDF = np.linspace(0, 0.1, 22)
levels_Mahal_D = np.linspace(0, 10, 22)

### 3. 绘制概率密度箱体外立面
# 定义等高线高度
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
# 绘制三维等高线，填充
ax.contourf(xxx1[:, :, -1],
            xxx2[:, :, -1],
            pdf_zz[:, :, -1],
            levels = levels_PDF,
            zdir='z', offset=xxx3.max(),
            cmap = 'turbo') # RdYlBu_r

ax.contour(xxx1[:, :, -1],
            xxx2[:, :, -1],
            pdf_zz[:, :, -1],
            levels = levels_PDF,
            zdir='z', offset=xxx3.max(),
            linewidths = 0.25,
            colors = '1')

ax.contourf(xxx1[0, :, :],
            pdf_zz[0, :, :],
            xxx3[0, :, :],
            levels = levels_PDF,
            zdir='y',
            cmap = 'turbo',
            offset=xxx2.min())

ax.contour(xxx1[0, :, :],
            pdf_zz[0, :, :],
            xxx3[0, :, :],
            levels = levels_PDF,
            zdir='y',
            colors = '1',
            linewidths = 0.25,
            offset=xxx2.min())

CS = ax.contourf(pdf_zz[:, 0, :],
            xxx2[:, 0, :],
            xxx3[:, 0, :],
            levels = levels_PDF,
            cmap = 'turbo',
            zdir='x',
            offset=xxx1.min())

ax.contour(pdf_zz[:, 0, :],
            xxx2[:, 0, :],
            xxx3[:, 0, :],
            levels = levels_PDF,
            zdir='x',
            colors = '1',
            linewidths = 0.25,
            offset=xxx1.min())

fig.colorbar(CS, ticks=np.linspace(0,0.1,6))
# Set limits of the plot from coord limits
xmin, xmax = xxx1.min(), xxx1.max()
ymin, ymax = xxx2.min(), xxx2.max()
zmin, zmax = xxx3.min(), xxx3.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# 绘制框线
edges_kw = dict(color='0.6', linewidth=1, zorder=1e5)
# zorder 控制呈现 artist 的先后顺序
# zorder 越小，artist 置于越底层
# zorder 赋值很大的数，这样确保 zorder 置于最顶层
ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.set_xticks([-2,0,2])
ax.set_yticks([-2,0,2])
ax.set_zticks([-2,0,2])
ax.view_init(azim=-125, elev=30)
ax.set_proj_type('ortho')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.set_box_aspect((1, 1, 1))
ax.grid(False)

### 4. 分层绘制等高线，沿x3
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

for idx in np.arange(0,len(x3),5):
    x3_idx = x3[idx]
    ax.contourf(xxx1[:, :, idx],
                xxx2[:, :, idx],
                pdf_zz[:, :, idx],
                levels = levels_PDF,
                zdir='z',
                offset=x3_idx,
                cmap = 'turbo')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)

ax.set_xticks([-2,0,2])
ax.set_yticks([-2,0,2])
ax.set_zticks([-2,0,2])
ax.view_init(azim=-125, elev=30)
ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_box_aspect((1, 1, 1))
ax.grid(False)
# fig.savefig('Figures/箱体分层x3_概率密度.svg', format='svg')


### 5. 将等高线展开，沿x3
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x3),5)):
    ax = fig.add_subplot(len(np.arange(0,len(x3),5)), 1, fig_idx + 1, projection='3d')
    x3_idx = x3[idx]
    ax.contourf(xxx1[:, :, idx],
                xxx2[:, :, idx],
                pdf_zz[:, :, idx],
                levels = levels_PDF,
                zdir='z',
                offset=x3_idx,
                cmap = 'turbo')
    ax.contour(xxx1[:, :, idx],
                xxx2[:, :, idx],
                pdf_zz[:, :, idx],
                levels = levels_PDF,
                zdir='z',
                offset=x3_idx,
               linewidths = 0.25,
                colors = '1')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # 绘制框线
    edges_kw = dict(color='0.5', linewidth=1, zorder=1e3)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)

    ax.view_init(azim=-125, elev=30)
    ax.set_box_aspect(None)
    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
# fig.savefig('Figures/分层分图_x3_概率密度.svg', format='svg')

### 6. 将等高线展开，沿x2
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x2),5)):
    ax = fig.add_subplot(len(np.arange(0,len(x2),5)), 1, fig_idx + 1, projection='3d')
    x2_idx = x2[idx]
    ax.contourf(xxx1[idx, :, :],
                pdf_zz[idx, :, :],
                xxx3[idx, :, :],
                levels = levels_PDF,
                zdir='y',
                offset=x2_idx,
                cmap = 'turbo')
    ax.contour(xxx1[idx, :, :],
                pdf_zz[idx, :, :],
                xxx3[idx, :, :],
                levels = levels_PDF,
                zdir='y',
                offset=x2_idx,
               linewidths = 0.25,
                colors = '1')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    # Plot edges
    edges_kw = dict(color='0.5', linewidth=1, zorder=1e3)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)
    # Set zoom and angle view
    ax.view_init(azim=-125, elev=30)
    ax.set_box_aspect(None)
    ax.set_proj_type('ortho')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
# fig.savefig('Figures/分层分图_x2_概率密度.svg', format='svg')















































