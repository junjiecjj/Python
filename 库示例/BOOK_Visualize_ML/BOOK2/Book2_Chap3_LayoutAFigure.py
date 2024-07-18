




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch03_01 图片对象规格



import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, exp, diff

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")


plt.rcParams  # 调取所有默认绘图设置
print(plt.rcParams.get('figure.figsize')) # 调取图片默认大小


# 设置图片参数
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5



# 定义函数
x, y = symbols("x, y")
f = exp(-x**2)
f_fcn = lambdify(x, f)

x_array = np.linspace(-3,3,100)
f_array = f_fcn(x_array)

df_dx = diff(f, x)
df_dx_fcn = lambdify(x, df_dx)
df_dx_array = df_dx_fcn(x_array)



# 一元函数图像
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_array, f_array)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.2)
# fig.savefig('Figures/默认大小.svg', format='svg')
plt.show()






# 图片大小
fig = plt.figure(figsize = (3, 2))
# 图片宽度：3 inches；图片高度：2 inches
ax = fig.add_subplot(1, 1, 1)
# 或者 111
ax.plot(x_array, f_array)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.2)
plt.xticks(fontname = "Times New Roman")
plt.xticks(fontsize = "8")
plt.yticks(fontname = "Times New Roman")
plt.yticks(fontsize = "8")
plt.show()
# fig.savefig('3 inch x 2 inch.svg', format='svg')




cm = 1/2.54
# 将厘米 cm 转化为 inch

fig = plt.figure(figsize = (9 * cm, 6 * cm))
# 图片宽度：9 厘米；图片高度：6 厘米
ax = fig.add_subplot(1, 1, 1)
# 或者 111
ax.plot(x_array, f_array)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.2)

# fig.savefig('Figures/9 cm x 6 cm.svg', format='svg')



# 调整边距
fig, ax = plt.subplots()

# 调整边距
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

# 显示图像
plt.show()




# 轴对象位置、大小
fig = plt.figure()
ax1 = fig.add_axes([0.5,0.5,0.5,0.5])
ax2 = fig.add_axes([0.25, 0.25, 0.5, 0.5])

# 2行1列
fig = plt.figure()
ax_1 = fig.add_subplot(2, 1, 1)
ax_1.plot(x_array, f_array)
ax_1.set_xlim(-3, 3)
ax_1.set_ylim(0, 1.2)

ax_2 = fig.add_subplot(2, 1, 2)
ax_2.plot(x_array, df_dx_array)
ax_2.set_xlim(-3, 3)
ax_2.set_ylim(-1, 1)

# fig.savefig('Figures/两行一列.svg', format='svg')




fig = plt.figure()

ax1 = fig.add_axes([0.125,0.11 + 0.42,.775,.35])
ax1.plot(x_array, f_array)
ax1.set_xlim(-3, 3)
ax1.set_ylim(0, 1.2)

ax2 = fig.add_axes([0.125,0.11,.775,.35])
ax2.plot(x_array, df_dx_array)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-1, 1)


# 1行2列
fig = plt.figure()
ax_1 = fig.add_subplot(1,2,1)
ax_1.plot(x_array, f_array)
ax_1.set_xlim(-3, 3)
ax_1.set_ylim(-1, 1)

ax_2 = fig.add_subplot(1,2,2)
ax_2.plot(x_array, df_dx_array)
ax_2.set_xlim(-3, 3)
ax_2.set_ylim(-1, 1)

# fig.savefig('一行两列.svg', format='svg')


# 二元函数
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2)\
    - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2)\
    - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数

# Reference:
# https://www.mathworks.com/help/matlab/ref/peaks.html

## 自定义mesh函数
def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

# 填充等高线
xx, yy = mesh(num = 6*100 + 1)
ff = f_xy_fcn(xx,yy)

fig, ax = plt.subplots(figsize = (3,3))

ax.contourf(xx, yy, ff,
           levels = np.linspace(-8,9,18),
           cmap = 'RdYlBu_r')

# ax.set_xlabel('$\it{x_1}$', fontsize=8)
# ax.set_ylabel('$\it{x_2}$', fontsize=8)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
# plt.xticks(fontname = "Times New Roman")
# plt.xticks(fontsize = "8")
# plt.yticks(fontname = "Times New Roman")
# plt.yticks(fontsize = "8")
ax.set_aspect('equal', adjustable='box')

# fig.savefig('Figures/3 inch X 3 inch.svg', format='svg')



# 图中图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_array, f_array)
ax.set_xlim(-3, 3)
ax.set_ylim(0, 1.2)

ax_zoom = fig.add_axes([0.65,0.6,0.2, 0.2])
ax_zoom.plot(x_array, df_dx_array)
ax_zoom.set_xlim(-3, 3)
ax_zoom.set_ylim(-1, 1)
ax_zoom.set_xticks(np.arange(-3,4))
ax_zoom.set_yticks(np.arange(-3,4))
plt.show()
# fig.savefig('Figures/图中图.svg', format='svg')






fig, ax = plt.subplots()

ax.contourf(xx, yy, ff,
           levels = np.linspace(-8,9,18),
           cmap = 'RdYlBu_r')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_aspect('equal', adjustable='box')

ax_zoom = fig.add_axes([0.6, 0.6,0.3, 0.3], projection='3d')
ax_zoom.plot_wireframe(xx,yy, ff,
                  color = [0.6, 0.6, 0.6],
                  rstride=30, cstride=30,
                  linewidth = 0.25)

ax_zoom.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax_zoom.set_xlim(xx.min(), xx.max())
ax_zoom.set_ylim(yy.min(), yy.max())
ax_zoom.set_zlim(-8, 8)
ax_zoom.patch.set_alpha(0.6)
ax_zoom.view_init(azim=-135, elev=30)
ax_zoom.grid(False)
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])
ax_zoom.set_zticks([])
plt.show()
# fig.savefig('Figures/三维图中图.svg', format='svg')









#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk_2_Ch03_02  混合二维、三维可视化方案¶
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, lambdify, exp, diff

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")



x, y = symbols("x, y")
# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2)\
    - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2)\
    - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数

# Reference:
# https://www.mathworks.com/help/matlab/ref/peaks.html


def mesh(num = 101):

    # number of mesh grids
    x_array = np.linspace(-3,3,num)
    y_array = np.linspace(-3,3,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy


xx, yy = mesh(num = 6*100 + 1)
ff = f_xy_fcn(xx,yy)



# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=(5,10))

ax = fig.add_subplot(2, 1, 1, projection = '3d')

ax.plot_wireframe(xx, yy, ff, color = '0.8', lw = 0.2)

ax.contour(xx, yy, ff,
           levels = np.linspace(-8,9,18),
           cmap = 'RdYlBu_r', linewidths = 1)

ax.set_proj_type('ortho')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

ax = fig.add_subplot(2, 1, 2)

ax.contourf(xx, yy, ff,
           levels = np.linspace(-8,9,18),
           cmap = 'RdYlBu_r')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')

# fig.savefig('Figures/add_subplot混合二维、三维.svg', format='svg')



import matplotlib.pyplot as plt
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

fig = plt.figure(figsize=(5,10))
ax_2D = fig.add_subplot(3, 1, 1, aspect = 1)
# ax_2D.grid()

ax_3D = fig.add_subplot(3, 1, 2, projection = '3d')
ax_3D.set_proj_type('ortho')

ax_polar = fig.add_subplot(3, 1, 3, projection = 'polar')

# fig.savefig('Figures/三种不同projections.svg', format='svg')





































































































































































































