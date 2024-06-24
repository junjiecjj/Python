#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:05:06 2024

@author: jack
 Chapter 22 隐函数 | Book 2《可视之美》


"""




#%% 绘制线段
# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


x = np.linspace(-4, 4, num = 1001)
y = np.linspace(-4, 4, num = 1001)

xx, yy = np.meshgrid(x, y);

# 绘制 x + y = c
fig, ax = plt.subplots(figsize=(5, 5))
levels = np.arange(-6, 6 + 1)
CS = plt.contour(xx, yy, xx + yy,
            levels = levels,
            cmap = 'rainbow',
            inline = True)

ax.clabel(CS, inline=True, fontsize=10)

ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()

# fig.savefig('Figures/直线，1.svg', format='svg')
plt.show()



#%% 绘制抛物线
# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


x = np.linspace(-4,4,num = 1001)
y = np.linspace(-4,4,num = 1001)

xx,yy = np.meshgrid(x,y);


# 绘制 x - y**2 = c
fig, ax = plt.subplots(figsize=(5, 5))
levels = np.arange(-4,3 + 1)
CS = plt.contour(xx,yy,xx - yy**2,
            levels = levels,
            cmap = 'rainbow',
            inline = True)

ax.clabel(CS, inline=True, fontsize=10)

ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()

# fig.savefig('Figures/抛物线，4.svg', format='svg')
plt.show()




#%% 离心率可视化一组圆锥曲线

# 导入包
import matplotlib.pyplot as plt
import numpy as np

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")


# 产生数据
x = np.linspace(-4, 4, num = 1001)
y = np.linspace(-4, 4, num = 1001)

xx,yy = np.meshgrid(x, y);

# 一组离心率取值
e_array = np.linspace(0, 3, num = 51)


# 离心率绘制椭圆
# 𝑦2−(𝑒2−1)𝑥2−2𝑥=0, 其中， 𝑒 为离心率

fig, ax = plt.subplots(figsize=(5, 5))

colors = plt.cm.rainbow(np.linspace(0,1,len(e_array)))
# 利用色谱生成一组渐变色，颜色数量和 e_array 一致

for i in range(0,len(e_array)):

    e = e_array[i]

    ellipse = yy**2 - (e**2 - 1)*xx**2 - 2*xx;

    color_code = colors[i,:].tolist()

    plt.contour(xx,yy,ellipse,levels = [0], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/圆锥曲线，随离心率变化.svg', format='svg')
plt.show()


#%% 用等高线绘制几何体
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 0. 可视化隐函数
def plot_implicit(fn, X_plot, Y_plot, Z_plot, ax, bbox):

    # 等高线的起止范围
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    ax.set_proj_type('ortho')

    # 绘制三条参考线
    k = 1.5
    ax.plot((xmin * k, xmax * k), (0, 0), (0, 0), 'k')
    ax.plot((0, 0), (ymin * k, ymax * k), (0, 0), 'k')
    ax.plot((0, 0), (0, 0), (zmin * k, zmax * k), 'k')

    # 等高线的分辨率
    A = np.linspace(xmin, xmax, 500)
    # 产生网格数据
    A1, A2 = np.meshgrid(A, A)

    # 等高线的分割位置
    B = np.linspace(xmin, xmax, 20)

    # 绘制 XY 平面等高线
    if X_plot == True:
        for z in B:
            X, Y = A1, A2
            Z = fn(X, Y, z)
            cset = ax.contour(X, Y, Z+z, [z],
                              zdir='z',
                              linewidths = 0.25,
                              colors = '#0066FF',
                              linestyles = 'solid')

    # 绘制 XZ 平面等高线
    if Y_plot == True:
        for y in B:
            X,Z = A1,A2
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y],
                              zdir='y',
                              linewidths = 0.25,
                              colors = '#88DD66',
                              linestyles = 'solid')

    # 绘制 YZ 平面等高线
    if Z_plot == True:
        for x in B:
            Y, Z = A1, A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x],
                              zdir='x',
                              linewidths = 0.25,
                              colors = '#FF6600',
                              linestyles = 'solid')

    ax.set_zlim(zmin * k, zmax * k)
    ax.set_xlim(xmin * k, xmax * k)
    ax.set_ylim(ymin * k, ymax * k)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(azim=-120, elev=30)
    ax.axis('off')
    # plt.show()
    return

def visualize_four_ways(fn, title, bbox=(-2.5,2.5)):

    fig = plt.figure(figsize=(20, 8))

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    plot_implicit(fn, True, False, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    plot_implicit(fn, False, True, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    plot_implicit(fn, False, False, True, ax, bbox)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    plot_implicit(fn, True, True, True, ax, bbox)

    # fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()
    return




# 1. 单位球
def unit_sphere(x,y,z):
    return x**2 + y**2 + z**2 - 1

visualize_four_ways(unit_sphere, '单位球', bbox = (-1,1))



# 2. 椭球
# Ellipsoid
def Ellipsoid(x,y,z):
    a = 1
    b = 2
    c = 1
    return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1

visualize_four_ways(Ellipsoid, '椭球', bbox = (-2,2))



# 3. 双曲抛物面
# 双曲抛物面是一个二次曲面，其形状像一个双曲面和抛物面的组合。
# 𝑥2𝑎2−𝑦2𝑏2−𝑧=0
# Hyperbolic_paraboloid
def Hyperbolic_paraboloid(x,y,z):
    a = 1
    b = 1
    return x**2/a**2 - y**2/b**2 - z

visualize_four_ways(Hyperbolic_paraboloid, '双曲抛物面', bbox = (-2,2))




# 4. 旋转双曲抛物面:𝑥𝑦−𝑧=0
# Hyperbolic_paraboloid, rotated
def Hyperbolic_paraboloid_rotated(x,y,z):
    return x*y - z

visualize_four_ways(Hyperbolic_paraboloid_rotated, '旋转双曲抛物面', bbox = (-2,2))


# 5A. 正圆抛物面，开口朝上
# 𝑥2+𝑦2−𝑧−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + y**2 - 2 - z

visualize_four_ways(circular_paraboloid, '正圆抛物面，开口朝上', bbox = (-2,2))


# 5B. 正圆抛物面，开口朝下
# 𝑥2+𝑦2+𝑧−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + y**2 - 2 + z

visualize_four_ways(circular_paraboloid, '正圆抛物面，开口朝下', bbox = (-2,2))



# 5C. 正圆抛物面，x轴
# 𝑦2+𝑧2−𝑥−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return y**2 + z**2 - 2 - x

visualize_four_ways(circular_paraboloid, '正圆抛物面，开口沿x轴', bbox = (-2,2))

# 5C. 正圆抛物面，y轴
# 𝑥2+𝑧2−𝑦−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + z**2 - 2 - y

visualize_four_ways(circular_paraboloid, '正圆抛物面，开口沿y轴', bbox = (-2,2))




# 6A. 单叶双曲面，z轴
# 𝑥2+𝑦2−𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return x**2 + y**2 - z**2 - 2

visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，z轴', bbox = (-4,4))




# 6B. 单叶双曲面，y轴
# 𝑥2−𝑦2+𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return x**2 - y**2 + z**2 - 2

visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，y轴', bbox = (-4,4))



# 6C. 单叶双曲面，x轴
# −𝑥2+𝑦2+𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return - x**2 + y**2 + z**2 - 2

visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，x轴', bbox = (-4,4))


# 7A. 双叶双曲面，z轴
# 𝑥2+𝑦2−𝑧2+1=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return x**2 + y**2 - z**2 + 1

visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，z轴', bbox = (-4,4))



# 7B. 双叶双曲面，y轴
# 𝑥2−𝑦2+𝑧2+2=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return x**2 - y**2 + z**2 + 2

visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，y轴', bbox = (-4,4))



# 7C. 双叶双曲面，x轴
# −𝑥2+𝑦2+𝑧2+1=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return - x**2 + y**2 + z**2 + 1

visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，x轴', bbox = (-4,4))



# 8A. 圆锥面，z轴
# 𝑥2+𝑦2−𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return x**2 + y**2 - z**2

visualize_four_ways(Circular_cone, '圆锥面', bbox = (-4, 4))



# 8B. 圆锥面，y轴
# 𝑥2−𝑦2+𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return x**2 - y**2 + z**2

visualize_four_ways(Circular_cone, '圆锥面_y_轴', bbox = (-4, 4))



# 8C. 圆锥面，x轴
# −𝑥2+𝑦2+𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return -x**2 + y**2 + z**2

visualize_four_ways(Circular_cone, '圆锥面_x_轴', bbox = (-4, 4))



# 9A. 圆柱面，z轴
# 𝑥2+𝑦2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + y**2 - 1

visualize_four_ways(Circular_cylinder, '圆柱面，z轴', bbox = (-1,1))




# 9B. 圆柱面，y轴
# 𝑥2+𝑧2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，y轴', bbox = (-1,1))

#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，y轴', bbox = (-1,1))






# 9C. 圆柱面，x轴
# 𝑦2+𝑧2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return y**2 + z**2 - 1

visualize_four_ways(Circular_cylinder, '圆柱面，x轴', bbox = (-1,1))
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return y**2 + z**2 - 1

visualize_four_ways(Circular_cylinder, '圆柱面，x轴', bbox = (-1,1))



# 10. 古尔萨特结
def Tanglecube(x,y,z):
    a,b,c = 0.0,-5.0,11.8
    return x**4+y**4+z**4+a*(x**2+y**2+z**2)**2+b*(x**2+y**2+z**2)+c

visualize_four_ways(Tanglecube, '古尔萨特结')






# 11. 心形
# (𝑥2+9/4𝑦2+𝑧2−1)3−𝑥2𝑧3−9/80𝑦2𝑧3=0
def heart(x,y,z):
    return (x**2 + 9/4*y**2 + z**2 - 1)**3 - x**2*z**3 - 9/80 * y**2 * z**3

visualize_four_ways(heart, '心形', (-1.2,1.2))





# 12. 环面
# 参考： https://en.wikipedia.org/wiki/Implicit_surface

# (𝑥2+𝑦2+𝑧2+𝑅2−𝑎2)2−4𝑅2(𝑥2+𝑧2)=0
def Torus(x,y,z):
    R = 2.5
    a = 0.8
    return (x**2 + y**2 + z**2 + R**2 - a**2)**2 - 4*R**2*(x**2 + z**2)

visualize_four_ways(Torus, '环面', (-3,3))





# 范数
def vector_norm(x,y,z):
    p = 0.6
    # 非范数。Lp范数，p >=1
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_0.6', bbox = (-1,1))




def vector_norm(x,y,z):
    p = 1
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_1', bbox = (-1,1))




def vector_norm(x,y,z):
    p = 1.5
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_1.5', bbox = (-1,1))



def vector_norm(x,y,z):
    p = 2
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_2', bbox = (-1,1))



def vector_norm(x,y,z):
    p = 3
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_3', bbox = (-1,1))





def vector_norm(x,y,z):
    p = 8
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_8', bbox = (-1,1))










































































































































































































































































































