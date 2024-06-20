#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://blog.csdn.net/liuchengzimozigreat/article/details/84566650


Created on Fri Dec  1 15:45:10 2023

@author: jack

这里是一些常用的图：

柱状图：pyplot.bar

直方图：pyplot.barh

水平直方图：pyplot.broken_barh

等高线图：pyplot.contour

误差线：pyplot.errorbar

柱形图：pyplot.hist

水平柱状图：pyplot.hist2d

饼状图：pyplot.pie

量场图：pyplot.quiver

散点图：pyplot.scatter

matplotlib.pyplot.quiver(*args, data=None, **kwargs)[source]
Plot a 2D field of arrows.

Call signature:

quiver([X, Y], U, V, [C], **kwargs)
X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.
"""
#%%================================================================================================
import numpy as np
import matplotlib.pyplot as plt

n = 8

# 二维网格坐标
X, Y = np.mgrid[0:n, 0:n]

# U,V 定义方向
U = X + 1
V = Y + 1

# C 定义颜色
C = X + Y
fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)
axs.quiver(X, Y, U, V, C)
plt.show()






#%%================================================================================================
# https://deepinout.com/matplotlib/matplotlib-examples/t_calculate-the-curl-of-a-vector-field-in-python-and-plot-it-with-matplotlib.html#google_vignette
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


x, y, z = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

vx = np.sin(x) * np.cos(y)
vy = np.sin(y) * np.cos(z)
vz = np.sin(z) * np.cos(x)

vec_field = np.stack([vx, vy, vz], axis=-1)

curl = np.gradient(vec_field)

curl_x = curl[2][:, :, :, 1] - curl[1][:, :, :, 2]
curl_y = curl[0][:, :, :, 2] - curl[2][:, :, :, 0]
curl_z = curl[1][:, :, :, 0] - curl[0][:, :, :, 1]


fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

Ax, Ay, Az = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

ax.quiver(Ax, Ay, Az, curl_x, curl_y, curl_z,  length=3, arrow_length_ratio=0.3, pivot='middle', )
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)


filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
# out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')

plt.show()


#%%================================================================================================

import sympy
import numpy as np
from sympy.functions import exp

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')

f_x = x1*exp(-(x1**2 + x2**2))

print(f_x)

#take the gradient symbolically
grad_f = [sympy.diff(f_x,var) for var in (x1,x2)]
print(grad_f)

f_x_fcn = sympy.lambdify([x1,x2],f_x)

#turn into a bivariate lambda for numpy
grad_fcn = sympy.lambdify([x1,x2],grad_f)

import matplotlib.pyplot as plt

xx1, xx2 = np.meshgrid(np.linspace(-2,2,40),np.linspace(-2,2,40))

# coarse mesh
xx1_, xx2_ = np.meshgrid(np.linspace(-2,2,20),np.linspace(-2,2,20))
V = grad_fcn(xx1_,xx2_)


ff_x = f_x_fcn(xx1,xx2)

color_array = np.sqrt(V[0]**2 + V[1]**2)

# 3D visualization
ax = plt.figure().add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, ff_x, rstride=1,
                  cstride=1, color = [0.5,0.5,0.5],
                  linewidth = 0.2)
ax.contour3D(xx1, xx2, ff_x, 20, cmap = 'RdBu_r')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.xlim(-2,2)
plt.ylim(-2,2)
ax.view_init(30, -125)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
plt.tight_layout()
plt.show()

# 2D visualization
fig, ax = plt.subplots()

plt.contourf(xx1, xx2, ff_x,20, cmap = 'RdBu_r')

plt.quiver (xx1_, xx2_, V[0], V[1],
            angles='xy', scale_units='xy',
            edgecolor='none', facecolor= 'k')

plt.show()
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.tight_layout()
plt.show()


#%%================================================================================================
# https://blog.csdn.net/weixin_43718675/article/details/104589175

import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V)
ax.quiverkey(q, X = 0.1, Y = 1.1, U=10, label = r'Quiver key, length = 10', labelpos='E')

plt.show()

##
import matplotlib.pyplot as plt
import numpy as np

fig1,ax1 = plt.subplots(1,1,figsize = (10,6))

X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16,
        }

##画出风速
h1 = ax1.quiver(X,Y,U,V,     #X,Y,U,V 确定位置和对应的风速
                width = 0.003, #箭杆箭身宽度
                scale = 100,    # 箭杆长度,参数scale越小箭头越长
                )
# plt.show()


#画出风场，和箭头箭轴后，得说明 箭轴长度与风速的对应关系
#调用quiver可以生成 参考箭头 + label。
# ax1.quiverkey(h1,                      #传入quiver句柄
#               X=0.09, Y = 0.051,       #确定 label 所在位置，都限制在[0,1]之间
#               U = 5,                    #参考箭头长度 表示风速为5m/s。
#               angle = 0,            #参考箭头摆放角度。默认为0，即水平摆放
#              label='v:5m/s',        #箭头的补充：label的内容  +
#              labelpos='S',          #label在参考箭头的哪个方向; S表示南边
#              color = 'b',labelcolor = 'b', #箭头颜色 + label的颜色
#              fontproperties = font,        #label 的字体设置：大小，样式，weight
#              )
# plt.show()


#由于风有U\V两个方向，最好设置两个方向的参考箭头 + label
ax1.quiverkey(h1, X=0.07, Y = 0.071,
              U = 5,
              angle = 90,           #参考箭头摆放角度，90即垂直摆放
              label = 'w:5cm/s',    #label内容
              labelpos='N',         #label在参考箭头的北边
              color = 'r',          #箭头颜色
              labelcolor = 'r',     #label颜色
              fontproperties = font)

##虽然也可以用plt.text来设置label和箭头，但是会比较繁琐。
plt.show()

#%%================================================================================================
#  https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_demo.html#sphx-glr-gallery-images-contours-and-fields-quiver-demo-py

import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U = np.cos(X)
V = np.sin(Y)

fig1, ax1 = plt.subplots()
ax1.set_title('Arrows scale with plot width, not view')
Q = ax1.quiver(X, Y, U, V, units='width')
qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',  coordinates='figure')



fig2, ax2 = plt.subplots()
ax2.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax2.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],  pivot='mid', units='inches')
qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',  coordinates='figure')
ax2.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)


fig3, ax3 = plt.subplots()
ax3.set_title("pivot='tip'; scales with x view")
M = np.hypot(U, V)
Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,  scale=1 / 0.15)
qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',  coordinates='figure')
ax3.scatter(X, Y, color='0.5', s=1)

plt.show()

#%%================================================================================================
# https://blog.csdn.net/qq_41345173/article/details/111352817
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
# meshgrid 生成网格，此处生成两个 shape = (20,20) 的 ndarray, 详见参考资料2,3
U, V = np.meshgrid(X, Y)
C = np.sin(U) # C 定义颜色
fig, ax = plt.subplots()
# 绘制箭头
q = ax.quiver(X, Y, U, V, C, #  U、V是箭头数据（data），X、Y是箭头的位置，C是箭头的颜色。
              width = 0.003, #箭杆箭身宽度
              scale = 200,    # 箭杆长度,参数scale越小箭头越长
              )
# 如果你多次使用quiver()，只要保证参数scale一致，那么箭头长度就会与风速\sqrt{U^2+V^2}的值成正比，可按照下面我贴出的代码那样设置参数。建议scale设置成30-50，100之内也都还可以。箭头宽度可以通过width=0.005开始设置。箭头颜色可以通过传入颜色列表来控制。


# 该函数绘制一个箭头标签在 (X, Y) 处， 长度为U, 详见参考资料4
ax.quiverkey(q,
             X=0.3,  Y=1.1, #  #确定 label 所在位置，都限制在[0,1]之间
             angle = 0, #参考箭头摆放角度。默认为0，即水平摆放
             U=10, #参考箭头长度 表示风速为5m/s。
             label='Quiver key, length = 10',
             labelpos='E', # label在参考箭头的哪个方向; S表示南边
             color = 'b',labelcolor = 'b', #箭头颜色 + label的颜色
             )
ax.legend()
plt.show()

##================================================================================================

fig, ax = plt.subplots()
# 以水平轴按照 angles 参数逆时针旋转得到箭头方向， units='xy' 指出了箭头长度计算方法
ax.quiver((0, 0), (0, 0), (1, 0), (1, 3), angles=[60, 300], units='xy', scale=1, color='r')
plt.axis('equal')
plt.xticks(range(-5, 6))
plt.yticks(range(-5, 6))
plt.grid()
plt.show()


##================================================================================================

# pivot: {'tail', 'mid', 'middle', 'tip'}, 默认为 'tail'。该参数指定了箭头的基点(旋转点)。
# width:此参数是轴宽，以箭头为单位。
# headwidth:该参数是杆头宽度乘以杆身宽度的倍数。
# headlength:此参数是长度宽度乘以轴宽度。
# headwidth:该参数是杆头宽度乘以杆身宽度的倍数。
# headaxislength:此参数是轴交点处的头部长度。
# minshaft:此参数是箭头缩放到的长度，以头部长度为单位。
# minlength:此参数是最小长度，是轴宽度的倍数。


X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U, V = np.cos(X), np.sin(Y)
fig, ax = plt.subplots()

ax.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], units='inches', pivot='mid', color='g')
qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
ax.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
plt.show()


##================================================================================================
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U, V = np.cos(X), np.sin(Y)
fig, ax = plt.subplots()
# M 为颜色矩阵
M = np.hypot(U, V)
ax.set_title("pivot='tip'; scales with x view")
Q = ax.quiver(X, Y, U, V, M, units='xy', scale = 1 / 0.15, pivot='tip')
qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',  coordinates='figure')
ax.scatter(X, Y, color='r', s=1)
plt.show()




#%%================================================================================================
# https://krajit.github.io/sympy/vectorFields/vectorFields.html
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))

u = -y/np.sqrt(x**2 + y**2)
v = x/np.sqrt(x**2 + y**2)

plt.quiver(x,y,u,v)
plt.show()




import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

# Make the direction data for the arrows
u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

plt.show()































































































































