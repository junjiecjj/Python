#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:52:23 2023

@author: jack

https://matplotlib.org/stable/gallery/mplot3d/index.html


"""

import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




#===============================================================================
#  三维作图 ===================================
# https://blog.csdn.net/qq_40811682/article/details/117027899
#===============================================================================
#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#生成三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图 同时还可以将等高线投影到不同的面上：
ax4.plot_surface(X,Y,Z, alpha=0.3, cmap='winter')     #生成表面， alpha 用于控制透明度
ax4.contour(X,Y,Z,zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
ax4.contour(X,Y,Z,zdir='x', offset=-6, cmap="rainbow")  #生成x方向投影，投到y-z平面
ax4.contour(X,Y,Z,zdir='y', offset=6, cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数

#设定显示范围
ax4.set_xlabel('X')
ax4.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax4.set_ylabel('Y')
ax4.set_ylim(-4, 6)
ax4.set_zlabel('Z')
ax4.set_zlim(-3, 3)


out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')

plt.show()

#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.contour(X,Y,Z, zdim='z',offset=-2,cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')

plt.show()


#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')


[x,t] = np.meshgrid(np.array(range(25))/24.0, np.arange(0,575.5,0.5)/575*17*np.pi-2*np.pi)
p = (np.pi/2)*np.exp(-t/(8*np.pi))
u = 1-(1-np.mod(3.6*t,2*np.pi)/np.pi)**4/2
y = 2*(x**2-x)**2*np.sin(p)
r = u*(x*np.sin(p)+y*np.cos(p))
surf = ax.plot_surface(r*np.cos(t), r*np.sin(t), u*(x*np.cos(p)-y*np.sin(p)), rstride=1, cstride=1, cmap=cm.gist_rainbow_r, linewidth=0, antialiased=True)
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()


#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
#ax.plot_surface(x, y, z, color='b')
ax.plot_surface(x, y, z,cmap='rainbow')
out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')

plt.show()

#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.contour(X,Y,Z,  offset=-2, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()

#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
ax3.contour(X,Y,Z,offset=-2, cmap = 'rainbow')#绘制等高线
plt.show()

#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

import math
fig = plt.figure(figsize=(10,10))  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-20,20,0.5)
yy = np.arange(-20,20,0.5)
X, Y = np.meshgrid(xx, yy)#将两个一维数组变为二维矩阵
Z = X*Y**2


#作图
#ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1)
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()


#===============================================================================
#  1.三维曲面
#===============================================================================

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值

#如果加入渲染时的步长，会得到更加清晰细腻的图像：
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
# ,其中的row和cloum_stride为横竖方向的绘图采样步长，越小绘图越精细。
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()

#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.mgrid[0:6*np.pi:0.25, 0:4*np.pi:0.25]
Z = np.sqrt(np.abs(np.cos(X) + np.cos(Y)))
ax.plot_surface(X + 1e5, Y + 1e5, Z, cmap='autumn', cstride=2, rstride=2)
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")
ax.set_zlim(0, 2)
plt.show()


#===============================================================================
# 1 表面图（Surface plots）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

# Make data.

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#===============================================================================
#   3D等高线图
#===============================================================================

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
def f(x, y):
   return np.sin(np.sqrt(x ** 2 + y ** 2))
#构建x、y数据
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
#将数据网格化处理
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
#50表示在z轴方向等高线的高度层级，binary颜色从白色变成黑色
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()


#===============================================================================
# 3D曲面图
#===============================================================================


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#求向量积(outer()方法又称外积)
x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
#矩阵转置
y = x.copy().T
#数据z
z = np.cos(x ** 2 + y ** 2)
#绘制曲面图
fig = plt.figure()
ax = plt.axes(projection='3d')
#调用plot_surface()函数
ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'plotfig.eps',bbox_inches = 'tight')
plt.show()

#===============================================================================
# 2 三维曲线和散点
#===============================================================================

#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

"""
#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)
"""
##################我是分割线#####################
import numpy as np
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='b')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()



#===============================================================================
# 2 三维曲线和散点
#===============================================================================
"""
基本用法：

ax.scatter(xs, ys, zs, s=20, c=None, depthshade=True, *args, *kwargs)

xs,ys,zs：输入数据；
s:scatter点的尺寸
c:颜色，如c = 'r’就是红色；
depthshase:透明化，True为透明，默认为True，False为不透明
*args等为扩展变量，如maker = ‘o’，则scatter结果为’o‘的形状

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()

#===============================================================================
#3D散点图
#===============================================================================

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
#创建绘图区域
ax = plt.axes(projection='3d')
#构建xyz
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y
ax.scatter3D(x, y, z, c=c)
ax.set_title('3d Scatter plot')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()

#===============================================================================
# 3 np.meshgrid使用方法
#===============================================================================

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax3 = plt.axes(projection='3d')

xx = np.arange(-20,20,0.5)
yy = np.arange(-20,20,0.5)
X, Y = np.meshgrid(xx, yy)

Z=X**2+Y**2#Change Here

ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()







#===============================================================================
# 四、线框图（Wireframe plots）
#===============================================================================

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})

# Get the test data
X, Y, Z = axes3d.get_test_data(0.05)

# Plot the data
for ax in axs:
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# Set the orthographic projection.
axs[0].set_proj_type('ortho')  # FOV = 0 deg
axs[0].set_title("'ortho'\nfocal_length = ∞", fontsize=10)

# Set the perspective projections
axs[1].set_proj_type('persp')  # FOV = 90 deg
axs[1].set_title("'persp'\nfocal_length = 1 (default)", fontsize=10)

# axs[2].set_proj_type('persp', focal_length=0.2)  # FOV = 157.4 deg
axs[2].set_title("'persp'\nfocal_length = 0.2", fontsize=10)

plt.show()

#===============================================================================
#  四、3D线框图
#===============================================================================

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#要绘制函数图像
def f(x, y):
   return np.sin(np.sqrt(x ** 2 + y ** 2))
#准备x,y数据
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
#生成x、y网格化数据
X, Y = np.meshgrid(x, y)
#准备z值
Z = f(X, Y)
#绘制图像
fig = plt.figure()
ax = plt.axes(projection='3d')
#调用绘制线框图的函数plot_wireframe()
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'plotfig.eps',bbox_inches = 'tight')
plt.close()


#===============================================================================
# 五、表面图（Surface plots）
#===============================================================================


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = plt.axes(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)



# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig .savefig(filepath2+'plotfig.eps',bbox_inches = 'tight')
plt.show()



#===============================================================================
# 六、三角表面图（Tri-Surface plots）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = np.sin(-x*y)

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()

#===============================================================================
# 七、等高线（Contour plots）
#===============================================================================

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
ax.clabel(cset, fontsize=9, inline=1)

plt.show()



# 二维的等高线，同样可以配合三维表面图一起绘制：from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

# 设置投影类型为正交投影 (orthographic projection)
# ax.set_proj_type('ortho')

# 设置投影类型为透视投影 (perspective projection)
# ax.set_proj_type('persp')

# 设置观察者的仰角为30度，方位角为30度，即改变三维图形的视角
# ax.view_init(azim=30, elev=30, )

# 设置观察者的仰角为30度，方位角为30度，即改变三维图形的视角
# ax.view_init(azim=-60, elev=30, )

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()


# 也可以是三维等高线在二维平面的投影：
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()



#===============================================================================
# 八、Bar plots（条形图）
#===============================================================================

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
    xs = np.arange(20)
    ys = np.random.rand(20)

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


#===============================================================================
# 九、子图绘制（subplot）
#===============================================================================


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
# Plot a sin curve using the x and y axes.
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='curve in (x,y)')

# Plot scatterplot data (20 2D points per colour) on the x and z axes.
colors = ('r', 'g', 'b', 'k')
x = np.random.sample(20*len(colors))
y = np.random.sample(20*len(colors))
c_list = []
for c in colors:
    c_list.append([c]*20)
# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x,y) points are plotted on the x and z axes.
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x,z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()

#===============================================================================
# 九、子图绘制（subplot）
#===============================================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

#===============
#  First subplot
#===============
# set up the axes for the first plot
ax = fig.add_subplot(2, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,  linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

#===============
# Second subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(2,1,2, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.show()



#===============================================================================
# 九、子图绘制（subplot）
# 文本注释的基本用法：
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params



# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



fig = plt.figure(figsize = (8, 8))
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
# Demo 1: zdir
zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
xs = (1, 4, 4, 9, 4, 1)
ys = (2, 5, 8, 10, 1, 2)
zs = (10, 3, 8, 9, 1, 8)


# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':22, 'color':'#9400D3'}

for zdir, x, y, z in zip(zdirs, xs, ys, zs):
    label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    ax.text(x, y, z, label, zdir,  fontproperties=font3, fontdict = font2)

# Demo 2: color
font2  = {'family':'Times New Roman','style':'normal','size':32, 'color':'green'}
ax.text(9, 0, 0, "red",  fontdict = font2) # color='red',

# Demo 3: text2D
font2  = {'family':'Times New Roman','style':'normal','size':42, 'color':'blue'}
# Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes,  fontdict = font2)

# Tweaking display region and labels
font3  = {'family':'Times New Roman','style':'normal','size':22}
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X axis',  fontproperties=font3, labelpad = 12.5)
ax.set_ylabel('Y axis', fontproperties=font3, labelpad = 12.5)
ax.set_zlabel('Z axis', fontproperties=font3, labelpad = 12.5)


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
ax.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3,)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5,y=0.96,)

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'hh3D.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.close()


#===============================================================================
# https://blog.csdn.net/u014636245/article/details/82799573
# 1.创建三维坐标轴对象Axes3D
# 创建Axes3D主要有两种方式，一种是利用关键字projection='3d'来实现，另一种则是通过从mpl_toolkits.mplot3d导入对象Axes3D来实现，目的都是生成具有三维格式的对象Axes3D.
#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
# ax1 = plt.axes(projection='3d')
ax1 = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

# 2.三维曲线和散点


import numpy as np
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'plotfig.eps',  bbox_inches = 'tight')
plt.close()

#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)





#===============================================================================
# 4.等高线
# 同时还可以将等高线投影到不同的面上：

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#生成三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
ax4.plot_surface(X,Y,Z,alpha=0.3,cmap='winter')     #生成表面， alpha 用于控制透明度
ax4.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面
ax4.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
ax4.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数

#设定显示范围
ax4.set_xlabel('X')
ax4.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax4.set_ylabel('Y')
ax4.set_ylim(-4, 6)
ax4.set_zlabel('Z')
ax4.set_zlim(-3, 3)

plt.show()

#=======================================================
# 5.随机散点图
# 可以利用scatter()生成各种不同大小，颜色的散点图，其参数如下：
#函数定义
# matplotlib.pyplot.scatter(x, y,
# 	s=None,   #散点的大小 array  scalar
# 	c=None,   #颜色序列   array、sequency
# 	marker=None,   #点的样式
# 	cmap=None,    #colormap 颜色样式
# 	norm=None,    #归一化  归一化的颜色camp
# 	vmin=None, vmax=None,    #对应上面的归一化范围
#  	alpha=None,     #透明度
# 	linewidths=None,   #线宽
# 	verts=None,   #
# 	edgecolors=None,  #边缘颜色
# 	data=None,
# 	**kwargs
# 	)
#ref:https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

#生成三维数据
xx = np.random.random(20)*10-5   #取100个随机数，范围在5~5之间
yy = np.random.random(20)*10-5
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
ax4.scatter(X,Y,Z,alpha=0.3,c=np.random.random(400),s=np.random.randint(10,20, size=(20, 40)))     #生成散点.利用c控制颜色序列,s控制大小

#设定显示范围

plt.show()





#=======================================================
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


















