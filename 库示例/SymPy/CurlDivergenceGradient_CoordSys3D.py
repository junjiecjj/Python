#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################

# 此程序的功能是： Sympy和Numpy、scipy

"""
梯度、散度和旋度是向量场的重要性质，它们在物理学、工程学等领域有广泛的应用。
    梯度：表示函数在某一点处的变化率，是一个向量，指向函数值增最快的方向。
    散度：表示向量场在某一点处的流量密度，是一个标量，描述向量场的源和汇。
    旋度：表示向量场在某一点处的旋转程度，是一个向量，描述向量场的旋转。


在程序中计算梯度散度旋度有一下几种方法：
(一) 纯数值计算，先用np计算出各个方向的上述物理量再画图，可以涉及也可以不涉及sympy工具的使用，总之这种方法的核心就是数值为中心，纯靠人工保证数据的正确性;如很多C程序的输出结果就是这些物理量，这时候直接画图。
(二) 使用sympy的符号计算和求导，先给出场的表达式，再根据公式计算出梯度/散度/旋度，这时候还是符号运算，最后再是把符号运算转化(lambdify)为np函数，最后带入格点，画图;

(三) 使用  sympy.vector 中的 CoordSys3D, 然后利用sympy.vector.Del(cross,gradient,dot) 或 sympy.vector.gradient, curl, divergence等自动化简，但是化简后是sympy的符号公式，怎么转为numpy并画图还未解决.


    sympy中计算梯度、散度和旋度主要有两种方式：
    一个是使用∇算子，sympy提供了类Del()，该类的方法有：cross、dot和gradient，cross就是叉乘，计算旋度的，dot是点乘，用于计算散度，gradient自然就是计算梯度的。
    另一种方法就是直接调用相关的API：curl、divergence和gradient，这些函数都在模块sympy.vector 下面。

    使用sympy计算梯度、散度和旋度之前，首先要确定坐标系，sympy.vector模块里提供了构建坐标系的类，常见的是笛卡尔坐标系， CoordSys3D，根据下面的例子可以了解到相应应用。


"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



#%%==================================== 根据sympy的求导 diff 和 lambdify 手动生成梯度并画图 ======================================
import numpy as np
import sympy as sy
# from sympy import symbols, Function, diff
from sympy.vector import CoordSys3D, gradient


#定义三维数据
xx = np.arange(-3, 3, 0.1)
yy = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(xx, yy)

# 定义变量和函数
x, y = sy.symbols('x y')
f = sy.Function('f')(x, y)
f = x*sy.exp(-x**2 - y**2)
# 计算梯度
Fx = sy.diff(f, x)
Fy = sy.diff(f, y)

# 将表达式转为numpy函数
fn = sy.lambdify([x,y], f, 'numpy')
fx = sy.lambdify([x,y], Fx, 'numpy')
fy = sy.lambdify([x,y], Fy, 'numpy')

## 计算在网格中的值
Z = fn(X, Y)
gx = fx(X, Y)
gy = fy(X, Y)

## 画三维图
fig, axs = plt.subplots(1, 1,  figsize=(12, 10), subplot_kw={'projection': '3d'})
# fig = plt.figure(figsize=(12, 10),)  #定义新的三维坐标轴
# axs = plt.axes(projection='3d')

## 1
axs.plot_surface(X,Y,Z, alpha=0.4,  cmap='winter')     #生成表面， alpha 用于控制透明度
axs.contour(X, Y, Z,  levels = 30, linewidths = 2, cmap = "RdYlBu_r")   # 3维等高线
CS = axs.contour(X, Y, Z, zdir='z', offset= Z.min(), levels = 30, linewidths = 2, cmap = "RdYlBu_r")  # 生成z方向投影，投到x-y平面

## colorbar
cb = fig.colorbar(CS, ax = axs, fraction = 0.025, pad = 0.02, label = "color bar", )
cb.ax.tick_params(labelsize=22)  #设置色标刻度字体大小。
font2  = {'family':'Times New Roman','style':'normal','size':26,  }
cb.set_label('colorbar', fontdict=font2) #设置colorbar的标签字体及其大小


# ## 2
# axs.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = [0.5, 0.5, 0.5], linewidth = 1)
# axs.contour(X, Y, Z,  levels = 30, linewidths = 2, cmap = "RdYlBu_r")   # 3维等高线
# axs.contour(X, Y, Z, levels = [0.2],  colors = 'green', linestyles = '-', linewidths = 4)
# CS = axs.contour(X, Y, Z, zdir='z', offset= Z.min(), levels = 30, linewidths = 2, cmap = "RdYlBu_r")  # 生成z方向投影，投到x-y平面

# ## colorbar
# cb = fig.colorbar(CS, ax = axs, fraction = 0.03, pad = 0.06, label = "color bar", )
# cb.ax.tick_params(labelsize=22)  #设置色标刻度字体大小。
# font2  = {'family':'Times New Roman','style':'normal','size':26,  }
# cb.set_label('colorbar', fontdict=font2) #设置colorbar的标签字体及其大小

# 设置观察者的仰角为30度，方位角为30度，即改变三维图形的视角
axs.view_init(azim=-60, elev=30, )

#设定显示范围
font2  = {'family':'Times New Roman','style':'normal','size':22,  }
axs.set_xlabel('X', fontdict = font2, labelpad=12.5)
axs.set_ylabel('Y', fontdict = font2, labelpad=12.5)
axs.set_zlabel('$f(x,y)$', fontdict = font2, labelpad=12.5)

axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3,  )
labels = axs.get_xticklabels() + axs.get_yticklabels() +  axs.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号
axs.set_proj_type('ortho')
axs.grid(False)


out_fig = plt.gcf()
out_fig.savefig('grad3D.pdf', bbox_inches = 'tight')
out_fig.savefig('grad3D.eps', bbox_inches = 'tight')
plt.show()

##================= 画2D图，展示梯度与等高线垂直 ========================
fig, axs = plt.subplots(1, 1,  figsize=(12, 10), )

CS = axs.contour(X, Y, Z, levels = 20, cmap = 'RdYlBu_r', linewidths = 2)  #生成z方向投影，投到x-y平面
axs.contour(X, Y, Z, levels = [0.2],  colors = 'green', linestyles = '-', linewidths = 4)
# 绘制等高线数据, 等高线的描述
axs.clabel(CS, inline = True, fontsize = 16)

## colorbar
cb = fig.colorbar(CS, ax = axs, fraction = 0.05, pad = 0.02, label = "color bar", )
cb.ax.tick_params(labelsize=22)  #设置色标刻度字体大小。
font2  = {'family':'Times New Roman','style':'normal','size':26,  }
cb.set_label('colorbar', fontdict=font2) #设置colorbar的标签字体及其大小


C = np.hypot(gx, gy)
axs.quiver(X, Y, gx, gy, C,  pivot='tail',)

#设定显示范围
font2  = {'family':'Times New Roman','style':'normal','size':26,  }
axs.set_xlabel('X', fontdict = font2, labelpad=12.5)
axs.set_ylabel('Y', fontdict = font2, labelpad=12.5)

axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=24, width=3,  )
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(24) for label in labels]  # 刻度值字号


out_fig = plt.gcf()
# out_fig.savefig('grad2D.eps', bbox_inches = 'tight')
plt.show()



#%% https://deepinout.com/numpy/numpy-questions/136_numpy_compute_divergence_of_vector_field_using_python.html#google_vignette
import numpy as np
import matplotlib.pyplot as plt

x, y, z = np.mgrid[0:4, 0:4, 0:4]  # 定义网格
F = np.sin(x) * np.cos(y) + np.cos(z)  # 定义矢量场

F_x, F_y, F_z = np.gradient(F)  # 计算矢量场在每个维度的偏导数
div_F = F_x + F_y + F_z  # 计算散度

fig = plt.figure(figsize=(10, 5))

# 绘制散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x.flatten(), y.flatten(), z.flatten(), c=div_F.flatten())
ax1.set_title('散点图')

# 绘制等值面
ax2 = fig.add_subplot(122, projection='3d')
ax2.contour(x, y, z, div_F, cmap='RdBu')
ax2.set_title('等值面')

plt.show()




#%% https://blog.csdn.net/ouening/article/details/80712269
from sympy.vector import CoordSys3D, Del, gradient, curl, divergence
from sympy import init_printing
init_printing()


### （1）计算梯度
#  gradient
C = CoordSys3D('C')
delop = Del() # nabla算子

# 标量场 f = x**2*y-xy
f = C.x**2*C.y - C.x*C.y

# res = delop.gradient(f, doit=True) # 使用nabla算子
# # res = delop(f).doit()
# print(res)

grad_f = gradient(f) # 直接使用gradient
print(grad_f) # (2*C.x*C.y - C.y)*C.i + (C.x**2 - C.x)*C.j


#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)


grad_f.subs([(x, 1), (y, 2)])









import numpy as np
from scipy import optimize

def func(x):
    return x[0]**2 + x[1]**2

x0 = np.array([3.0, 4.0])
res = optimize.minimize(func, x0, method='BFGS', jac=True, tol=1e-10)
# print(res['jac'])


##
import numpy as np
f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
f = np.arange(48).reshape(6,8)
fx = np.gradient(f, axis=0)
fy = np.gradient(f, axis=1)
grad = np.sqrt(fx ** 2 + fy ** 2)
print(grad)












#%% （2）计算散度
C = CoordSys3D('C')
delop = Del() # nabla算子

# 向量场 f = x**2*y*i-xy*j
f = C.x**2*C.y*C.i - C.x*C.y*C.j

res = delop.dot(f, doit=True) # 使用nabla算子
print(res)

res = divergence(f)
print(res)  # 2*C.x*C.y - C.x，即2xy-x，向量场的散度是标量


#%% （3）计算旋度
## curl
C = CoordSys3D('C')
delop = Del() # nabla算子

# 向量场 f = x**2*y*i-xy*j
f = C.x**2*C.y*C.i - C.x*C.y*C.j

res = delop.cross(f, doit=True)
print(res)

res = curl(f)
print(res)  # (-C.x**2 - C.y)*C.k，即(-x**2-y)*k，向量场的旋度是向量








































































































































































































































































































































































































































