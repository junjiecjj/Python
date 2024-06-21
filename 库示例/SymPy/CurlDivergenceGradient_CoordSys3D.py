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
import numpy as np
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

res = delop.gradient(f, doit=True) # 使用nabla算子
# # res = delop(f).doit()
print(res)

grad_f = gradient(f) # 直接使用gradient
print(grad_f) # (2*C.x*C.y - C.y)*C.i + (C.x**2 - C.x)*C.j


#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)






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








































































































































































































































































































































































































































