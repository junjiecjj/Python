#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:29:50 2023

@author: jack

https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide


https://zhuanlan.zhihu.com/p/367067235

https://blog.csdn.net/xfijun/article/details/108422317

https://blog.csdn.net/dgvv4/article/details/124226759

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




# M = 10
# n = 80


# fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)


# for i in range(M):
#     white = 1
#     black = 1
#     res = []
#     for j in range(n):
#         p1 = white/(white + black)
#         p2 = 1 - p1
#         # flag = np.random.binomial(1, white/(white + black), size = 1)
#         flag = np.random.choice(['white', 'black'], 1, p=[p1, p2])
#         if flag == "white":
#             white += 1
#         elif flag == "black":
#             black += 1
#         res.append(white)

#     # Mean = np.mean(res)
#     # label = f"平均值:{Mean:.3f}"
#     axs.plot(range(len(res)), res,   lw = 2,  )


# font1  = {'family':'Times New Roman','style':'normal','size':25, }
# # font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
# axs.set_xlabel('n',  fontproperties=font1, labelpad = 12.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。)

# font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
# axs.set_ylabel('白球的个数',  fontproperties=font1 )
# #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
# font1 = FontProperties(fname=fontpath+"simsun.ttf", size=20)
# #  edgecolor='black',
# # facecolor = 'y', # none设置图例legend背景透明
# # legend1 = axs.legend(loc='best',  prop = font1,  facecolor = 'w', edgecolor = 'k', labelcolor = 'r', borderaxespad=0,)
# # frame1 = legend1.get_frame()
# # frame1.set_alpha(1)
# # # frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


# axs.tick_params(direction='in',axis='both', top=True, right=True, labelsize=16,   )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号


# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig .savefig(filepath2+'box_play100.eps', format='eps',  bbox_inches = 'tight')
# #out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
# plt.show()





##==========================================================================
##                          单变量积分
##==========================================================================

a = integrate.quad(lambda x:x**2,0,1)
print(a)

# 返回值为：

# (0.33333333333333337, 3.700743415417189e-15)
# 前者是积分值，后者是误差。


## SciPy提供的quad还支持对多变量函数的其中某个变量积分，对下式积分：
# integrate(f,a,b,(x_2,x_3,...,x_n))
# 即需要在上下限后加入后面其他变量的值。
# 比如对于下式的计算：

def f(y,x,z):
    return np.sin(x)*np.cos(y)*np.tan(z)

print(integrate.quad(f,0,1,(2,3)))





###  “向量值函数”积分

# 为什么这里要打引号呢？因为SciPy中的向量值函数和我们理解的不太一样。
# 先介绍一下SciPy函数：
# integrate.quad_vec(f,a,b)
# 其返回的值是：
# 这里说明一下，输入的y可以是一个列表或者数组，在积分命令之前定义。
# 这里举个例子你就明白了，比如计算函数
#  的x从0到2的定积分，这个定积分是a的一个函数：

a=np.linspace(0,2,1000) # a is a vector here.
f=lambda x:x**a
y,err=integrate.quad_vec(f,0,2)
fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)
axs.plot(a,y)
axs.set_xlabel('a')
axs.set_ylabel(r"$\int_{0}^{2}x^2 \mathrm{d}x$")
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'box_play.eps', format='eps',  bbox_inches = 'tight')
plt.show()
plt.close()


from scipy.integrate import quad
def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print(I)



from scipy.integrate import quad
import numpy as np
def integrand(t, n, x):
    return np.exp(-x*t) / t**n
def expint(n, x):
    return quad(integrand, 1, np.inf, args=(n, x))[0]

vec_expint = np.vectorize(expint)



##==========================================================================
##                         二重积分
##==========================================================================

def f(y,x):
    return np.sin(x**2)

def y_1(x):
    return 0

def y_2(x):
    return x

a,b = 0,np.sqrt(np.pi/2)
result = integrate.dblquad(f,a,b,y_1,y_2)
print(result)
# (0.4999999999999998, 1.3884045385060701e-14)





##==========================================================================
##                         三重积分
##==========================================================================

def f(z,y,x):
    return x*y

def y_1(x):
    return 0

def y_2(x):
    return 1-x

def z_1(x,y):
    return 0

def z_2(x,y):
    return x*y

a,b=0,1
print(integrate.tplquad(f,a,b,y_1,y_2,z_1,z_2))
# (0.005555555555555556, 6.9087921361902645e-16)






##==========================================================================
##                         常微分方程
##==========================================================================
###  1.给出初始条件的情况
# SciPy提供了有力的常微分方程求解工具，即函数integrate.solve_ivp
# 该函数用于解决形如下式的ODE：
# 这里的y可以为标量也可以为矢量。
# 函数的具体形式为：
# integrate.solve_ivp(f,(t_0,t_1),y_0,method='RK45',dense_output=False)
# f即上式左侧的函数，(t_0, t_1)为参数t的范围，y0为初值条件。method='RK45'为选择的方法，默认为5（4）阶龙格库塔法。dense_output选择的是是否稠密输出，这个主要是画图的时候需要，默认为否。
# 以解下列方程为例：

def f(t,y):
    return y*np.sin(t)

def analytical_solution(t):
    return np.e**(1-np.cos(t))

result = integrate.solve_ivp(f,(0,5),[1],dense_output=True)
t = np.linspace(0,5,1001)

fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)
axs.plot(t,analytical_solution(t),label='analytical solution')
axs.plot(t,result.sol(t)[0],label='numerical solution')
axs.grid()
plt.legend()
axs.set_xlabel('t')
axs.set_ylabel('y(t)')
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig.savefig(filepath2+'box_play.eps', format='eps',  bbox_inches = 'tight')
plt.show()
plt.close()


# 可以发现数值解非常准确，盖住了解析解的曲线。

# integrate.solve_ivp中的method的可选参数还有很多，包括：

# 参数	方法
# 'RK23'	显式2(3)阶龙格库塔法
# 'DOP853'	显式8阶龙格库塔法
# 'Radau'	隐式5阶龙格库塔法
# 'BDF'	向后微分公式法





###  2.给出边界条件的情况
def f(x,y,p):
    k = p[0]
    return np.vstack((y[1],-k**2*y[0]))

def bc(y_a,y_b,p):
    k = p[0]
    return np.array([y_a[0],y_b[0],y_a[1]-k])

x = np.linspace(0,1,5)
y = np.zeros((2,x.size))
y[0,1] = 1
y[0,3] = -1
result = integrate.solve_bvp(f,bc,x,y,p=[6])
x_plot = np.linspace(0,1,1000)
y_plot = result.sol(x_plot)[0]
plt.plot(x_plot,y_plot)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()





# https://blog.csdn.net/dgvv4/article/details/124226759
##定积分计算


from scipy  import integrate

# 定义函数x的n次方，自变量x
def myfunction(x, n):
    return x**n

# 计算函数对x的积分，下限0上限2，被积函数中的n=2
data = integrate.quad(func=myfunction, a=0, b=2, args=(2,))
print(data)
# 积分结果2.666, 计算误差2.9605e-14
# (2.666666666666667, 2.960594732333751e-14)




# 定义函数k乘x的n次方，自变量x
def myfunction(x, n, k):
    return k * x**n

# 计算函数对x的积分，下限0上限2，被积函数中的n=2,k=3
data = integrate.quad(func=myfunction, a=0, b=2, args=(2,3))
print(data)
# 积分结果8.0, 误差8.881e-14
# (8.0, 8.881784197001252e-14)







## 二重积分

from scipy.integrate import dblquad

# 定义空间平面
def myfunction(x, y):
    # z = 4-x-y的平面
    return 4-x-y

# 计算二重积分, x的下限为0上限为2，y的下限为0上限为1
# 先对y积分，再对x积分
value, error = dblquad(func=myfunction, a=0, b=2, gfun=0, hfun=1)
print('value:', value, 'error:', error)
# value: 5.0   error: 5.551115123125783e-14







# 案例二：计算围成扇形面积的积分
from scipy.integrate import dblquad

# 定义被积函数
def myfunction(x, y):
    data = 3 * (x**2) * (y**2)
    return data

# 如果外围积分区间和内部积分区间是相互依赖的就要写成函数形式
# y的定义域函数
def y_area(x):
    return 1-x**2

# 计算二重积分, 先对y积分，后对x积分
# # x定义域[0, 1], y定义域[0, 1-x^2]
value, error = dblquad(func=myfunction, a=0, b=1, gfun=0, hfun=y_area)
print(value)  # 0.050793650793650794




## 1
from scipy.integrate import dblquad
area = dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda x: 1-2*x)
print(area)

## 2
from scipy import integrate
def f(x, y):
    return x*y

def bounds_y():
    return [0, 0.5]

def bounds_x(y):
    return [0, 1-2*y]
area = integrate.nquad(f, [bounds_x, bounds_y])
print(area)
# (0.010416666666666668, 4.101620128472366e-16)


from scipy import integrate
N = 5
def f(t, x):
   return np.exp(-x*t) / t**N

a = integrate.nquad(f, [[1, np.inf],[0, np.inf]])

print(a)










































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































