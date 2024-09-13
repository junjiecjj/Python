#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:00 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247487403&idx=1&sn=a58720db20be83a7b91a02123713dcce&chksm=ceecd243b15f4bef85b0c25e40165191ba9654c85e14a72d727902a8d0688056c759fe30f54e&mpshare=1&scene=1&srcid=09124lVHSIu2XH7x1mojd1BM&sharer_shareinfo=324f06c052067781c4fe073ebc5e5b5c&sharer_shareinfo_first=324f06c052067781c4fe073ebc5e5b5c&exportkey=n_ChQIAhIQEbwLpg6phOQBbwnvjo481xKfAgIE97dBBAEAAAAAAJU%2BA4FTHjcAAAAOpnltbLcz9gKNyK89dVj0TFB2IwCWpOPCaRyAyesrwpcdGuipxnl2z2MIvhSAa9xXqwECieSEM6tNR82mTGbJSbv0bMc4sr4f0gY6aUbQtQXyEWT%2BL%2F3%2F0QHn3WNvtToMBt%2FJYk4HNHCgVM3ZEmGw%2BFRill%2FtgX%2BpxQIddGhKCEF1stTC%2BSK274Cs4%2B4XZ%2Fg3mGEtaOhGXW4DoW7T5xcWKlj4ywLAiYLK6JzE0Je8Wa6GQZtY4uy1LfY3gJhWu0OP81Y%2Fp9%2BdMkpfBHLGNY%2BW%2FwF%2BdQgAfXIgTNJ38HpYMdVhzncXFenUVDdFHwekGS2ZIpQwU%2F67TaqZ6VNRqdkRtengwqHFBHRW&acctmode=0&pass_ticket=LJB%2FPmNnNyCasxS0y1iEvSzAKH3xO8O6Lt2Cp5vJ%2FAscI%2BCnJQ0XDgRFwuMHTKJW&wx_header=0#rd


"""


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  非线性规划示例
import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

# Define the constraints
constraints = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})

# Initial guess
x0 = np.array([0.5, 0.5])

# Perform the optimization
result = minimize(objective_function, x0, constraints=constraints)

print('Optimal value:', result.fun)
print('Optimal solution:', result.x)




#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 不可行非线性规划问题

constraints = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 3})
result = minimize(objective_function, x0, constraints=constraints)
print('Message:', result.message)
print('Success:', result.success)



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 无界非线性规划问题
def objective_function_unbounded(x):
    return -x[0]**2 - x[1]**2

constraints = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})
result = minimize(objective_function_unbounded, x0, constraints=constraints)
print('Message:', result.message)
print('Success:', result.success)


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 资源分配问题


from scipy.optimize import linprog

c = [-1, -2]
A = [[1, 2], [3, 4]]
b = [4, 10]
x0_bounds = (0, None)
x1_bounds = (0, None)

res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')
print('Optimal value:', res.fun)
print('Optimal solution:', res.x)





#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1.Rosenbrock函数
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 目标函数
def  rosenbrock ( vars ):
    x, y = vars
    a, b = 1, 100
    return (a - x)** 2 + b * (y - x** 2 )** 2

# 初始猜测
initial_guess = [ 0 , 0 ]

# 执行优化
result = minimize(rosenbrock, initial_guess)

# 提取结果
x_opt, y_opt = result.x
optimal_value = rosenbrock(result.x)

# 打印结果
print ( f"x 的最佳值：{x_opt} " )
print ( f"y 的最佳值：{y_opt} " )
print ( f"Min of Rosenbrock fun：{optimal_value} " )

# 绘图
x = np.linspace(- 2 , 2 , 100 )
y = np.linspace(- 1 , 3 , 100 )
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

# 3D 表面图
fig = plt.figure(figsize=( 14 , 8 ))
ax = fig.add_subplot( 111 , projection= '3d' )
ax.set_proj_type('ortho')

norm_plt = plt.Normalize(X.min(), X.max())
colors = cm.hsv(norm_plt(X))
surf = ax.plot_surface(X, Y, Z, facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       linewidth = 0.5, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。


# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.scatter(x_opt, y_opt, optimal_value, color= 'red' , s= 50 , label= 'optimal sol' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'fun' )
ax.set_title( 'Rosenbrock 3D' )
ax.legend()

plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2. 球面函数

# 问题陈述：最小化球体函数，这是一个用于优化的简单测试函数。
# 目标函数： f（x，y）= x² + y²

# 目标函数
def  sphere_function ( vars ):
    x, y = vars
    return x** 2 + y** 2

# 初始猜测
initial_guess = [ 1 , 1 ]

# 执行优化
result = minimize(sphere_function, initial_guess)

# 提取结果
x_opt, y_opt = result.x
optimal_value = sphere_function(result.x)

# 打印结果
print ( f"x 的最佳值：{x_opt} " )
print ( f"y 的最佳值：{y_opt} " )
print ( f"Sphere 函数的最小值：{optimal_value} " )

# 绘图
x = np.linspace(- 2 , 2 , 100 )
y = np.linspace(- 2 , 2 , 100 )
X, Y = np.meshgrid(x, y)
Z = sphere_function([X, Y])

# 3D 表面图
fig = plt.figure(figsize=( 14 , 8 ))

ax = fig.add_subplot( 111 , projection= '3d' )
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(X.min(), X.max())
colors = cm.RdYlBu_r(norm_plt(X))
surf = ax.plot_surface(X, Y, Z, facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       linewidth = 0.5, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.grid()
# ax.plot_surface(X, Y, Z, cmap= 'viridis' , edgecolor= 'none' )
ax.scatter(x_opt, y_opt, optimal_value, color= 'red' , s = 80 , label= 'optimal sol' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'fun' )
ax.set_title( 'bold 3D' )
ax.legend()

plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 具有多个局部最小值的函数测试优化算法寻找全局最小值的能力。

# 问题陈述：最小化 Himmelblau 函数，该函数具有多个局部最小值。

# 目标函数： f（x，y）=（x²+ y — 11）²+（x + y² — 7）²


# 目标函数
def  himmelblau ( vars ):
    x, y = vars
    return (x** 2 + y - 11 )** 2 + (x + y** 2 - 7 )** 2

# 初始猜测
initial_guess = [ 0 , 0 ]

# 执行优化
result = minimize(himmelblau, initial_guess)

# 提取结果
x_opt, y_opt = result.x
optimal_value = himmelblau(result.x)

# 打印结果
print ( f"x 的最佳值：{x_opt} " )
print ( f"y 的最佳值：{y_opt} " )
print ( f"Himmelblau 函数的最小值：{optimal_value} " )

# 绘图
x = np.linspace(- 5 , 5 , 100 )
y = np.linspace(- 5 , 5 , 100 )
X, Y = np.meshgrid(x, y)
Z = himmelblau([X, Y])

# 3D 表面图
fig = plt.figure(figsize=( 14 , 8 ))

ax = fig.add_subplot( 111 , projection= '3d' )
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(X.min(), X.max())
colors = cm.plasma_r(norm_plt(X))
surf = ax.plot_surface(X, Y, Z, facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       linewidth = 0.5, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.grid()


ax.scatter(x_opt, y_opt, optimal_value, color= 'red' , s= 50 , label= 'optimal sol' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'fun' )
ax.set_title( 'Himmelblau 3D' )
ax.legend()

plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4. Beale 函数


# 目标函数
def  beale ( vars ):
    x, y = vars
    return ( 1.5 - x + x * y)** 2 + ( 2.25 - x + x * y** 2 )** 2 + ( 2.625 - x + x * y** 3 )** 2

# 初始猜测
initial_guess = [ 1 , 1 ]

# 执行优化
result = minimize(beale, initial_guess)

# 提取结果
x_opt, y_opt = result.x
optimal_value = beale(result.x)

# 打印结果
print ( f"x 的最佳值：{x_opt} " )
print ( f"y 的最佳值：{y_opt} " )
print ( f"Beale 函数的最小值：{optimal_value} " )

# 绘图
x = np.linspace(- 4 , 4 , 100 )
y = np.linspace(- 4 , 4 , 100 )
X, Y = np.meshgrid(x, y)
Z = beale([X, Y])

# 3D 表面图
fig = plt.figure(figsize=( 14 , 8 ))

ax = fig.add_subplot( 111 , projection= '3d' )
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(X.min(), X.max())
colors = cm.hsv(norm_plt(X))
surf = ax.plot_surface(X, Y, Z, facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       linewidth = 0.5, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.grid()

ax.scatter(x_opt, y_opt, optimal_value, color= 'red' , s= 50 , label= '最优解' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'Beale' )
ax.set_title( '3D Beale' )
ax.legend()

plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5.二次规划问题

from scipy.optimize import minimize, LinearConstraint

# 目标函数
def  quadratic_function ( vars ):
    x, y = vars
    return x** 2 + 2 *x*y + y** 2

# 约束
def  constrain1 ( vars ):
    x, y = vars
    return x + y - 1

def  constrain2 ( vars ):
    x, y = vars
    return x - y

# 定义约束
constrains = [
    { 'type' : 'ineq' , 'fun' : constrain1},
    { 'type' : 'ineq' , 'fun' : constrain2}
]

# 初始猜测
initial_guess = [ 0 , 0 ]

# 执行优化
result = minimize(quadratic_function, initial_guess,constraints=constraints)

# 提取结果
x_opt, y_opt = result.x
optimal_value = quadratic_function(result.x)

# 打印结果
print ( f"x 的最佳值：{x_opt} " )
print ( f"y 的最佳值：{y_opt} " )
print ( f"二次函数的最小值：{optimal_value} " )

# 绘图
x = np.linspace(- 1 , 2 , 100 )
y = np.linspace(- 1 , 2 , 100 )
X, Y = np.meshgrid(x, y)
Z = quadratic_function([X, Y])

# 3D 表面图
fig = plt.figure(figsize=( 14 , 8 ))

ax = fig.add_subplot( 111 , projection= '3d' )
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(X.min(), X.max())
colors = cm.hsv(norm_plt(X))
surf = ax.plot_surface(X, Y, Z, facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       linewidth = 0.5, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 设置X、Y、Z面的背景是白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.grid()
ax.scatter(x_opt, y_opt, optimal_value, color= 'red' , s= 50 , label= 'optimal sol' )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'quadratic function' )
ax.set_title('3D quadratic function' )
ax.legend()

plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>












