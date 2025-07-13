





#%% Bk3_Ch19_01

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from sympy import lambdify, diff, exp
from sympy.abc import x

f_x = -2*x*exp(-x**2)
obj_f = lambdify(x, f_x)
# objective function

# def obj_f(x):
#     return -4*x*np.exp(-x**2)

result = optimize.minimize_scalar(obj_f)

print('=== Success ===')
print(result.success)

x_min = result.x

x_array = np.linspace(-2,2,100)
y_array = obj_f(x_array)

fig, ax = plt.subplots()
plt.plot(x_array,y_array, color = 'b')
# plot the optimal solution
plt.plot(x_min, obj_f(x_min), color = 'r', marker = 'x', markersize = 12)

plt.xlabel('x'); plt.ylabel('f(x)')
plt.xticks(np.linspace(-2, 2, 5)); plt.yticks(np.linspace(-2, 2, 5))
plt.axis('scaled'); ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()

### 一次函数图像和一阶导函数图像,极小值点位置
f_x_1_diff = diff(f_x,x)
f_x_1_diff_fcn = lambdify(x, f_x_1_diff)
f_x_1_diff_array = f_x_1_diff_fcn(x_array)

fig, ax = plt.subplots()
plt.plot(x_array, f_x_1_diff_array, color = 'b')
# plot the optimal solution
plt.plot(x_min, f_x_1_diff_fcn(x_min), color = 'r', marker = 'x',  markersize = 12)

plt.xlabel('x'); plt.ylabel('f\'(x)')
plt.xticks(np.linspace(-2, 2, 5)); plt.yticks(np.linspace(-2, 2, 5))
plt.axis('scaled'); ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()



#%% Bk3_Ch19_02

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds, NonlinearConstraint
import matplotlib.pyplot as plt

def obj_f(x):
    x1 = x[0]
    x2 = x[1]
    obj = -2*x1*np.exp(-x1**2 - x2**2)
    return obj

x0 = [1,1]  # initial guess
# linear_constraint = LinearConstraint([1,1],[1],[1])

def nonlinear_c(x):
    x1 = x[0]
    x2 = x[1]
    nlc = np.abs(x1) + np.abs(x2+1) - 1
    return nlc

nlc = NonlinearConstraint(nonlinear_c, -np.inf, 0)
bounds = Bounds([-1.5, -1.5], [1.5, 1.5])
res = minimize(obj_f, x0, method='trust-constr', bounds = bounds, constraints=[nlc])
optimized_x = res.x

print("==== Optimal solution ====")
print(res.x)
print("==== Optimized objective ====")
print(res.fun)

# Visualization
num = 201; # number of mesh grids
rr = np.linspace(-2,2,num)
xx1, xx2 = np.meshgrid(rr,rr);

yy = obj_f(np.vstack([xx1.ravel(), xx2.ravel()])).reshape((num,num))

fig, ax = plt.subplots()

ax.contourf(xx1, xx2, yy, levels = 20, cmap="RdYlBu_r")
yy_nlc = nonlinear_c(np.vstack([xx1.ravel(), xx2.ravel()])).reshape((num, num))
ax.contour(xx1, xx2, yy_nlc, levels = [0], colors="k")

plt.plot(optimized_x[0], optimized_x[1], 'rx', markersize = 12)

ax.set_xlabel(r'$\it{x}_1$')
ax.set_ylabel(r'$\it{x}_2$')
ax.axis('square')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim([rr.min(),rr.max()])
ax.set_ylim([rr.min(),rr.max()])
plt.show()
plt.close()






































































































































































































































































































