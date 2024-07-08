

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     Bk3_Ch17_01
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from sympy import lambdify, diff, evalf, sin, exp
from sympy.abc import x
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

f_x = exp(-x**2)

x_array   = np.linspace(-3, 3, 100)
x_0_array = np.linspace(-2.5, 2.5, 5)
# f_x.evalf(subs = {x: 0})

f_x_fcn = lambdify(x, f_x)
f_x_array = f_x_fcn(x_array)

#  高斯函数不同点处二次近似
plt.close('all')

colors = plt.cm.rainbow(np.linspace(0, 1, len(x_0_array)))

f_x_1_diff = diff(f_x, x)
# f_x_1_diff_fcn = lambdify(x, f_x_1_diff)

f_x_2_diff = diff(f_x, x,2)
# f_x_2_diff_fcn = lambdify(x, f_x_2_diff)

fig, ax = plt.subplots(figsize = (12, 12))

ax.plot(x_array, f_x_array, linewidth = 6, color = 'black')
ax.set_xlabel("$\it{x}$")
ax.set_ylabel("$\it{f}(\it{x})$")

for i in np.arange(len(x_0_array)):
    color = colors[i,:]
    x_0 = x_0_array[i]
    y_0 = f_x.evalf(subs = {x: x_0})
    x_t_array = np.linspace(x_0-1, x_0+1, 50)

    b = f_x_1_diff.evalf(subs = {x: x_0})
    a = f_x_2_diff.evalf(subs = {x: x_0})

    second_order_f = a/2*(x - x_0)**2 + b*(x - x_0) + y_0
    second_order_f_fcn = lambdify(x, second_order_f)
    second_order_f_array = second_order_f_fcn(x_t_array)

    ax.plot(x_t_array, second_order_f_array, ls = '-', linewidth = 2, color = color)
    ax.plot(x_0, y_0, marker = '.', color = color, markersize = 22)

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim((x_array.min(),x_array.max()))

ax.set_xlim(-3,3)
ax.set_ylim(-0.25,1.25)
plt.show()


#### 高斯函数一阶导数不同点处二次近似
f_x_1_diff_new = diff(f_x, x)
print(f_x_1_diff)
f_x_1_diff_fcn_new = lambdify(x, f_x_1_diff_new)
f_x_1_diff_array_new = f_x_1_diff_fcn_new(x_array)

colors = plt.cm.rainbow(np.linspace(0, 1, len(x_0_array)))

f_x_1_diff = diff(f_x, x, 2)
f_x_1_diff_fcn = lambdify(x,f_x_1_diff)

f_x_2_diff = diff(f_x, x, 3)
f_x_2_diff_fcn = lambdify(x,f_x_2_diff)

fig, ax = plt.subplots(figsize = (12, 12))
ax.plot(x_array, f_x_1_diff_array_new, linewidth = 4)
ax.set_xlabel("$\it{x}$")
ax.set_ylabel("$\it{f}(\it{x})$")

for i in np.arange(len(x_0_array)):
    color = colors[i,:]
    x_0 = x_0_array[i]
    y_0 = f_x_1_diff_new.evalf(subs = {x: x_0})
    x_t_array = np.linspace(x_0 - 1, x_0 + 1, 50)

    b = f_x_1_diff.evalf(subs = {x: x_0})
    a = f_x_2_diff.evalf(subs = {x: x_0})

    second_order_f = a/2*(x - x_0)**2 + b*(x - x_0) + y_0
    second_order_f_fcn = lambdify(x,second_order_f)
    second_order_f_array = second_order_f_fcn(x_t_array)

    ax.plot(x_t_array, second_order_f_array, linewidth = 2, color = color)
    ax.plot(x_0,y_0,marker = '.', color = color, markersize = 12)

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim((x_array.min(),x_array.max()))

ax.set_xlim(-3,3)
ax.set_ylim(-1.25,1.25)
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     Bk3_Ch17_02
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sympy import latex, lambdify, diff, sin, log, exp, series
from sympy.abc import x
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

f_x = exp(x)
x_array = np.linspace(-2,2,100)
x_0 = 0 # expansion point

# f_x = sin(x)
# x_array = np.linspace(-10,10,100)
# x_0 = np.pi/6 # expansion

# f_x = log(x + 1) # ln(y + 1) = r
# x_array = np.linspace(-0.8,2,100)

y_0 = f_x.evalf(subs = {x: x_0})

f_x_fcn = lambdify(x, f_x)
f_x_array = f_x_fcn(x_array)

# Visualization
plt.close('all')
fig, ax = plt.subplots(figsize = (12, 12))

ax.plot(x_array, f_x_array, 'k', linewidth = 4)
ax.plot(x_0, y_0, 'xr', markersize = 12)
ax.set_xlabel("$\it{x}$")
ax.set_ylabel("$\it{f}(\it{x})$")

highest_order = 5
order_array = np.arange(0, highest_order + 1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(order_array)))

i = 0
for order in order_array:
    f_series = f_x.series(x, x_0, order + 1).removeO()
    # order + 1 = number of terms
    f_series_fcn = lambdify(x, f_series)
    f_series_array = f_series_fcn(x_array)
    ax.plot(x_array, x_array*0 + f_series_array, linewidth = 2, color = colors[i,:], label = 'Order = %0.0f'%order)
    i += 1

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim(x_array.min(),x_array.max())
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# ax.set_ylim(x_array.min(),x_array.max())
# ax.set_aspect('equal', 'box')
plt.legend()
plt.show()


#%% Error
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)

ax.plot(x_array, f_x_array, 'k', linewidth = 1.5)
ax.plot(x_0, y_0, 'xr', markersize = 12)
ax.set_xlabel("$\it{x}$")
ax.set_ylabel("$\it{f}(\it{x})$")

highest_order = 2

f_series = f_x.series(x, x_0, highest_order + 1).removeO()
# order + 1 = number of terms

f_series_fcn = lambdify(x,f_series)
f_series_array = f_series_fcn(x_array)
f_series_array = x_array*0 + f_series_array

ax.plot(x_array, f_series_array, linewidth = 1.5, color = 'b')

ax.fill_between(x_array, f_x_array, x_array*0 + f_series_array, color = '#DEEAF6')

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim(x_array.min(),x_array.max())

ax.set_ylim(np.floor(f_x_array.min()), np.ceil(f_x_array.max()))
# ax.set_aspect('equal', 'box')
# plt.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

## 2
ax = fig.add_subplot(1, 2, 2)
error = f_x_array - f_series_array
ax.plot(x_array, error, 'r', linewidth = 1.5)
ax.fill_between(x_array, error, color = '#DEEAF6')
plt.axhline(y=0, color='k', linestyle='--', linewidth = 0.25)
ax.set_xlabel("$\it{x}$")
ax.set_ylabel("Error")

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim(x_array.min(),x_array.max())
ax.set_ylim(np.floor(error.min()),np.ceil(error.max()))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     Bk3_Ch17_03, 二元泰勒展开
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.close('all')
import numpy as np
from sympy import lambdify, diff, exp, latex, simplify
from sympy.abc import x, y
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


num = 301       # number of mesh grids
x_array = np.linspace(-1.5, 1.5, num)  # (301,)
y_array = np.linspace(-1.5, 1.5, num)  # (301,)

# global mesh
xx, yy = np.meshgrid(x_array, y_array) #  (301, 301)
num_stride = 5

f_xy = exp(-x**2 - y**2)
f_xy_fcn = lambdify([x, y], f_xy)
f_xy_zz = f_xy_fcn(xx, yy)  # (301, 301)

# expansion point
x_a = -0.1
y_b = -0.2

# local mesh
x_a_array = np.linspace(x_a - 0.5, x_a + 0.5, 101)
y_b_array = np.linspace(y_b - 0.5, y_b + 0.5, 101)

xx_local, yy_local = np.meshgrid(x_a_array, y_b_array)
f_xy_zz_local = f_xy_fcn(xx_local, yy_local)  #  (101, 101)


# expansion point
f_ab = f_xy_fcn(x_a, y_b) # 0.9512


#%% constant approximation, 用常数函数估计二元高斯函数
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))

ax.plot_wireframe(xx, yy, f_xy_zz, color = [0.5,0.5,0.5], rstride=num_stride, cstride=num_stride, linewidth = 0.6)
ax.plot(x_a, y_b, f_ab, marker = 'x', color = 'r', markersize = 12)
approx_zero_order = f_ab + xx_local*0 # (101, 101)
ax.plot_wireframe(xx_local, yy_local, approx_zero_order, color = [1,0,0], rstride = num_stride, cstride = num_stride, linewidth = 0.6)

ax.set_proj_type('ortho')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2) = e^{-x^2-y^2}$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(f_xy_zz.min(), 1.5)

ax.view_init(azim=-145, elev=30)
# ax.view_init(azim=-90, elev=0)

plt.tight_layout()
ax.grid(False)
plt.show()

#%% first order approximation, 用二元一次函数估计二元高斯函数
df_dx = f_xy.diff(x)
df_dx_fcn = lambdify([x,y], df_dx)
df_dx_a_b = df_dx_fcn(x_a, y_b) # # 一个数

df_dy = f_xy.diff(y)
df_dy_fcn = lambdify([x, y], df_dy)
df_dy_a_b = df_dy_fcn(x_a, y_b) # 一个数

approx_first_order = approx_zero_order + df_dx_a_b*(xx_local - x_a) + df_dy_a_b*(yy_local - y_b) # (101, 101)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))
ax.plot_wireframe(xx,yy, f_xy_zz,  color = [0.5,0.5,0.5],  rstride=num_stride, cstride=num_stride, linewidth = 0.25)
ax.plot_wireframe(xx_local, yy_local, approx_first_order, color = [1,0,0], rstride=num_stride, cstride=num_stride, linewidth = 0.25)
ax.plot(x_a,y_b,f_ab, marker = 'x', color = 'r', markersize = 12)
ax.set_proj_type('ortho')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$ ')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(f_xy_zz.min(), 1.5)

ax.view_init(azim=-145, elev=30)
# ax.view_init(azim=-90, elev=0)

plt.tight_layout()
ax.grid(False)
plt.show()

#%% second order approximation, 用二次函数估计二元高斯函数

d2f_dxdx = f_xy.diff(x,2)
d2f_dxdx_fcn = lambdify([x,y],d2f_dxdx)
d2f_dxdx_a_b = d2f_dxdx_fcn(x_a,y_b)

d2f_dxdy = f_xy.diff(x,y)
d2f_dxdy_fcn = lambdify([x,y],d2f_dxdy)
d2f_dxdy_a_b = d2f_dxdy_fcn(x_a,y_b)

d2f_dydy = f_xy.diff(y,2)
d2f_dydy_fcn = lambdify([x,y],d2f_dydy)
d2f_dydy_a_b = d2f_dydy_fcn(x_a,y_b)

approx_second_order = approx_first_order + (d2f_dxdx_a_b*(xx_local - x_a)**2 + 2*d2f_dxdy_a_b*(xx_local - x_a)*(yy_local - y_b) + d2f_dydy_a_b*(yy_local - y_b)**2)/2 # (101, 101)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (10, 10))
ax.plot_wireframe(xx,yy, f_xy_zz, color = [0.5,0.5,0.5], rstride=num_stride, cstride=num_stride, linewidth = 0.5)
ax.plot_wireframe(xx_local,yy_local, approx_second_order, color = [1,0,0], rstride=num_stride, cstride=num_stride, linewidth = 0.5)
ax.plot(x_a, y_b, f_ab, marker = '*', color = 'g', markersize = 12, markeredgewidth = 4)
ax.set_proj_type('ortho')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(f_xy_zz.min(), 1.5)

ax.view_init(azim=-145, elev=30)
# ax.view_init(azim=-90, elev=0)

plt.tight_layout()
ax.grid(False)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Bk3_Ch17_04.py, 数值微分:估算一阶导数
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import x
from sympy import latex, lambdify, diff, sin, log, exp

def num_diff(f, a, method, dx):
    # f: function handle
    # a: expansion point
    # method: 'forward', 'backward', and 'central'
    # dx: step size

    if method == 'central':
        return (f(a + dx) - f(a - dx))/(2*dx)
    elif method == 'forward':
        return (f(a + dx) - f(a))/dx
    elif method == 'backward':
        return (f(a) - f(a - dx))/dx
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

f_x = exp(-x**2)
x_array = np.linspace(-3, 3, 100)
a_array = np.linspace(-2.5, 2.5, 11)

f_x_fcn = lambdify(x, f_x)
f_x_array = f_x_fcn(x_array)

f_x_1_diff = diff(f_x, x)
f_x_1_diff_fcn = lambdify(x, f_x_1_diff)
f_x_1_diff_array = f_x_1_diff_fcn(x_array)

#%% visualization
fig = plt.figure( figsize = (10, 10))
ax = fig.add_subplot(2, 1, 1)

ax.plot(x_array, f_x_array, '#0070C0', linewidth = 1.5)
ax.set_ylim(np.floor(f_x_array.min()), np.ceil(f_x_array.max()))
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_xlim((x_array.min(),x_array.max()))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


ax = fig.add_subplot(2, 1, 2)
ax.plot(x_array, f_x_1_diff_array, '#0070C0', linewidth = 1.5)
ax.set_xlabel('x')
ax.set_ylabel('f\'(x)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim((x_array.min(),x_array.max()))
plt.show()


#%% numerical methods
dx = 0.2
diff_central  = num_diff(f_x_fcn, a_array, 'central', dx)
diff_forward  = num_diff(f_x_fcn, a_array, 'forward', dx)
diff_backward = num_diff(f_x_fcn, a_array, 'backward', dx)

fig, ax = plt.subplots( figsize = (10, 10))
ax.plot(x_array, f_x_1_diff_array, '#0070C0', linewidth = 1.5)

ax.plot(a_array, diff_central, marker = '.', markersize = 12, linestyle = 'none', label = 'central')

ax.plot(a_array, diff_forward, marker = '>', markersize = 12, linestyle = 'none', label = 'forward')

ax.plot(a_array, diff_backward, marker = '<', markersize = 12, linestyle = 'none', label = 'backward')

ax.set_xlabel('x')
ax.set_ylabel('f\'(x)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim((x_array.min(),x_array.max()))
plt.axhline(y=0, color='k', linestyle='--', linewidth = 0.25)
plt.legend()
plt.show()


#%% varying step size
dx_array = np.linspace(0.01,0.2,20)
a = 1
diff_central  = num_diff(f_x_fcn, a, 'central', dx_array)
diff_forward  = num_diff(f_x_fcn, a, 'forward', dx_array)
diff_backward = num_diff(f_x_fcn, a, 'backward',dx_array)

f_x_1_diff_a = f_x_1_diff_fcn(a)

fig, ax = plt.subplots(figsize = (10, 10))

ax.plot(dx_array, diff_central, linewidth = 1.5, marker = '.',label = 'central')

ax.plot(dx_array, diff_forward, linewidth = 1.5, marker = '>',label = 'forward')

ax.plot(dx_array, diff_backward, linewidth = 1.5, marker = '<',label = 'backward')

plt.axhline(y=f_x_1_diff_a, color='k', linestyle='--', linewidth = 0.25,label = 'analytical')

ax.set_xlim((dx_array.min(),dx_array.max()))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
ax.set_xlabel('\u0394x')
ax.set_ylabel('f\'(x)')
plt.show()














