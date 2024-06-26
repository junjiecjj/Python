


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_01
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from sympy import *
from matplotlib import pyplot as plt

x = Symbol('x')

f_x = x**2 + 1/2
# f_x = x**2 - 1/2

f_x_fcn = lambdify([x],f_x)

integral_f_x = integrate(f_x, x)
integral_f_x_fcn = lambdify([x], integral_f_x)

a = 0 # lower bound
b = 1 # upper bound

num = 201; # number of mesh grids
x_array     = np.linspace(-0.2,1.2, num)
x_a_b_array = np.linspace(a, b, num)

y_array     = f_x_fcn(x_array)
y_a_b_array = f_x_fcn(x_a_b_array)

fig, ax = plt.subplots()
ax.plot(x_array, y_array, 'b')
ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')

ax.fill_between(x_a_b_array, y_a_b_array, edgecolor = 'none', facecolor = '#DBEEF3')

ax.set_xlim(-0.2, 1.2)
# ax.set_ylim(np.floor(y_array.min()),
#             np.ceil(y_array.max()))
ax.set_ylim(-1,2)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

integral_a_b = integral_f_x_fcn(b) - integral_f_x_fcn(a)

integral_a_b_v2 = integrate(f_x, (x, a, b))
integral_a_b_v2 = float(integral_a_b_v2)

ax.set_title(r'$\int_a^b  f(x) dx= %0.4f$'%integral_a_b)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_02
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from sympy import *
from matplotlib import pyplot as plt

x = Symbol('x')

f_x = exp(-x**2)

# integrate(f_x,(x,-oo,oo))

# integrate(exp(-x**2/2),(x,-oo,oo))

f_x_fcn = lambdify([x],f_x)

integral_f_x = integrate(f_x, x)
integral_f_x_fcn = lambdify([x],integral_f_x)

a = -0.5
b = 1

num = 201; # number of mesh grids
x_array     = np.linspace(-3,3,num)
x_a_b_array = np.linspace(a,b,num)

y_array     = f_x_fcn(x_array)
y_a_b_array = f_x_fcn(x_a_b_array)

fig, ax = plt.subplots()
ax.plot(x_array, y_array, 'b')
ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')

ax.fill_between(x_a_b_array, y_a_b_array, edgecolor = 'none', facecolor = '#DBEEF3')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(np.floor(y_array.min()), np.ceil(y_array.max()))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

integral_a_b = integral_f_x_fcn(b) - integral_f_x_fcn(a)

integral_a_b_v2 = integrate(f_x, (x, a, b))
integral_a_b_v2 = float(integral_a_b_v2)

ax.set_title(r'$\int_a^b  f(x) = %0.3f$'%integral_a_b)
plt.show()
#%% plot integral function

t = Symbol('t')
integral_f_x_oo_t = integrate(f_x, (x,-oo,t))
integral_f_x_oo_t_fcn = lambdify([t],integral_f_x_oo_t)

t_array     = np.linspace(-3,3,num)

integral_f_x_oo_t_array = integral_f_x_oo_t_fcn(t_array)

fig, ax = plt.subplots()
ax.plot(t_array, integral_f_x_oo_t_array, 'b')
ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')

ax.axhline(y = integral_f_x_oo_t_fcn(a), color = 'r', linestyle = '-')

ax.axhline(y = integral_f_x_oo_t_fcn(b), color = 'r', linestyle = '-')

ax.set_xlim(t_array.min(), t_array.max())
ax.set_ylim(np.floor(integral_f_x_oo_t_array.min()), np.ceil(integral_f_x_oo_t_array.max()))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax.set_xlabel('x')
ax.set_ylabel('Integral, F(x)')
ax.grid(linestyle='--', linewidth=0.25, color=[0.75,0.75,0.75])

plt.show()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_03
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Bk3_Ch18_03

from sympy import oo, erf, lambdify
import numpy as np

x_array = np.linspace(-3,3,100)

erf_x_fcn = lambdify(x, erf(x))
y_array = erf_x_fcn(x_array)

fig, ax = plt.subplots()
ax.plot(x_array, y_array, 'b')

ax.axhline(y = erf(oo), color = 'r', linestyle = '-')
ax.axhline(y = erf(-oo), color = 'r', linestyle = '-')
ax.axhline(y = erf(0), color = 'r', linestyle = '-')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(-1,1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
ax.set_xlabel('x')
ax.set_ylabel('erf(x)')
ax.grid(linestyle='--', linewidth=0.25, color=[0.75,0.75,0.75])
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_04,  二重积分:类似二重求和
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from sympy.abc import x, y, s, t
from sympy import *

f_xy = exp(- x**2 - y**2);

f_x_y_double_integrate = integrate(f_xy, (y, -oo, y), (x, -oo, x))
print(f_x_y_double_integrate)

f_x_y_volume = integrate(f_xy, (y, -oo, oo), (x, -oo, oo))
print(f_x_y_volume)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_05, “偏积分”:类似偏求和
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from sympy.abc import x, y, s, t
from sympy import *

f_xy = exp(- x**2 - y**2);

f_x_partial_integrate = integrate(f_xy, (y,-oo,oo))
print(f_x_partial_integrate)

f_y_partial_integrate = integrate(f_xy, (x,-oo,oo))
print(f_y_partial_integrate)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_06,  估算圆周率:牛顿法
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from scipy.special import factorial

n_array = np.linspace(0,10,11)

expansion = factorial(2*n_array)/2**(4*n_array + 2)/(factorial(n_array))**2/(2*n_array - 1)/(2*n_array + 3)

est_pi = 24*(np.sqrt(3)/32 - np.cumsum(expansion))

fig, ax = plt.subplots()

plt.axhline(y=np.pi, color='r', linestyle='-')
plt.plot(n_array,est_pi, color = 'b', marker = 'x')

plt.tight_layout()
plt.xlabel('n')
plt.ylabel('Estimate of $\pi$')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_07
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
from sympy import *

x = Symbol('x')
# f_x = x**2 + x + 1
f_x = x**2
# f_x = exp(-x**2)

f_x_fcn = lambdify([x], f_x)

integral_f_x = integrate(f_x, x)
integral_f_x_fcn = lambdify([x], integral_f_x)

a = 0 # lower bound
b = 1 # upper bound

integral_a_b = integral_f_x_fcn(b) - integral_f_x_fcn(a)

integral_a_b_v2 = integrate(f_x, (x, a, b))
integral_a_b_v2 = float(integral_a_b_v2)

print('$\int_a^b  f(x)dx = %0.3f$'%integral_a_b)


#%% Visualizations

num_interval = 20
delta_x = (b - a)/num_interval

x_array = np.linspace(a, b, num_interval+1)
y_array = f_x_fcn(x_array)

x_array_fine = np.linspace(a, b, 200)
y_array_fine = f_x_fcn(x_array_fine)


fig = plt.figure(figsize=(15,5))

#########(1) Left Riemann sum
ax = fig.add_subplot(1,3,1)

plt.plot(x_array_fine, y_array_fine, color = '#0070C0')

# left endpoints
x_left = x_array[:-1]
y_left = y_array[:-1]

plt.plot(x_left, y_left,'rx',markersize=10)

# plot the rectangles
plt.bar(x_left, y_left, width=delta_x, facecolor = '#DEEAF6', align='edge', edgecolor='#B2B2B2')

ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')

left_riemann_sum = np.sum(f_x_fcn(x_left) * delta_x)

plt.title('Left Riemann sum (N = %0.0f) = %0.3f' %(num_interval,left_riemann_sum))
plt.xlim((a,b))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('x')
plt.ylabel('f(x)')

#########(2) Middle Riemann sum
ax = fig.add_subplot(1,3,2)
plt.plot(x_array_fine, y_array_fine, color = '#0070C0')

# middle endpoints
x_mid = (x_array[:-1] + x_array[1:])/2
y_mid = f_x_fcn(x_mid)

plt.plot(x_mid,y_mid,'rx',markersize=10)

# plot the rectangles
plt.bar(x_mid,y_mid, width=delta_x,  facecolor = '#DEEAF6', edgecolor='#B2B2B2')

ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')

mid_riemann_sum = np.sum(f_x_fcn(x_mid) * delta_x)

plt.title('Middle Riemann sum (N = %0.0f) = %0.3f' %(num_interval,mid_riemann_sum))
plt.xlim((a,b))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


#########(3) Right Riemann sum
ax = fig.add_subplot(1,3,3)
plt.plot(x_array_fine,y_array_fine, color = '#0070C0')

# right endpoints
x_right = x_array[1:]
y_right = f_x_fcn(x_right)

plt.plot(x_right,y_right,'rx',markersize=10)

# plot the rectangles
plt.bar(x_right, y_right, width = -delta_x, facecolor = '#DEEAF6', align='edge', edgecolor='#B2B2B2')

ax.axvline(x = a, color = 'r', linestyle = '-')
ax.axvline(x = b, color = 'r', linestyle = '-')

right_riemann_sum = np.sum(f_x_fcn(x_right) * delta_x)

plt.title('Right Riemann sum (N = %0.0f) = %0.3f' %(num_interval,right_riemann_sum))
plt.xlim((a,b))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##########     # Bk3_Ch18_08,  数值积分:用简单几何形状近似
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sympy.abc import x, y
from sympy import *

plt.close('all')

f_xy = exp(- x**2 - y**2);

f_xy_fcn = lambdify([x, y], f_xy)

a = -2; b = 1
c = -1; d = 2

x_array_fine = np.linspace(a, b, 300)
y_array_fine = np.linspace(c, d, 300)

xx_fine,yy_fine = np.meshgrid(x_array_fine,y_array_fine)
zz_fine = f_xy_fcn(xx_fine, yy_fine)

### 二元函数曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(xx_fine,yy_fine, zz_fine, color = '#0070C0', rstride=10, cstride=10, linewidth = 0.25)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z = f(x,y)')

ax.set_xlim((a,b))
ax.set_ylim((c,d))
ax.set_zlim((0,zz_fine.max()))
ax.grid(False)
ax.view_init(azim=-135, elev=30)
ax.set_proj_type('ortho')
plt.show()

#%% 不断减小步长提高估算精度
num_array = [5, 10, 15, 20]

for num in num_array:
    x_array = np.linspace(a, b - (b - a)/num,num)
    y_array = np.linspace(c, d - (d - c)/num,num)

    xx,yy = np.meshgrid(x_array,y_array)
    xx_array = xx.ravel()
    yy_array = yy.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zz_array = np.zeros_like(yy_array)

    dx = np.ones_like(yy_array)/num*(b - a)
    dy = np.ones_like(yy_array)/num*(d - c)

    dz = f_xy_fcn(xx_array, yy_array)

    ax.bar3d(xx_array, yy_array, zz_array, dx, dy, dz, shade=False, color = '#DEEAF6', edgecolor = '#B2B2B2')

    # ax.scatter(xx_array, yy_array, dz, c=dz, cmap='RdYlBu_r',marker = '.')

    # ax.plot_wireframe(xx_fine,yy_fine, zz_fine,
    #                   color = '#0070C0',
    #                   rstride=10, cstride=10,
    #                   linewidth = 0.25)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z = f(x,y)')

    ax.set_xlim((a,b))
    ax.set_ylim((c,d))
    ax.set_zlim((0,zz_fine.max()))
    ax.grid(False)
    ax.view_init(azim=-135, elev=30)
    ax.set_proj_type('ortho')
    estimated_volume = dz.sum()*(b - a)/num*(d - c)/num
    ax.set_title('Estimated volume = %0.3f'%estimated_volume)
plt.show()
volume = integrate(f_xy, (y, c, d), (x, a, b))
volume = volume.evalf()
print('==== real Volume ====')
print(volume)


































































































































































































































































