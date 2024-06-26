








# Bk3_Ch15_01

from sympy import latex, lambdify, limit, log, oo
from sympy.abc import x
import numpy as np
from matplotlib import pyplot as plt

f_x = (1 + 1/x)**x

x_array = np.linspace(0.1,1000,1000)

f_x_fcn = lambdify(x,f_x)
f_x_array = f_x_fcn(x_array)

f_x_oo_limit = limit(f_x,x,oo)

# visualization

plt.close('all')

fig, ax = plt.subplots()

ax.plot(x_array, f_x_array, linewidth = 1.5)
ax.axhline(y = f_x_oo_limit, color = 'r')

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim(x_array.min(),x_array.max())
plt.xscale("log")
ax.set_xlabel('$\it{x}$',fontname = 'Times New Roman')
ax.set_ylabel('$%s$' % latex(f_x), fontname = 'Times New Roman')

plt.grid(True, which="both", ls="-")
plt.show()



# Bk3_Ch15_02

from sympy import latex, lambdify, limit, log, oo
from sympy.abc import x
import numpy as np
from matplotlib import pyplot as plt

f_x = 1/(1 + 2**(-1/x))

f_x_fcn = lambdify(x,f_x)

# right limit
x_array_right = np.linspace(0.01,4,500)
f_x_array_right = f_x_fcn(x_array_right)
f_x_0_limit_right = limit(f_x,x,0,'+')

# left limit
x_array_left = np.linspace(-4,-0.01,500)
f_x_array_left = f_x_fcn(x_array_left)
f_x_0_limit_left = limit(f_x,x,0,'-')

# visualization

plt.close('all')

fig, ax = plt.subplots()

ax.plot(x_array_right, f_x_array_right, linewidth = 1.5, color = 'b')
ax.axhline(y = f_x_0_limit_right, color = 'r')

ax.plot(x_array_left, f_x_array_left, linewidth = 1.5, color = 'b')
ax.axhline(y = f_x_0_limit_left, color = 'r')

ax.axvline(x = 0,   color = 'k')
ax.axhline(y = 0.5, color = 'k')

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xlim(x_array_left.min(),x_array_right.max())

ax.set_xlabel('$\it{x}$',fontname = 'Times New Roman')
ax.set_ylabel('$%s$' % latex(f_x), fontname = 'Times New Roman')
plt.show()





# Bk3_Ch15_03,  导数就是变化率

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
from sympy import latex, lambdify

def plot_secant(x0, y0, x1, y1, color):
    k = (y1 - y0)/(x1 - x0)
    x = np.linspace(-1,4,100)
    secant_y_x = k*(x - x0) + y0
    plt.plot(x, secant_y_x, color = color, linewidth = 0.25)

delta_Xs = np.linspace(0.1, 1, 2)

from sympy.abc import x

f_x = x**2

x_array = np.linspace(-1,4,100)

f_x_fcn = lambdify(x,f_x)
y_array = f_x_fcn(x_array)

x0 = 1
y0 = f_x_fcn(x0)
fig, ax = plt.subplots(figsize = (8,8))

plt.plot(x_array, y_array, color = '#00448A',  linewidth = 1.25)
plt.plot(x0, y0, color = '#92D050', marker = 'x', markersize = 12)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(delta_Xs)))

for i in np.linspace(1,len(delta_Xs),len(delta_Xs)):
    x1 = x0 + delta_Xs[int(i)-1]
    y1 = f_x_fcn(x1)
    plt.plot(x1, y1, color = '#00448A', marker = 'x', markersize = 12)
    plot_secant(x0, y0, x1, y1, colors[int(i)-1])

plt.xlabel('X')
plt.ylabel('$y = f(x)$')
ax.set_title('$f(x) = %s$' % latex(f_x))
ax.set_xlim(0, 2)
ax.set_ylim(-1, 4)
plt.show()



######
fig, ax = plt.subplots()

plt.plot(x0, y0, color = '#92D050', marker = 'x', markersize = 12)

colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(delta_Xs)))

for i in np.linspace(1,len(delta_Xs),len(delta_Xs)):
    x1 = x0 + delta_Xs[int(i)-1]
    y1 = f_x_fcn(x1)
    k = (y1 - y0)/(x1 - x0)
    plt.plot(delta_Xs[int(i)-1], k, color = colors[int(i)-1], marker = 'x', markersize = 12)

plt.xlabel('$\Delta$X')
plt.ylabel('$k$')
ax.set_xlim(0, 1)
ax.set_ylim(2, 3)
plt.show()




# Bk3_Ch15_04,  二次函数、一阶导数、二阶导数

from sympy import latex, lambdify, diff, sin
from sympy.abc import x
import numpy as np
from matplotlib import pyplot as plt

# function
f_x = x**2 - 2

x_array = np.linspace(-2,2,100)

f_x_fcn = lambdify(x,f_x)
f_x_array = f_x_fcn(x_array)

# first order derivative

f_x_1_diff = diff(f_x,x)
print(f_x_1_diff)
f_x_1_diff_fcn = lambdify(x,f_x_1_diff)
f_x_1_diff_array = f_x_1_diff_fcn(x_array)

# second order derivative

f_x_2_diff = diff(f_x, x, 2)
print(f_x_2_diff)
f_x_2_diff_fcn = lambdify(x, f_x_2_diff)
f_x_2_diff_array = f_x_2_diff_fcn(x_array)
f_x_2_diff_array = f_x_2_diff_array + x_array*0

#%% plot first-, second-order derivatives as functions

fig, ax = plt.subplots(3,1)

# original function

ax[0].plot(x_array, f_x_array, linewidth = 1.5)
ax[0].hlines(y=0, xmin = x_array.min(), xmax = x_array.max(),
             color='r', linestyle='--')
ax[0].vlines(x=0, ymin = f_x_array.min(), ymax = f_x_array.max(),
             color='r', linestyle='--')
ax[0].set_title('$f(x) = %s$' % latex(f_x))
ax[0].set_ylabel('$f(x)$')
ax[0].set_xlim((x_array.min(),x_array.max()))
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xticklabels([])
ax[0].grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# first-order derivative

ax[1].plot(x_array, f_x_1_diff_array, linewidth = 1.5)
ax[1].hlines(y=0, xmin = x_array.min(), xmax = x_array.max(),
             color='r', linestyle='--')
ax[1].vlines(x=0,
             ymin = f_x_1_diff_array.min(),
             ymax = f_x_1_diff_array.max(),
             color='r', linestyle='--')

ax[1].set_xlabel("$\it{x}$")
ax[1].set_title('$f\'(x) = %s$' % latex(f_x_1_diff))
ax[1].set_ylabel('$f\'(x)$')
ax[1].grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax[1].set_xlim((x_array.min(),x_array.max()))
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

# second-order derivative

ax[2].plot(x_array, f_x_2_diff_array, linewidth = 1.5)
ax[2].hlines(y=0, xmin = x_array.min(), xmax = x_array.max(),
             color='r', linestyle='--')
ax[2].vlines(x=0,
             ymin = f_x_2_diff_array.min(),
             ymax = f_x_2_diff_array.max(),
             color='r', linestyle='--')

ax[2].set_xlabel("$\it{x}$")
ax[2].set_title('$f^{(2)}(x) = %s$' % latex(f_x_2_diff))
ax[2].set_ylabel('$f\'(x)$')
ax[2].grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax[2].set_xlim((x_array.min(),x_array.max()))
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
plt.show()




























































































































































































































































































































































































































































































































































































