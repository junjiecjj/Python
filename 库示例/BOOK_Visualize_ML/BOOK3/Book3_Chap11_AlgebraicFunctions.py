



# Bk3_Ch11_01

import numpy as np
import matplotlib.pyplot as plt

w_array = np.array([1/5,1/4,1/3,1/2,1,2,3,4,5])
x_array = np.linspace(-2,2,100)

ww, xx = np.meshgrid(w_array,x_array)

b = 0 # y intercept
ff = ww*xx + b

fig, ax = plt.subplots()

colors = plt.cm.jet(np.linspace(0,1,len(w_array)))

for i in np.linspace(1,len(w_array),len(w_array)):
    plt.plot(x_array,ff[:,int(i)-1],
             color = colors[int(i)-1],
             label = '$w_1 = {lll:.2f}$'.format(lll = w_array[int(i)-1]))

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.xticks(np.arange(-2, 2.5, step=0.5))
plt.yticks(np.arange(-2, 2.5,  step=0.5))
plt.axis('scaled')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])



# Bk3_Ch11_02

import numpy as np
import matplotlib.pyplot as plt

a_array = np.linspace(1,6,6)
x_array = np.linspace(-2,2,100)

aa, xx = np.meshgrid(a_array,x_array)
ww = aa*xx**2

fig, ax = plt.subplots()

colors = plt.cm.jet(np.linspace(0,1,6))

for i in np.linspace(1,6,6):
    plt.plot(x_array,ww[:,int(i)-1], color = colors[int(i)-1], label = '$a = {lll:.0f}$'.format(lll = a_array[int(i)-1]))

plt.xlabel('x'); plt.ylabel('f(x)')
plt.legend()
plt.xticks(np.arange(-2, 2.5, step=0.5)); plt.yticks(np.arange(0, 4.5,  step=0.5))
plt.axis('scaled')
ax.set_xlim(-2,2); ax.set_ylim(0,4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


# Bk3_Ch11_03

import math
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-2,2,100);

def plot_curve(x, y):

    fig, ax = plt.subplots()

    plt.xlabel("$\it{x}$")
    plt.ylabel("$\it{f}(\it{x})$")
    plt.plot(x, y, linewidth = 1.5)
    plt.axhline(y=0, color='k', linewidth = 1.5)
    plt.axvline(x=0, color='k', linewidth = 1.5)
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    plt.axis('equal')
    plt.xticks(np.arange(-2, 2.5, step=0.5))
    plt.yticks(np.arange(y.min(), y.max() + 0.5, step=0.5))
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('square')

#%% plot linear, quadratic, and cubic functions

plt.close('all')

# linear function
y = x + 1;
plot_curve(x, y)

# linear function
y = -x + 1;
plot_curve(x, y)

# quadratic function, parabola opens upwards
# y = np.power(x,2) - 2;
y = x**2 - 2;
plot_curve(x, y)

# quadratic function, parabola opens downwards
# y = -np.power(x,2) + 2;
y = -x**2 + 2;
plot_curve(x, y)

# cubic function
# y = np.power(x,3) - x;
y = x**3 - x;
plot_curve(x, y)

# cubic function
# y = -np.power(x,3) + x;
y = -x**3 + x;
plot_curve(x, y)





















































































































































































































































































