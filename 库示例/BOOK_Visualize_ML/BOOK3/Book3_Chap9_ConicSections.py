

#%% Bk3_Ch9_01

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-4, 4, num = 201)
y = np.linspace(-4, 4, num = 201)
m = 1
n = 1.5
xx, yy = np.meshgrid(x,y);
e_array = np.linspace(0, 3,num = 11)
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.RdYlBu(np.linspace(0,1,len(e_array)))
for i in range(0,len(e_array)):
    e = e_array[i]
    ellipse = yy**2 - (e**2 - 1)*xx**2 - 2*xx;
    color_code = colors[i,:].tolist()
    plt.contour(xx, yy, ellipse, levels = [0], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_02

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

a = 1
b = 1

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)

xx,yy = np.meshgrid(x,y);
# k_array = np.linspace(0, 2, num = 11)
# k_array = np.linspace(0, 1, num = 21)
# k_array = np.linspace(0, -2, num = 21)
k_array = np.linspace(0, -1, num = 21)
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.RdYlBu(np.linspace(0,1,len(k_array)))
for i in range(0,len(k_array)):
    k = k_array[i]
    ellipse = (xx/a)**2 + (yy/b)**2 - 2*k*(xx/a)*(yy/b);
    color_code = colors[i,:].tolist()
    plt.contour(xx,yy,ellipse,levels = [1], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_03

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)
m = 1
n = 1.5

xx,yy = np.meshgrid(x, y);
rho_array = np.linspace(-0.95,0,num = 20)
fig, ax = plt.subplots(figsize=(8, 8))
# Create a Rectangle patch
rect = patches.Rectangle((-m, -n), 2*m, 2*n, linewidth = 0.25, edgecolor='k', linestyle = '--', facecolor = 'none')
# Add the patch to the Axes
ax.add_patch(rect)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(rho_array)))
for i in range(0,len(rho_array)):
    rho = rho_array[i]
    ellipse = ((xx/m)**2 - 2*rho*(xx/m)*(yy/n) + (yy/n)**2)/(1 - rho**2);
    color_code = colors[i,:].tolist()
    plt.contour(xx,yy,ellipse,levels = [1], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_04, 超椭圆：和范数有关

import matplotlib.pyplot as plt
import numpy as np

a = 1;
b = 1;

p = [0.5, 1, 2, 3]
q = p

pp, qq = np.meshgrid(p, q)
pp = pp.flatten()
qq = qq.flatten()

x1 = np.linspace(-2, 2, num=11);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)
fig, axes = plt.subplots(ncols=4, nrows=4,  figsize=(12, 12))
for p, q, ax in zip(pp, qq, axes.flat):
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1/a),np.abs(xx2/b))
    else:
        zz = ((np.abs((xx1/a))**p) + (np.abs((xx2/b))**q))**(1./q)
    # plot contour of Lp
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # plot contour of Lp = 1
    ax.contour(xx1, xx2, zz, [1], colors='k', linewidths = 2)
    # decorations

    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('p = ' + str(p) + 'q = ' + str(q))
    ax.set_aspect('equal', adjustable='box')
plt.show()
plt.close()





























































































































































































































































































