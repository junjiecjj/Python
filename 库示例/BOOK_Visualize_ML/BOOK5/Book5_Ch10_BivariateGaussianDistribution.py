




#%% Bk5_Ch10_01


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm # Colormaps

rho     = 0.75
sigma_X = 1
sigma_Y = 2
mu_X = 0
mu_Y = 0

mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.linspace(-width, width, 321)
Y = np.linspace(-width, width, 321)
XX, YY  = np.meshgrid(X, Y)    # (321, 321)
XXYY    = np.dstack((XX, YY))  # (321, 321, 2)
bi_norm = multivariate_normal(mu, Sigma)

#%% visualize joint PDF surface
f_X_Y_joint = bi_norm.pdf(XXYY) # (321, 321)
# f_X_Y_joint = bi_norm.cdf(XXYY) # (321, 321)

# 3D visualization
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, constrained_layout=True)
ax.set_proj_type('ortho')
ax.plot_wireframe(XX,YY, f_X_Y_joint, rstride=10, cstride=10, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f_{X,Y}(x,y)$')
# ax.set_proj_type('ortho')
# ax.xaxis._axinfo["grid"].update({"linewidth":1.25, "linestyle" : ":"})
# ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

# 3D坐标区的背景设置为白色
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(False)
# ax.set_xlim(-width, width)
# ax.set_ylim(-width, width)
# ax.set_zlim(f_X_Y_joint.min(), f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.show()
plt.close()

#%% surface projected along Y to X-Z plane
# 图 2. PDF 函数曲面 fX,Y(x,y)，沿 x 方向的剖面线，σX = 1, σY = 2, ρX,Y = 0.75
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, constrained_layout=True)
ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=10, cstride=0, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour(XX, YY, f_X_Y_joint, levels = 33, zdir='y', offset = XX.max(), cmap = cm.RdYlBu_r)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.show()
plt.close()


fig, ax = plt.subplots()
colors = plt.cm.RdYlBu_r(np.linspace(0, 1,len(X)))
for i in np.arange(1,len(X),5):
    plt.plot(X, f_X_Y_joint[int(i)-1,:], color = colors[int(i)-1])
plt.xlabel(r'x')
plt.ylabel(r'$f_{X,Y}(x,y)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_X_Y_joint.max())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
plt.close()

#%% surface projected along Y to Y-Z plane
# 图 3. PDF 函数曲面 fX,Y(x,y)，沿 y 方向的剖面线，σX = 1, σY = 2, ρX,Y = 0.75
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=0, cstride=10, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour(XX, YY, f_X_Y_joint, levels = 33, zdir='x', offset=YY.max(), cmap=cm.RdYlBu_r)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f_{X,Y}(x,y)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
ax.view_init(azim=-120, elev=30)
plt.show()
plt.close()

fig, ax = plt.subplots()
colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Y)))
for i in np.arange(1,len(X),5):
    plt.plot(X,f_X_Y_joint[:,int(i)-1], color = colors[int(i)-1])

plt.xlabel(r'y')
plt.ylabel(r'$f_{X,Y}(x,y)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_X_Y_joint.max())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
plt.close()


#%% surface projected along Z to X-Y plane
# 图 4. PDF 函数曲面 fX,Y(x,y)，空间等高线和平面填充等高线，σX = 1, σY = 2, ρX,Y = 0.75
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 10))
ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=10, cstride=10, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour3D(XX,YY, f_X_Y_joint, 12,  cmap = 'RdYlBu_r')
# ax.contour(XX, YY, f_X_Y_joint, levels = 12, zdir='z',  offset=0, cmap=cm.RdYlBu_r)
# ax.contour(XX, YY, f_X_Y_joint, levels = 12, zdir='z', cmap=cm.RdYlBu_r)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$f_{X,Y}(x,y)$')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_X_Y_joint.min(),f_X_Y_joint.max())
plt.show()
plt.close()

# Plot filled contours
fig, ax = plt.subplots(figsize=(10, 10))
# Plot bivariate normal
ax.contourf(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
ax.axvline(x = mu_X, color = 'r', linestyle = '--')
ax.axhline(y = mu_Y, color = 'r', linestyle = '--')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
# Plot bivariate normal
ax.contour(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
ax.axvline(x = mu_X, color = 'r', linestyle = '--')
ax.axhline(y = mu_Y, color = 'r', linestyle = '--')
ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.show()
plt.close()

###
import matplotlib.patches as patches
fig, ax = plt.subplots(figsize=(12, 12))
# x1 = mu_X + rho * sigma_X
# x2 = mu_Y + sigma_Y
x1 = mu_X + sigma_X
x2 = mu_Y + rho * sigma_Y
tmp = bi_norm.pdf([x1, x2])
# Plot bivariate normal
ax.contour(XX, YY, f_X_Y_joint, levels = [tmp], cmap=cm.RdYlBu_r)
ax.axvline(x = mu_X, color = 'r', linestyle = '--')
ax.axhline(y = mu_Y, color = 'r', linestyle = '--')

# left_x = bi_norm.pdf([mu_X - sigma_X, mu_Y - sigma_Y])
rect = patches.Rectangle((mu_X - sigma_X, mu_Y - sigma_Y), 2*sigma_X, 2*sigma_Y, linewidth = 1, edgecolor='k', linestyle = '--', facecolor = 'none')
# Add the patch to the Axes
ax.add_patch(rect)

## 椭圆的四个切点
A = [mu_X + rho * sigma_X, mu_Y + sigma_Y]
B = [mu_Y + sigma_X, mu_Y + rho * sigma_Y]
C = [mu_X - rho * sigma_X, mu_Y - sigma_Y]
D = [mu_Y - sigma_X, mu_Y - rho * sigma_Y]
ax.plot(A[0], A[1], marker = 'x', color = 'r', ms = 12)
ax.plot(B[0], B[1], marker = 'x', color = 'b', ms = 12)
ax.plot(C[0], C[1], marker = 'x', color = 'r', ms = 12)
ax.plot(D[0], D[1], marker = 'x', color = 'b', ms = 12)
ax.plot([A[0], C[0]], [A[1], C[1]],  color = 'r',  )
ax.plot([B[0], D[0]], [B[1], D[1]],  color = 'b',  )

# LAMBDA, V = np.linalg.eig(Sigma)
## 椭圆的四个顶点
LAMBDA, V = np.linalg.eig(Sigma)
ax.plot(mu_X + V[0,0]*np.sqrt(LAMBDA[0]), mu_Y + V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = 'b', ms = 12)
ax.plot(mu_X - V[0,0]*np.sqrt(LAMBDA[0]), mu_Y - V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = 'b', ms = 12)
ax.plot(mu_X + V[0,1]*np.sqrt(LAMBDA[1]), mu_Y + V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = 'b', ms = 12)
ax.plot(mu_X - V[0,1]*np.sqrt(LAMBDA[1]), mu_Y - V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = 'b', ms = 12)

ax.set_xlim(-width , width )
ax.set_ylim(-width , width )
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.show()
plt.close()


#%% Bk5_Ch10_02
# 图 7. 二元高斯分布 PDF 和边缘 PDF，σX = 1, σY = 2, ρX,Y = 0.75
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from scipy.stats import multivariate_normal
from scipy.stats import norm

rho     = 0.75 # -0.75, 0, 0.75
sigma_X = 1 # 1, 2
sigma_Y = 1 # 1, 2

mu_X = 0
mu_Y = 0
mu   = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 6
X = np.arange(-width, width, 0.05)
Y = np.arange(-width, width, 0.05)
XX, YY = np.meshgrid(X, Y) # (160, 160)

XXYY = np.dstack((XX, YY))  # (160, 160, 2)
bi_norm = multivariate_normal(mu, Sigma)

# visualize PDF
f_X_Y_joint = bi_norm.pdf(XXYY) # (160, 160)
# f_X_Y_joint = bi_norm.cdf(XXYY) # (160, 160)
# Plot the conditional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

# # gs.update(wspace=0., hspace=0.)
# plt.suptitle('Marginal distributions', y=0.93)

# Plot surface on top left
ax1 = plt.subplot(gs[0])

# Plot bivariate normal
ax1.contourf(XX, YY, f_X_Y_joint, 15, cmap=cm.RdYlBu_r)
ax1.axvline(x = mu_X, color = 'k', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')

ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')
ax1.yaxis.set_label_position('right')
# ax1.set_xticks([])
# ax1.set_yticks([])

# Plot Y marginal
ax2 = plt.subplot(gs[1])
f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)
ax2.plot(f_Y, Y, 'b', label='$f_{Y}(y)$')
ax2.axhline(y = mu_Y, color = 'r', linestyle = '--')
ax2.fill_between(f_Y,Y, edgecolor = 'none', facecolor = '#DBEEF3')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(-width, width)
ax2.set_xlim(0, 0.5)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X marginal
ax3 = plt.subplot(gs[2])
f_X = norm.pdf(X, loc=mu_X, scale=sigma_X)
ax3.plot(X, f_X, 'b', label='$f_{X}(x)$')
ax3.axvline(x = mu_X, color = 'r', linestyle = '--')
ax3.fill_between(X,f_X, edgecolor = 'none', facecolor = '#DBEEF3')
ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(-width, width)
ax3.set_ylim(0, 0.5)
ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()
plt.close()

#%%
# 图 15. 椭圆和中心在 (µX, µY) 长 2σX、宽 2σY 的矩形相切
# 图 16. 三种标准差 σX、σY 大小不同的情况

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-4, 4, num = 201)
y = np.linspace(-4, 4, num = 201)
sigma_X = 1
sigma_Y = 2

xx, yy = np.meshgrid(x, y);
rho_array = np.linspace(-0.95, 0.95, num = 5)
fig, ax = plt.subplots(figsize=(8, 8))
# Create a Rectangle patch
rect = patches.Rectangle((-sigma_X, -sigma_Y), 2*sigma_X, 2*sigma_Y, linewidth = 0.25, edgecolor='k', linestyle = '--', facecolor = 'none')
# Add the patch to the Axes
ax.add_patch(rect)
colors = plt.cm.hsv(np.linspace(0,1,len(rho_array)))
for i in range(0,len(rho_array)):
    rho = rho_array[i]
    ellipse = ((xx/sigma_X)**2 - 2*rho*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - rho**2);
    color_code = colors[i,:].tolist()
    ax.contour(xx, yy, ellipse, levels = [1], colors = [color_code])
    ## 椭圆的四个切点
    ax.plot(sigma_X, rho * sigma_Y, marker = 'x', color = color_code, ms = 12)
    ax.plot(rho * sigma_X, sigma_Y, marker = 'x', color = color_code, ms = 12)
    ax.plot(-sigma_X, -rho * sigma_Y, marker = 'x', color = color_code, ms = 12)
    ax.plot(-rho * sigma_X, -sigma_Y, marker = 'x', color = color_code, ms = 12)

    ## 椭圆的四个顶点
    Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]
    LAMBDA, V = np.linalg.eig(Sigma)
    ax.plot(V[0,0]*np.sqrt(LAMBDA[0]), V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = color_code, ms = 12)
    ax.plot(-V[0,0]*np.sqrt(LAMBDA[0]), -V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = color_code, ms = 12)
    ax.plot(V[0,1]*np.sqrt(LAMBDA[1]), V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = color_code, ms = 12)
    ax.plot(-V[0,1]*np.sqrt(LAMBDA[1]), -V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = color_code, ms = 12)


plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()


#%% Bk5_Ch10_03, 图 18. 椭圆和矩形切点随 σX、σY、ρX,Y变化关系
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)
sigma_X = 1
sigma_Y = 2

xx, yy = np.meshgrid(x,y);
kk = np.linspace(-0.8,0.8,num = 9)

## 1
fig = plt.figure(figsize=(30,5))
for i in range(0,len(kk)):
    k = kk[i]
    ax = fig.add_subplot(1,len(kk),int(i+1))
    ellipse = ((xx/sigma_X)**2 - 2*k*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - k**2);

    plt.contour(xx, yy, ellipse, levels = [1], colors = '#0099FF')
    rect = Rectangle(xy = [- sigma_X, - sigma_Y] , width = 2*sigma_X, height = 2*sigma_Y, edgecolor = 'k',facecolor="none")
    ax.add_patch(rect)

    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_position('zero')
    ax.spines['bottom'].set_color('none')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('\u03C1 = %0.1f' %k)
plt.show()
plt.close()
## 2
colors = plt.cm.hsv(np.linspace(0, 1, len(kk)))
fig, ax = plt.subplots(figsize=(7, 7))
for i in range(0, len(kk)):
    k = kk[i]
    ellipse = ((xx/sigma_X)**2 - 2*k*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - k**2)
    color_code = colors[i,:].tolist()
    ax.contour(xx, yy, ellipse, levels = [1], colors = [color_code]) # '#0099FF')
rect = Rectangle(xy = [- sigma_X, - sigma_Y] , width = 2*sigma_X, height = 2*sigma_Y, edgecolor = 'k',facecolor="none")
ax.add_patch(rect)

ax.set_xlim([-2.5,2.5])
ax.set_ylim([-2.5,2.5])
ax.set_xticks([])
ax.set_yticks([])
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_position('zero')
ax.spines['bottom'].set_color('none')
plt.show()
plt.close()

#%% Bk5_Ch10_04
# 图 21. 相关性系数的几种可视化方案
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
import numpy as np
from matplotlib.patches import Rectangle

np.random.seed(2)

rho = -0.9
mu_X = 0
mu_Y = 0
MU = [mu_X, mu_Y]
sigma_X = 1
sigma_Y = 1

# covariance
SIGMA = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]
num = 500
## 生成数据点
X, Y = multi_norm(MU, SIGMA, num).T  # (500,)

center_X = np.mean(X)
center_Y = np.mean(Y)

### plot center of data
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(X, Y, '.', color = '#00448A', alpha = 0.25, markersize = 10)
ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0, color = 'k', linestyle = '--')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.show()
plt.close()

### 3D visualization
from scipy.stats import multivariate_normal
X_grid = np.linspace(-3,3,200)
Y_grid = np.linspace(-3,3,200)

XX, YY = np.meshgrid(X_grid, Y_grid)
XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(MU, SIGMA)
# visualize PDF
pdf_fine = bi_norm.pdf(XXYY)

fig, ax = plt.subplots(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.plot_wireframe(XX,YY, pdf_fine, cstride = 2, rstride = 2, color = [0.7,0.7,0.7], linewidth = 0.25)
ax.contour3D(XX, YY, pdf_fine, 15, cmap = 'RdYlBu_r')
ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
# ax.xaxis.set_ticks([])
# ax.yaxis.set_ticks([])
# ax.zaxis.set_ticks([])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PDF')
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.set_xlim3d([-3,3])
# ax.set_ylim3d([-3,3])
# ax.set_zlim3d([0,0.3])
plt.show()
plt.close()

### 2D visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.contour(XX,YY,pdf_fine,15, cmap = 'RdYlBu_r')
ax.axvline(x = 0, color = 'k', linestyle = '--')
ax.axhline(y = 0, color = 'k', linestyle = '--')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
plt.close()

###
def draw_vector(vector,RBG):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)
theta = np.arccos(rho)
fig, ax = plt.subplots()
draw_vector([1,0],np.array([0,112,192])/255)
draw_vector([np.cos(theta), np.sin(theta)],np.array([255,0,0])/255)

circle_theta = np.linspace(0, 2*np.pi, 100)
circle_X = np.cos(circle_theta)
circle_Y = np.sin(circle_theta)
ax.plot(circle_X, circle_Y, color = 'k', linestyle = '--')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('scaled')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
plt.close()












































































































































































































































































