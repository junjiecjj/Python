


#%% Bk5_Ch12_01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from scipy.stats import norm

rho     = 0.5
sigma_X = 1.5
sigma_Y = 1
mu_X = 0
mu_Y = 0
mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]

width = 4
X = np.linspace(-width,width,81)
Y = np.linspace(-width,width,81)
XX, YY = np.meshgrid(X, Y) # (81, 81)
XXYY = np.dstack((XX, YY)) # (81, 81, 2)
bi_norm = multivariate_normal(mu, Sigma)

#%% visualize PDF,
# 图 2. y = −2 时，联合 PDF、边缘 PDF、条件 PDF 的关系
y_cond_i = 60 # 20, 30, 40, 50, 60, index
f_X_Y_joint = bi_norm.pdf(XXYY) # (81, 81)


## 3D
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(XX, YY, f_X_Y_joint, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
# ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()


### Plot the tional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])

# Plot surface on top left
ax1 = plt.subplot(gs[0])

# Plot bivariate normal
ax1.contour(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
ax1.axvline(x = mu_X, color = 'k', linestyle = '--')
ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax1.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')

x_sym_axis = mu_X + rho*sigma_X/sigma_Y*(Y[y_cond_i] - mu_Y)
ax1.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.yaxis.set_label_position('right')
# ax1.set_xticks([])
# ax1.set_yticks([])

# Plot Y marginal
ax2 = plt.subplot(gs[1])
f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

ax2.plot(f_Y, Y, 'k', label='$f_{Y}(y)$')
ax2.axhline(y = mu_Y, color = 'k', linestyle = '--')
ax2.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')
ax2.plot(f_Y[y_cond_i], Y[y_cond_i], marker = 'x', markersize = 15)
plt.title('$f_{Y}(y_{} = %.2f) = %.2f$' %(Y[y_cond_i],f_Y[y_cond_i]))

ax2.fill_between(f_Y,Y, edgecolor = 'none', facecolor = '#D9D9D9')
ax2.legend(loc=0)
ax2.set_xlabel('PDF')
ax2.set_ylim(-width, width)
ax2.set_xlim(0, 0.5)
ax2.invert_xaxis()
ax2.yaxis.tick_right()

# Plot X and Y joint
ax3 = plt.subplot(gs[2])
f_X_Y_cond_i = f_X_Y_joint[y_cond_i,:]
ax3.plot(X, f_X_Y_cond_i, 'r', label='$f_{X,Y}(x,y_{} = %.2f)$' %(Y[y_cond_i]))

ax3.axvline(x = mu_X, color = 'k', linestyle = '--')
ax3.axvline(x = x_sym_axis, color = 'r', linestyle = '--')
ax3.legend(loc=0)
ax3.set_ylabel('PDF')
ax3.yaxis.set_label_position('left')
ax3.set_xlim(-width, width)
ax3.set_ylim(0, 0.5)
ax3.set_yticks([0, 0.25, 0.5])

ax4 = plt.subplot(gs[3])
ax4.set_visible(False)

plt.show()

#%% compare joint, marginal and tional
# 图 2. y = −2 时，联合 PDF、边缘 PDF、条件 PDF 的关系
f_X = norm.pdf(X, loc=mu_X, scale=sigma_X)
fig, ax = plt.subplots()
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(Y)))
f_X_given_Y_cond_i = f_X_Y_cond_i/f_Y[y_cond_i]

plt.plot(X,f_X, color = 'k', label='$f_{X}(x)$') # marginal
ax.axvline(x = mu_X, color = 'k', linestyle = '--')


plt.plot(X, f_X_Y_cond_i, color = 'r', label='$f_{X,Y}(x,y_{} = %.2f$)' %(Y[y_cond_i])) # joint
ax.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

plt.plot(X, f_X_given_Y_cond_i, color = 'b', label='$f_{X|Y}(x|y_{} = %.2f$)' %(Y[y_cond_i])) # tional
ax.fill_between(X, f_X_given_Y_cond_i, edgecolor = 'none', facecolor = '#DBEEF3')
ax.fill_between(X, f_X_Y_cond_i, edgecolor = 'none', hatch='/')


plt.xlabel('X')
plt.ylabel('PDF')
ax.set_xlim(-width, width)
ax.set_ylim(0, 0.35)
plt.title('$f_{Y}(y_{} = %.2f) = %.2f$' %(Y[y_cond_i],f_Y[y_cond_i]))
ax.legend()


#%% Bk5_Ch12_02
# 图 7. fY|X(y|x) 曲面网格线
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d

def fcn_Y_given_X(mu_X, mu_Y, sigma_X, sigma_Y, rho, X, Y):
    coeff = 1/sigma_Y/np.sqrt(1 - rho**2)/np.sqrt(2*np.pi)
    sym_axis = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)
    quad  = -1/2*((Y - sym_axis)/sigma_Y/np.sqrt(1 - rho**2))**2
    f_Y_given_X  = coeff*np.exp(quad)
    return f_Y_given_X

# parameters
rho     = 0.5
sigma_X = 1
sigma_Y = 1

mu_X = 0
mu_Y = 0

width = 3
X = np.linspace(-width,width,31)
Y = np.linspace(-width,width,31)

XX, YY = np.meshgrid(X, Y)
f_Y_given_X = fcn_Y_given_X(mu_X, mu_Y, sigma_X, sigma_Y, rho, XX, YY)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_wireframe(XX, YY, f_Y_given_X, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour(XX, YY, f_Y_given_X, 20, cmap=cm.RdYlBu_r)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(), f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()

## surface projected along X to Y-Z plane
# 图 8. fY|X(y|x) 曲面在 yz 平面上投影
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(XX, YY, f_Y_given_X, rstride=0, cstride=1, color = [0.3,0.3,0.3], linewidth = 0.25)
ax.contour(XX, YY, f_Y_given_X, levels = 20, zdir='x', offset=YY.max(), cmap=cm.RdYlBu_r)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()


## add X marginal
f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)
fig, ax = plt.subplots()
colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(X)))
for i in np.linspace(1,len(X),len(X)):
    plt.plot(Y,f_Y_given_X[:,int(i)-1], color = colors[int(i)-1])

plt.plot(Y,f_Y, color = 'k')
plt.xlabel('y')
plt.ylabel('$f_{Y|X}(y|x)$')
ax.set_xlim(-width, width)
ax.set_ylim(0, f_Y_given_X.max())

#%% surface projected along Z to X-Y plane
# 图 9. fY|X(y|x) 曲面等高线
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(XX, YY, f_Y_given_X, color = [0.3,0.3,0.3], linewidth = 0.25)

ax.contour3D(XX,YY,f_Y_given_X,12, cmap = 'RdYlBu_r')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f_{Y|X}(y|x)$')
ax.set_proj_type('ortho')
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

ax.set_xlim(-width, width)
ax.set_ylim(-width, width)
ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
plt.tight_layout()
ax.view_init(azim=-120, elev=30)
plt.show()

# Plot filled contours
# 图 10. fY|X(y|x) 平面等高线
E_Y_given_X = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)

from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7, 7))

# Plot bivariate normal
plt.contourf(XX, YY, f_Y_given_X, 12, cmap=cm.RdYlBu_r)
plt.plot(X,E_Y_given_X, color = 'k', linewidth = 1.25)
plt.axvline(x = mu_X, color = 'k', linestyle = '--')
plt.axhline(y = mu_Y, color = 'k', linestyle = '--')

x = np.linspace(-width,width,num = 201)
y = np.linspace(-width,width,num = 201)

xx,yy = np.meshgrid(x,y);

ellipse = ((xx/sigma_X)**2 - 2*rho*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - rho**2);
plt.contour(xx,yy,ellipse,levels = [1], colors = 'k')
rect = Rectangle(xy = [- sigma_X, - sigma_Y] , width = 2*sigma_X, height = 2*sigma_Y, edgecolor = 'k',facecolor="none")
ax.add_patch(rect)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')




#%% Bk5_Ch12_03

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import norm
import scipy
import seaborn as sns
from numpy.linalg import inv

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

x_array = np.linspace(0, 8, 100)
# Convert X array to dataframe
X_Y_df = pd.DataFrame(X, columns=feature_names)

#%% Heatmap of centroid vector, MU
# 图 26. 质心向量、协方差矩阵热图
MU = X_Y_df.mean()
MU = np.array([MU]).T

fig, axs = plt.subplots()
h = sns.heatmap(MU,cmap='RdYlBu_r', linewidths=.05,annot=True,fmt = '.2f')
h.set_aspect("equal")
h.set_title('Vector of $\mu$')

#%% Heatmap of covariance matrix
# 图 26. 质心向量、协方差矩阵热图
SIGMA = X_Y_df.cov()

fig, axs = plt.subplots()
h = sns.heatmap(SIGMA,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Covariance matrix')

#%%
SIGMA = np.array(SIGMA)
from sympy import symbols
x1, x2, x3 = symbols('x1 x2 x3')

SIGMA_XX = SIGMA[0:3, 0:3]

SIGMA_YX = SIGMA[3, 0:3]
SIGMA_YX = np.matrix(SIGMA_YX)

MU_Y = MU[3]
MU_Y = np.matrix(MU_Y)

MU_X = MU[0:3]

x_vec = np.array([[x1,x2,x3]]).T
y = SIGMA_YX@inv(SIGMA_XX)@(x_vec - MU_X) + MU_Y
print(y)

#%% matrix computation
b  = SIGMA_YX@inv(SIGMA_XX) # coefficients
b0 = MU_Y - b@MU_X          # constant

#%% computate coefficient vector
# 图 27. 矩阵计算系数向量 b
fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b, cmap ='RdYlBu_r',  linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 10, vmin = -5)

ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(SIGMA_YX, cmap='RdYlBu_r', linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 10, vmin = -5)
ax.set_aspect("equal")
plt.title('$\Sigma_{YX}$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(inv(SIGMA_XX), cmap='RdYlBu_r', linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 10, vmin = -5)
ax.set_aspect("equal")
plt.title('$\Sigma^{-1}_{XX}$')

#%% compute constant b0
# 图 28. 矩阵计算常数 b0
fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(b0, cmap ='RdYlBu_r',  linewidths=.05,annot=True,  cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 5, vmin = -0.5)

ax.set_aspect("equal")
plt.title('$b_0$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(MU_Y,cmap='RdYlBu_r', linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$\mu_Y$')

plt.sca(axs[3])
plt.title('-')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(b,cmap='RdYlBu_r', linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f', vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$b$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(MU_X,cmap='RdYlBu_r', linewidths=.05,annot=True, cbar_kws={"orientation": "horizontal"},fmt = '.3f',  vmax = 5, vmin = -0.5)
ax.set_aspect("equal")
plt.title('$\mu_X$')

#%% use statsmodels

import statsmodels.api as sm

X_df = X_Y_df[feature_names[0:3]]
y_df = X_Y_df[feature_names[3]]

# add a column of ones
X_df = sm.add_constant(X_df)

model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())

p = model.fit().params
print(p)

































































































































































































































































































