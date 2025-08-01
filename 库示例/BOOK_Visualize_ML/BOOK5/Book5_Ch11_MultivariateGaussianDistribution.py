





#%% Bk5_Ch11_01.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm # Colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from scipy.stats import norm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Rectangle

# parameters
rho     = 0.5
sigma_X = 1
sigma_Y = 2
mu_X = 0
mu_Y = 0
width = 5
mu    = [mu_X, mu_Y]
Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], [sigma_X*sigma_Y*rho, sigma_Y**2]]

X = np.linspace(-width, width, 101)
Y = np.linspace(-width, width, 101)
XX, YY = np.meshgrid(X, Y)

XXYY = np.dstack((XX, YY))
bi_norm = multivariate_normal(mu, Sigma)

f_X_Y_joint = bi_norm.pdf(XXYY)

# expectation of Y given X
E_Y_given_X = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)
# expectation of X given Y
E_X_given_Y = mu_X + rho*sigma_X/sigma_Y*(Y - mu_Y)

## 椭圆旋转角度 θ: Book05_Chap11_page22, Eq.(41)
theta = 1/2*np.arctan(2*rho*sigma_X*sigma_Y/(sigma_X**2 - sigma_Y**2))
k = np.tan(theta)

axis_minor = mu_Y + k*(X - mu_X)
axis_major = mu_Y - 1/k*(X - mu_X)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
# Plot bivariate normal
# ax.contour(XX, YY, f_X_Y_joint, 25, cmap=cm.RdYlBu_r)

x1 = mu_X + sigma_X
x2 = mu_Y + rho * sigma_Y
tmp = bi_norm.pdf([x1, x2])
ax.contour(XX, YY, f_X_Y_joint, levels = [tmp], cmap=cm.RdYlBu_r)

# tmp = bi_norm.pdf([mu_X + 2 * sigma_X, mu_Y + 2 * sigma_Y])
# ax.contour(XX, YY, f_X_Y_joint, levels = [tmp], cmap=cm.RdYlBu_r)
ax.axvline(x = mu_X, color = 'k', linestyle = '--')
ax.axhline(y = mu_Y, color = 'k', linestyle = '--')

# 椭圆的四个切点的直线
ax.plot(E_X_given_Y, Y, color = 'k', linewidth = 1.25)
ax.plot(X, E_Y_given_X, color = 'k', linewidth = 1.25)

# 椭圆长半轴和短半轴
ax.plot(X, axis_minor, color = 'r', linewidth = 1.25)
ax.plot(X, axis_major, color = 'r', linewidth = 1.25)

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
ax.plot(mu_X + V[0,0]*np.sqrt(LAMBDA[0]), mu_Y + V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = 'b', ms = 8)
ax.plot(mu_X - V[0,0]*np.sqrt(LAMBDA[0]), mu_Y - V[1,0]*np.sqrt(LAMBDA[0]), marker = 'd', color = 'b', ms = 8)
ax.plot(mu_X + V[0,1]*np.sqrt(LAMBDA[1]), mu_Y + V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = 'b', ms = 8)
ax.plot(mu_X - V[0,1]*np.sqrt(LAMBDA[1]), mu_Y - V[1,1]*np.sqrt(LAMBDA[1]), marker = 'd', color = 'b', ms = 8)

rect = Rectangle(xy = [mu_X -  sigma_X, mu_Y - sigma_Y] , width = 2*sigma_X, height = 2*sigma_Y, edgecolor = 'k',facecolor="none")
ax.add_patch(rect)

ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_xlim(-width , width )
ax.set_ylim(-width , width )
plt.show()
plt.close()



#%% Bk5_Ch11_02.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

x = np.linspace(-4, 4, num = 201)
y = np.linspace(-4, 4, num = 201)
xx, yy = np.meshgrid(x, y)
sigma_X = 1
sigma_Y = 2

RHOs = np.linspace(-0.8, 0.8, num = 9)
fig = plt.figure(figsize=(30, 5), constrained_layout = True)
for i in range(0,len(RHOs)):
    rho = RHOs[i]
    ax = fig.add_subplot(1, len(RHOs), int(i+1))
    ellipse = ((xx/sigma_X)**2 - 2*rho*(xx/sigma_X)*(yy/sigma_Y) + (yy/sigma_Y)**2)/(1 - rho**2);

    plt.contour(xx, yy, ellipse, levels = [1], colors = '#0099FF')

    A = (sigma_X**2 + sigma_Y**2)/2
    B = np.sqrt((rho*sigma_X*sigma_Y)**2 + ((sigma_X**2 - sigma_Y**2)/2)**2)
    length_major = np.sqrt(A + B)
    length_minor = np.sqrt(A - B)

    if sigma_X == sigma_Y and rho >= 0:
        theta = 45

    elif sigma_X == sigma_Y and rho < 0:
        theta = -45
    else:
        theta = 1/2*np.arctan(2*rho*sigma_X*sigma_Y/(sigma_X**2 - sigma_Y**2))
        theta = theta*180/np.pi

    if sigma_X >= sigma_Y:
        rect = Rectangle([-length_major, -length_minor] ,
                         width = 2*length_major,
                         height = 2*length_minor,
                         edgecolor = 'k',facecolor="none",
                         transform=Affine2D().rotate_deg_around(*(0,0), theta)+ax.transData)
    else:
        rect = Rectangle([-length_minor, -length_major] ,
                         width = 2*length_minor,
                         height = 2*length_major,
                         edgecolor = 'k',facecolor="none",
                         transform=Affine2D().rotate_deg_around(*(0,0), theta)+ax.transData)

    ax.add_patch(rect)

    X = np.linspace(-2.5, 2.5, 101)
    k = np.tan(theta/180*np.pi)
    axis_minor = k*X
    axis_major = - 1/k*X
    plt.plot(X, axis_minor, color = 'r', linewidth = 1.25)
    plt.plot(X, axis_major, color = 'r', linewidth = 1.25)


    ax.set_xlim([-2.5,2.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_position('zero')
    ax.spines['bottom'].set_color('none')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_title('\u03C1 = %0.1f' %rho)
plt.show()
plt.close()


































































































































































































































































