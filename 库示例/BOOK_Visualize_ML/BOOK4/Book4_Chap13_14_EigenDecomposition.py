

#%% Bk_2_Ch17_03 可视化特征向量
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "SimSun"

### 产生数据
A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])

xx1, xx2 = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-8, 8, 9))
num_vecs = np.prod(xx1.shape);

thetas = np.linspace(0, 2*np.pi, num_vecs)

thetas = np.reshape(thetas, (-1, 9))
thetas = np.flipud(thetas);

uu = np.cos(thetas);
vv = np.sin(thetas);

# 矩阵乘法
V = np.array([uu.flatten(), vv.flatten()]).T;
W = V@A;
# 矩阵A线性映射
uu_new = np.reshape(W[:,0], (-1, 9));
vv_new = np.reshape(W[:,1], (-1, 9));

fig, ax = plt.subplots(figsize = (6,6))
# 绘制线性映射之前的向量
ax.quiver(xx1, xx2, # 向量始点位置坐标，网格化数据
          uu, vv,   # 两个方向的投影量
          angles = 'xy', scale_units='xy',
          scale = 0.8, # 稍微放大
          width = 0.0025, # 宽度，默认0.005
          edgecolor = 'none', facecolor = 'b')

# 绘制线性映射之后的向量
ax.quiver(xx1, xx2, uu_new, vv_new, angles='xy', scale_units='xy', scale=0.8, width = 0.0025, edgecolor='none', facecolor= 'r')

plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title("特征向量")
# fig.savefig('Figures/特征向量.svg', format='svg')
plt.show()
plt.close()


fig, ax = plt.subplots(figsize = (6,6))
import matplotlib
cm = matplotlib.cm.rainbow
norm = matplotlib.colors.Normalize()
ax.quiver(xx1*0, xx2*0, uu, vv, angles='xy', scale_units='xy',scale=1, width = 0.0025, edgecolor='none', facecolor=cm(norm(range(len(xx1.ravel())))))
ax.quiver(xx1*0, xx2*0, uu_new, vv_new, angles='xy', scale_units='xy',scale=1, width = 0.0025, edgecolor='none', facecolor=cm(norm(range(len(xx1.ravel())))))
plt.axis('scaled')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-2,2,5));
ax.set_yticks(np.linspace(-2,2,5));

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title("特征向量,单位圆")
# fig.savefig('Figures/特征向量，单位圆.svg', format='svg')
plt.show()
plt.close()

#%%  Bk_2_Ch17_04 # 可视化几何变换
import sympy
import numpy as np
import matplotlib.pyplot as plt
# 创建网格坐标数据
xx1_, xx2_ = np.meshgrid(np.linspace(-2,2,18), np.linspace(-2,2,18))
# 构造复数
zz_ = xx1_ + xx2_ * 1j
# 计算辐角
zz_angle_ = np.angle(zz_)
# 全零矩阵
zeros = np.zeros_like(xx1_)
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, # 向量场起点
            xx1_, xx2_,   # 横纵轴分量
            zz_angle_,    # 颜色映射依据
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')

ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
ax.set_xticks(np.arange(-2,3))
ax.set_yticks(np.arange(-2,3))
# plt.grid()
ax.axis('off')
# fig.savefig('Figures/用辐角大小给箭头着色，原图.svg', format='svg')
ax.set_title("用辐角大小给箭头着色，原图")
plt.show()
plt.close()

#>>>>>>>>>>>>>>>>>>
A = np.array([[2,  0],
              [0, 1]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A
uu_new = np.reshape(W[:,0],xx1_.shape)
vv_new = np.reshape(W[:,1],xx1_.shape)
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_, angles='xy', scale_units='xy', scale = 1, edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
ax.set_title("用辐角大小给箭头着色，缩放")
# fig.savefig('Figures/用辐角大小给箭头着色，缩放.svg', format='svg')
plt.show()
plt.close()

#>>>>>>>>>>>>>>>>>>
theta = np.pi/3
A = np.array([[np.cos(theta),  -np.sin(theta)],
              [np.sin(theta),   np.cos(theta)]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A.T;

uu_new = np.reshape(W[:,0],xx1_.shape);
vv_new = np.reshape(W[:,1],xx1_.shape);

fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_, angles='xy', scale_units='xy', scale = 1, edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
ax.set_title("用辐角大小给箭头着色，旋转")
# fig.savefig('Figures/用辐角大小给箭头着色，旋转.svg', format='svg')
plt.show()
plt.close()
#>>>>>>>>>>>>>>>>>>
A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A;
uu_new = np.reshape(W[:,0],xx1_.shape);
vv_new = np.reshape(W[:,1],xx1_.shape);
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_, angles='xy', scale_units='xy', scale = 1, edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
ax.set_title("用辐角大小给箭头着色，旋转+缩放")
# fig.savefig('Figures/用辐角大小给箭头着色，旋转 + 缩放.svg', format='svg')
plt.show()
plt.close()

#%% Bk4_Ch13_01.py

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])

xx1, xx2 = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-8, 8, 9))
num_vecs = np.prod(xx1.shape);

thetas = np.linspace(0, 2*np.pi, num_vecs)

thetas = np.reshape(thetas, (-1, 9))
thetas = np.flipud(thetas);

uu = np.cos(thetas);
vv = np.sin(thetas);

fig, ax = plt.subplots()
ax.quiver(xx1,xx2,uu,vv, angles='xy', scale_units='xy',scale=1, edgecolor='none', facecolor= 'b')
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));
plt.show()

# Matrix multiplication
V = np.array([uu.flatten(),vv.flatten()]).T;
W = V@A;

uu_new = np.reshape(W[:,0],(-1, 9));
vv_new = np.reshape(W[:,1],(-1, 9));

fig, ax = plt.subplots()
ax.quiver(xx1,xx2,uu,vv, angles='xy', scale_units='xy',scale=1, edgecolor='none', facecolor= 'b')
ax.quiver(xx1,xx2,uu_new,vv_new, angles='xy', scale_units='xy',scale=1, edgecolor='none', facecolor= 'r')
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));
plt.show()


fig, ax = plt.subplots()
ax.quiver(xx1*0, xx2*0, uu,vv, angles='xy', scale_units='xy', scale=1, edgecolor='none', facecolor= 'b')
ax.quiver(xx1*0, xx2*0, uu_new, vv_new, angles='xy', scale_units='xy', scale=1, edgecolor='none', facecolor= 'r')

plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.axis('scaled')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-2,2,5));
ax.set_yticks(np.linspace(-2,2,5));
plt.show()


#%% Bk4_Ch13_02.py
import numpy as np
import matplotlib.pyplot as plt

def visualize(X_circle,X_vec,title_txt):
    fig, ax = plt.subplots()
    plt.plot(X_circle[0,:], X_circle[1,:],'k', linestyle = '--', linewidth = 0.5)
    plt.quiver(0, 0, X_vec[0,0], X_vec[1,0], angles='xy', scale_units='xy',scale=1, color = [0, 0.4392, 0.7529])
    plt.quiver(0, 0, X_vec[0,1], X_vec[1,1], angles='xy', scale_units='xy',scale=1, color = [1,0,0])
    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)

    plt.ylabel('$x_2$')
    plt.xlabel('$x_1$')
    ax.set_aspect(1)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xticks(np.linspace(-2,2,5));
    ax.set_yticks(np.linspace(-2,2,5));
    plt.title(title_txt)
    plt.show()

theta = np.linspace(0, 2*np.pi, 100)
circle_x1 = np.cos(theta)
circle_x2 = np.sin(theta)

V_vec = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],
                  [np.sqrt(2)/2,  np.sqrt(2)/2]])

X_circle = np.array([circle_x1, circle_x2])
# plot original circle and two vectors
visualize(X_circle, V_vec, 'Original')
A = np.array([[1.25, -0.75],
              [-0.75, 1.25]])
# plot the transformation of A
visualize(A@X_circle, A@V_vec,'$A$')

#  Eigen deomposition
# A = V @ D @ V.T
lambdas, V = np.linalg.eig(A)
# D = np.diag(np.flip(lambdas))
# V = V.T # reverse the order

D = np.diag(lambdas)
# V = V.T # reverse the order

print('=== LAMBDA ===')
print(D)
print('=== V ===')
print(V)

# plot the transformation of V.T
visualize(V.T@X_circle, V.T@V_vec,r'$V^T$')

# plot the transformation of D @ V.T
visualize(D@V.T@X_circle, D@V.T@V_vec,'$\u039BV^T$')

# plot the transformation of V @ D @ V.T
visualize(V@D@V.T@X_circle, V@D@V.T@V_vec,'$V\u039BV^T$')

# plot the transformation of A
visualize(A@X_circle, A@V_vec,r'$A$')






#%% Bk4_Ch14_01.py  利用特征值分解完成方阵开方
# 能特征值分解的矩阵存在平方根矩阵
import numpy as np
import scipy

A = np.matrix([[1.25, -0.75],
               [-0.75, 1.25]])
LAMBDA, V = np.linalg.eig(A)
B = V@np.diag(np.sqrt(LAMBDA))@np.linalg.inv(V)
print(B)
A_reproduced = B@B
print(A_reproduced)

B_hat = scipy.linalg.sqrtm(A)  # == B
print(B_hat)

# 可以用 scipy.linalg.expm() 计算矩阵指数。
import scipy
expA = scipy.linalg.expm(A)
print(expA)


#%% Bk4_Ch14_02.py

import numpy as np
import matplotlib.pyplot as plt

# transition matrix
T = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])

# steady state
sstate = np.linalg.eig(T)[1][:,1]
sstate = sstate/sstate.sum()
print(sstate)

# initial states
initial_x_array = np.array([[1, 0, 0.5, 0.4],  # Chicken
                            [0, 1, 0.5, 0.6]]) # Rabbit

num_iterations = 10;
for i in np.arange(0,4):
    initial_x = initial_x_array[:,i][:, None]
    x_i = np.zeros_like(initial_x)
    x_i = initial_x
    X =   initial_x.T;
    # matrix power through iterations
    for x in np.arange(0,num_iterations):
        x_i = T@x_i;
        X = np.concatenate([X, x_i.T],axis = 0)
    fig, ax = plt.subplots()
    itr = np.arange(0,num_iterations+1);
    plt.plot(itr,X[:,0],marker = 'x',color = (1,0,0))
    plt.plot(itr,X[:,1],marker = 'x',color = (0,0.6,1))

    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xlim(0, num_iterations)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iteration, k')
    ax.set_ylabel('State')


#%% Bk4_Ch14_03.py
import sympy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as L

def mesh_circ(c1, c2, r, num):
    theta = np.linspace(0, 2*np.pi, num)
    r     = np.linspace(0,r, num)
    theta,r = np.meshgrid(theta,r)
    xx1 = np.cos(theta)*r + c1
    xx2 = np.sin(theta)*r + c2
    return xx1, xx2

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')
A = np.array([[0.5, -0.5],
              [-0.5, 0.5]])
Lambda, V = L.eig(A)
x = np.array([[x1,x2]]).T
f_x = x.T@A@x
f_x = f_x[0][0]
f_x_fcn = sympy.lambdify([x1,x2],f_x)
xx1, xx2 = mesh_circ(0, 0, 1, 50)
ff_x = f_x_fcn(xx1,xx2)
if Lambda[1] > 0:
    levels = np.linspace(0,Lambda[0],21)
else:
    levels = np.linspace(Lambda[1],Lambda[0],21)

t = np.linspace(0,np.pi*2,100)

# 2D visualization
fig, ax = plt.subplots()
ax.plot(np.cos(t), np.sin(t), color = 'k')
cs = plt.contourf(xx1, xx2, ff_x, levels=levels, cmap = 'RdYlBu_r')
plt.show()
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
clb = fig.colorbar(cs, ax=ax)
clb.set_ticks(levels)
plt.show()
plt.close()


#  3D surface of f(x1,x2)
x1_ = np.linspace(-1.2,1.2,31)
x2_ = np.linspace(-1.2,1.2,31)

xx1_fine, xx2_fine = np.meshgrid(x1_,x2_)
ff_x_fine = f_x_fcn(xx1_fine,xx2_fine)
f_circle = f_x_fcn(np.cos(t), np.sin(t))
# 3D visualization
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
ax.plot(np.cos(t), np.sin(t), f_circle, color = 'k')
# circle projected to f(x1,x2)

ax.plot_wireframe(xx1_fine, xx2_fine, ff_x_fine, color = [0.8,0.8,0.8], linewidth = 0.25)
ax.contour3D(xx1_fine, xx2_fine, ff_x_fine, 15, cmap = 'RdYlBu_r')

ax.view_init(elev=30, azim=60)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.set_xlim(xx1_fine.min(),xx1_fine.max())
ax.set_ylim(xx2_fine.min(),xx2_fine.max())
plt.tight_layout()
ax.set_proj_type('ortho')
plt.show()

# %% Bk4_Ch13_03.py
import numpy as np
import matplotlib.pyplot as plt
theta = np.deg2rad(30)
r = 0.8 # 1.2, scaling factor
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[r, 0],
              [0, r]])
A = R@S
# A = np.array([[1, -1],
#               [1, 1]])
Lamb, V = np.linalg.eig(A)
theta_array = np.arange(0,np.pi*2,np.pi*2/18)
colors = plt.cm.rainbow(np.linspace(0,1,len(theta_array)))

fig, ax = plt.subplots()
for j, theat_i in enumerate(theta_array):
    # initial point
    x = np.array([[5*np.cos(theat_i)],
                  [5*np.sin(theat_i)]])
    plt.plot(x[0],x[1], marker = 'x',color = colors[j], markersize = 15)
    # plot the initial point
    x_array = x
    for i in np.arange(20):
        x = A@x
        x_array = np.column_stack((x_array,x))
    # colors_j = colors[j,:]
    plt.plot(x_array[0,:],x_array[1,:], lw = 1, marker = '.',color = colors[j,:])
plt.axis('scaled')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')


# #%% Bk4_Ch14_04.py
# import numpy as np
# import matplotlib.pyplot as plt
# theta = np.deg2rad(30)
# r = 0.8 # 1.2, scaling factor
# R = np.array([[np.cos(theta), -np.sin(theta)],
#               [np.sin(theta), np.cos(theta)]])
# S = np.array([[r, 0],
#               [0, r]])
# A = R@S
# # A = np.array([[1, -1],
# #              [1, 1]])
# Lamb, V = np.linalg.eig(A)
# theta_array = np.arange(0,np.pi*2,np.pi*2/18)
# colors = plt.cm.rainbow(np.linspace(0,1,len(theta_array)))
# fig, ax = plt.subplots(figsize = (10, 10))
# for j, theat_i in enumerate(theta_array):
#     # initial point
#     x = np.array([[5*np.cos(theat_i)],
#                   [5*np.sin(theat_i)]])
#     plt.plot(x[0], x[1], marker = 'x',color = colors[j], markersize = 15)
#     # plot the initial point
#     x_array = x
#     for i in np.arange(20):
#         x = A@x
#         x_array = np.column_stack((x_array, x))
#     # colors_j = colors[j,:]
#     plt.plot(x_array[0,:], x_array[1,:], marker = '.', color = colors[j,:])
# plt.axis('scaled')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axvline(x=0,color = 'k')
# ax.axhline(y=0,color = 'k')
# plt.show()



































































































































































































































































