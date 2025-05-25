


#%% 可视化 2 * 2 方阵奇异值分解
# 导入包
import numpy as np
import matplotlib.pyplot as plt

## 创建数据
x1 = np.arange(-5, 5 + 1, step=1)
x2 = np.arange(-5, 5 + 1, step=1)

XX1,XX2 = np.meshgrid(x1,x2)
X = np.column_stack((XX1.ravel(),XX2.ravel()))

colors = np.arange(len(x1) * len(x2))

## 二维散点
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)
plt.axis('scaled')
ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


## 定义矩阵A
A = np.array([[1.25, -0.75],
              [-0.75, 1.25]])

U, s, VH = np.linalg.svd(A)
S = np.zeros(A.shape)
np.fill_diagonal(S, s)
# S = np.diag(s)

print(f"U@S@VH = \n{U@S@VH}")
print(f"A = \n{A}")


## 用矩阵A完成映射
Z = X@A.T
fig, ax = plt.subplots(figsize = (5,5))
plt.scatter(Z[:,0].ravel(), Z[:,1].ravel(), c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)
plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.savefig('Figures/A转换.svg', format='svg')

## 奇异值分解
U,S,VT = np.linalg.svd(A, full_matrices=True)
V = VT.T
S = np.diag(S)

A1 = U @ S @ VT
A1T = V @ S.T @ U.T

## 平面旋转
Z = X@V
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0].ravel(),
            Z[:,1].ravel(),
            c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/V旋转.svg', format='svg')
## 缩放
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0].ravel(),
            Z[:,1].ravel(),
            c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/V旋转，S缩放.svg', format='svg')


## 再次旋转
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0].ravel(),
            Z[:,1].ravel(),
            c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/V旋转，S缩放，U旋转.svg', format='svg')




#%%# 可视化3 * 3方阵奇异值分解

# 导入包
import numpy as np
import matplotlib.pyplot as plt


###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis

def _get_coord_info_new(self, renderer):
    mins, maxs, cs, deltas, tc, highs = self._get_coord_info_old(renderer)
    correction = deltas * [0.25,
                           0.25,
                           0.25]
    mins += correction
    maxs -= correction
    return mins, maxs, cs, deltas, tc, highs
if not hasattr(Axis, "_get_coord_info_old"):
    Axis._get_coord_info_old = Axis._get_coord_info
Axis._get_coord_info = _get_coord_info_new
###patch end###
## 创建数据
num = 21

array_1_0 = np.linspace(0,1,num)
array_0_0 = np.ones_like(array_1_0)
array_1_1 = np.zeros_like(array_1_0)

A1 = np.column_stack([array_1_0,array_0_0,array_0_0])
A2 = np.column_stack([array_1_0,array_1_1,array_0_0])
A3 = np.column_stack([array_1_0,array_0_0,array_1_1])
A4 = np.column_stack([array_1_0,array_1_1,array_1_1])

A = np.vstack((A1,A2,A3,A4))
B = np.roll(A, 1)
C = np.roll(A, 2)

X   = np.vstack((A,B,C))
colors = np.vstack((A,B,C))


## 三维散点
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/矮胖矩阵SVD，变换前.svg', format='svg')


## 定义矩阵A
A = np.array([[0,1,1],
              [1,0,1],
              [-1,-1,0]])

np.linalg.det(A)
# 三维体积放大倍数
# 负数表示镜像

from sympy import Matrix
A_ = Matrix(A)
U__, S__, V__ = A_.singular_value_decomposition()

print(A)
print(np.array(U__@S__@V__.T))

## 用矩阵A完成映射
Z = X@A.T
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)

ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/A转换.svg', format='svg')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)

ax.set_proj_type('ortho')
ax.view_init(elev=90, azim=-90) # xy
# ax.view_init(elev=0, azim=-90)  # xz
# ax.view_init(elev=0, azim=0)    # yz
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/A转换_投影.svg', format='svg')


## 奇异值分解

U,S,VT = np.linalg.svd(A, full_matrices=True)
V = VT.T
S = np.diag(S)
# A = U @ S @ VT
# AT = V @ ST @ UT

## 三维旋转
Z = X@V

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/V旋转.svg', format='svg')

## 缩放
Z = X@V@S.T
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/V旋转，S缩放.svg', format='svg')



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)

ax.set_proj_type('ortho')
ax.view_init(elev=90, azim=-90) # xy
# ax.view_init(elev=0, azim=-90)  # xz
# ax.view_init(elev=0, azim=0)    # yz
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/V旋转，S缩放_投影视角.svg', format='svg')



## 再次三维旋转
Z = X@V@S.T@U.T
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/V旋转，S缩放，U旋转.svg', format='svg')




#%%# 可视化细高矩阵奇异值分解
## 创建数据
x1 = np.arange(-5, 5 + 1, step=1)
x2 = np.arange(-5, 5 + 1, step=1)

XX1,XX2 = np.meshgrid(x1,x2)
X = np.column_stack((XX1.ravel(),XX2.ravel()))

colors = np.arange(len(x1) * len(x2))

## 二维散点
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(XX1.ravel(), XX2.ravel(), c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

## 定义矩阵A
A = np.array([[0, 1],
              [1, 1],
              [1, 0]])

from sympy import Matrix
A_ = Matrix(A)
U__, S__, V__ = A_.singular_value_decomposition()
# 注意，并非完全SVD分解，不同于后文的SVD分解结果

print(A)
print(np.array(U__@S__@V__.T))
## 用矩阵A完成映射

Z = X@A.T;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
ZZ3 = Z[:,2].reshape((len(x1), len(x2)))

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
ax.scatter(ZZ1, ZZ2, ZZ3, c = colors, s = 15, cmap = 'RdYlBu', alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-10,0,0],[0,-10,0],[0,0,-10]])
u, v, w = np.array([[20,0,0],[0,20,0],[0,0,20]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/SVD，变换后.svg', format='svg')



## 奇异值分解
U,S,VT = np.linalg.svd(A, full_matrices=True)
V = VT.T
S = np.diag(S)
S = np.vstack((S,np.array([[0,0]])))

# A = U @ S @ VT
# AT = V @ ST @ UT
## 平面旋转
Z = X@V
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(ZZ1.ravel(), ZZ2.ravel(), c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/SVD, V旋转.svg', format='svg')



## 缩放 + 升维
Z = X@V@S.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
ZZ3 = Z[:,2].reshape((len(x1), len(x2)))

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
ax.scatter(ZZ1, ZZ2, ZZ3, c = colors, s = 15, cmap = 'RdYlBu', alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-10,0,0],[0,-10,0],[0,0,-10]])
u, v, w = np.array([[20,0,0],[0,20,0],[0,0,20]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")

# fig.savefig('Figures/SVD，V旋转-S缩放.svg', format='svg')

fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0], Z[:,1], c = colors, s = 15, cmap = 'RdYlBu', zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/SVD，V旋转-S缩放，平面上看.svg', format='svg')



## 再次旋转，三维空间
Z = X@V@S.T@U.T
ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))
ZZ3 = Z[:,2].reshape((len(x1), len(x2)))

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='3d')
ax.scatter(ZZ1, ZZ2, ZZ3, c = colors, s = 15, cmap = 'RdYlBu', alpha = 1)

ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-10,0,0],[0,-10,0],[0,0,-10]])
u, v, w = np.array([[20,0,0],[0,20,0],[0,0,20]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# ax.set_axis_off()

# fig.savefig('Figures/SVD，V旋转-S缩放-UT旋转.svg', format='svg')



#%% # 可视化矮胖矩阵奇异值分解
## 创建数据
num = 21

array_1_0 = np.linspace(0,1,num)
array_0_0 = np.ones_like(array_1_0)
array_1_1 = np.zeros_like(array_1_0)

A1 = np.column_stack([array_1_0,array_0_0,array_0_0])
A2 = np.column_stack([array_1_0,array_1_1,array_0_0])
A3 = np.column_stack([array_1_0,array_0_0,array_1_1])
A4 = np.column_stack([array_1_0,array_1_1,array_1_1])

A = np.vstack((A1,A2,A3,A4))
B = np.roll(A, 1)
C = np.roll(A, 2)

X   = np.vstack((A,B,C))
colors = np.vstack((A,B,C))


## 三维散点
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/矮胖矩阵SVD，变换前.svg', format='svg')

#
### 定义矩阵A
A = np.array([[0, 1],
              [1, 1],
              [1, 0]])
# A = np.array([[0, 0],
#               [1, -0.5],
#               [1, -0.5]])

A = A.T

from sympy import Matrix
A_ = Matrix(A)
U__, S__, V__ = A_.singular_value_decomposition()
# 注意，并非完全SVD分解，不同于后文的SVD分解结果

# 验证书中结果
U_test = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],
                   [np.sqrt(2)/2, np.sqrt(2)/2]])

S_test = np.array([[np.sqrt(3), 0, 0],
                   [0, 1, 0]])

V_T_test = np.array([[1/np.sqrt(6),2/np.sqrt(6),1/np.sqrt(6)],
                     [np.sqrt(2)/2, 0, -np.sqrt(2)/2],
                     [np.sqrt(3)/3,-np.sqrt(3)/3,np.sqrt(3)/3]])


U_test @ S_test @ V_T_test

V_T_test.T @ S_test.T @ U_test.T


U_test.T

## 用矩阵A完成映射
Z = X@A.T

fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0], Z[:,1], c = colors, s = 15, zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# fig.savefig('Figures/矮胖矩阵SVD，变换后.svg', format='svg')


## 奇异值分解
U,S,VT = np.linalg.svd(A, full_matrices=True)
V = VT.T
S = np.diag(S)
S = np.column_stack((S,np.array([[0],[0]])))
S
# A = U @ S @ VT
# AT = V @ ST @ UT

## 三维旋转
Z = X@V

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/矮胖矩阵SVD，V旋转.svg', format='svg')

## 缩放 + 降维
Z = X@V@S.T
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0], Z[:,1], c = colors, s = 15, zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# fig.savefig('Figures/矮胖矩阵SVD，V旋转-S缩放，2D.svg', format='svg')



# 在三维空间呈现
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,1] * 0, c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/矮胖矩阵SVD，V旋转-S缩放，3D.svg', format='svg')

## 平面旋转

Z = X@V@S.T@U.T
fig, ax = plt.subplots(figsize = (5,5))

plt.scatter(Z[:,0], Z[:,1], c = colors, s = 15, zorder=1e3)

plt.axis('scaled')

ax.axhline(y = 0, color = 'k')
ax.axvline(x = 0, color = 'k')
plt.xticks([])
plt.yticks([])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/矮胖矩阵SVD，V旋转-S缩放-UT旋转，2D.svg', format='svg')


# 在三维空间呈现
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,1] * 0, c = colors, s = 15, alpha = 1)


ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
x, y, z = np.array([[-3,0,0],[0,-3,0],[0,0,-3]])
u, v, w = np.array([[6,0,0],[0,6,0],[0,0,6]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.05, color="black")
# fig.savefig('Figures/矮胖矩阵SVD，V旋转-S缩放-UT旋转，3D.svg', format='svg')




















































































































































































































































































































































































































































































































































































































