


#%% Bk4_Ch15_01.py
import numpy as np
import matplotlib.pyplot as plt

def visualize(X_circle, X_vec, title_txt):
    fig, ax = plt.subplots()
    plt.plot(X_circle[:,0], X_circle[:,1], 'k', linestyle = '--', linewidth = 0.5)
    plt.quiver(0, 0, X_vec[0,0], X_vec[0,1], angles='xy', scale_units='xy',scale=1, color = [0, 0.4392, 0.7529])
    plt.quiver(0, 0, X_vec[1,0], X_vec[1,1], angles='xy', scale_units='xy',scale=1, color = [1,0,0])
    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)

    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')

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

X_vec = np.array([[1,0], [0,1]])

X_circle = np.array([circle_x1, circle_x2]).T

# plot original circle and two vectors
visualize(X_circle, X_vec,'Original')

A = np.array([[1.6250, 0.6495],
              [0.6495, 0.8750]])

# plot the transformation of A
visualize(X_circle@A.T, X_vec@A.T, '$A$')

################# SVD ##################
# A = U @ S @ V.T
U, S, V = np.linalg.svd(A)
S = np.diag(S)
# V[:,0] = -V[:,0] # reverse sign of first vector of V
# U[:,0] = -U[:,0] # reverse sign of first vector of U

print('=== U ===')
print(U)
print('=== S ===')
print(S)
print('=== V ===')
print(V)

# plot the transformation of V
visualize(X_circle@V, X_vec@V,'$V^T$')

# plot the transformation of V @ S
visualize(X_circle@V@S, X_vec@V@S,'$SV^T$')

# plot the transformation of V @ S @ U.T
visualize(X_circle@V@S@U.T, X_vec@V@S@U.T,'$USV^T$')

e1 = np.array([[1],
               [0]])

e2 = np.array([[0],
               [1]])

# Calculate step by step from e1 and e2
VT_e1 = V.T@e1
VT_e2 = V.T@e2

S_VT_e1 = S@VT_e1
S_VT_e2 = S@VT_e2

U_S_VT_e1 = U@S_VT_e1
U_S_VT_e2 = U@S_VT_e2

#%% Bk4_Ch15_02_A
import numpy as np
import scipy
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'
import seaborn as sns
PRECISION = 3

def svd(X):
    full_matrices = True
    U, s, Vt = np.linalg.svd(X, full_matrices = full_matrices)
    # Put the vector singular values into a padded matrix
    if full_matrices:
        S = np.zeros(X.shape)
        np.fill_diagonal(S, s)
    else:
        S = np.diag(s)
    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)

def visualize_svd(X, title_X, title_U, title_S, title_V, fig_height=5):
    # Run SVD, as defined above
    U, S, V = svd(X)
    # all_ = np.r_[X.flatten(order='C'), U.flatten(order='C'), S.flatten(order='C'), V.flatten(order='C')]

    # all_max = max(all_.max(),all_.min())
    # all_min = -max(all_.max(),all_.min())
    all_max = 6
    all_min = -6
    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(12, fig_height))

    plt.sca(axs[0])
    ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(U, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_U)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(S, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_S)

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(V.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_V)
    return X, U, S, V

# Repeatability
np.random.seed(1)

# Generate random matrix
X = np.random.randn(6, 4)

# manipulate X and reduce rank to 3
# X[:,3] = X[:,0] + X[:,1]

X, U, S, V = visualize_svd(X,'$X$','$U$','$S$','$V^T$', fig_height=3)
X_2, U_2, S_2, V_2 = visualize_svd(X.T@X,'$X^TX$','$V$','$S^TS$','$V^T$', fig_height=3)
X_3, U_3, S_3, V_3 = visualize_svd(X@X.T,'$XX^T$','$U$','$SS^T$','$U^T$', fig_height=3)

# V == U_3 = V_3
# V = U_2 = V_2
scipy.linalg.inv(U) @ X@X.T @ U == S @ S.conj().T # 说明U是X*X^T的特征向量
scipy.linalg.inv(V) @ X.T@X @ V == S.conj().T @ S # 说明V是X^T*X的特征向量


#%% Bk4_Ch15_02_B
#  U*U.T = I
all_max = 6
all_min = -6

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(U, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U$')


plt.sca(axs[1])
ax = sns.heatmap(U.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U^T$')

plt.sca(axs[2])
ax = sns.heatmap(U@U.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$I$')


#%% Bk4_Ch15_02_C
## V*V.T = I

all_max = 6
all_min = -6

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(V,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V$')

plt.sca(axs[1])
ax = sns.heatmap(V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V^T$')

plt.sca(axs[2])
ax = sns.heatmap(V@V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$I$')

#%% Bk4_Ch15_02_D
# analysis of singular value matrix

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
# 四幅热图叠加还原原始图像
for j in [0, 1, 2, 3]:
    X_j = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
    plt.sca(axs[j])
    ax = sns.heatmap(X_j, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    title_txt = '$s_'+ str(j+1) + 'u_'+ str(j+1) + 'v_'+ str(j+1) + '^T$'
    plt.title(title_txt)

#  projection,  X 向 v_j 映射结果为 s_j u_j
for j in [0, 1, 2, 3]:
    fig, axs = plt.subplots(1, 7, figsize=(12, 3))
    v_j = V[:,j]
    v_j = np.matrix(v_j).T
    s_j = S[j,j]
    s_j = np.matrix(s_j)
    u_j = U[:,j]
    u_j = np.matrix(u_j).T

    plt.sca(axs[0])
    ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('X')

    plt.sca(axs[1])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(v_j,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('v_'+ str(j+1))

    plt.sca(axs[3])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(s_j,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('s_'+ str(j+1))

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(u_j,cmap='RdBu_r', vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('u_'+ str(j+1))



#%% Bk4_Ch16_01.py
#%% Bk4_Ch16_01_A
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
PRECISION = 3
def svd(X, full_matrices):
    U, s, Vt = np.linalg.svd(X, full_matrices = full_matrices)
    # Put the vector singular values into a padded matrix
    if full_matrices:
        S = np.zeros(X.shape)
        np.fill_diagonal(S, s)
    else:
        S = np.diag(s)
    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)
# Repeatability
np.random.seed(1)
# Generate random matrix
X = np.random.randn(6, 4)

# manipulate X and reduce rank to 3
# X[:,3] = X[:,0] + X[:,1]
all_max = 2
all_min = -2

#%% full, 数据 X 完全型 SVD 分解矩阵热图
print("数据 X 完全型 SVD 分解矩阵热图")
U, S, V = svd(X, full_matrices = True)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$X$')

plt.sca(axs[1])
ax = sns.heatmap(U,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U$')

plt.sca(axs[2])
ax = sns.heatmap(S,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$S$')

plt.sca(axs[3])
ax = sns.heatmap(V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V^T$')

#%% Bk4_Ch16_01_B
#  Economy-size, thin, 数据 X 经济型 SVD 分解热图
print("Economy-size, thin, 数据 X 经济型 SVD 分解热图")
U, S, V = svd(X, full_matrices = False)
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$X$')

plt.sca(axs[1])
ax = sns.heatmap(U, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U$')

plt.sca(axs[2])
ax = sns.heatmap(S, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$S$')

plt.sca(axs[3])
ax = sns.heatmap(V.T, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V^T$')

#%% Bk4_Ch16_01_C,
# Compact, 数据 X 紧凑型 SVD 分解热图
print("数据 X 紧凑型 SVD 分解热图")
import copy

X_rank_3 = copy.deepcopy(X);
# manipulate X and reduce rank to 3, 用 X 第一、二列数据之和替代 X 矩阵第四列,即 x4 = x1 + x2。这样 X 矩阵列向量线性相关,rank(X) = 3,而 s4 = 0。
X_rank_3[:,3] = X[:,0] + X[:,1]
U_rank_3, S_rank_3, V_rank_3 = svd(X_rank_3, full_matrices = False)

fig, axs = plt.subplots(1, 4, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X_rank_3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$X$')

plt.sca(axs[1])
ax = sns.heatmap(U_rank_3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U$')

plt.sca(axs[2])
ax = sns.heatmap(S_rank_3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$S$')

plt.sca(axs[3])
ax = sns.heatmap(V_rank_3.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V^T$')

#%% Bk4_Ch16_01_D
# Truncated,  采用截断型 SVD 分解还原数据运算热图
print("采用截断型 SVD 分解还原数据运算热图")
num_p = 3;
U_truc = U[:,0:num_p]
S_truc = S[0:num_p, 0:num_p]
V_truc = V[:, 0:num_p]
X_hat = U_truc@S_truc@(V_truc.T)

# reproduce
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X_hat, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}$')

plt.sca(axs[1])
ax = sns.heatmap(U_truc,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$U$')

plt.sca(axs[2])
ax = sns.heatmap(S_truc,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$S$')

plt.sca(axs[3])
ax = sns.heatmap(V_truc.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V^T$')

######### Error
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_hat,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}$')

plt.sca(axs[2])
ax = sns.heatmap(X - X_hat,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$E = X - \hat{X}$')

#%% Bk4_Ch16_01_E
# Tensor products, 计算张量积,以及绘制还原原始数据过程热图
u1_outer_v1 = np.outer(U[:,0][:, None], V[:,0][:, None]);
u2_outer_v2 = np.outer(U[:,1][:, None], V[:,1][:, None]);
u3_outer_v3 = np.outer(U[:,2][:, None], V[:,2][:, None]);
u4_outer_v4 = np.outer(U[:,3][:, None], V[:,3][:, None]);

# visualize tensor products
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(u1_outer_v1,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$u1v1^T$')

plt.sca(axs[1])
ax = sns.heatmap(u2_outer_v2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$u2v2^T$')

plt.sca(axs[2])
ax = sns.heatmap(u3_outer_v3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$u3v3^T$')

plt.sca(axs[3])
ax = sns.heatmap(u4_outer_v4,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$u4v4^T$')

X_1 = S[0,0]*u1_outer_v1
# X_1 = S[0,0]*U[:,0][:, None]@V[:,0][None, :];
X_2 = S[1,1]*u2_outer_v2
# X_2 = S[1,1]*U[:,1][:, None]@V[:,1][None, :];
X_3 = S[2,2]*u3_outer_v3
# X_3 = S[2,2]*U[:,2][:, None]@V[:,2][None, :];
X_4 = S[3,3]*u4_outer_v4
# X_4 = S[3,3]*U[:,3][:, None]@V[:,3][None, :];

# visualize components
fig, axs = plt.subplots(1, 4, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X_1, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_1$')

plt.sca(axs[1])
ax = sns.heatmap(X_2, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_2$')

plt.sca(axs[2])
ax = sns.heatmap(X_3, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_3$')

plt.sca(axs[3])
ax = sns.heatmap(X_4, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_4$')

#%% Bk4_Ch16_01_F
# Reproduction and error,  绘制本节数据还原和误差热图

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_1 + X_2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_1 + \hat{X}_2$')


plt.sca(axs[2])
ax = sns.heatmap(X - (X_1 + X_2),cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X - (\hat{X}_1 + \hat{X}_2)$')


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_1 + X_2 + X_3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_1 + \hat{X}_2 + \hat{X}_3$')


plt.sca(axs[2])
ax = sns.heatmap(X - (X_1 + X_2 + X_3),cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X - (\hat{X}_1 + \hat{X}_2 + \hat{X}_3)$')























































































































































































































































































