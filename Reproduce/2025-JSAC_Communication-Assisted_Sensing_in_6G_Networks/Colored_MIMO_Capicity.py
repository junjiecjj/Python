

import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cp

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)


#%%
def align_eigenvectors(U_c, Psi_c):
    """
    通过计算列向量间的内积来对齐U_c和Psi_c的特征向量顺序
    参数:
        U_c: Σc的特征向量矩阵
        Psi_c: Rc的特征向量矩阵
    返回:
        Psi_c_aligned: 对齐后的Psi_c
        mapping: 列映射关系
    """
    N = U_c.shape[1]
    # 计算相关系数矩阵
    correlation_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 计算归一化内积 (a·b*)/(|a|·|b|)
            a = U_c[:, i]
            b = Psi_c[:, j]
            correlation = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            correlation_matrix[i, j] = correlation

    reorder = correlation_matrix.argmax(axis = 1)
    Psi_c_aligned = Psi_c[:, reorder]

    return Psi_c_aligned, reorder

#%% 按行展开, 是错的，对不上
M = 2
N = 3
T = 4
I = np.eye(T)
H = np.arange(M*N).reshape(M, N)
X = np.random.randn(N, T)
Y = H@X

y = Y.flatten('C')

Hhat = np.kron(I, H)
x = X.flatten('C')

yhat = Hhat @ x

#%% 按列展开,对的, Eq.(12)
M = 2
N = 3
T = 4

Hs = np.random.randn(M, N) + 1j * np.random.randn(M, N)
Xs = np.random.randn(N, T) + 1j * np.random.randn(N, T)
Ys = Hs@Xs

ys = Ys.conj().T.flatten('F')
I = np.eye(M)
Xhat = np.kron(I, Xs.conj().T)
hs = Hs.conj().T.flatten('F')

yhat = Xhat @ hs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MIMO Capacity maximization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = 6  # recv annt, M >= N
L = 100
N = 4  # transmit annt
PT = 1
sigma_c2 = 1

Hc = np.random.randn(M, N) + 1j * np.random.randn(M, N)
Sigma_H = Hc.conj().T @ Hc
Lambda_h_hat, V_h = np.linalg.eig(Sigma_H)
Lambda_h_hat = np.real(Lambda_h_hat)

#%% IID的高斯噪声
Sigma_Z = np.eye(M) * sigma_c2   #  M x M
Lambda_z = np.array([1]* M) * sigma_c2

# idx = np.argsort(Lambda_h_hat)  # 从小到大, 对于高斯噪声，排序这步可要可不要
# Lambda_h_hat = Lambda_h_hat[idx]
# U_h = U_h[:, idx]

## Theoretical solution
from WaterFilling import water_filling, plot_waterfilling

Lambda_x1, water_level = water_filling(Lambda_z[M-N:], Lambda_h_hat, PT)
print(f"最优功率分配: {Lambda_x1}/{np.sum(Lambda_x1):.4f}")

plot_waterfilling(Lambda_z[M-N:]/Lambda_h_hat, Lambda_x1, water_level)
Sigma_X = V_h @ np.diag(Lambda_x1) @ V_h.conj().T

## Use CVX
Sigma_x = cp.Variable((N, N), hermitian = True)

constraints = [0 << Sigma_x,
               cp.trace(Sigma_x) <= PT,
              ]
obj = cp.Maximize(cp.log_det(Hc@Sigma_x@Hc.conj().T + Sigma_Z))
prob = cp.Problem(obj, constraints)
prob.solve()

if prob.status=='optimal':
     print(f"{prob.value}")
      # print(f"{Rc.value}")

Sigma_x = Sigma_x.value
Lambda_x, U_x = np.linalg.eig(Sigma_x)
Lambda_x = np.abs(Lambda_x)

# U_c 和 Psi_c的列的顺序不同，但是值确实是一样的，也就是列的顺序打乱了
Psi_c_aligned, reorder = align_eigenvectors(V_h, U_x)
Lambda_x = Lambda_x[reorder]


## Lambda_x == Lambda_x1, Sigma_X == Sigma_x, checked, amazing !!!
print(f"Lambda_x, cvx         = {Lambda_x}")
print(f"Lambda_x, Theoretical = {Lambda_x1}")


#%% Colored 噪声：正确版本

Z = np.random.randn(M, L) + 1j * np.random.randn(M, L)
Sigma_Z = Z @ Z.conj().T / L
Sigma_Z = (Sigma_Z + Sigma_Z.conj().T) / 2

# 等效信道矩阵对应的 Gram 矩阵：
# A_eff = H^H Sigma_Z^{-1} H
A_eff = Hc.conj().T @ np.linalg.solve(Sigma_Z, Hc)
A_eff = (A_eff + A_eff.conj().T) / 2

# Hermitian 矩阵使用 eigh
Lambda_eff, V_eff = np.linalg.eigh(A_eff)

# 按等效信道增益从大到小排序
idx = np.argsort(Lambda_eff)[::-1]
Lambda_eff = np.real(Lambda_eff[idx])
V_eff = V_eff[:, idx]
Lambda_eff = np.maximum(Lambda_eff, 0)

# 理论注水解
Lambda_x1, water_level = water_filling(np.ones(N), Lambda_eff, PT)
print(f"最优功率分配: {Lambda_x1}/{np.sum(Lambda_x1):.4f}")

plot_waterfilling(1 / Lambda_eff, Lambda_x1, water_level)

# 理论最优协方差
Sigma_X = V_eff @ np.diag(Lambda_x1) @ V_eff.conj().T
Sigma_X = (Sigma_X + Sigma_X.conj().T) / 2

# 理论容量
C1_mat = Hc @ Sigma_X @ Hc.conj().T + Sigma_Z
C1_mat = (C1_mat + C1_mat.conj().T) / 2
C1 = np.linalg.slogdet(C1_mat)[1]

#%% Use CVX

Sigma_x_var = cp.Variable((N, N), hermitian=True)

constraints = [
    Sigma_x_var >> 0,
    cp.trace(Sigma_x_var) <= PT
]

obj = cp.Maximize(cp.log_det(Hc @ Sigma_x_var @ Hc.conj().T + Sigma_Z))
prob = cp.Problem(obj, constraints)
prob.solve()

if prob.status == 'optimal':
    print(f"CVX objective = {prob.value}")

Sigma_x_cvx = Sigma_x_var.value
Sigma_x_cvx = (Sigma_x_cvx + Sigma_x_cvx.conj().T) / 2

# CVX 容量
C2_mat = Hc @ Sigma_x_cvx @ Hc.conj().T + Sigma_Z
C2_mat = (C2_mat + C2_mat.conj().T) / 2
C2 = np.linalg.slogdet(C2_mat)[1]

# 比较特征值
Lambda_x_cvx, U_x_cvx = np.linalg.eigh(Sigma_x_cvx)

idx = np.argsort(Lambda_x_cvx)[::-1]
Lambda_x_cvx = np.real(Lambda_x_cvx[idx])
U_x_cvx = U_x_cvx[:, idx]
Lambda_x_cvx = np.maximum(Lambda_x_cvx, 0)

print(f"Lambda_x, CVX         = {Lambda_x_cvx}")
print(f"Lambda_x, Theoretical = {Lambda_x1}")
print(f"C1 = {C1}, C2 = {C2}")
print(f"Capacity gap = {abs(C1 - C2)}")
print(f"Covariance gap = {np.linalg.norm(Sigma_X - Sigma_x_cvx, 'fro')}")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Radar Capacity maximization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%% Colored 噪声
# Z = np.random.randn(M, L) + 1j * np.random.randn(M, L)
# Sigma_Z = Z @ Z.conj().T / L  #  + np.diag(np.random.randint(10, size  = M))   #  N x N
# Lambda_z, U_z = np.linalg.eig(Sigma_Z)
# Lambda_z = np.abs(Lambda_z)
# idx = np.argsort(Lambda_z)[::-1]  # 从大到小
# Lambda_z = Lambda_z[idx]
# U_z = U_z[:, idx]

# idx = np.argsort(Lambda_h_hat)    # 从小到大
# Lambda_h_hat = Lambda_h_hat[idx]
# V_h = V_h[:, idx]


# ## Theoretical solution
# from WaterFilling import water_filling, plot_waterfilling

# Lambda_x1, water_level = water_filling(Lambda_z[M-N:], Lambda_h_hat, PT)
# print(f"最优功率分配: {Lambda_x1}/{np.sum(Lambda_x1):.4f}")

# plot_waterfilling(Lambda_z[M-N:]/Lambda_h_hat, Lambda_x1, water_level)
# Sigma_X = V_h @ np.diag(Lambda_x1) @ V_h.conj().T
# C1 = np.log(np.linalg.det(Hc@Sigma_X@Hc.conj().T + Sigma_Z))

# ## Use CVX
# Sigma_x = cp.Variable((N, N), hermitian = True)

# constraints = [0 << Sigma_x,
#                cp.trace(Sigma_x) <= PT,
#               ]
# obj = cp.Maximize(cp.log_det(Hc@Sigma_x@Hc.conj().T + Sigma_Z))
# prob = cp.Problem(obj, constraints)
# prob.solve()

# if prob.status=='optimal':
#      print(f"{prob.value}")
#       # print(f"{Rc.value}")

# Sigma_x = Sigma_x.value
# C2 = np.log(np.linalg.det(Hc@Sigma_x@Hc.conj().T + Sigma_Z))
# Lambda_x, U_x = np.linalg.eig(Sigma_x)
# Lambda_x = np.abs(Lambda_x)

# Psi_c_aligned, reorder = align_eigenvectors(V_h, U_x)
# Lambda_x = Lambda_x[reorder]


# # 好像上面的理论值和CVX仿真的 Lambda_x 结果有出入，
# print(f"Lambda_x, cvx         = {Lambda_x}")
# print(f"Lambda_x, Theoretical = {Lambda_x1}")
# print(f"C1 = {C1}, C2 = {C2}")  # 但是容量是差不多的，因此这里需要确认，到底是python带来的误差还是理论分析有误



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



















































































































