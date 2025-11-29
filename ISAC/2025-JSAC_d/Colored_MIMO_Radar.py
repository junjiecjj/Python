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
np.random.seed(12)


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
Lambda_h_hat = np.abs(Lambda_h_hat)

#%% IID的高斯噪声
Sigma_W = np.eye(M) * sigma_c2   #  M x M
Lambda_w = np.array([1]* M) * sigma_c2

# idx = np.argsort(Lambda_h_hat)  # 从小到大, 对于高斯噪声，排序这步可要可不要
# Lambda_h_hat = Lambda_h_hat[idx]
# V_h = V_h[:, idx]

## Theoretical solution
from WaterFilling import water_filling, plot_waterfilling

Lambda_x1, water_level = water_filling(Lambda_w[M-N:], Lambda_h_hat, PT)
print(f"最优功率分配: {Lambda_x1}/{np.sum(Lambda_x1):.4f}")

plot_waterfilling(Lambda_w[M-N:]/Lambda_h_hat, Lambda_x1, water_level)
Sigma_X = V_h @ np.diag(Lambda_x1) @ V_h.conj().T

## Use CVX
Sigma_x = cp.Variable((N, N), hermitian = True)

constraints = [0 << Sigma_x,
               cp.trace(Sigma_x) <= PT,
              ]
obj = cp.Maximize(cp.log_det(Hc@Sigma_x@Hc.conj().T + Sigma_W))
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
print(f"Lambda_x = {Lambda_x}")
print(f"Lambda_x1 = {Lambda_x1}")


#%% Colored 噪声
W = np.random.randn(M, L) + 1j * np.random.randn(M, L)
Sigma_W = W @ W.conj().T / L  + np.diag(np.random.randint(10, size  = M))   #  N x N
Lambda_w, U_w = np.linalg.eig(Sigma_W)
Lambda_w = np.abs(Lambda_w)
idx = np.argsort(Lambda_w)[::-1]  # 从大到小
Lambda_w = Lambda_w[idx]
U_w = U_w[:, idx]


idx = np.argsort(Lambda_h_hat)  # 从小到大
Lambda_h_hat = Lambda_h_hat[idx]
V_h = V_h[:, idx]


## Theoretical solution
from WaterFilling import water_filling, plot_waterfilling

Lambda_x1, water_level = water_filling(Lambda_w[M-N:], Lambda_h_hat, PT)
print(f"最优功率分配: {Lambda_x1}/{np.sum(Lambda_x1):.4f}")

plot_waterfilling(Lambda_w[M-N:]/Lambda_h_hat, Lambda_x1, water_level)
Sigma_X = V_h @ np.diag(Lambda_x1) @ V_h.conj().T

## Use CVX
Sigma_x = cp.Variable((N, N), hermitian = True)

constraints = [0 << Sigma_x,
               cp.trace(Sigma_x) <= PT,
              ]
obj = cp.Maximize(cp.log_det(Hc@Sigma_x@Hc.conj().T + Sigma_W))
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


## Lambda_x == Lambda_x1, checked, amazing !!!
print(f"Lambda_x = {Lambda_x}")
print(f"Lambda_x1 = {Lambda_x1}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Radar Capacity maximization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



















































































































