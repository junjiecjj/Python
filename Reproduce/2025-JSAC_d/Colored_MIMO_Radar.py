

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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Radar Capacity maximization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  MMSE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  MI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%














#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



















































































































