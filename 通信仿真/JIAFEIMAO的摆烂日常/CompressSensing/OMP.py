
"""
https://zhuanlan.zhihu.com/p/322180659

https://scikit-learn.org/stable/auto_examples/decomposition/plot_sparse_coding.html#sphx-glr-auto-examples-decomposition-plot-sparse-coding-py

https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py

https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
"""

# https://wenku.csdn.net/answer/7u4oktb81d
import numpy as np
np.random.seed(42)

def OMP111(A, y, sparsity):
    """
    OMP算法的Python实现

    参数：
    A: 测量矩阵，形状为(m, n)
    y: 观测向量，形状为(m, 1)
    k: 稀疏度，即信号的非零元素个数

    返回：
    x: 重构的稀疏信号，形状为(n, 1)
    """
    m, n = A.shape
    residual = y.copy()  # 初始化残差
    support = []  # 初始化支持集合

    for _ in range(sparsity):
        # 计算投影系数
        projections = np.abs(A.T @ residual)
        # 选择最相关的原子
        index = np.argmax(projections)
        support.append(index)
        # 更新估计信号
        x = np.linalg.lstsq(A[:, support], y, rcond=None)[0]
        # 更新残差
        residual = y - A[:, support] @ x

    # 构造稀疏信号
    x_sparse = np.zeros((n, 1))
    x_sparse[support] = x

    return x_sparse

#%% ###################################### 实例
# 重建一个稀疏信号。设定信号  具有稀疏性，即仅有少数元素非零。
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

def find_chinese_font():
    font_list = font_manager.fontManager.ttflist
    for font in font_list:
        if "SimHei" in font.name:
            return font.fname
        elif "SimSun" in font.name:
            return font.fname
    return None

font_path = find_chinese_font()
if font_path:
    my_font = font_manager.FontProperties(fname=font_path)
else:
    print("未找到中文字体")

rcParams['axes.unicode_minus'] = False
# rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

N = 100
K = 5
signal = np.zeros(N)
non_zero_indices = np.random.choice(N, K, replace=False)
signal[non_zero_indices] = np.random.randn(K)

M = 50
phi = np.random.randn(M, N)
y = phi @ signal

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K)
omp.fit(phi, y)
reconstructed_signal = omp.coef_

plt.figure(figsize=(10, 6), constrained_layout=True)
plt.plot(signal, label="原始稀疏信号", color="blue", linewidth=2)
plt.plot(reconstructed_signal, linestyle="--", color="red", label="重建信号", linewidth=2)
if font_path:
    plt.legend(loc='upper right', prop = my_font, fontsize = 22,  )
    plt.xlabel("信号索引", fontsize=24, fontproperties=my_font)
    plt.ylabel("幅值", fontsize=24, fontproperties=my_font)
    plt.title("OMP 重建效果", fontproperties=my_font, fontsize=24, )
else:
    plt.legend(loc='upper right', prop = my_font, fontsize = 22,  )
    plt.xlabel("信号索引", fontsize = 24)
    plt.ylabel("幅值", fontsize = 24)
    plt.title("OMP 重建效果", fontproperties=my_font, fontsize = 24)

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
# 这表明在无噪声条件下，OMP 能够准确重建稀疏信号，重建精度高且未引入额外噪声或误差。



#%% 随着噪声水平的增加，OMP 的重建效果
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

def OMP2(A, y, sparsity):
    """
    OMP算法的Python实现
    参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m, 1)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    m, n = A.shape
    y = y.reshape(-1,1)
    residual = y.copy()  # 初始化残差
    support = []  # 初始化支持集合

    for _ in range(sparsity):
        # 计算投影系数
        projections = np.abs(A.T @ residual)
        # 选择最相关的原子
        index = np.argmax(projections)
        support.append(index)
        # 更新估计信号
        x = np.linalg.lstsq(A[:, support], y, rcond = None)[0]
        # 更新残差
        residual = y - A[:, support] @ x
    # 构造稀疏信号
    x_sparse = np.zeros((n, 1))
    x_sparse[support] = x

    return x_sparse

def OMP1(phi, y, sparsity):
    """
    OMP算法的Python实现
        参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m,)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    N = phi.shape[1]
    residual = y.copy()
    index_set = []
    theta = np.zeros(N)
    for _ in range(sparsity):
        correlations = phi.T @ residual
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)
        phi_selected = phi[:, index_set]
        theta_selected, _, _, _ = np.linalg.lstsq(phi_selected, y, rcond = None)
        for i, idx in enumerate(index_set):
            theta[idx] = theta_selected[i]
        residual = y - phi @ theta
        if np.linalg.norm(residual) < 1e-6:
            break
    return theta

def find_chinese_font():
    font_list = font_manager.fontManager.ttflist
    for font in font_list:
        if "SimHei" in font.name:
            return font.fname
        elif "SimSun" in font.name:
            return font.fname
    return None

font_path = find_chinese_font()
if font_path:
    my_font = font_manager.FontProperties(fname=font_path)
else:
    print("未找到中文字体")

rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

N = 100
M = 50
phi = np.random.randn(M, N)

#%% ############# 1. 对噪声敏感性实验
noise_levels = [0, 0.05, 0.1, 0.2, 0.5, 1]
K = 5
signal = np.zeros(N)
signal[np.random.choice(N, K, replace=False)] = np.random.randn(K)

plt.figure(figsize=(15, 8), constrained_layout=True)
for i, noise_level in enumerate(noise_levels, 1):
    y = phi @ signal + noise_level * np.random.randn(M)
    reconstructed_signal = OMP1(phi, y, sparsity=K)

    plt.subplot(2, 3, i)
    plt.plot(signal, label="原始信号", color="blue", linewidth=1.5)
    plt.plot(reconstructed_signal, linestyle="--", color="red", label="重建信号", linewidth=1.5)
    plt.title(f"噪声水平: {noise_level}", fontproperties=my_font, fontsize = 18)
    plt.legend(prop=my_font, fontsize = 18)

plt.suptitle("OMP 对不同噪声水平的重建效果", fontproperties = my_font, fontsize = 20)
plt.show()

#%% ############# 2. 稀疏度依赖性实验
estimated_sparsities = [2, 5, 7, 10, 15, 20]
y = phi @ signal

plt.figure(figsize=(15, 8), constrained_layout=True)
for i, estimated_K in enumerate(estimated_sparsities, 1):
    reconstructed_signal = OMP1(phi, y, sparsity=estimated_K)

    plt.subplot(2, 3, i)
    plt.plot(signal, label="原始信号", color="blue", linewidth=1.5)
    plt.plot(reconstructed_signal, linestyle="--", color="red", label="重建信号", linewidth=1.5)
    plt.title(f"预设稀疏度: {estimated_K}", fontproperties=my_font, fontsize = 18)
    plt.legend(prop=my_font, fontsize = 18)
plt.suptitle("OMP 对不同预设稀疏度的重建效果", fontproperties = my_font, fontsize = 20)
plt.show()

#%% ############# 3. 适用范围实验（不同信号稀疏度）
true_sparsities = [2, 5, 10, 15, 20, 30]

plt.figure(figsize=(15, 8), constrained_layout = True)
for i, current_K in enumerate(true_sparsities, 1):
    signal = np.zeros(N)
    signal[np.random.choice(N, current_K, replace=False)] = np.random.randn(current_K)
    y = phi @ signal
    reconstructed_signal = OMP1(phi, y, sparsity=current_K)

    plt.subplot(2, 3, i)
    plt.plot(signal, label = "原始信号", color = "blue", linewidth = 1.5)
    plt.plot(reconstructed_signal, linestyle = "--", color = "red", label = "重建信号", linewidth = 1.5)
    plt.title(f"信号稀疏度: {current_K}", fontproperties = my_font, fontsize = 18)
    plt.legend(prop = my_font, fontsize = 18)

plt.suptitle("OMP 对不同信号稀疏度的重建效果", fontproperties = my_font, fontsize = 20)

plt.show()
plt.close('all')



















