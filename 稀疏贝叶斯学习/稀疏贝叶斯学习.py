#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:46:24 2025

@author: jack

https://zhuanlan.zhihu.com/p/1893622601374468027

https://blog.csdn.net/qq_44648285/article/details/143313531

https://blog.csdn.net/weixin_40735720/article/details/148583124

https://blog.csdn.net/qq_45471796/article/details/130487580

https://www.cnblogs.com/shuangli0824/p/10811244.html

https://github.com/al5250/sparse-bayes-learn
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_rvm import EMRVC

# 1. 生成模拟的二分类数据
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=10, n_classes=2, random_state=42)

# 2. 数据预处理：标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 初始化并训练 RVM 分类器（线性核）
rvm = EMRVC(kernel='linear', verbose=False)
rvm.fit(X_train, y_train)

# 5. 预测并评估
y_pred = rvm.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))



#%% DeepSeek, SBL for Compress senesing
# https://zhuanlan.zhihu.com/p/1893622601374468027
import numpy as np
import matplotlib.pyplot as plt

def CS_SBL(Phi, y, max_iter=100, tol=1e-4, prune_threshold=1e3):
    """
    Sparse Bayesian Learning (SBL) via EM algorithm.
    Parameters:
        Phi: Design matrix (N x M)
        y: Observation vector (N x 1)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        prune_threshold: Threshold for pruning small weights
    Returns:
        w: Estimated sparse weights (M x 1)
        alpha: Hyperparameters (M x 1)
        sigma2: Noise variance
    """
    N, M = Phi.shape
    alpha = np.ones(M)  # Initialize hyperparameters
    sigma2 = np.var(y)  # Initialize noise variance

    for _ in range(max_iter):
        # E-Step: Compute posterior mean and covariance
        A = np.diag(alpha)
        Sigma = np.linalg.inv(sigma2 ** (-1) * Phi.T @ Phi + A)
        mu = sigma2 ** (-1) * Sigma @ Phi.T @ y
        # M-Step: Update alpha and sigma2
        gamma = 1 - alpha * np.diag(Sigma)
        alpha_new = gamma / (mu ** 2 + 1e-10)  # Avoid division by zero
        # sigma2_new = np.linalg.norm(y - Phi @ mu) ** 2 / (N - np.sum(gamma))
        ## or
        sigma2_new = (np.linalg.norm(y - Phi @ mu) ** 2 + sigma2 * np.sum(gamma))/ N

        # Check convergence
        if np.max(np.abs(alpha_new - alpha)) < tol and np.abs(sigma2_new - sigma2) < tol:
            break
        alpha, sigma2 = alpha_new, sigma2_new

    # Prune small weights (set to zero if alpha > prune_threshold)
    w = mu * (alpha < prune_threshold)
    return w, alpha, sigma2

def plot_results(w_true, w_est):
    """
    Plot true weights vs. estimated weights.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.stem(w_true, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.title("True Weights")
    plt.xlabel("Index")
    plt.ylabel("Weight Value")

    plt.subplot(1, 2, 2)
    plt.stem(w_est, linefmt='r-', markerfmt='ro', basefmt=' ')
    plt.title("Estimated Weights (SBL)")
    plt.xlabel("Index")
    plt.ylabel("Weight Value")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    N, M = 100, 200
    K = 10
    Phi = np.random.randn(N, M)
    w_true = np.zeros(M)
    nonzero = np.random.randint(M, size = K)
    nonzero = np.sort(nonzero)
    w_true[nonzero] = 1.0 + np.abs(np.random.randn(K,))  # True sparse weights (non-zero at indices 10-14)
    y = Phi @ w_true + 0.01 * np.random.randn(N)  # Observations with noise

    # Run SBL
    w_est, alpha, sigma2 = CS_SBL(Phi, y)

    # Plot results
    plot_results(w_true, w_est)

    print("Estimated noise variance (σ²):", sigma2)
    print("True Non-zero weight indices:", nonzero)
    print("Estimated Non-zero weight indices:", np.where(np.abs(w_est) > 0.1)[0])


#%% DeepSeek, SBL of DOA estimate

####>>>>>>>>>>>>>>>>>>>>>>>>>>.原始接收信号矩阵Y, T = 1
import numpy as np
import matplotlib.pyplot as plt

def sparse_bayesian_doa(y, array_geometry, theta_grid, max_iter=100, tol=1e-4):
    """
    SBL-based DOA estimation with division-by-zero fix.
    """
    N = len(y)
    M = len(theta_grid)

    # Construct dictionary matrix
    Phi = np.zeros((N, M), dtype=complex)
    for m in range(M):
        phi_m = np.exp(-2j * np.pi * array_geometry * np.sin(np.deg2rad(theta_grid[m])))
        Phi[:, m] = phi_m

    # SBL initialization
    alpha = np.ones(M)
    sigma2 = np.var(y)

    for _ in range(max_iter):
        # E-Step
        A = np.diag(alpha)
        Sigma = np.linalg.inv((1 / sigma2) * Phi.conj().T @ Phi + A)
        mu = (1 / sigma2) * Sigma @ Phi.conj().T @ y

        # M-Step: Safe division
        gamma = 1 - alpha * np.diag(Sigma)
        alpha_new = gamma / (np.abs(mu) ** 2 + 1e-10)  # Add epsilon to avoid division by zero

        # Update sigma2
        # sigma2_new = np.linalg.norm(y - Phi @ mu) ** 2 / (N - np.sum(gamma))
        ## or
        sigma2_new = (np.linalg.norm(y - Phi @ mu) ** 2 + sigma2 * np.sum(gamma))/ N
        # Check convergence
        if np.max(np.abs(alpha_new - alpha)) < tol and np.abs(sigma2_new - sigma2) < tol:
            break

        alpha, sigma2 = alpha_new, sigma2_new

    # Detect DOAs
    w_est = mu
    doa_estimates = theta_grid[np.abs(w_est) > 0.1 * np.max(np.abs(w_est))]

    return w_est, doa_estimates

def plot_doa_spectrum(theta_grid, w_est, true_doas=None):
    plt.figure(figsize=(10, 4))
    plt.plot(theta_grid, 20 * np.log10(np.abs(w_est) / np.max(np.abs(w_est))), 'b-')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('DOA Spectrum (SBL)')
    plt.grid(True)
    if true_doas is not None:
        for doa in true_doas:
            plt.axvline(doa, color='r', linestyle='--', label=f'True DOA: {doa}°')
        plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    N = 8  # Number of sensors
    wavelength = 1.0
    array_geometry = np.arange(N) * 0.5 * wavelength  # ULA
    theta_true = np.array([-30, 30])  # True DOAs
    theta_grid = np.linspace(-80, 80, 180)  # Angle grid

    # Generate signal
    A = np.zeros((N, len(theta_true)), dtype = complex)
    for k in range(len(theta_true)):
        A[:, k] = np.exp(-2j * np.pi * array_geometry * np.sin(np.deg2rad(theta_true[k])))
    s = np.random.randn(len(theta_true)) + 1j * np.random.randn(len(theta_true))
    y = A @ s + 0.01 * (np.random.randn(N) + 1j * np.random.randn(N))

    # Run SBL-DOA
    w_est, doa_estimates = sparse_bayesian_doa(y, array_geometry, theta_grid)

    # Plot results
    plot_doa_spectrum(theta_grid, w_est, true_doas = theta_true)
    print("Estimated DOAs (degrees):", doa_estimates)

####>>>>>>>>>>>>>>>>>>>>>>>>>>.原始接收信号矩阵Y, T = 1000
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv

def SBL_DOA(Y, array_pos, theta_grid, max_iter=100, tol=1e-4):
    """
    基于稀疏贝叶斯学习的DOA估计（数值稳定版本）
    """
    N, T = Y.shape
    M = len(theta_grid)

    # 构造字典矩阵（添加微小正则项确保数值稳定）
    Phi = np.zeros((N, M), dtype=complex)
    for m in range(M):
        Phi[:, m] = np.exp(-2j * np.pi * array_pos * np.sin(np.deg2rad(theta_grid[m])))
    Phi += 1e-10 * np.random.randn(N, M)  # 微小扰动避免奇异矩阵

    # 初始化（确保非零）
    alpha = np.ones(M) + 1e-6
    beta = 1 / (np.var(Y) + 1e-10)
    mu_x = np.zeros((M, T), dtype=complex)

    for it in range(max_iter):
        try:
            # E-step: 添加正则项确保矩阵可逆
            Sigma_x = inv(np.diag(alpha) + beta * Phi.conj().T @ Phi + 1e-8 * np.eye(M))
            mu_x = beta * Sigma_x @ Phi.conj().T @ Y

            # M-step: 添加保护避免除零
            gamma = np.clip(1 - alpha * np.diag(Sigma_x), 1e-10, 1)
            mu_power = np.mean(np.abs(mu_x)**2, axis=1)
            alpha_new = gamma / (mu_power + 1e-10)

            # 噪声方差更新（确保正值）
            residual = Y - Phi @ mu_x
            beta_new = (N*T) / (np.linalg.norm(residual, 'fro')**2 + 1e-10)
            beta_new = np.clip(beta_new, 1e-10, 1e10)

            # 检查收敛
            if np.max(np.abs(alpha_new - alpha)) < tol and np.abs(beta_new - beta) < tol:
                break

            alpha, beta = alpha_new, beta_new

        except np.linalg.LinAlgError:
            print(f"矩阵求逆失败于迭代 {it}，添加更强的正则项")
            Sigma_x = inv(np.diag(alpha) + beta * Phi.conj().T @ Phi + 1e-6 * np.eye(M))
            continue

    # 计算空间谱（对数尺度处理零值）
    P = np.mean(np.abs(mu_x)**2, axis=1)
    P = P / (np.max(P) + 1e-10)

    # 提取DOA估计（基于峰值检测）
    peaks = np.where(P > 0.5 * np.max(P))[0]
    theta_est = theta_grid[peaks]

    return theta_est, P

# 示例使用
if __name__ == "__main__":
    # 参数设置
    N = 8  # 阵元数
    wavelength = 1.0
    array_pos = np.arange(N) * 0.5 * wavelength  # ULA
    theta_true = np.array([-20, 10, 30])  # 真实DOA
    T = 1000  # 快拍数
    theta_grid = np.linspace(-40, 40, 181)  # 角度网格

    # 生成接收信号
    A = np.zeros((N, len(theta_true)), dtype=complex)
    for k in range(len(theta_true)):
        A[:, k] = np.exp(-2j * np.pi * array_pos * np.sin(np.deg2rad(theta_true[k])))
    S = (np.random.randn(len(theta_true), T) + 1j * np.random.randn(len(theta_true), T)) / np.sqrt(2)
    Y = A @ S + 0.1 * (np.random.randn(N, T) + 1j * np.random.randn(N, T)) / np.sqrt(2)

    # DOA估计
    theta_est, P = SBL_DOA(Y, array_pos, theta_grid)

    # 绘制结果
    plt.figure(figsize=(10, 5))
    plt.plot(theta_grid, 10 * np.log10(P + 1e-10))
    plt.scatter(theta_true, np.zeros(len(theta_true)), c='r', marker='x', label='True DOA')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title('DOA Estimation using SBL')
    plt.legend()
    plt.grid()
    plt.show()

    print("Estimated DOAs:", theta_est)


###>>>>>>>>>>>>>>>>>>>>>>>>>> 仅使用协方差矩阵 Rxx 的稀疏贝叶斯学习（SBL）DOA估计的完整Python实现, 版本一
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm
from scipy.signal import find_peaks

def SBL_DOA_Rxx(Rxx, array_pos, theta_grid, max_iter=100, tol=1e-4):
    """
    基于协方差矩阵的稀疏贝叶斯学习DOA估计

    参数:
        Rxx: 样本协方差矩阵 (N x N)
        array_pos: 阵列位置（单位：波长）
        theta_grid: 角度搜索网格（度）
        max_iter: 最大迭代次数
        tol: 收敛阈值

    返回:
        theta_est: 估计的DOA角度（度）
        power_spectrum: 空间谱（dB）
    """
    # 输入校验
    assert Rxx.shape[0] == Rxx.shape[1], "Rxx必须是方阵"
    N = Rxx.shape[0]
    M = len(theta_grid)

    # 构造字典矩阵（带正则化）
    theta_rad = np.deg2rad(theta_grid)
    Phi = np.exp(-2j * np.pi * array_pos[:, None] * np.sin(theta_rad[None, :]))
    Phi += 1e-10 * (np.random.randn(*Phi.shape) + 1j * np.random.randn(*Phi.shape))

    # 初始化参数
    Gamma = np.eye(M) * 0.1  # 稀疏性参数
    noise_var = np.real(np.mean(np.linalg.eigvalsh(Rxx)[:N//2]))  # 噪声方差估计

    # EM算法主循环
    for it in range(max_iter):
        try:
            # E-step: 计算后验统计量 (使用Woodbury恒等式)
            PhiH_Phi = Phi.conj().T @ Phi
            K = inv(Gamma) + PhiH_Phi / noise_var
            Sigma_x = inv(K + 1e-8 * np.eye(M))  # 正则化
            Mu_x = Sigma_x @ Phi.conj().T @ Rxx / noise_var

            # M-step: 更新参数
            diag_Mu = np.real(np.diag(Mu_x @ Mu_x.conj().T))
            diag_Sigma = np.real(np.diag(Sigma_x))
            Gamma_new = np.diag(diag_Mu + diag_Sigma)  # T=1时的简化形式

            # 噪声更新
            residual = Rxx - Phi @ Mu_x
            noise_var_new = np.real(np.trace(residual)) / N

            # 收敛检查
            if np.linalg.norm(Gamma_new - Gamma, 'fro') < tol:
                break

            Gamma, noise_var = Gamma_new, noise_var_new

        except np.linalg.LinAlgError:
            print(f"Iter {it}: 矩阵求逆失败，增加正则化")
            K = inv(Gamma) + PhiH_Phi/noise_var + 1e-6 * np.eye(M)
            Sigma_x = inv(K)

    # 计算空间谱（对数尺度）
    power = np.real(np.diag(Gamma))
    power_db = 10 * np.log10(power / np.max(power) + 1e-10)

    # 峰值检测（自适应阈值）
    peaks, _ = find_peaks(power_db,
                         height=np.median(power_db) + 5,  # 高于中值5dB
                         distance=len(theta_grid)//20)    # 最小角度间隔

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(theta_grid, power_db, label='SBL Power Spectrum')
    plt.scatter(theta_grid[peaks], power_db[peaks], c='r',
               label=f'Estimated DOAs: {theta_grid[peaks]}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('SBL DOA Estimation using Covariance Matrix')
    plt.legend()
    plt.grid()
    plt.show()

    return theta_grid[peaks], power_db

# 示例使用
if __name__ == "__main__":
    # 参数设置
    np.random.seed(0)
    N = 8  # 阵元数
    wavelength = 1.0
    array_pos = np.arange(N) * 0.5 * wavelength  # ULA
    theta_true = np.array([-30, 0, 45])  # 真实DOA
    T = 100  # 快拍数
    theta_grid = np.arange(-80, 80, 0.5)  # 1度分辨率

    # 生成接收信号（含相干信号）
    A = np.exp(-2j * np.pi * array_pos[:, None] * np.sin(np.deg2rad(theta_true)[None, :]))
    S = np.random.randn(len(theta_true), T) + 1j * np.random.randn(len(theta_true), T)  # 相干信号源
    Y = A @ S + 0.5 * (np.random.randn(N, T) + 1j * np.random.randn(N, T)) / np.sqrt(2)

    # 方法1：直接处理原始信号
    # theta_est1, ps1 = SBL_DOA_enhanced(Y, array_pos, theta_grid, use_covariance=False)

    # 方法2：使用协方差矩阵输入
    Rxx = Y @ Y.conj().T / T
    theta_est2, ps2 = SBL_DOA_Rxx(Rxx, array_pos, theta_grid,  )

    # print(f"原始信号输入结果: {theta_est1}")
    print(f"协方差矩阵输入结果: {theta_est2}")

###>>>>>>>>>>>>>>>>>>>>>>>>>> 仅使用协方差矩阵 Rxx 的稀疏贝叶斯学习（SBL）DOA估计的完整Python实现, 版本2

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, eigvalsh
from scipy.signal import find_peaks

def SBL_DOA_Rxx(Rxx, array_pos, theta_grid, max_iter=100, tol=1e-6):
    """
    严格理论实现的SBL DOA估计 (复数域运算)

    参数:
        Rxx: 样本协方差矩阵 (N x N)，可为复数Hermitian矩阵
        array_pos: 阵列位置 (波长单位)
        theta_grid: 角度网格 (度)
        max_iter: 最大迭代次数
        tol: 收敛阈值

    返回:
        theta_est: 估计的DOA (度)
        power_spectrum: 空间谱 (dB)
    """
    # ===== 1. 初始化 =====
    N = Rxx.shape[0]
    M = len(theta_grid)

    # 构造字典矩阵 (保持复数)
    theta_rad = np.deg2rad(theta_grid)
    Phi = np.exp(-2j * np.pi * array_pos[:, None] * np.sin(theta_rad[None, :]))

    # 初始化超参数 (理论形式)
    alpha = np.ones(M, dtype=complex)  # 复数初始化
    beta = 1 / np.mean(np.real(eigvalsh(Rxx)[:N//2]))  # 实数噪声方差

    # ===== 2. EM算法迭代 =====
    for iter in range(max_iter):
        # --- E-step: 严格理论实现 ---
        Sigma_x = inv(np.diag(alpha) + beta * Phi.conj().T @ Phi + 1e-10 * np.eye(M))
        Mu_x = beta * Sigma_x @ Phi.conj().T @ Rxx

        # --- M-step: 理论更新公式 ---
        gamma = 1 - alpha * np.diag(Sigma_x)
        alpha_new = gamma / np.diag(Mu_x @ Mu_x.conj().T + 1e-10)

        # 噪声方差更新 (保持实数)
        residual = Rxx - Phi @ Mu_x
        beta_new = N / np.real(np.trace(residual) + 1e-10)

        # --- 复数模收敛检查 ---
        if np.max(np.abs(alpha_new - alpha) / (np.abs(alpha) + 1e-10)) < tol:
            break

        alpha, beta = alpha_new, beta_new

    # ===== 3. 空间谱计算 (理论形式) ====
    power = 1 / (np.abs(alpha) + 1e-10)  # 取模处理
    power_db = 10 * np.log10(power / np.max(power) + 1e-10)

    # ===== 4. 峰值检测 (复数兼容) ====
    peaks, _ = find_peaks(np.real(power_db), height=-5, distance=10)
    theta_est = theta_grid[peaks]

    # ===== 5. 可视化 =====
    plt.figure(figsize=(10, 5))
    plt.plot(theta_grid, np.real(power_db))
    plt.scatter(theta_est, np.real(power_db[peaks]), c='r', label=f'Estimated DOAs: {theta_est}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title('Theoretical SBL DOA Estimation (Complex Domain)')
    plt.legend()
    plt.grid()
    plt.show()

    return theta_est, power_db


# 示例使用
if __name__ == "__main__":
    # 参数设置
    np.random.seed(0)
    N = 8  # 阵元数
    wavelength = 1.0
    array_pos = np.arange(N) * 0.5 * wavelength  # ULA
    theta_true = np.array([-20, 0, 45])  # 真实DOA
    T = 100  # 快拍数
    theta_grid = np.arange(-80, 80, 0.5)  # 1度分辨率

    # 生成接收信号（含相干信号）
    A = np.exp(-2j * np.pi * array_pos[:, None] * np.sin(np.deg2rad(theta_true)[None, :]))
    S = np.random.randn(len(theta_true), T) + 1j * np.random.randn(len(theta_true), T)  # 相干信号源
    Y = A @ S + 0.5 * (np.random.randn(N, T) + 1j * np.random.randn(N, T)) / np.sqrt(2)

    # 方法1：直接处理原始信号
    # theta_est1, ps1 = SBL_DOA_enhanced(Y, array_pos, theta_grid, use_covariance=False)

    # 方法2：使用协方差矩阵输入
    Rxx = Y @ Y.conj().T / T
    theta_est2, ps2 = SBL_DOA_Rxx(Rxx, array_pos, theta_grid,  )

    # print(f"原始信号输入结果: {theta_est1}")
    print(f"协方差矩阵输入结果: {theta_est2}")






#%% https://blog.csdn.net/qq_45471796/article/details/130487580
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
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
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2

def OGSBI(params):
    """OGSBI算法实现"""
    Y = params['Y']
    reso_num = params['reso_num']
    reso_grid = np.linspace(-90, 90, reso_num).reshape(-1, 1)
    reso = 180 / (reso_num - 1)
    array_num, snapshot_num = Y.shape
    r = reso * np.pi / 180
    maxiter = params['maxiter']
    tol = params['tolerance']
    index_b = np.random.permutation(len(reso_grid))[:array_num]
    converged = False
    iter_num = 0

    # 初始化导向矩阵
    A = np.exp(-1j * np.arange(array_num).reshape(-1, 1) * np.pi * np.sin(reso_grid.reshape(1, -1) * np.pi / 180))
    B = -1j * np.pi * np.arange(array_num).reshape(-1, 1) * np.cos(reso_grid.reshape(1, -1) * np.pi / 180) * A

    alpha = np.mean(np.abs(A.conj().T @ Y), axis=1).reshape(-1, 1).astype(complex)
    beta = np.zeros((reso_num, 1))

    # 超参数初始化
    c_sigma0_init = 1e-4
    d_sigma0_init = 1e-4
    c_gamma_init = 1
    d_gamma_init = 1e-4
    alpha_0 = 0.01
    alpha_0_seq = np.zeros(maxiter, dtype=complex)
    Phi = A.copy()

    while not converged and iter_num < maxiter:
        iter_num += 1
        Phi[:, index_b] = np.exp(-1j * np.arange(array_num).reshape(-1, 1) * np.pi * np.sin(reso_grid[index_b].reshape(1, -1) * np.pi / 180))
        B[:, index_b] = -1j * np.pi * np.arange(array_num).reshape(-1, 1) * np.cos(reso_grid[index_b].reshape(1, -1) * np.pi / 180) * A[:, index_b]
        alpha_last = alpha.copy()

        # 更新X的后验概率密度函数的均值与方差
        C = (1/alpha_0) * np.eye(array_num) + Phi @ np.diag(alpha.flatten()) @ Phi.conj().T
        Cinv = inv(C)
        Sigma = np.diag(alpha.flatten()) - np.diag(alpha.flatten()) @ Phi.conj().T @ Cinv @ Phi @ np.diag(alpha.flatten())
        mu = alpha_0 * Sigma @ Phi.conj().T @ Y

        # 更新alpha
        musq = np.mean(np.abs(mu)**2, axis=1).reshape(-1, 1)
        alpha = musq + np.real(np.diag(Sigma)).reshape(-1, 1).astype(complex)

        for ik in range(reso_num):
            numerator = mu[ik, :] @ mu[ik, :].conj().T + snapshot_num * Sigma[ik, ik]
            alpha[ik] = (-snapshot_num + np.sqrt(snapshot_num**2 + 4 * d_gamma_init * numerator)) / (2 * d_gamma_init)

        # 更新alpha_0
        numerator = snapshot_num * array_num + c_sigma0_init - 1
        denominator = np.linalg.norm(Y - Phi @ mu, 'fro')**2 + snapshot_num * np.trace(Phi @ Sigma @ Phi.conj().T) + d_sigma0_init
        alpha_0 = numerator / denominator

        alpha_0_seq[iter_num-1] = alpha_0
        # 判断是否停止迭代
        if np.linalg.norm(alpha - alpha_last) / np.linalg.norm(alpha_last) < tol:
            converged = True

        # 更新beta
        beta, index_b = off_grid_operation(Y, alpha, array_num, mu, Sigma, Phi, B, beta, r)
        reso_grid[index_b] = reso_grid[index_b] + np.real(beta[index_b]) * 180 / np.pi  # 添加np.real()取实部

    result = {
        'mu': mu,
        'Sigma': Sigma,
        'beta': beta,
        'alpha': alpha,
        'iter': iter_num,
        'sigma2': 1/alpha_0,
        'sigma2seq': 1/alpha_0_seq[:iter_num],
        'reso_grid': reso_grid.T
    }
    return result

def off_grid_operation(Y, gamma, iter_size, mu, Sigma, Phi, B, beta, r):
    """离网操作函数"""
    reso_num = B.shape[1]
    snapshot_num = Y.shape[1]
    index_b = np.argsort(gamma.flatten())[::-1][:iter_size]
    temp = beta.copy()
    # beta = np.zeros((reso_num, 1))
    # 确保beta是复数类型
    beta = np.zeros((reso_num, 1), dtype=float)
    beta[index_b] = temp[index_b]

    BHB = B.conj().T @ B
    P = np.real(BHB[index_b, :][:, index_b].conj() * (mu[index_b, :] @ mu[index_b, :].conj().T + snapshot_num * Sigma[index_b, :][:, index_b]))

    # 确保v是实数
    v = np.zeros((len(index_b), 1), dtype=float)  # 改为实数类型
    for t in range(snapshot_num):
        term = B[:, index_b].conj().T @ (Y[:, t] - Phi @ mu[:, t])
        v += np.real(mu[index_b, t].conj() * term).reshape(-1, 1)  # 确保是实数
    v = v - snapshot_num * np.real(np.diag(B[:, index_b].conj().T @ Phi @ Sigma[:, index_b])).reshape(-1, 1)
    eigP = np.linalg.svd(P, compute_uv=False)
    eigP = np.sort(eigP)[::-1]

    if eigP[-1] / eigP[0] > 1e-5 or np.any(np.diag(P)) == 0:
        for n in range(iter_size):
            temp_beta = beta[index_b].copy()
            temp_beta[n] = 0
            beta[index_b[n]] = np.real((v[n] - P[n, :] @ temp_beta) / P[n, n])  # 取实部
            if np.abs(beta[index_b[n]]) > r/2:
                beta[index_b[n]] = r/2 * np.sign(beta[index_b[n]])
            if P[n, n] == 0:
                beta[index_b[n]] = 0
    else:
        beta = np.zeros((reso_num, 1), dtype=complex)
        beta[index_b] = np.real(np.linalg.solve(P, v))  # 取实部

    return beta, index_b

# 主程序
if __name__ == "__main__":
    # 参数设置
    array_num = 10  # 阵元数
    snapshot_num = 100  # 快拍数
    source_aoa = np.array([-30, 0, 45])  # 信源到达角
    c = 340  # 波速
    f = 1000  # 频率
    wavelength = c / f  # 波长
    d = 0.5 * wavelength
    source_num = len(source_aoa)  # 信源数
    sig_nr = np.array([20, 20, 20])  # 信噪比
    reso_num = 91  # 网格数

    # 生成信号
    X = np.zeros((source_num, snapshot_num), dtype=complex)
    A = np.exp(-1j * np.arange(array_num).reshape(-1, 1) * 2 * np.pi * (d/wavelength) * np.sin(source_aoa.reshape(1, -1) * np.pi / 180))

    for ik in range(len(sig_nr)):
        X[ik, :] = np.sqrt(10**(sig_nr[ik]/10)) * (np.random.randn(snapshot_num) + 1j * np.random.randn(snapshot_num)) / np.sqrt(2)

    n = (np.random.randn(array_num, snapshot_num) + 1j * np.random.randn(array_num, snapshot_num)) / np.sqrt(2)
    Y = A @ X + n

    # OGSBI算法输入参数
    params = {
        'Y': Y,
        'reso_num': reso_num,
        'maxiter': 2000,
        'tolerance': 1e-4,
        'sigma2': np.mean(np.var(Y, axis=1)) / 100
    }

    # 运行OGSBI算法
    res = OGSBI(params)
    xp_rec = res['reso_grid']
    x_rec = res['mu']
    xpower_rec = np.mean(np.abs(x_rec)**2, axis=1) + np.real(np.diag(res['Sigma'])) * source_num / snapshot_num
    xpower_rec = np.abs(xpower_rec) / np.max(xpower_rec)

    # 排序结果
    sort_idx = np.argsort(xp_rec.flatten())
    xp_rec_sorted = xp_rec.flatten()[sort_idx]
    xpower_rec_sorted = xpower_rec[sort_idx]

    # 绘图
    plt.figure()
    plt.plot(xp_rec_sorted, 10 * np.log10(xpower_rec_sorted))
    plt.xlabel("角度/°")
    plt.ylabel("归一化功率/dB")
    plt.title("DOA估计结果")

    # 标记真实DOA
    for aoa in source_aoa:
        plt.axvline(aoa, color='r', linestyle='--')

    plt.grid(True)
    plt.show()































































































