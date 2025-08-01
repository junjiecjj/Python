#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:55:49 2025

@author: jack

https://github.com/chenhui07c8/Radio_Localization?tab=readme-ov-file
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt


# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%%%%%%% Uniform Linear Array %%%%%%%%
# create a manifold vector
def steering_vector(k, N):
    n = np.arange(N)
    return np.exp(-1j * np.pi * np.sin(k) * n)

def MUSIC(Rxx, K, N):
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
    Un = eigvector[:, K:N]

    # Un = eigvector
    UnUnH = Un @ Un.T.conjugate()
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pmusic = np.zeros(angle.size)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * np.arange(N) * np.sin(ang)).reshape(-1, 1)
        Pmusic[i] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]

    Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
    Pmusic = 10 * np.log10(Pmusic)
    peaks, _ =  scipy.signal.find_peaks(Pmusic, height = -10, distance = 10)

    angle_est = Thetalst[peaks]

    return Thetalst, Pmusic, angle_est, peaks

def MUSIC1(Rxx, K, N):
    # Eigenvalue Decomposition
    eigvals, eigvecs = np.linalg.eigh(Rxx)
    U_n = eigvecs[:, :-K]  # noise sub-space
    UnUnH = U_n @ U_n.conj().T
    # MUSIC pseudo-spectrum
    Thetalst = np.arange(-90, 90.1, 0.5)
    k_scan = np.deg2rad(Thetalst)
    P_music = np.zeros_like(k_scan, dtype = float)

    for i, k in enumerate(k_scan):
        a_k = steering_vector(k, N)
        P_music[i] = 1 / np.abs(a_k.conj().T @ UnUnH @ a_k)

    # normalize
    P_music = np.abs(P_music) / np.abs(P_music).max()
    P_music = 10 * np.log10(P_music)
    peaks, _ =  scipy.signal.find_peaks(P_music, height=-10, distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, P_music, angle_est, peaks

def ESPRIT(Rxx, K, N):
    # 特征值分解
    D, U = np.linalg.eigh(Rxx)             # 特征值分解
    idx = np.argsort(D)                    # 将特征值排序 从小到大
    U = U[:, idx]
    U = U[:,::-1]                          # 对应特征矢量排序
    Us = U[:, 0:K]

    ## 角度估计
    Ux = Us[0:K, :]
    Uy = Us[1:K+1, :]

    # ## 方法一:最小二乘法
    # Psi = np.linalg.inv(Ux)@Uy
    # Psi = np.linalg.solve(Ux,Uy)    # or Ux\Uy

    ## 方法二：完全最小二乘法
    Uxy = np.hstack([Ux,Uy])
    Uxy = Uxy.T.conjugate() @ Uxy
    eigenvalues, eigvector = np.linalg.eigh(Uxy)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    eigvector = eigvector[:,::-1]                          # 对应特征矢量排序
    F0 = eigvector[0:K, K:2*K]
    F1 = eigvector[K:2*K, K:2*K]
    Psi = -F0 @ np.linalg.inv(F1)

    # 特征值分解
    D, U = np.linalg.eig(Psi)          # 特征值分解
    Theta = np.rad2deg(np.arcsin(-np.angle(D)/np.pi ))

    Theta = np.sort(Theta)
    return Theta

# https://github.com/highskyno1/MIMO_DOA
def DOA_ESPRIT(X, K, N, lamda = 2, d = 1):
    # DOA_ESPRIT 基于旋转不变子空间法实现DOA
    #   x_sig       每个阵元接收到的信号矩阵，阵元数*快拍数
    #   target_len  目标数量
    #   lamda       载波波长
    #   d           阵元间隔
    #   DOA_esp_ml  基于最大似然估计准则得到的估计结果
    #   DOA_esp_tls 基于最小二乘准则得到的估计结果
    N = X.shape[0]
    # 回波子阵列合并
    x_esp = np.vstack((X[:N-1,:], X[1:N, :]))
    #  计算协方差
    R_esp = np.cov(x_esp.conj())
    # 特征分解
    D, W = np.linalg.eig(R_esp.T)
    D1, W1 = np.linalg.eig(R_esp)
    # 获取信号子空间
    # W = np.fliplr(W)
    U_s = W[:,:K]
    # 拆分
    U_s1 = U_s[:N-1,:]
    U_s2 = U_s[N-1:,:]

    ## LS-ESPRIT法
    mat_esp_ml = scipy.linalg.pinv(U_s1) @ U_s2;
    # 获取对角线元素并解算来向角
    DOA_esp_ml = -np.angle(np.linalg.eig(mat_esp_ml)[0])
    DOA_esp_ml = np.arcsin(DOA_esp_ml * lamda / 2 / np.pi / d)
    DOA_esp_ml = np.rad2deg(DOA_esp_ml)

    ## TLS-ESPRIT
    Us12 = np.hstack((U_s1, U_s2))
    U, s, VH = np.linalg.svd(Us12)
    V = VH.conj().T
    ## 提取E12和E22
    E12 = V[:K, K:]
    E22 = V[K:,K:]
    mat_esp_tls = - E12 @ scipy.linalg.inv(E22)
    # 获取对角线元素并解算来向角
    DOA_esp_tls = -np.angle(np.linalg.eig(mat_esp_tls)[0])
    DOA_esp_tls = np.arcsin(DOA_esp_tls * lamda / 2 / np.pi / d)
    DOA_esp_tls = np.rad2deg(DOA_esp_tls);

    return DOA_esp_ml, DOA_esp_tls


def CBF(Rxx, K, N):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pcbf = np.zeros(angle.size)
    d = np.arange(0, N).reshape(-1, 1)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * d * np.sin(ang))
        Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]

    Pcbf = np.abs(Pcbf) / np.abs(Pcbf).max()
    Pcbf = 10 * np.log10(Pcbf)
    peaks, _ =  scipy.signal.find_peaks(Pcbf, height=-2,  distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, Pcbf, angle_est, peaks

def Capon(Rxx, K, N):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    Pcapon = np.zeros(angle.size)
    # for i, ang in enumerate(angle):
    #     a = np.exp(-1j * np.pi * d * np.sin(ang))
    #     Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]
    d = np.arange(0, N).reshape(-1, 1)
    for i, ang in enumerate(angle):
        a = np.exp(-1j * np.pi * d * np.sin(ang))
        Pcapon[i] = 1/np.real(a.T.conjugate() @ scipy.linalg.inv(Rxx) @ a)[0,0]

    Pcapon = np.abs(Pcapon) / np.abs(Pcapon).max()
    Pcapon = 10 * np.log10(Pcapon)
    peaks, _ =  scipy.signal.find_peaks(Pcapon, height=-2,  distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, Pcapon, angle_est, peaks

# https://blog.csdn.net/weixin_44705592/article/details/131500890
# https://zhuanlan.zhihu.com/p/22897428966
# https://blog.csdn.net/qq_42233059/article/details/126524639
# 对求根MUSIC算法， 作如下说明。
# （1）求根MUSIC算法与谱搜索方式的MUSIC算法原理是一样的，只不过是用一个关于z的矢量来代替导向矢量，从而用求根过程代替搜索过程；
# （2）由于噪声的存在，求出的根不可能在单位圆上，可选择接近单位圆上的根为真实信号的根，这就存在一定的误差；
# （3）求根MUSIC算法与谱搜索的MUSIC算法相似，同样存在两种表达方式，一个是利用噪声子空间，另一个是利用信号子空间。
def ROOT_MUSIC(Rxx, K, d = 0.5, wavelength = 1.0):
    """
    Root-MUSIC 算法进行 DOA 估计（适用于 ULA）。

    参数:
        R: 接收信号的样本协方差矩阵 (num_sensors x num_sensors)
        num_sources: 信号数（需要估计的 DOA 数量）
        d: 传感器间距（以波长为单位，默认 0.5）
        wavelength: 信号波长（默认 1.0）

    返回:
        doa_estimates_deg: 估计的 DOA（单位：度，按从小到大排序）
    """
    N = Rxx.shape[0]
    eigvals, eigvecs = np.linalg.eigh(Rxx)  # # 对协方差矩阵进行特征值分解
    En = eigvecs[:, :N - K]      # 选取噪声子空间：使用最小的 (num_sensors - num_sources) 个特征向量
    Pn = En @ En.conj().T        # 构造噪声子空间投影矩阵

    # 利用 Toeplitz 结构提取多项式系数: 对于 ULA, Pn 的每条对角线理论上应相等, 这里对每条对角线求和, 得到系数 c[k] (k 从 -M+1 到 M-1)
    c = np.array([np.sum(np.diag(Pn, k)) for k in range(-N+1, N)])
    c = c / c[N - 1] # # 归一化：令 k=0（主对角线）的系数为 1，这不会改变根的位置

    poly_coeffs = c[::-1] # 构造多项式系数，注意 np.roots 要求系数按降幂排列
    roots_all = np.roots(poly_coeffs) # 求解多项式的所有根
    roots_inside = roots_all[np.abs(roots_all) < 1] # 只考虑位于单位圆内部的根（理论上信号相关根应落在单位圆附近）

    # 根据距离单位圆的距离排序，选择最接近单位圆的 num_sources 个根
    distances = np.abs(np.abs(roots_inside) - 1)
    sorted_indices = np.argsort(distances)
    selected_roots = roots_inside[sorted_indices][:K]

    # 由理论，根的相位与 DOA 满足: angle(z) = -2π*d*sin(θ)/wavelength
    # beta = 2π*d/wavelength
    beta = 2 * np.pi * d / wavelength
    phi = np.angle(selected_roots)

    doa_estimates_rad = np.arcsin(-phi / beta)
    doa_estimates_deg = np.rad2deg(doa_estimates_rad)

    return np.sort(doa_estimates_deg), roots_all

def DOA_ML(Rxx):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    P_ml = np.zeros(angle.size, dtype = complex)
    for i, ang in enumerate(angle):
        scan = np.exp(-1j * np.pi * np.arange(N) * np.sin(ang)).reshape(-1, 1)
        Pa = scan/(scan.conjugate().T @ scan) * scan.conjugate().T
        P_ml[i] = np.trace(Pa @ Rxx) / N

    P_ml = np.abs(P_ml) / np.abs(P_ml).max()
    P_ml = 10 * np.log10(P_ml)

    peaks, _ =  scipy.signal.find_peaks(P_ml, height = -10, distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, P_ml, angle_est, peaks

def DOA_FOCUSS(Rxx, spar = 0, reg = 1e-4, err = 1e-4, maxIter = 1000):
    #   DOA_FOCUSS 基于欠定系统局灶解法(Focal Under determinedSystem Solver)
    #   实现稀疏恢复获得DOA估计结果
    #   scan_a      DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    #   u           对回波自相关矩阵做酉对角化后，最大特征值对应的酉向量
    #   lamda_spe   稀疏因子，效果类似于结果的范数约束
    #   lamda_reg   正则化因子，过大会趋于0解，过小结果发散
    #   lamda_err   迭代结束误差
    #   P_focuss    通过FOCUSS法得到的归一化来波方向功率估计
    N = Rxx.shape[0]   # 计算阵元数
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    # DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    Dg = np.exp(-1j * np.pi * np.arange(N)[:,None] * np.sin(angle))

    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    u = eigvector[:,-1]     # 对回波自相关矩阵做酉对角化后，最大特征值对应的酉向量

    s0 = Dg.conjugate().T @ scipy.linalg.inv(Dg @ Dg.conjugate().T) @ u

    for _ in range(maxIter):
        W = np.diag(s0 ** (1-spar/2))
        s = W @ W.conjugate().T @ Dg.conjugate().T @ scipy.linalg.inv(Dg@(W @ W.conjugate().T)@Dg.conjugate().T + reg * np.eye(N) ) @ u
        if np.linalg.norm(s - s0)/np.linalg.norm(s0) < err:
            break
        s0 = s
    P_focuss = np.abs(s)
    P_focuss = np.abs(P_focuss) / np.abs(P_focuss).max()
    P_focuss[np.where(P_focuss < 1e-4)] = 1e-4
    P_focuss = 10 * np.log10(P_focuss)

    peaks, _ =  scipy.signal.find_peaks(P_focuss, height = -10, distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, P_focuss, angle_est, peaks

def DOA_PINV(Rxx, ):
    # DOA_PINV 基于伪逆法实现稀疏恢复，效果等同于最大似然估计法
    #   u       对回波自相关矩阵做酉对角化后，最大特征值对应的酉向量
    #   scan_a  DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    #   s_pinv  基于PINV法得到的不同来波方向的归一化功率
    N = Rxx.shape[0]   # 计算阵元数
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    # DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    scan_a = np.exp(-1j * np.pi * np.arange(N)[:,None] * np.sin(angle))

    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    u = eigvector[:,-1]     # 对回波自相关矩阵做酉对角化后，最大特征值对应的酉向量

    s_pinv = u.conjugate().T @ scipy.linalg.pinv(scan_a).conjugate().T
    s_pinv = np.abs(s_pinv)

    # P_focuss = np.abs(s)
    s_pinv = np.abs(s_pinv) / np.abs(s_pinv).max()
    # P_focuss[np.where(P_focuss < 1e-4)] = 1e-4
    s_pinv = 10 * np.log10(s_pinv)

    peaks, _ =  scipy.signal.find_peaks(s_pinv, height = -10, distance = 10)
    angle_est = Thetalst[peaks]

    return Thetalst, s_pinv, angle_est, peaks

# https://github.com/highskyno1/MIMO_DOA
def DOA_EM_SBL(noisevar, Rxx, Ns, err = 1e-3, timelim = 30):
    # 基于期望最大化-稀疏贝叶斯学习方法实现DOA估计
    #   noisevar    估计的噪声方差
    #   scan_a      DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    #   Rxx         回波自相关矩阵
    #   Ns          快拍数
    #   err         迭代误差限，迭代退出的条件之一
    #   timelim     迭代次数限制，迭代退出的条件之二
    N = Rxx.shape[0]   # 计算阵元数
    Thetalst = np.arange(-90, 90.1, 0.5)

    angle = np.deg2rad(Thetalst)
    # DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    scan_a = np.exp(-1j * np.pi * np.arange(N)[:,None] * np.sin(angle))
    scan_len = scan_a.shape[-1]
    times_cnt = 0
    Gamma = np.eye(scan_len, dtype = float) * 0.1
    while 1:
        times_cnt += 1
        # E-step
        Sigma_x = scipy.linalg.pinv(noisevar * (scan_a.conjugate().T @ scan_a) + scipy.linalg.pinv(Gamma))
        Mu_x = Sigma_x / noisevar @ scan_a.conjugate().T @ Rxx
        # M-step
        Gamma_new = Gamma
        for i in range(scan_len):
            mu_xn = Mu_x[i, :]
            Gamma_new[i, i] = mu_xn @ mu_xn.conjugate().T / Ns + Sigma_x[i, i]
        if np.sum(np.abs(np.diag(Gamma_new - Gamma))) < err or times_cnt > timelim:
            break
        Gamma = Gamma_new
    Gamma_new = np.abs(np.diag(Gamma_new))
    s_sbl = Gamma_new / np.max(Gamma_new)
    s_sbl = 10 * np.log10(s_sbl)
    peaks, _ =  scipy.signal.find_peaks(s_sbl, height = -10, distance = 10)
    angle_est = Thetalst[peaks]
    return Thetalst, s_sbl, angle_est, peaks

# 仅使用协方差矩阵 Rxx 的稀疏贝叶斯学习（SBL）DOA估计的完整Python实现, 版本一
def SBL_DOA_Rxx(Rxx, max_iter=100, tol=1e-4):
    from scipy.linalg import inv
    from scipy.signal import find_peaks
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

    wavelength = 1.0
    array_pos = np.arange(N) * 0.5 * wavelength  # ULA
    # theta_true = np.array([-20, 0, 45])  # 真实DOA
    theta_grid = np.arange(-90, 90.1, 0.5)
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

    # Pmusic = np.abs(Pmusic) / np.abs(Pmusic).max()
    # Pmusic = 10 * np.log10(Pmusic)
    # peaks, _ =  scipy.signal.find_peaks(Pmusic, height = -10, distance = 10)

    angle_est = Thetalst[peaks]

    return theta_grid, power_db, angle_est, peaks
    # return theta_grid[peaks], power_db

import cvxpy as cpy
def DOA_CVX(Rxx, p_norm, tor_lim = 1e-1):
    Thetalst = np.arange(-90, 90.1, 0.5)
    angle = np.deg2rad(Thetalst)
    # DOA估计的栅格，在稀疏恢复理论中也称为"超完备字典"
    scan_a = np.exp(-1j * np.pi * np.arange(N)[:,None] * np.sin(angle))
    scan_len = scan_a.shape[-1]
    # 特征值分解
    eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
    idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
    eigvector = eigvector[:, idx]
    u = eigvector[:,-1][:,None]     # 对回波自相关矩阵做酉对角化后，最大特征值对应的酉向量

    s_cvx = cpy.Variable((scan_len, 1))
    constraints = [cpy.norm(u - scan_a @ s_cvx, 2) <= tor_lim]
    prob = cpy.Problem(cpy.Minimize(cpy.sum(cpy.pnorm(s_cvx, p_norm))), constraints)
    prob.solve()
    s_cvx = np.abs(s_cvx.value.flatten())
    s_cvx = s_cvx / np.max(s_cvx)
    s_cvx = 10 * np.log10(s_cvx)
    peaks, _ =  scipy.signal.find_peaks(s_cvx, height = -10, distance = 20)
    angle_est = Thetalst[peaks]
    return Thetalst, s_cvx, angle_est, peaks


derad = np.pi/180             # 角度->弧度
N = 8                         # 阵元个数
K = 3                         # 信源数目
doa_deg = [0, 30, 60]    # 待估计角度
doa_rad = np.deg2rad(doa_deg) # beam angles
f0 = 1e6
f = np.array([0.1, 0.2, 0.3,]) * f0  # 为了保持各个用户的信号正交，需要满足各个用户的频率不等或者是各个用户的信号为随机噪声
snr = 20                                  # 信噪比
Ns = 1000                                 # 快拍数
fs = 1e8                                  # 满足采样定理，fs >> f0
Ts = 1/fs
t = np.arange(Ns) * Ts
SNR = 20                                  # 信噪比(dB)

# generate signals
X = np.zeros((N, Ns), dtype = complex)
for i in range(K):
    a_k = steering_vector(doa_rad[i], N)
    # s = np.exp(1j * 2 * np.pi * np.random.rand(Ns))  # 信源信号，入射信号，不相干即可，也可以用正弦替代
    s = np.exp(1j * 2 * np.pi * f[i] * t)  # 正弦 signals
    X += np.outer(a_k, s)

# add noise
Xpow = np.mean(np.abs(X)**2)
noisevar = Xpow * 10 ** (-SNR / 10)
noise = np.sqrt(noisevar/2) * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))
X += noise
Rxx = X @ X.T.conjugate() / Ns

Thetalst, Pmusic, angle_music, peak_music = MUSIC(Rxx, K, N)
Thetalst, Pcbf, angle_cbf, perak_cbf = CBF(Rxx, K, N)
Thetalst, Pcapon, angle_capon, peak_capon = Capon(Rxx, K, N)
Theta_esprit = ESPRIT(Rxx, K, N)
Theta_esprit_ml, Theta_esprit_tsl = DOA_ESPRIT(X, K, N)
Theta_root, roots_all = ROOT_MUSIC(Rxx, K )
Thetalst, P_ml, angle_ml, peak_ml = DOA_ML(Rxx)
Thetalst, P_focuss, angle_focuss, peak_focuss = DOA_FOCUSS(Rxx)
Thetalst, P_pinv, angle_pinv, peak_pinv = DOA_PINV(Rxx)
Thetalst, P_sbl, angle_sbl, peak_sbl = DOA_EM_SBL(noisevar, Rxx, Ns)
Thetalst1, P_sbl1, angle_sbl1, peak_sbl1 = SBL_DOA_Rxx(Rxx)

print(f"True = {doa_deg}")
print(f"MUSIC = {angle_music}")
print(f"Root MUSIC = {Theta_root}")
print(f"CBF = {angle_cbf}")
print(f"Capon = {angle_capon}")
print(f"ESPRIT = {Theta_esprit}")

###>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.plot(Thetalst, Pmusic , color = colors[0], linestyle='-', lw = 2, label = "MUSIC", )
axs.plot(angle_music, Pmusic[peak_music], linestyle='', marker = 'o', color=colors[0], markersize = 12)

axs.plot(Thetalst, Pcbf , color = colors[1], linestyle='--', lw = 2, label = "CBF", )
axs.plot(angle_cbf, Pcbf[perak_cbf], linestyle='', marker = 'd', color=colors[1], markersize = 12)

axs.plot(Thetalst, Pcapon , color = colors[2], linestyle='-.', lw = 2, label = "CAPON", )
axs.plot(angle_capon, Pcapon[peak_capon], linestyle='', marker = 's', color=colors[2], markersize = 12)

# axs.plot(Thetalst, P_ml , color = colors[3], linestyle='-.', lw = 2, label = "ML", )
# axs.plot(angle_ml, P_ml[peak_ml], linestyle='', marker = 's', color=colors[3], markersize = 12)

# axs.plot(Thetalst, P_focuss , color = colors[4], linestyle='-.', lw = 2, label = "Focuss", )
# axs.plot(angle_focuss, P_focuss[peak_focuss], linestyle='', marker = 's', color=colors[4], markersize = 12)


# axs.plot(Thetalst, P_pinv , color = colors[5], linestyle='-.', lw = 2, label = "Pinv", )
# axs.plot(angle_pinv, P_pinv[peak_pinv], linestyle='', marker = 's', color=colors[5], markersize = 12)

axs.plot(Theta_esprit, np.zeros(K), linestyle='', marker = '*', color=colors[3], markersize = 12, label = "ESPRIT", )
axs.plot(Theta_root, np.zeros(K)-5, linestyle='', marker = 'v', color='r', markersize = 12, label = "ROOT MUSIC", )

axs.set_xlabel( "DOA/(degree)",)
axs.set_ylabel('Normalized Spectrum/(dB)',)
axs.legend()

plt.show()
plt.close('all')


###>>>>>>>>>> ESPIRT
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.plot(Theta_esprit_tsl, np.zeros(K)-2, linestyle='', marker = 'd', color=colors[1], markersize = 12, label = "ESPRIT TSL",)

axs.plot(Theta_esprit_ml, np.zeros(K)-1, linestyle='', marker = 's', color=colors[2], markersize = 12, label = "ESPRIT ML",)

axs.plot(Theta_esprit, np.zeros(K), linestyle='', marker = '*', color=colors[3], markersize = 12, label = "ESPRIT", )

axs.set_xlabel( "DOA/(degree)",)
axs.set_ylabel('Normalized Spectrum/(dB)',)
axs.legend()

plt.show()
plt.close('all')

##>>>>>>>>>>>>>

##>>>>>>>>>>>>> ROOT
fig, axs = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)

theta = np.linspace(0, 2*np.pi, 400)
axs.plot(np.cos(theta), np.sin(theta), 'k--', lw = 1, label='unit circle')
axs.scatter(np.real(roots_all), np.imag(roots_all), marker='o', color='b', label='roots of polynomial')
axs.set_xlabel('Real', fontsize = 12, )
axs.set_ylabel('Imaginary', fontsize = 12,)
axs.set_title('Root-MUSIC Root Distribution', fontsize = 12,)
axs.axis('equal')
axs.legend( fontsize = 12,)
axs.grid(True)
plt.show()
plt.close('all')


###>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.plot(Thetalst, P_ml , color = colors[0], linestyle='-.', lw = 2, label = "ML", )
axs.plot(angle_ml, P_ml[peak_ml], linestyle='', marker = 's', color=colors[0], markersize = 12)

axs.plot(Thetalst, P_focuss , color = colors[1], linestyle='-.', lw = 2, label = "Focuss", )
axs.plot(angle_focuss, P_focuss[peak_focuss], linestyle='', marker = 's', color=colors[1], markersize = 12)

axs.plot(Thetalst, P_pinv , color = colors[2], linestyle='-.', lw = 2, label = "Pinv", )
axs.plot(angle_pinv, P_pinv[peak_pinv], linestyle='', marker = 's', color=colors[2], markersize = 12)

axs.plot(Thetalst, P_sbl , color = colors[3], linestyle='-.', lw = 2, label = "SBL", )
axs.plot(angle_sbl, P_sbl[peak_sbl], linestyle='', marker = 's', color=colors[3], markersize = 12)

axs.plot(Thetalst1, P_sbl1 , color = colors[4], linestyle='-.', lw = 2, label = "SBL1", )
axs.plot(angle_sbl1, P_sbl1[peak_sbl1], linestyle='', marker = 's', color=colors[4], markersize = 12)

axs.set_xlabel( "DOA/(degree)",)
axs.set_ylabel('Normalized Spectrum/(dB)',)
axs.legend()

plt.show()
plt.close('all')

#%% CVX
Thetalst, P_cvx1, angle_cvx1, peak_cvx1 = DOA_CVX(Rxx, 1, tor_lim = 1e-1)
Thetalst, P_cvx15, angle_cvx15, peak_cvx15 = DOA_CVX(Rxx, 1.5, tor_lim = 1e-1)
Thetalst, P_cvx2, angle_cvx2, peak_cvx2 = DOA_CVX(Rxx, 2, tor_lim = 1e-1)

###>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.plot(Thetalst, P_cvx1 , color = colors[0], linestyle='-.', lw = 2, label = "Cvx1", )
axs.plot(angle_cvx1, P_cvx1[peak_cvx1], linestyle='', marker = 's', color=colors[0], markersize = 12)

axs.plot(Thetalst, P_cvx15 , color = colors[1], linestyle='-.', lw = 2, label = "Cvx1.5", )
axs.plot(angle_cvx15, P_cvx15[peak_cvx15], linestyle='', marker = 's', color=colors[1], markersize = 12)

axs.plot(Thetalst, P_cvx2 , color = colors[2], linestyle='-.', lw = 2, label = "Cvx2", )
axs.plot(angle_cvx2, P_cvx2[peak_cvx2], linestyle='', marker = 's', color=colors[2], markersize = 12)


axs.set_xlabel( "DOA/(degree)",)
axs.set_ylabel('Normalized Spectrum/(dB)',)
axs.legend()

plt.show()
plt.close('all')










