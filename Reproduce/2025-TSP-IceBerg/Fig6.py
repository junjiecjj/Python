#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:24:53 2025

@author: jack
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy
import commpy
from Modulations import modulator

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
# 产生傅里叶矩阵
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     ll = np.arange(L)
     for i in range(L):
         mat[i,:] = 1.0*np.exp(-1j*2.0*np.pi*i*ll/L) / (np.sqrt(L)*1.0)
     return mat
def srrcFunction(beta, L, span, Tsym = 1):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    t = np.arange(-span*Tsym/2, span*Tsym/2 + 0.5/L, Tsym/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # power normaize.
    return p, t, filtDelay

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def solve_iceberg_shaping_psl(N, L, alpha, K_s1, ):
    """
    求解冰山谱形优化问题（PSL目标函数）
    参数:
        N: 滤波器长度
        alpha: 滚降因子
        K_s1: 延迟区域索引集合
        f_hat: 频域向量序列 f_hat_{k+1} for k in K_s1
    返回:
        g_opt: 最优的时域滤波器系数
    """

    # 计算相关参数
    N_alpha       = int(alpha * N)  # 滚降部分长度
    N_non_rolloff = N - N_alpha     # 非滚降部分长度
    N_zeros = N_non_rolloff // 2    # 前导零的数量
    N_ones  = N_non_rolloff // 2    # 尾部一的数量

    # 定义优化变量
    g = cp.Variable(N, nonneg = True)  # g_n ≥ 0

    # 构建约束条件
    constraints = []

    # 约束(45): 前N_zeros个元素为1
    constraints.append(g[0:N_zeros] == 1)

    # 约束(46): 后N_ones个元素为0
    constraints.append(g[N - N_ones:N] == 0)

    # 约束(47): 单调递增约束 g_{n+1} - g_n ≥ 0
    for n in range(N - 1):
        constraints.append(g[n + 1] - g[n] <= 0)

    # 约束(48): 总能量约束
    constraints.append(cp.sum(g) == N / 2)

    # 构建PSL目标函数(44)
    psl_terms = []
    for k in K_s1:
        # 假设 f_hat[k] 包含 f_hat_{k+1} 向量
        f_k = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))
        gk = g + (1 - g) * np.exp(-1j * 2 * pi * k / L)
        psl_terms.append(cp.abs(f_k.conj().T @ gk)**2)

    # PSL是这些项中的最大值
    objective = cp.Minimize(cp.max(cp.hstack(psl_terms)))

    # 定义优化问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    # problem.solve(solver=cp.ECOS, verbose=True)
    problem.solve(verbose=False)

    if problem.status == cp.OPTIMAL:
        print("优化成功!")
        print(f"最优PSL值: {problem.value}")
        return g.value
    else:
        print(f"优化失败，状态: {problem.status}")
        return None

# 使用示例
# if __name__ == "__main__":
Tsym = 1
pi = np.pi
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.35  # 滚降因子
# span = 6      # 滤波器跨度（根据旁瓣要求调整）

# p, t, filtDelay = srrcFunction(alpha, L, span, Tsym = Tsym)
# p = np.pad(p, (0, L*N - p.size))

t, p = commpy.filters.rrcosfilter(L*N , alpha, Tsym, L/Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))

norm2p = np.linalg.norm(p)
FLN = FFTmatrix(L*N )
FN = FFTmatrix(N)

###>>>>>>>>>>>>>>>>>> The squared spectra of the designed pulse
K_s1 = np.arange(5, 16) # 延迟区域索引

# 求解优化问题
gN = solve_iceberg_shaping_psl(N, L, alpha, K_s1, )

if gN is not None:
    print("最优滤波器系数:")
    print(gN)

g_N = 1 - gN
g_design = np.hstack((gN, np.zeros((L-2)*N), g_N))
g_Design = np.fft.fftshift(g_design)


###>>>>>>>>>>>>>>>>>> The squared spectra of the RRC
M = 100
kappa = 1.32

g = (N * (FLN@p) * (FLN.conj() @ p.conj()))
g_rrc = np.fft.fftshift(g)

##>>>>>>>>>>>>>>>>>>>>  Plot Fig.6d
colors = plt.cm.jet(np.linspace(0, 1, 5))
# x = np.arange(-N//2, N//2, 1/((L)))
x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
axs.plot(x, np.abs(g_Design), color='r', linestyle='--', label='Spectrum of the Designed Pulse',)
axs.plot(x, np.abs(g_rrc), color='b', linestyle='-', label='Spectrum of the RRC',)

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', fontsize = 18)
axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
axs.set_xlim([-200, 200])

out_fig = plt.gcf()
# out_fig.savefig('Fig6_d.png', )
# out_fig.savefig('Fig6_d.pdf', )
plt.show()
plt.close()

























#%%








#%%










































































































































































































































































































































































































































