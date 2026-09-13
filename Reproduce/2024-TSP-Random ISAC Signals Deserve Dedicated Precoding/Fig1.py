#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
复现 Fig.1：样本协方差并不总是严格等于统计协方差，
只有当帧长 L 足够大时，(1/L)SS^H 才逐渐接近 I_{N_t}。

模型：
    s(ell) ~ CN(0, I_{N_t}), ell = 1, 2, ..., L
    S = [s(1), ..., s(L)] in C^{N_t x L}
    R_hat = (1/L) S S^H

误差度量：
    e = ||R_hat - I_{N_t}||_F^2 / ||I_{N_t}||_F^2

绘图：
    纵轴 = 10 log10(e)
    横轴 = Frame Length = 2^2, 2^4, ..., 2^16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ===================== 全局画图设置（参考你的风格） =====================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 22

np.random.seed(42)


def generate_signal_matrix(Nt, L):
    r"""
    生成随机信号矩阵 S \in C^{Nt x L}，其列向量独立同分布，且
        s(ell) ~ CN(0, I_Nt).
    """
    S = (np.random.randn(Nt, L) + 1j * np.random.randn(Nt, L)) / np.sqrt(2)
    return S


def approximation_error_single_trial(Nt, L):
    """
    单次仿真下的归一化近似误差：
        e = ||(1/L)SS^H - I||_F^2 / ||I||_F^2.
    """
    I_Nt = np.eye(Nt, dtype=complex)
    S = generate_signal_matrix(Nt, L)
    R_hat = (S @ S.conj().T) / L

    err = np.linalg.norm(R_hat - I_Nt, ord='fro')**2
    err = err / (np.linalg.norm(I_Nt, ord='fro')**2)
    return np.real(err)


def approximation_error_mc(Nt, L, num_mc=1):
    """
    Monte Carlo 平均误差。
    - num_mc = 1：更接近论文图中的“单次模拟”风格，曲线会有轻微随机起伏；
    - num_mc > 1：曲线更平滑。
    """
    err_list = np.zeros(num_mc)
    for mc in range(num_mc):
        err_list[mc] = approximation_error_single_trial(Nt, L)
    return np.mean(err_list)


def theoretical_mean_error(Nt, L):
    """
    理论平均误差：
        E{ ||(1/L)SS^H - I||_F^2 / ||I||_F^2 } = Nt / L.

    这可作为仿真结果的参考验证。
    """
    return Nt / L



# ===================== 参数设置 =====================
Nt_list = [128, 64, 32, 16]
log2L_list = np.arange(2, 17, 2)   # 2,4,...,16
L_list = 2 ** log2L_list

# 你如果想更平滑，可以改成 50、100
num_mc = 1

# ===================== 仿真 =====================
sim_result = {}
th_result = {}

for Nt in Nt_list:
    sim_result[Nt] = np.zeros(len(L_list))
    th_result[Nt] = np.zeros(len(L_list))

    for idx, L in enumerate(L_list):
        sim_result[Nt][idx] = approximation_error_mc(Nt, L, num_mc=num_mc)
        th_result[Nt][idx] = theoretical_mean_error(Nt, L)

print('Simulation result in dB:')
for Nt in Nt_list:
    print(f'Nt = {Nt}:', np.round(10 * np.log10(sim_result[Nt]), 2))

# ===================== 绘图 =====================
colors = ['#F65314', '#00A1F1', '#77AC30', '#8A2BE2', '#00A8BB', 'k']
markers = ['s','v','d', 'o', '*', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D']

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

for iNt, Nt in enumerate(Nt_list):
    axs.plot(log2L_list, 10*np.log10(sim_result[Nt]), color=colors[iNt], linestyle='-', linewidth=2, marker=markers[iNt], ms=12, markerfacecolor='white', label=rf'$N={Nt}$', zorder=10-iNt)

# 如果需要同时画理论平均误差 Nt/L，可取消下面注释
# for iNt, Nt in enumerate(Nt_list):
#     axs.plot(log2L_list, 10*np.log10(th_result[Nt]), color=colors[iNt], linestyle='--', linewidth=1.5, label=rf'Theory, $N_t={Nt}$')

font1 = FontProperties(family='Times New Roman', style='normal', size=22)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black',  labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)

axs.set_xlabel(r'Frame Length', fontsize = 24)
axs.set_ylabel(r'Approximation Error: $\left\|\mathbf{S}\mathbf{S}^{H}/L-\mathbf{I}\right\|_F^2$ [dB]')
axs.set_xlim([2, 16])
axs.set_ylim([-40, 20])
axs.set_xticks(log2L_list)
axs.set_xticklabels([rf'$2^{{{k}}}$' for k in log2L_list])
axs.set_yticks(np.arange(-40, 21, 10))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)

labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]

axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)

plt.savefig('Fig1_sample_covariance.pdf')
plt.show()
plt.close()
