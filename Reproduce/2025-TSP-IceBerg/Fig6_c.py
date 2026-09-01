#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 6(c): Direct comparison between the designed pulse and the RRC pulse."""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import commpy
from matplotlib.font_manager import FontProperties
from Modulations import modulator

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22                     # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22                # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22                # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18               # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18               # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False         # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]            # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300                 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2                # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6               # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22
np.random.seed(42)

#%%
def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j*2.0*np.pi*i*ll/L)/np.sqrt(L)
    return mat

def solve_iceberg_shaping_psl(N, L, alpha, K_s1):
    N_alpha = int(alpha*N)
    N_non_rolloff = N-N_alpha
    N_zeros = int(np.floor(N_non_rolloff/2))
    N_ones = int(np.floor(N_non_rolloff/2))

    g = cp.Variable(N, nonneg=True)
    psl_terms = []

    for k in K_s1:
        f_k = np.exp(-1j*2*np.pi*k*np.arange(N)/(L*N))
        gk = g+(1-g)*np.exp(-1j*2*np.pi*k/L)
        psl_terms.append(cp.square(cp.abs(f_k.conj() @ gk)))

    constraints = []
    constraints.append(g[0:N_zeros] == 1)
    constraints.append(g[N-N_ones:N] == 0)

    for n in range(N-1):
        constraints.append(g[n+1]-g[n] <= 0)

    constraints.append(cp.sum(g) == N/2)
    problem = cp.Problem(cp.Minimize(cp.max(cp.hstack(psl_terms))), constraints)
    problem.solve(verbose=False)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print('Optimization successful!')
        print(f'Optimal PSL value: {problem.value:.12e}')
        g_opt = np.asarray(g.value).reshape(-1)
    else:
        raise RuntimeError(f'CVXPY failed to solve the PSL problem. Status: {problem.status}')

    return g_opt

#%% Parameters
Tsym = 1
pi_value = np.pi
N = 128
L = 10
alpha = 0.35
kappa = 1.32
M = 10000
runNumericalSimulation = False

FLN = FFTmatrix(L*N)
FN = FFTmatrix(N)

#%% Generate the designed pulse according to (44)-(49)
K_s1 = np.arange(5*L, 15*L+1)
gN = solve_iceberg_shaping_psl(N, L, alpha, K_s1)
g_N = 1-gN
g_design = np.concatenate((gN, np.zeros((L-2)*N), g_N))
P_spectrum = np.sqrt(np.maximum(np.real(g_design), 0)/N)
p_Designed = FLN.conj().T @ P_spectrum
p_Designed = p_Designed/np.sqrt(np.sum(np.abs(p_Designed)**2))
g_Designed = N*(FLN @ p_Designed)*(FLN.conj() @ p_Designed.conj())
g_Designed = np.real(g_Designed)

pulseSpectrumError = np.linalg.norm(g_Designed-g_design)/max(np.linalg.norm(g_design), np.finfo(float).eps)
print(f'Relative squared-spectrum reconstruction error: {pulseSpectrumError:.3e}')

#%% Generate the RRC pulse
t, p_RRC = commpy.filters.rrcosfilter(L*N, alpha, Tsym, L/Tsym)
filtDelay = (len(p_RRC)-1)/2
p_RRC = p_RRC/np.sqrt(np.sum(np.abs(p_RRC)**2))
g_RRC = N*(FLN @ p_RRC)*(FLN.conj() @ p_RRC.conj())
g_RRC = np.real(g_RRC)

#%% Generate the 16-QAM constellation
MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)

U = FN.conj().T
V = np.eye(N)
tilde_V = V * V.conj()

#%% Theoretical average squared ACFs with 10,000 coherent integrations, Eq. (36)
TheoAveACF_RRC_M10000 = np.zeros(L*N)
TheoAveACF_Designed_M10000 = np.zeros(L*N)

for k in range(L*N):
    gk_RRC = g_RRC[:N]+(1-g_RRC[:N])*np.exp(-1j*2*pi_value*k/L)
    gk_Designed = g_Designed[:N]+(1-g_Designed[:N])*np.exp(-1j*2*pi_value*k/L)
    f_k = np.exp(-1j*2*pi_value*k*np.arange(N)/(L*N))
    r1_RRC = np.abs(gk_RRC @ f_k.conj())**2
    r1_Designed = np.abs(gk_Designed @ f_k.conj())**2
    r2_RRC = (kappa-1)/M*(N-2*(1-np.cos(2*pi_value*k/L))*np.sum(g_RRC[:N]*(1-g_RRC[:N])))
    r2_Designed = (kappa-1)/M*(N-2*(1-np.cos(2*pi_value*k/L))*np.sum(g_Designed[:N]*(1-g_Designed[:N])))
    TheoAveACF_RRC_M10000[k] = r1_RRC+r2_RRC
    TheoAveACF_Designed_M10000[k] = r1_Designed+r2_Designed

# normalization_Theo = TheoAveACF_RRC_M10000[0]
TheoAveACF_RRC_M10000 = TheoAveACF_RRC_M10000/np.abs(TheoAveACF_RRC_M10000).max() + 1e-14
TheoAveACF_Designed_M10000 = TheoAveACF_Designed_M10000/np.abs(TheoAveACF_Designed_M10000).max()+1e-14
TheoAveACF_RRC_M10000 = np.fft.fftshift(TheoAveACF_RRC_M10000)
TheoAveACF_Designed_M10000 = np.fft.fftshift(TheoAveACF_Designed_M10000)

#%% Numerical ACFs with 10,000 coherent integrations
if runNumericalSimulation:
    Iter = 100
    SimAveACF_RRC_M10000 = np.zeros((M, Iter, L*N), dtype=complex)
    SimAveACF_Designed_M10000 = np.zeros((M, Iter, L*N), dtype=complex)

    for k in range(L*N):
        fk = FLN[:,k]
        fk_tilde = fk[:N]
        gk_RRC = g_RRC[:N]+(1-g_RRC[:N])*np.exp(-1j*2*pi_value*k/L)
        gk_Designed = g_Designed[:N]+(1-g_Designed[:N])*np.exp(-1j*2*pi_value*k/L)

        for m in range(M):
            for it in range(Iter):
                d = np.random.randint(Order, size=N)
                s = Constellation[d]
                VHs = np.abs(V.conj().T @ s)**2
                SimAveACF_RRC_M10000[m,it,k] = np.sum(gk_RRC*VHs*fk_tilde.conj())
                SimAveACF_Designed_M10000[m,it,k] = np.sum(gk_Designed*VHs*fk_tilde.conj())

    #%% Coherent averaging
    RkBar_RRC = np.mean(SimAveACF_RRC_M10000, axis=0, keepdims=True)
    RkBar_Designed = np.mean(SimAveACF_Designed_M10000, axis=0, keepdims=True)
    RkBar2_RRC = np.abs(RkBar_RRC)**2
    RkBar2_Designed = np.abs(RkBar_Designed)**2

    Sim_RRC_M10000_avg = np.squeeze(np.mean(RkBar2_RRC, axis=1))
    Sim_Designed_M10000_avg = np.squeeze(np.mean(RkBar2_Designed, axis=1))

    # normalization = Sim_RRC_M10000_avg[0]
    Sim_RRC_M10000_avg = Sim_RRC_M10000_avg/np.abs(Sim_RRC_M10000_avg).max()+1e-14
    Sim_Designed_M10000_avg = Sim_Designed_M10000_avg/np.abs(Sim_Designed_M10000_avg).max()+1e-14
    Sim_RRC_M10000_avg = np.fft.fftshift(Sim_RRC_M10000_avg)
    Sim_Designed_M10000_avg = np.fft.fftshift(Sim_Designed_M10000_avg)

#%% Plot Fig. 6(c)
x = np.arange(-N*L//2, N*L//2)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

if runNumericalSimulation:
    axs.plot(x, 10*np.log10(Sim_RRC_M10000_avg), color='#8A2BE2', linestyle='none', marker='x', ms=8, markevery=20, label='RRC, 10k Coh Integration, Numerical', zorder = 1)
    axs.plot(x, 10*np.log10(Sim_Designed_M10000_avg), color='#05C349', linestyle='none', marker='+', ms=10, markevery=20, label='Designed Pulse, 10k Coh Integration, Numerical', zorder = 10)

axs.plot(x, 10*np.log10(TheoAveACF_RRC_M10000), color='#F65314', linestyle='-', linewidth=2, label='RRC, 10k Coh Integration', zorder = 1)
axs.plot(x, 10*np.log10(TheoAveACF_Designed_M10000), color='#00A1F1', linestyle='-', linewidth=1, label='Designed Pulse, 10k Coh Integration', zorder = 10)

# font1 = {'family':'Times New Roman', 'style':'normal', 'size':12}
font1 = FontProperties(family='Times New Roman', style='normal', size=20)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', fontsize=20, labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)
axs.set_xlabel(r'Delay Index')
axs.set_ylabel(r'Ambiguity Level (dB)')
axs.set_title(r'OFDM with 16-QAM, Iceberg Shaping Design')
axs.set_xlim([-300, 300])
axs.set_ylim([-80, 0])
axs.set_xticks(np.arange(-300, 301, 100))
axs.set_yticks(np.arange(-80, 1, 10))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)

plt.savefig("Fig6c_TSP.pdf",)
# plt.savefig("Fig6c_TSP.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
