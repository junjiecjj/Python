#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:10:14 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现论文 Fig. 8(a): Random ISAC Signals Deserve Dedicated Precoding, IEEE TSP 2024.

Fig. 8(a): L:N_T = 3:4，对应 (N_T,L)=(32,24) 和 (128,96)。
比较：Water-Filling、DIP(SGP)、DIP(MB-SGP)、DDP。
"""

import numpy as np
import matplotlib.pyplot as plt

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 11

np.random.seed(42)

#%% 基础函数

def project_power(W, P):
    power = np.linalg.norm(W, 'fro')**2
    if power <= P:
        return W
    return W*np.sqrt(P/power)


def find_water_level(power_function, P):
    mu_l = 0.0
    mu_r = 1.0
    while power_function(mu_r) < P:
        mu_r = 2*mu_r
    for _ in range(100):
        mu = (mu_l+mu_r)/2
        if power_function(mu) < P:
            mu_l = mu
        else:
            mu_r = mu
    return (mu_l+mu_r)/2


def water_filling(lam, P, L, sigma2, NR):
    c = sigma2*NR/L
    def power_function(mu):
        return np.sum(c*np.maximum(mu-1/lam, 0))
    mu = find_water_level(power_function, P)
    p = c*np.maximum(mu-1/lam, 0)
    W = np.diag(np.sqrt(p)).astype(complex)
    return W


def elmmse(W, C_set, lam, sigma2, NR):
    Rinv = np.diag(1/lam).astype(complex)
    c = 1/(sigma2*NR)
    Delta = Rinv[None,:,:]+c*(W[None,:,:]@C_set@W.conj().T[None,:,:])
    eigvalue = np.linalg.eigvalsh(Delta)
    J = np.mean(np.sum(1/eigvalue, axis=1))
    return np.real(J)


def gradient_f(W, C_batch, lam, sigma2, NR):
    Rinv = np.diag(1/lam).astype(complex)
    c = 1/(sigma2*NR)
    WC = W[None,:,:]@C_batch
    Delta = Rinv[None,:,:]+c*(WC@W.conj().T[None,:,:])
    temp1 = np.linalg.solve(Delta, WC)
    temp2 = np.linalg.solve(Delta, temp1)
    grad = -c*np.mean(temp2, axis=0)
    return grad


def dip_sgp(C_set, lam, P, sigma2, NR, batch_index, W0, a=10, b=10):
    W = W0.copy()
    for r in range(len(batch_index)):
        grad = gradient_f(W, C_set[batch_index[r]], lam, sigma2, NR)
        eta = a/(b+r+1)
        W = project_power(W-eta*grad, P)
    return W


def dip_mbsgp(C_set, lam, P, sigma2, NR, batch_index, W0, a=10, b=10, beta1=0.6, beta2=0.999, epsilon0=1e-8):
    W = W0.copy()
    M = np.zeros_like(W)
    v = 0.0
    for r in range(len(batch_index)):
        grad = gradient_f(W, C_set[batch_index[r]], lam, sigma2, NR)
        M = beta1*M+(1-beta1)*grad
        v = beta2*v+(1-beta2)*np.linalg.norm(grad, 'fro')**2
        M_hat = M/(1-beta1**(r+1))
        v_hat = v/(1-beta2**(r+1))
        eta = a/(b+r+1)
        W = project_power(W-eta*M_hat/(np.sqrt(v_hat)+epsilon0), P)
    return W


def ddp_elmmse(theta_set, lam, P):
    J_set = np.zeros(len(theta_set))
    for n in range(len(theta_set)):
        theta = theta_set[n]
        active = theta > 1e-14
        lam_active = lam[active]
        theta_active = theta[active]
        def power_function(mu):
            return np.sum(np.maximum(mu/np.sqrt(theta_active)-1/(lam_active*theta_active), 0))
        mu = find_water_level(power_function, P)
        p = np.zeros(len(lam))
        p[active] = np.maximum(mu/np.sqrt(theta_active)-1/(lam_active*theta_active), 0)
        J_set[n] = np.sum(1/(1/lam+theta*p))
    return np.mean(J_set)


#%% Fig. 8(a) 参数
sigma2 = 1
N_train = 100
N_test = 100
batch_size = 10
rmax = 1000
snr_db = np.arange(0, 51, 5)
case_list = [(32, 24), (128, 96)]

result = {}

#%% 分别计算 N_T=32 和 N_T=128
for case_index, (NT, L) in enumerate(case_list):
    NR = NT

    # R_H 的特征值服从 U[1,10]，固定一次 realization 并在所有算法/SNR 中共用
    rng_RH = np.random.default_rng(42+case_index)
    lam = np.sort(rng_RH.uniform(1, 10, NT))

    # DIP 用独立的 Gaussian training set 做离线优化
    rng_train = np.random.default_rng(100+case_index)
    S_train = (rng_train.standard_normal((N_train,NT,L))+1j*rng_train.standard_normal((N_train,NT,L)))/np.sqrt(2)
    C_train = S_train@np.swapaxes(S_train.conj(), 1, 2)

    # 性能评估使用独立 test set，避免 MB-SGP 对有限 training set 的过拟合造成虚假优势
    rng_test = np.random.default_rng(1000+case_index)
    S_test = (rng_test.standard_normal((N_test,NT,L))+1j*rng_test.standard_normal((N_test,NT,L)))/np.sqrt(2)
    C_test = S_test@np.swapaxes(S_test.conj(), 1, 2)

    # DDP 对每个 test realization 都根据瞬时 S_n 重新设计，因此提前计算其 Theta_n
    theta_test = np.zeros((N_test,NT))
    for n in range(N_test):
        singular_value = np.linalg.svd(S_test[n], compute_uv=False, full_matrices=False)
        singular_value_square = np.zeros(NT)
        singular_value_square[0:len(singular_value)] = singular_value**2
        theta_test[n] = singular_value_square[::-1]/(sigma2*NR)

    J_WF = np.zeros(len(snr_db))
    J_SGP = np.zeros(len(snr_db))
    J_MBSGP = np.zeros(len(snr_db))
    J_DDP = np.zeros(len(snr_db))

    # Fig.8(a) 比较的是最终 DIP 性能而不是收敛速度。
    # 论文没有给 Algorithm 1/2 的 W^(1)，因此扫 SNR 时采用同一个 continuation 初始点：
    # 第一个 SNR 用等功率，后续 SNR 都从上一个 SNR 的收敛 DIP 解按功率比例缩放。
    W_DIP_prev = None
    P_prev = None

    for iSNR in range(len(snr_db)):
        SNR = snr_db[iSNR]
        P = sigma2*10**(SNR/10)

        # Water-Filling, Eq.(8)
        W_WF = water_filling(lam, P, L, sigma2, NR)
        J_WF[iSNR] = elmmse(W_WF, C_test, lam, sigma2, NR)

        # DDP, Theorem 1 / Eq.(19)-(21)
        J_DDP[iSNR] = ddp_elmmse(theta_test, lam, P)

        # SGP 和 MB-SGP 使用完全相同的初始点
        if W_DIP_prev is None:
            W0 = np.sqrt(P/NT)*np.eye(NT, dtype=complex)
        else:
            W0 = W_DIP_prev*np.sqrt(P/P_prev)

        # SGP 和 MB-SGP 使用完全相同的 mini-batch 序列
        rng_batch = np.random.default_rng(10000+case_index*100+iSNR)
        batch_index = np.zeros((rmax,batch_size), dtype=int)
        for r in range(rmax):
            batch_index[r] = rng_batch.choice(N_train, batch_size, replace=False)

        # DIP(SGP), Algorithm 1
        W_SGP = dip_sgp(C_train, lam, P, sigma2, NR, batch_index, W0)
        J_SGP[iSNR] = elmmse(W_SGP, C_test, lam, sigma2, NR)

        # DIP(MB-SGP), Algorithm 2
        W_MBSGP = dip_mbsgp(C_train, lam, P, sigma2, NR, batch_index, W0)
        J_MBSGP[iSNR] = elmmse(W_MBSGP, C_test, lam, sigma2, NR)

        # MB-SGP 的作用主要是更快达到同一个 DIP 解，因此用其收敛解作为下一 SNR 的公共 continuation 点
        W_DIP_prev = W_MBSGP.copy()
        P_prev = P

        gap = 10*np.log10(J_SGP[iSNR]/J_MBSGP[iSNR])
        print(f'N_T={NT:3d}, L={L:3d}, SNR={SNR:2d} dB, SGP-MB-SGP gap={gap:.4f} dB')

    result[NT] = {}
    result[NT]['Water-Filling'] = 10*np.log10(J_WF/(NT*NR))
    result[NT]['DIP (SGP)'] = 10*np.log10(J_SGP/(NT*NR))
    result[NT]['DIP (MB-SGP)'] = 10*np.log10(J_MBSGP/(NT*NR))
    result[NT]['DDP'] = 10*np.log10(J_DDP/(NT*NR))


#%% Plot Fig. 8(a)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

axs.plot(snr_db, result[32]['Water-Filling'], color='c', marker='o', markerfacecolor='white', label=r'Water-Filling, $N_t=32$')
axs.plot(snr_db, result[32]['DIP (SGP)'], color='g', marker='x', linestyle='None', label=r'DIP (SGP), $N_t=32$')
axs.plot(snr_db, result[32]['DIP (MB-SGP)'], color='b', linestyle='-', label=r'DIP (MB-SGP), $N_t=32$')
axs.plot(snr_db, result[32]['DDP'], color='k', marker='d', markerfacecolor='white', label=r'DDP, $N_t=32$')

axs.plot(snr_db, result[128]['Water-Filling'], color='c', marker='o', markerfacecolor='white', linestyle='--', label=r'Water-Filling, $N_t=128$')
axs.plot(snr_db, result[128]['DIP (SGP)'], color='g', marker='x', linestyle='None', label=r'DIP (SGP), $N_t=128$')
axs.plot(snr_db, result[128]['DIP (MB-SGP)'], color='b', linestyle='--', label=r'DIP (MB-SGP), $N_t=128$')
axs.plot(snr_db, result[128]['DDP'], color='k', marker='d', markerfacecolor='white', linestyle='--', label=r'DDP, $N_t=128$')

axs.set_xlabel(r'Transmit SNR [dB]')
axs.set_ylabel(r'Normalized ELMMSE [dB]')
axs.set_xlim([0, 50])
axs.set_ylim([-26, -8])
axs.set_xticks(np.arange(0, 51, 5))
axs.set_yticks(np.arange(-26, -7, 2))
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.3)
legend1 = axs.legend(loc='lower left', borderaxespad=0, edgecolor='black', fontsize=11)

out_fig = plt.gcf()
out_fig.savefig('./Fig8a_reproduced_v3.png', dpi=300)
out_fig.savefig('./Fig8a_reproduced_v3.pdf')
plt.show()
plt.close()
