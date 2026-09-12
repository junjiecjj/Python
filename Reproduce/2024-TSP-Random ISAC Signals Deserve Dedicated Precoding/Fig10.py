#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现论文 Fig.10：S&C tradeoff under different precoding schemes.

关键修正：
1) Eq.(40) 中 Omega 必须作为完整 Hermitian PSD 矩阵优化，不能限制为 A+V Delta V^H；
2) 之前 nu0=0.08 太小，30 次迭代后 DDP/DIP 还停留在高通信速率附近，导致多个 R0 得到同一个点；
3) W 更新采用论文 Eq.(43) 的 projected-gradient，并加入回溯以保证每一步在当前子问题上有效下降；
4) 每个 R0 都从 communication water-filling 初始化，保持与 Algorithm 4 一致；
5) 最终打印 |R(W)-R0|，用于检查是否真正落到 Fig.10 的 Pareto boundary 上。
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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
plt.rcParams['legend.fontsize'] = 15

np.random.seed(42)

#%% 基础函数

def project_power(W, P):
    power = np.linalg.norm(W, 'fro')**2
    if power <= P:
        return W
    return W*np.sqrt(P/power)


def matrix_sqrt_psd(A):
    eigvalue, eigvector = np.linalg.eigh((A+A.conj().T)/2)
    eigvalue = np.maximum(eigvalue, 0)
    return eigvector@np.diag(np.sqrt(eigvalue))@eigvector.conj().T


def rate_value(W, Hc, sigma2_c):
    M = np.eye(Hc.shape[0])+Hc@W@W.conj().T@Hc.conj().T/sigma2_c
    sign, logdet = np.linalg.slogdet((M+M.conj().T)/2)
    return np.real(logdet/np.log(2))


def communication_water_filling(Hc, P, sigma2_c):
    Uc, singular_value, Vhc = np.linalg.svd(Hc, full_matrices=False)
    gain = singular_value**2/sigma2_c
    mu_l = 0
    mu_r = 1
    while np.sum(np.maximum(mu_r-1/gain, 0)) < P:
        mu_r = 2*mu_r
    for _ in range(100):
        mu = (mu_l+mu_r)/2
        if np.sum(np.maximum(mu-1/gain, 0)) < P:
            mu_l = mu
        else:
            mu_r = mu
    mu = (mu_l+mu_r)/2
    p = np.maximum(mu-1/gain, 0)
    Vc = Vhc.conj().T
    W = Vc@np.diag(np.sqrt(p))@Vc.conj().T
    return W


def elmmse(W, C_set, lam, sigma2_s, NR):
    Rinv = np.diag(1/lam).astype(complex)
    c = 1/(sigma2_s*NR)
    Delta = Rinv[None,:,:]+c*(W[None,:,:]@C_set@W.conj().T[None,:,:])
    eigvalue = np.linalg.eigvalsh((Delta+np.swapaxes(Delta.conj(), 1, 2))/2)
    J = np.mean(np.sum(1/eigvalue, axis=1))
    return np.real(J)


def gradient_f(W, C_batch, lam, sigma2_s, NR):
    Rinv = np.diag(1/lam).astype(complex)
    c = 1/(sigma2_s*NR)
    WC = W[None,:,:]@C_batch
    Delta = Rinv[None,:,:]+c*(WC@W.conj().T[None,:,:])
    temp1 = np.linalg.solve(Delta, WC)
    temp2 = np.linalg.solve(Delta, temp1)
    grad = -c*np.mean(temp2, axis=0)
    return grad


def penalty_objective(W, Omega, C_batch, lam, sigma2_s, NR, rho):
    value = elmmse(W, C_batch, lam, sigma2_s, NR)+rho/2*np.linalg.norm(Omega-W@W.conj().T, 'fro')**2
    return np.real(value)


def solve_omega(W, Hc, sigma2_c, R0):
    # Eq.(40): min_Omega ||Omega-WW^H||_F^2, s.t. R(Omega)>=R0, Omega>=0.
    # 如果 WW^H 已满足速率约束，那么 Eq.(40) 的精确最优解就是 Omega=WW^H，无需调用 CVX。
    NT = W.shape[0]
    Nu = Hc.shape[0]
    A = W@W.conj().T
    current_rate = rate_value(W, Hc, sigma2_c)
    if current_rate >= R0-1e-8:
        return A

    Omega = cp.Variable((NT,NT), hermitian=True)
    rate_matrix = np.eye(Nu)+Hc@Omega@Hc.conj().T/sigma2_c
    constraints = [Omega >> 0, cp.log_det(rate_matrix)/np.log(2) >= R0]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(Omega-A))), constraints)
    problem.solve(verbose=False, warm_start=True)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f'Eq.(40) failed. Status: {problem.status}')

    Omega_value = np.asarray(Omega.value)
    Omega_value = (Omega_value+Omega_value.conj().T)/2
    return Omega_value


def solve_detopt(lam, Hc, P, sigma2_s, sigma2_c, NR, L, R0):
    NT = len(lam)
    Nu = Hc.shape[0]
    Rinv = np.diag(1/lam)
    Omega = cp.Variable((NT,NT), hermitian=True)
    Delta = Rinv+L/(sigma2_s*NR)*Omega
    rate_matrix = np.eye(Nu)+Hc@Omega@Hc.conj().T/sigma2_c
    constraints = [Omega >> 0, cp.real(cp.trace(Omega)) <= P, cp.log_det(rate_matrix)/np.log(2) >= R0]
    problem = cp.Problem(cp.Minimize(cp.matrix_frac(np.eye(NT), Delta)), constraints)
    problem.solve(verbose=False, warm_start=True)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f'DetOpt failed. Status: {problem.status}')

    Omega_value = np.asarray(Omega.value)
    Omega_value = (Omega_value+Omega_value.conj().T)/2
    W = matrix_sqrt_psd(Omega_value)
    return W


#%% Fig.10 参数
NT = 32
NR = 32
Nu = 4
sigma2_s = 1
sigma2_c = 1
SNR_dB = 16
P = sigma2_s*10**(SNR_dB/10)

N_train = 100
N_test = 100
batch_size = 10
tmax = 30
tau0 = 1e-3
xi0 = 0.1

# 论文只说明 rho 逐步增加、nu(t) 逐步减小，没有公开具体数值。
# nu0=0.08 会导致 30 次迭代远未收敛，这正是旧代码前 3 个点完全重合的主要原因。
rho0 = 0.02
rho_growth = 1.20
nu0 = 1.0
nu_decay = 0.15
backtracking_factor = 0.5
backtracking_max = 12

L_list = [24, 32]
rate_ratio = np.array([0.64, 0.76, 0.88, 0.96, 0.999])

#%% 产生 R_H 和通信信道 H_c
rng_RH = np.random.default_rng(42)
lam = np.sort(rng_RH.uniform(1, 10, NT))

# 论文没有给出 H_c 的具体 realization。
# 这里固定 i.i.d. CN(0,1) realization；seed=4 时 Rmax 与论文 Fig.10 右端约 32.7 bps/Hz 接近。
rng_Hc = np.random.default_rng(4)
Hc = (rng_Hc.standard_normal((Nu,NT))+1j*rng_Hc.standard_normal((Nu,NT)))/np.sqrt(2)

W_comm = communication_water_filling(Hc, P, sigma2_c)
Rmax = rate_value(W_comm, Hc, sigma2_c)
R0_list = rate_ratio*Rmax

print(f'Rmax = {Rmax:.6f} bps/Hz')
print('R0 =', np.round(R0_list, 6))

#%% 结果数组
result = {}

#%% 分别计算 L=24 和 L=32
for iL, L in enumerate(L_list):
    rng_train = np.random.default_rng(100+iL)
    S_train = (rng_train.standard_normal((N_train,NT,L))+1j*rng_train.standard_normal((N_train,NT,L)))/np.sqrt(2)
    C_train = S_train@np.swapaxes(S_train.conj(), 1, 2)

    rng_test = np.random.default_rng(1000+iL)
    S_test = (rng_test.standard_normal((N_test,NT,L))+1j*rng_test.standard_normal((N_test,NT,L)))/np.sqrt(2)
    C_test = S_test@np.swapaxes(S_test.conj(), 1, 2)

    rate_DetOpt = np.zeros(len(R0_list))
    rate_DIP = np.zeros(len(R0_list))
    rate_DDP = np.zeros(len(R0_list))
    J_DetOpt = np.zeros(len(R0_list))
    J_DIP = np.zeros(len(R0_list))
    J_DDP = np.zeros(len(R0_list))

    #%% 扫描通信速率约束
    for iR, R0 in enumerate(R0_list):
        print(f'\n================ L={L}, R0={R0:.6f} ================')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # DetOpt, Eq.(50)-(51)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_DetOpt = solve_detopt(lam, Hc, P, sigma2_s, sigma2_c, NR, L, R0)
        rate_DetOpt[iR] = rate_value(W_DetOpt, Hc, sigma2_c)
        J_DetOpt[iR] = elmmse(W_DetOpt, C_test, lam, sigma2_s, NR)
        print(f'DetOpt: Rate={rate_DetOpt[iR]:.6f}, |Rate-R0|={abs(rate_DetOpt[iR]-R0):.6f}, ELMMSE={10*np.log10(J_DetOpt[iR]/(NT*NR)):.6f} dB')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # DIP, Eq.(46): Penalty-Based SGP-AO
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_DIP = W_comm.copy()
        rho = rho0
        previous_sensing_objective = np.inf
        rng_batch = np.random.default_rng(20000+iL*100+iR)

        for t in range(tmax):
            Omega = solve_omega(W_DIP, Hc, sigma2_c, R0)
            A = W_DIP@W_DIP.conj().T
            index = rng_batch.choice(N_train, batch_size, replace=False)
            grad_sensing = gradient_f(W_DIP, C_train[index], lam, sigma2_s, NR)
            grad_penalty = rho*(A-Omega)@W_DIP
            grad = grad_sensing+grad_penalty

            nu = nu0/(1+nu_decay*t)
            h_old = penalty_objective(W_DIP, Omega, C_train[index], lam, sigma2_s, NR, rho)
            for _ in range(backtracking_max):
                W_candidate = project_power(W_DIP-nu*grad, P)
                h_new = penalty_objective(W_candidate, Omega, C_train[index], lam, sigma2_s, NR, rho)
                if h_new <= h_old+1e-10:
                    break
                nu = nu*backtracking_factor

            W_DIP = W_candidate
            sensing_objective = elmmse(W_DIP, C_train, lam, sigma2_s, NR)
            xi = abs(rate_value(W_DIP, Hc, sigma2_c)-R0)

            if previous_sensing_objective < np.inf and abs(previous_sensing_objective-sensing_objective) <= tau0 and xi <= xi0:
                break

            previous_sensing_objective = sensing_objective
            rho = rho*rho_growth

        rate_DIP[iR] = rate_value(W_DIP, Hc, sigma2_c)
        J_DIP[iR] = elmmse(W_DIP, C_test, lam, sigma2_s, NR)
        print(f'DIP:    Rate={rate_DIP[iR]:.6f}, |Rate-R0|={abs(rate_DIP[iR]-R0):.6f}, ELMMSE={10*np.log10(J_DIP[iR]/(NT*NR)):.6f} dB, Iter={t+1}')
        if abs(rate_DIP[iR]-R0) > xi0:
            print('WARNING: DIP has not reached the Fig.10 Pareto boundary. Increase tmax or adjust rho/step-size schedule.')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # DDP, Eq.(37)-(44): Penalty-Based AO
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        J_DDP_sample = np.zeros(N_test)
        rate_DDP_sample = np.zeros(N_test)

        for n in range(N_test):
            W_DDP = W_comm.copy()
            rho = rho0
            previous_sensing_objective = np.inf

            for t in range(tmax):
                Omega = solve_omega(W_DDP, Hc, sigma2_c, R0)
                A = W_DDP@W_DDP.conj().T
                grad_sensing = gradient_f(W_DDP, C_test[n:n+1], lam, sigma2_s, NR)
                grad_penalty = rho*(A-Omega)@W_DDP
                grad = grad_sensing+grad_penalty

                nu = nu0/(1+nu_decay*t)
                h_old = penalty_objective(W_DDP, Omega, C_test[n:n+1], lam, sigma2_s, NR, rho)
                for _ in range(backtracking_max):
                    W_candidate = project_power(W_DDP-nu*grad, P)
                    h_new = penalty_objective(W_candidate, Omega, C_test[n:n+1], lam, sigma2_s, NR, rho)
                    if h_new <= h_old+1e-10:
                        break
                    nu = nu*backtracking_factor

                W_DDP = W_candidate
                sensing_objective = elmmse(W_DDP, C_test[n:n+1], lam, sigma2_s, NR)
                xi = abs(rate_value(W_DDP, Hc, sigma2_c)-R0)

                if previous_sensing_objective < np.inf and abs(previous_sensing_objective-sensing_objective) <= tau0 and xi <= xi0:
                    break

                previous_sensing_objective = sensing_objective
                rho = rho*rho_growth

            J_DDP_sample[n] = elmmse(W_DDP, C_test[n:n+1], lam, sigma2_s, NR)
            rate_DDP_sample[n] = rate_value(W_DDP, Hc, sigma2_c)

        J_DDP[iR] = np.mean(J_DDP_sample)
        rate_DDP[iR] = np.mean(rate_DDP_sample)
        rate_violation_DDP = np.mean(np.abs(rate_DDP_sample-R0))
        print(f'DDP:    Rate={rate_DDP[iR]:.6f}, Mean|Rate-R0|={rate_violation_DDP:.6f}, ELMMSE={10*np.log10(J_DDP[iR]/(NT*NR)):.6f} dB')
        if rate_violation_DDP > xi0:
            print('WARNING: DDP has not reached the Fig.10 Pareto boundary. Increase tmax or adjust rho/step-size schedule.')

    result[L] = {}
    result[L]['DetOpt Rate'] = rate_DetOpt
    result[L]['DIP Rate'] = rate_DIP
    result[L]['DDP Rate'] = rate_DDP
    result[L]['DetOpt'] = 10*np.log10(J_DetOpt/(NT*NR))
    result[L]['DIP'] = 10*np.log10(J_DIP/(NT*NR))
    result[L]['DDP'] = 10*np.log10(J_DDP/(NT*NR))

    print(f'\nL={L} final rates:')
    print('R0           =', np.round(R0_list, 4))
    print('DetOpt Rate  =', np.round(rate_DetOpt, 4))
    print('DIP Rate     =', np.round(rate_DIP, 4))
    print('DDP Rate     =', np.round(rate_DDP, 4))


#%% Plot Fig.10
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

color_DetOpt = '#F65314'
color_DIP = '#00A1F1'
color_DDP = '#8A2BE2'

# L=24：实线
axs.plot(result[24]['DetOpt Rate'], result[24]['DetOpt'], color=color_DetOpt, linestyle='-', linewidth=2, marker='>', ms=8, markerfacecolor='white', label=r'DetOpt, $L=24$', zorder=3)
axs.plot(result[24]['DIP Rate'], result[24]['DIP'], color=color_DIP, linestyle='-', linewidth=2, marker='o', ms=7, markerfacecolor='white', label=r'DIP, $L=24$', zorder=4)
axs.plot(result[24]['DDP Rate'], result[24]['DDP'], color=color_DDP, linestyle='-', linewidth=2, marker='s', ms=7, markerfacecolor='white', label=r'DDP, $L=24$', zorder=5)

# L=32：虚线
axs.plot(result[32]['DetOpt Rate'], result[32]['DetOpt'], color=color_DetOpt, linestyle='--', linewidth=2, marker='>', ms=8, label=r'DetOpt, $L=32$', zorder=3)
axs.plot(result[32]['DIP Rate'], result[32]['DIP'], color=color_DIP, linestyle='--', linewidth=2, marker='o', ms=7, label=r'DIP, $L=32$', zorder=4)
axs.plot(result[32]['DDP Rate'], result[32]['DDP'], color=color_DDP, linestyle='--', linewidth=2, marker='s', ms=7, label=r'DDP, $L=32$', zorder=5)

font1 = FontProperties(family='Times New Roman', style='normal', size=14)
legend1 = axs.legend(loc='upper left', ncol=2, borderaxespad=0, edgecolor='black', fontsize=14, labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)
axs.set_xlabel(r'Communication Rate [bps/Hz]')
axs.set_ylabel(r'Normalized ELMMSE [dB]')
axs.set_xlim([20.5, 33])
axs.set_ylim([-16, -10])
axs.set_xticks(np.arange(22, 33, 2))
axs.set_yticks(np.arange(-16, -9, 1))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(18) for label in labels]
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)

plt.savefig('Fig10_TSP.pdf')
# plt.savefig('Fig10_TSP.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()




