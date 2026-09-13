
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现 2025 JSAC tutorial Fig.14：S&C performance tradeoff under different precoding designs.

论文明确参数：N_t=N_s=32，N=20 和 32，SNR=15 dB。
算法来自同团队 2024 TSP [70]：
1) Baseline：确定性 LMMSE + communication-rate / power constraints
2) DIP：Penalty-Based SGP-AO
3) DDP：Penalty-Based AO

说明：论文没有公开 H_c 的具体 realization，也没有公开 rho 与 nu(t) 的精确数值序列。
这里固定 H_c~CN(0,1) 的 seed，并把这些实现参数集中放在参数区。
Fig.14 中约有 9 个 operating points，这里按图反推 R0/Rmax=0.60:0.05:1.00，
最后一点用 0.999 代替 1.00，以避免容量边界上的数值不可行。

Python/CVXPY 版本主要用于交叉验证；当前更建议以 MATLAB/CVX+SeDuMi 版本为主。
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 全局画图设置
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
plt.rcParams['lines.markersize'] = 7
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 14
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
    eigvalue = np.linalg.eigvalsh((M+M.conj().T)/2)
    eigvalue = np.maximum(eigvalue, np.finfo(float).eps)
    return np.sum(np.log2(eigvalue)).real

def communication_water_filling(Hc, P, sigma2_c):
    _, singular_value, Vh = np.linalg.svd(Hc, full_matrices=False)
    gain = singular_value**2/sigma2_c
    mu_l = 0.0
    mu_r = 1.0
    while np.sum(np.maximum(mu_r-1/gain, 0)) < P:
        mu_r *= 2
    for _ in range(100):
        mu = (mu_l+mu_r)/2
        power_allocation = np.maximum(mu-1/gain, 0)
        if np.sum(power_allocation) < P:
            mu_l = mu
        else:
            mu_r = mu
    power_allocation = np.maximum((mu_l+mu_r)/2-1/gain, 0)
    V = Vh.conj().T
    Omega = V@np.diag(power_allocation)@V.conj().T
    return matrix_sqrt_psd(Omega)

def single_lmmse(W, C, RHinv, sigma2_s, Ns):
    Delta = RHinv+1/(sigma2_s*Ns)*W@C@W.conj().T
    Delta = (Delta+Delta.conj().T)/2
    return np.trace(np.linalg.solve(Delta, np.eye(Delta.shape[0]))).real

def elmmse(W, C_set, RHinv, sigma2_s, Ns):
    J = 0.0
    for m in range(C_set.shape[0]):
        J += single_lmmse(W, C_set[m], RHinv, sigma2_s, Ns)
    return J/C_set.shape[0]

def gradient_f(W, C_batch, RHinv, sigma2_s, Ns):
    c = 1/(sigma2_s*Ns)
    grad = np.zeros_like(W, dtype=complex)
    for m in range(C_batch.shape[0]):
        C = C_batch[m]
        Delta = RHinv+c*W@C@W.conj().T
        Delta = (Delta+Delta.conj().T)/2
        WC = W@C
        grad -= c*np.linalg.solve(Delta, np.linalg.solve(Delta, WC))
    return grad/C_batch.shape[0]

def solve_omega(W, Hc, sigma2_c, R0, solver_name):
    A = W@W.conj().T
    if rate_value(W, Hc, sigma2_c) >= R0-1e-10:
        return A
    NT = W.shape[0]
    Omega = cp.Variable((NT,NT), hermitian=True)
    RateMatrix = np.eye(Hc.shape[0])+Hc@Omega@Hc.conj().T/sigma2_c
    constraints = [Omega >> 0, cp.log_det(RateMatrix) >= R0*np.log(2)]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(Omega-A))), constraints)
    problem.solve(solver=solver_name, verbose=False, warm_start=True)
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f'Omega subproblem failed. Status: {problem.status}')
    return (Omega.value+Omega.value.conj().T)/2

def solve_baseline(RHinv, Hc, PT, sigma2_s, sigma2_c, Ns, N, R0, solver_name):
    NT = RHinv.shape[0]
    Omega = cp.Variable((NT,NT), hermitian=True)
    Z = cp.Variable((NT,NT), hermitian=True)
    Delta = RHinv+N/(sigma2_s*Ns)*Omega
    RateMatrix = np.eye(Hc.shape[0])+Hc@Omega@Hc.conj().T/sigma2_c
    block = cp.bmat([[Delta, np.eye(NT)], [np.eye(NT), Z]])
    constraints = [Omega >> 0, Z >> 0, block >> 0, cp.real(cp.trace(Omega)) <= PT, cp.log_det(RateMatrix) >= R0*np.log(2)]
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(Z))), constraints)
    problem.solve(solver=solver_name, verbose=False, warm_start=True)
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError(f'Baseline failed. Status: {problem.status}')
    Omega_value = (Omega.value+Omega.value.conj().T)/2
    return matrix_sqrt_psd(Omega_value)

#%% Fig.14 参数
NT = 32
Ns = 32
Nc = 4
sigma2_s = 1
sigma2_c = 1
SNR_dB = 15
PT = sigma2_s*10**(SNR_dB/10)
NumMC = 100
batch_size = 10
tmax = 30
tau0 = 1e-3
xi0 = 0.1
N_list = [20, 32]
rate_ratio = np.array([0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.999])

# 论文未明确给出的 AO 数值参数
rho0 = 0.02
rho_growth = 1.25
rho_max = 1e4
nu0 = 1.0
nu_decay = 0.15
max_backtracking = 20

# CVXPY solver：优先 CLARABEL；若未安装则退到 SCS
installed_solvers = cp.installed_solvers()
solver_name = cp.CLARABEL if 'CLARABEL' in installed_solvers else cp.SCS
print('CVXPY solver =', solver_name)

#%% 产生 R_H 和 H_c
rng_RH = np.random.default_rng(42)
lambda_H = np.sort(rng_RH.uniform(1, 10, NT))
RHinv = np.diag(1/lambda_H).astype(complex)

rng_Hc = np.random.default_rng(4)
Hc = (rng_Hc.standard_normal((Nc,NT))+1j*rng_Hc.standard_normal((Nc,NT)))/np.sqrt(2)

W_comm = communication_water_filling(Hc, PT, sigma2_c)
Rmax = rate_value(W_comm, Hc, sigma2_c)
R0_list = rate_ratio*Rmax
print(f'Rmax = {Rmax:.6f} bps/Hz')
print('R0 =', np.round(R0_list, 6))

#%% 结果数组
result = {}

#%% 分别计算 N=20 和 N=32
for iN, N in enumerate(N_list):
    rng_train = np.random.default_rng(100+iN)
    S_train = (rng_train.standard_normal((NumMC,NT,N))+1j*rng_train.standard_normal((NumMC,NT,N)))/np.sqrt(2)
    C_train = S_train@np.swapaxes(S_train.conj(), 1, 2)

    rng_test = np.random.default_rng(1000+iN)
    S_test = (rng_test.standard_normal((NumMC,NT,N))+1j*rng_test.standard_normal((NumMC,NT,N)))/np.sqrt(2)
    C_test = S_test@np.swapaxes(S_test.conj(), 1, 2)

    rate_Baseline = np.zeros(len(R0_list))
    rate_DIP = np.zeros(len(R0_list))
    rate_DDP = np.zeros(len(R0_list))
    J_Baseline = np.zeros(len(R0_list))
    J_DIP = np.zeros(len(R0_list))
    J_DDP = np.zeros(len(R0_list))

    #%% 扫描通信速率约束
    for iR, R0 in enumerate(R0_list):
        print(f'\nN={N}, R0={R0:.6f} bps/Hz')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Baseline：deterministic LMMSE optimization
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_Baseline = solve_baseline(RHinv, Hc, PT, sigma2_s, sigma2_c, Ns, N, R0, solver_name)
        rate_Baseline[iR] = rate_value(W_Baseline, Hc, sigma2_c)
        J_Baseline[iR] = elmmse(W_Baseline, C_test, RHinv, sigma2_s, Ns)
        print(f'Baseline: Rate={rate_Baseline[iR]:.6f}, RateViolation={max(R0-rate_Baseline[iR],0):.6f}, Normalized ELMMSE={10*np.log10(J_Baseline[iR]/(NT*Ns)):.6f} dB')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # DIP：Eq.(101), Penalty-Based SGP-AO
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_DIP = W_comm.copy()
        rho = rho0
        previous_sensing = elmmse(W_DIP, C_train, RHinv, sigma2_s, Ns)
        rng_batch = np.random.default_rng(20000+iN*100+iR)
        for t in range(tmax):
            Omega = solve_omega(W_DIP, Hc, sigma2_c, R0, solver_name)
            index = rng_batch.choice(NumMC, batch_size, replace=False)
            grad_sensing = gradient_f(W_DIP, C_train[index], RHinv, sigma2_s, Ns)
            grad_penalty = rho*(W_DIP@W_DIP.conj().T-Omega)@W_DIP
            grad = grad_sensing+grad_penalty
            h_old = elmmse(W_DIP, C_train[index], RHinv, sigma2_s, Ns)+rho/2*np.linalg.norm(Omega-W_DIP@W_DIP.conj().T, 'fro')**2
            nu = nu0/(1+nu_decay*t)
            accepted = False
            for _ in range(max_backtracking):
                W_candidate = project_power(W_DIP-nu*grad, PT)
                h_new = elmmse(W_candidate, C_train[index], RHinv, sigma2_s, Ns)+rho/2*np.linalg.norm(Omega-W_candidate@W_candidate.conj().T, 'fro')**2
                if h_new <= h_old+1e-12:
                    accepted = True
                    break
                nu *= 0.5
            if not accepted:
                W_candidate = W_DIP
            W_DIP = W_candidate
            current_sensing = elmmse(W_DIP, C_train, RHinv, sigma2_s, Ns)
            xi = max(R0-rate_value(W_DIP, Hc, sigma2_c), 0)
            if abs(previous_sensing-current_sensing) <= tau0 and xi <= xi0:
                break
            previous_sensing = current_sensing
            rho = min(rho*rho_growth, rho_max)

        rate_DIP[iR] = rate_value(W_DIP, Hc, sigma2_c)
        J_DIP[iR] = elmmse(W_DIP, C_test, RHinv, sigma2_s, Ns)
        print(f'DIP:      Rate={rate_DIP[iR]:.6f}, RateViolation={max(R0-rate_DIP[iR],0):.6f}, Normalized ELMMSE={10*np.log10(J_DIP[iR]/(NT*Ns)):.6f} dB, Iter={t+1}')

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # DDP：Eq.(99)-(100), 每个 realization 单独执行 Penalty-Based AO
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        J_DDP_sample = np.zeros(NumMC)
        rate_DDP_sample = np.zeros(NumMC)
        for m in range(NumMC):
            W_DDP = W_comm.copy()
            rho = rho0
            previous_sensing = single_lmmse(W_DDP, C_test[m], RHinv, sigma2_s, Ns)
            for t in range(tmax):
                Omega = solve_omega(W_DDP, Hc, sigma2_c, R0, solver_name)
                grad_sensing = gradient_f(W_DDP, C_test[m:m+1], RHinv, sigma2_s, Ns)
                grad_penalty = rho*(W_DDP@W_DDP.conj().T-Omega)@W_DDP
                grad = grad_sensing+grad_penalty
                h_old = single_lmmse(W_DDP, C_test[m], RHinv, sigma2_s, Ns)+rho/2*np.linalg.norm(Omega-W_DDP@W_DDP.conj().T, 'fro')**2
                nu = nu0/(1+nu_decay*t)
                accepted = False
                for _ in range(max_backtracking):
                    W_candidate = project_power(W_DDP-nu*grad, PT)
                    h_new = single_lmmse(W_candidate, C_test[m], RHinv, sigma2_s, Ns)+rho/2*np.linalg.norm(Omega-W_candidate@W_candidate.conj().T, 'fro')**2
                    if h_new <= h_old+1e-12:
                        accepted = True
                        break
                    nu *= 0.5
                if not accepted:
                    W_candidate = W_DDP
                W_DDP = W_candidate
                current_sensing = single_lmmse(W_DDP, C_test[m], RHinv, sigma2_s, Ns)
                xi = max(R0-rate_value(W_DDP, Hc, sigma2_c), 0)
                if abs(previous_sensing-current_sensing) <= tau0 and xi <= xi0:
                    break
                previous_sensing = current_sensing
                rho = min(rho*rho_growth, rho_max)
            J_DDP_sample[m] = single_lmmse(W_DDP, C_test[m], RHinv, sigma2_s, Ns)
            rate_DDP_sample[m] = rate_value(W_DDP, Hc, sigma2_c)

        J_DDP[iR] = np.mean(J_DDP_sample)
        rate_DDP[iR] = np.mean(rate_DDP_sample)
        mean_violation = np.mean(np.maximum(R0-rate_DDP_sample, 0))
        print(f'DDP:      Rate={rate_DDP[iR]:.6f}, MeanRateViolation={mean_violation:.6f}, Normalized ELMMSE={10*np.log10(J_DDP[iR]/(NT*Ns)):.6f} dB')

    result[N] = {}
    result[N]['Baseline Rate'] = rate_Baseline
    result[N]['DIP Rate'] = rate_DIP
    result[N]['DDP Rate'] = rate_DDP
    result[N]['Baseline'] = 10*np.log10(J_Baseline/(NT*Ns))
    result[N]['DIP'] = 10*np.log10(J_DIP/(NT*Ns))
    result[N]['DDP'] = 10*np.log10(J_DDP/(NT*Ns))

#%% 打印最终数组，便于逐点排查
for N in N_list:
    print(f'\nN={N} Baseline Rate =', result[N]['Baseline Rate'])
    print(f'N={N} DIP Rate      =', result[N]['DIP Rate'])
    print(f'N={N} DDP Rate      =', result[N]['DDP Rate'])
    print(f'N={N} Baseline      =', result[N]['Baseline'])
    print(f'N={N} DIP           =', result[N]['DIP'])
    print(f'N={N} DDP           =', result[N]['DDP'])

#%% Plot Fig.14
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
color_Baseline = '#77AC30'
color_DIP = '#00A1F1'
color_DDP = '#F65314'

# N=20：实线
axs.plot(result[20]['Baseline Rate'], result[20]['Baseline'], color=color_Baseline, linestyle='-', linewidth=2, marker='o', ms=7, markerfacecolor='white', label=r'Baseline ($N=20$)', zorder=3)
axs.plot(result[20]['DIP Rate'], result[20]['DIP'], color=color_DIP, linestyle='-', linewidth=2, label=r'DIP Scheme ($N=20$)', zorder=4)
axs.plot(result[20]['DDP Rate'], result[20]['DDP'], color=color_DDP, linestyle='-', linewidth=2, marker='s', ms=7, markerfacecolor='white', label=r'DDP Scheme ($N=20$)', zorder=5)

# N=32：虚线
axs.plot(result[32]['Baseline Rate'], result[32]['Baseline'], color=color_Baseline, linestyle='--', linewidth=2, marker='o', ms=7, markerfacecolor='white', label=r'Baseline ($N=32$)', zorder=3)
axs.plot(result[32]['DIP Rate'], result[32]['DIP'], color=color_DIP, linestyle='--', linewidth=2, label=r'DIP Scheme ($N=32$)', zorder=4)
axs.plot(result[32]['DDP Rate'], result[32]['DDP'], color=color_DDP, linestyle='--', linewidth=2, marker='s', ms=7, markerfacecolor='white', label=r'DDP Scheme ($N=32$)', zorder=5)

# 原 Fig.14 的 3 bps/s/Hz 标注；仅用于视觉对齐
axs.annotate('', xy=(26.5, -11.58), xytext=(23.5, -11.58), arrowprops=dict(arrowstyle='<->', lw=1.2))
axs.text(24.2, -11.35, '3 bps/s/Hz', fontname='Times New Roman', fontsize=14)

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
axs.set_xlabel(r'Communication Rate [bps/s/Hz]')
axs.set_ylabel(r'Normalized ELMMSE [dB]')
axs.set_xlim([18, 32])
axs.set_ylim([-15, -9])
axs.set_xticks(np.arange(18, 33, 2))
axs.set_yticks(np.arange(-15, -8, 1))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(18) for label in labels]
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)

plt.savefig('Fig14_JSAC_Python.pdf')
# plt.savefig('Fig14_JSAC_Python.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()









