

# https://github.com/PaddiZhou/ISAC_Research/tree/main


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func, hyp1f1
import math
# Constants
L = 10  # number of samples
sigma2 = 1  # noise variance
gamma_values = np.linspace(0, 10, 25)  # gamma值范围
gamma_R_values = [-10, 10, 20]         # 不同的直达路径SNR (dB)
gamma_dB = 5                           # 固定的检测阈值
gamma_C_dB = 10                        # 通信接收器的SNR (dB)
gamma_C = 10 ** (gamma_C_dB / 10)      # 将dB转换为线性刻度
# PT = 20
rate_values = np.linspace(0.1, 7.5, 30)  # 速率范围 (bpcu)
# DSNR values in dB and their linear scale
DSNR_dB = np.array([-10, 10, 20])
DSNR = 10 ** (DSNR_dB / 10)

# Number of Monte Carlo simulations
num_simulations = 10000

# Gamma range for PFA calculation
gamma_range = np.linspace(0, 10, 50)
# 计算假报警概率PFA的函数
def calculate_pfa(gamma, alpha_R, L):
    term1 = np.exp(-gamma)
    term2 = (2 * gamma * np.exp(-(gamma + alpha_R / 2))) / (2 ** L * gamma_func(L))
    sum_term1 = 0
    for k in range(L - 1):
        for p in range(k + 1):
            sum_term1 += (math.comb(k, p) * 2 ** (L - 1) * gamma ** (k - p) * gamma_func(L + p - k - 1) * hyp1f1(L + p - k - 1, L, alpha_R / 2))
    sum_term2 = 0
    for k in range(L - 1):
        for l in range(k + 1):
            for p in range(k + 1):
                sum_term2 += (math.comb(k, p) * 2 ** (k - p - l) * gamma ** (k - p) / gamma_func(l + 1) * gamma_func(L + l + p - k - 1) * hyp1f1(L + l + p - k - 1, L, alpha_R / 4))
    pfa = term1 + term2 * (sum_term1 - sum_term2)
    return pfa

# 蒙特卡洛模拟函数
def monte_carlo_simulation(dsnr, gamma_range, num_simulations, L, sigma2):
    PFA_values = []
    for gamma in gamma_range:
        false_alarms = 0
        for _ in range(num_simulations):
            # 生成参考信号和监视信号
            s_r = np.sqrt(dsnr) * np.ones(L) / np.sqrt(L) # 雷达波形
            n_d = np.sqrt(sigma2) * (np.random.randn(L)+1j*np.random.randn(L)) / np.sqrt(2)  # 参考通道噪声
            n_s = np.sqrt(sigma2) * (np.random.randn(L)+1j*np.random.randn(L)) / np.sqrt(2) # 监视通道噪声

            x_d = s_r + n_d
            x_s = n_s
            # 计算A矩阵
            A = np.outer(x_d, x_d.conj()) + np.outer(x_s, x_s.conj())
            # 计算最大特征值
            lambda_max = np.linalg.eigvalsh(A).max()

            # 计算GLRT统计量
            glrt_statistic = (lambda_max - np.linalg.norm(x_d) ** 2) / sigma2

            # 判断是否产生误警
            if glrt_statistic >= gamma:
                false_alarms += 1

        # 计算PFA
        PFA = false_alarms / num_simulations
        PFA_values.append(PFA)

    return PFA_values

# Perform simulations
results = {}
for dsnr in DSNR:
    results[dsnr] = monte_carlo_simulation(dsnr, gamma_range, num_simulations, L, sigma2)

pfa_gamma_values = {}
for gamma_R in gamma_R_values:
    alpha_R = 2*(10 ** (gamma_R / 10))  # 将dB转换为线性刻度
    pfa_gamma_values[gamma_R] = [calculate_pfa(gamma, alpha_R, L) for gamma in gamma_values]

# Plotting the results
plt.figure(figsize=(10, 6))
for dsnr in DSNR:
    plt.plot(gamma_range, results[dsnr],marker='o', label=f'DSNR = {10 * np.log10(dsnr)} dB')

for gamma_R in gamma_R_values:
    plt.plot(gamma_values, pfa_gamma_values[gamma_R], linestyle='-', label=f'Theo., D-SNR={gamma_R} dB')

approx_pfa_values = np.exp(-gamma_values)  # 修正后的高SNR近似
plt.plot(gamma_values, approx_pfa_values, linestyle='-', marker='>', label='High D-SNR approx.')

plt.xlabel('Gamma')
plt.ylabel('Probability of False Alarm (PFA)')
plt.title('PFA vs Gamma (L=10)')
plt.legend()
plt.grid(True)
plt.savefig('PFA vs Gamma (L=10).png')
plt.show()

##  2
# results_2 = {}
# for gamma_R in gamma_R_values :
#     results_2[gamma_R]={}
#     for rate in rate_values :
#         results_2[gamma_R] = monte_carlo_simulation(gamma_R, 5, num_simulations, L, sigma2, rate)

# plt.figure(figsize=(10, 6))
# for gamma_R in gamma_R_values:
#     plt.plot(rate_values, results_2[gamma_R], marker='o', label=f'γ_R={gamma_R} dB')

# plt.yscale('log')  # 使用对数刻度
# plt.ylim(1e-3, 1e1)  # 设置y轴范围
# plt.yticks([1e-3, 1e-2, 1e-1, 1, 1e1], ['0.001', '0.01', '0.1', '1', '10'])  # 设置y轴刻度
# plt.xlabel('Rate (bpcu)')
# plt.ylabel('PFA')
# plt.title('PFA vs Rate (PT = 20 W, γ_C = 10 dB, γ = 5, L = 10)')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.show()

