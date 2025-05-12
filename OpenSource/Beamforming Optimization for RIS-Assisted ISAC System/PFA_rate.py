
# https://github.com/PaddiZhou/ISAC_Research/tree/main



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func, hyp1f1
import math

def monte_carlo_simulation(Pt, gamma_c, gamma, L, gamma_R_dBs, num_iterations=1000):
    rates = np.linspace(0, 7.6, 30)  # Rates to be tested
    pfa_results = {gamma_R_dB: [] for gamma_R_dB in gamma_R_dBs}
    sigma = 1  # Noise power

    for gamma_R_dB in gamma_R_dBs:
        gamma_R = 10**(gamma_R_dB / 10)
        for rate in rates:
            Pc = (1 / gamma_c) * (2**rate - 1)
            Pr = Pt - Pc
            gammad = np.sqrt(gamma_R * sigma)  # gammad adjusted with noise power

            false_alarms = 0
            for _ in range(num_iterations):
                sr  = np.sqrt(Pr) * np.ones(L) / np.sqrt(L) # Reference signal
                n_d = np.sqrt(sigma) * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)  # 参考通道噪声
                n_s = np.sqrt(sigma) * (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2)  # 监视通道噪声
                x_d = gammad * sr + n_d
                x_s = n_s

                # Compute A
                A = np.outer(x_d, x_d.conj()) + np.outer(x_s, x_s.conj())

                # Find the largest eigenvalue of A
                lambda_max_A = np.linalg.eigvalsh(A).max()

                # GLRT Calculation using the provided formula
                glrt_stat = (lambda_max_A - np.linalg.norm(x_d)**2) / sigma
                if glrt_stat > gamma:
                    false_alarms += 1
            pfa = false_alarms / num_iterations
            pfa_results[gamma_R_dB].append(pfa)

    return rates, pfa_results

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


# Parameters
Pt = 20  # Total power in Watts
gamma_c_dB = 10  # SNR at communication receiver in dB
gamma_c = 10**(gamma_c_dB / 10)
gamma = 5  # Detection threshold

L = 10  # Number of samples
gamma_R_dBs = [-10, 10, 20]  # SNR values in dB

gamma_C = 10  # 通信接收器的SNR (dB)
gamma_R_values = [-10, 10, 20]  # 不同的直达路径SNR (dB)
# Run simulation
rates, pfa_results = monte_carlo_simulation(Pt, gamma_c, gamma, L, gamma_R_dBs)

# Plot results
plt.figure(figsize=(10, 6))
for gamma_R_dB in gamma_R_dBs:
    plt.plot(rates, pfa_results[gamma_R_dB],marker='o', label=f'γR = {gamma_R_dB} dB')

pfa_rate_values = {}
rate_values = np.linspace(0, 7.6, 30)  # 速率范围 (bpcu)
for gamma_R in gamma_R_values:
    pfa_rate_values[gamma_R] = []
    for rate in rate_values:
        PR = Pt - (1 / gamma_C) * (2 ** rate - 1)

        alpha_R = 2 * (10 ** (gamma_R / 10)) * PR
        pfa_rate_values[gamma_R].append(calculate_pfa(5, alpha_R, L))

# 绘制PFA vs Rate图
for gamma_R in gamma_R_values:
    plt.plot(rate_values, pfa_rate_values[gamma_R], label=f'theo.γ_R={gamma_R} dB')

plt.xlabel('Rate')
plt.ylabel('Probability of False Alarm (PFA)')
plt.title('PFA vs Rate')
plt.yscale('log')
plt.yticks([1, 0.1, 0.01], [r'$10^0$', r'$10^{-1}$', r'$10^{-2}$'])
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('PFA vs Rate test with max .png')
plt.show()
































































