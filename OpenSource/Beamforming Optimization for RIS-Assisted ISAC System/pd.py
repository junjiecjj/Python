
# https://github.com/PaddiZhou/ISAC_Research/tree/main


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2

def Q1(a, b):
    return 1 - ncx2.cdf(b**2, 2, a**2)

# Define constants
L = 10
PT_10W = 10  # Transmit power 10 W
PT_20W = 20  # Transmit power 20 W
gamma_C_0dB = 1  # 0 dB in linear scale
gamma_T_0dB = 1  # 0 dB in linear scale
gamma_R_10dB = 10  # 10 dB in linear scale
gamma_T_values = [-10, -5, 0, 5]  # gamma_T in dB
gamma_R_dBs = [-10, 10, 20]
PFA_target = 0.01
gamma = -np.log(PFA_target)

# Information rate (in bits per channel use)
rates_1 = np.linspace(0, 3.4, 20)
rates_2 = np.linspace(0, 4.2, 20)

# Functions to calculate PD
def PD_function(PT, rm, gamma_T):
    Pr = PT - (1 / gamma_C_0dB) * (2 ** rm - 1)
    a = np.sqrt(2 * Pr * gamma_T)
    b = np.sqrt(2 * gamma)
    return Q1(a, b)

PD_20W_10dB = {}
# Calculate PD for different gamma_T values
PD_10W_0dB = [PD_function(PT_10W, rm, gamma_T_0dB) for rm in rates_1]

for gamma_T_dB in gamma_T_values:
    gamma_T = 10 ** (gamma_T_dB / 10)
    PD_20W_10dB[gamma_T_dB] = [PD_function(PT_20W, rm, gamma_T) for rm in rates_2]

def monte_carlo_simulation(Pt, gamma_c, gamma, L, gamma_R_dBs, num_iterations=100000):
    rates = np.linspace(0, 3.4, 15)  # Rates to be tested
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
                x_s = sr + n_s

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

rates, pfa_results = monte_carlo_simulation(PT_10W, gamma_C_0dB, gamma, L, gamma_R_dBs)

# Plotting the results
plt.figure(figsize=(12, 6))

# Left plot: L=10, PT=10 W, gamma_T = gamma_C = 0 dB
plt.subplot(1, 2, 1)
for gamma_R_dB in gamma_R_dBs:
    plt.plot(rates, pfa_results[gamma_R_dB],marker='o', label=f'γR = {gamma_R_dB} dB')
plt.plot(rates_1, PD_10W_0dB, label='tho.', color='blue')
plt.xlabel('Information Rate (bits per channel use)')
plt.ylabel('Probability of Detection (PD)')
plt.title('PD vs. Rate (L=10, PT=10 W, γ_T = γ_C = 0 dB)')
plt.grid(True)
plt.legend()

# Right plot: L=10, PT=20 W, γ_R = 10 dB, γ_C = 0 dB
plt.subplot(1, 2, 2)
for gamma_T_dB in gamma_T_values:
    plt.plot(rates_2, PD_20W_10dB[gamma_T_dB], marker='o',label=f'γ_T = {gamma_T_dB} dB')
plt.xlabel('Information Rate (bits per channel use)')
plt.ylabel('Probability of Detection (PD)')
plt.title('PD vs. Rate (L=10, PT=20 W, γ_R = 10 dB, γ_C = 0 dB)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('PD vs Rate.png')
plt.show()




































