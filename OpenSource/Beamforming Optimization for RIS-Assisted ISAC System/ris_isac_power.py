
# https://github.com/PaddiZhou/ISAC_Research/tree/main


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

np.random.seed(42)
# 参数初始化
N = 169
M = 8
K = 4
noise_power_dbm = -80
noise_power_linear = 10 ** (noise_power_dbm / 10)

# 信道和路径损耗参数
d_BS_RIS = 50
d_RIS_User = 4
d_BS_User = 50 + 4 * np.random.randn()
C0_db = -30
D0 = 1
alpha_BR = 2.5
alpha_Ru = 2.5
alpha_Bu = 3.5
beta_BR = 10 ** (3 / 10)  # 基站到RIS的Rician因子
beta_Ru = 10 ** (3 / 10)  # RIS到用户的Rician因子
beta_Bu = 0  # 基站到用户的Rician因子（纯Rayleigh衰落）

def path_loss(d, alpha, C0_db, D0):
    return 10 ** (C0_db / 10) * (d / D0) ** (-alpha)

def rician_channel(N, M, beta, PL):
    LoS_component = np.sqrt(PL) * (np.ones((N, M)) + 1j * np.ones((N, M)))
    NLoS_component = np.sqrt(PL) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
    return np.sqrt(beta / (1 + beta)) * LoS_component + np.sqrt(1 / (1 + beta)) * NLoS_component

# 信道生成
PL_BS_RIS = path_loss(d_BS_RIS, alpha_BR, C0_db, D0)
PL_RIS_User = path_loss(d_RIS_User, alpha_Ru, C0_db, D0)
PL_BS_User = path_loss(d_BS_User, alpha_Bu, C0_db, D0)
G = rician_channel(N, M, beta_BR, PL_BS_RIS)
hr_k = rician_channel(K, N, beta_Ru, PL_RIS_User)
hd_k = rician_channel(K, M, beta_Bu, PL_BS_User)

# 随机相位角度生成和相位向量计算
theta = 2 * np.pi * np.random.rand(N, 1)
phi = np.exp(1j * theta)
Phi_random = np.diag(phi.flatten())
hk = hr_k @ Phi_random @ G + hd_k
iters = 100

# 对功率范围进行循环
def calculate_sum_rate(hk, power_range):
    sum_rate = np.zeros(len(power_range))
    counter = 0
    for power_dbm in power_range:
        P_total = 10 ** (power_dbm / 10)  # dBm to Watts
        Wc_0 = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # Communication signal matrix
        Wr_0 = np.random.randn(M, M) + 1j * np.random.randn(M, M)  # Radar detection signal matrix
        W_combined_0 = np.hstack([Wc_0, Wr_0])  # Concatenate horizontally
        normm = np.linalg.norm(W_combined_0, 'fro')  # Frobenius norm

        W_combined_0 = np.sqrt(P_total) * W_combined_0 / normm  # Normalize

        # Calculate received signals and SINRs for each user
        rec_signal = np.zeros(K, dtype=complex)
        c = np.zeros(K, dtype=complex)
        g = np.zeros(K, dtype=complex)

        for _ in range(iters):
            for k in range(K):

                rec_signal[k] = hk[k, :] @ W_combined_0[:, k]  # Received signal for user k

                signal_power = np.abs(rec_signal[k]) ** 2  # Signal power for user k

                other_signal_power = 0  # Initialize interference power

                for j in range(K):
                    if j != k:
                        denum_c = hk[k, :] @ W_combined_0[:, j]
                        other_signal_power += np.abs(denum_c) ** 2  # Accumulate interference power

                c[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k
                g[k] = (np.sqrt(1 + c[k]) * rec_signal[k]) / (
                            signal_power + other_signal_power + noise_power_linear)  # Calculate optimal combining coefficient for user k

            Wc = cp.Variable((M, K), complex=True)
            Wr = cp.Variable((M, M), complex=True)  # Radar detection signal matrix
            W_combined = cp.hstack([Wc, Wr])

            objective = 0
            for k in range(K):
                total_power = 0
                for j in range(K):
                    signal_j = hk[k, :] @ W_combined[:, j]  # Signal for each user and radar
                    total_power += cp.square(cp.abs(signal_j))  # Total power calculation

                objective = 0
                for k in range(K):
                    total_power = 0
                    for j in range(K):
                        signal_j = hk[k, :] @ W_combined[:, j]  # Signal for each user and radar
                        total_power += cp.square(cp.abs(signal_j))  # Total power calculation

                    objective += 2 * np.sqrt(1 + c[k].real) * cp.real(
                        cp.conj(g[k]) * (hk[k, :] @ W_combined[:, k])) - cp.square(
                        cp.abs(g[k])) * total_power  # Objective function: Maximize SINR-based utility

            constraints = [cp.sum_squares(cp.vec(W_combined)) <= np.sqrt(P_total)]
            prob = cp.Problem(cp.Maximize(objective), constraints)
            prob.solve()
            W_combined_0 = W_combined.value

        SNR = np.zeros(K)
        for k in range(K):
            rec_signal[k] = hk[k, :] @ W_combined_0[:, k]  # Received signal for user k
            signal_power = np.abs(rec_signal[k]) ** 2  # Signal power for user k
            other_signal_power = 0  # Initialize interference power
            for j in range(K):
                if j != k:
                    denum_c = hk[k, :] @ W_combined_0[:, j]
                    other_signal_power += np.abs(denum_c) ** 2  # Accumulate interference power

            SNR[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k

        sum_rate[counter] = np.sum(np.log2(1 + SNR))
        counter += 1
    return sum_rate

power_range = np.linspace(10, 20, 6)  # Power range in dBm from 10 to 20

# Calculate sum rate for the first scenario
sum_rate_scenario_1 = calculate_sum_rate(hk, power_range)

# Second scenario: hk = hd_k
hk = hd_k

# Calculate sum rate for the second scenario
sum_rate_scenario_2 = calculate_sum_rate(hk, power_range)

# First scenario: hk = hr_k @ Phi_random @ G + hd_k
hk = hr_k @ Phi_random @ G + hd_k
hk = hk.T

# Function to calculate sum rate
def calculate_sum_rate_c(hk, power_range):
    sum_rate = np.zeros(len(power_range))
    counter = 0
    for power_dbm in power_range:
        P_total = 10 ** (power_dbm / 10)  # dBm to Watts
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # Communication signal matrix
        normm = np.linalg.norm(W, 'fro')  # Frobenius norm
        W = np.sqrt(P_total) * W / normm  # Normalize

        for _ in range(iters):
            c = np.zeros(K, dtype=complex)
            g = np.zeros(K, dtype=complex)

            for k in range(K):
                rec_signal_k = hk[:, k].conj().T @ W[:, k]  # Received signal for user k
                signal_power = np.abs(rec_signal_k)**2  # Signal power for user k

                other_signal_power = 0  # Initialize interference power
                for j in range(K):
                    if j != k:
                        denum_c = hk[:, k].conj().T @ W[:, j]
                        other_signal_power += np.abs(denum_c)**2  # Accumulate interference power

                c[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k
                g[k] = (np.sqrt(1 + np.abs(c[k])) * rec_signal_k) / (signal_power + other_signal_power + noise_power_linear)  # Calculate optimal combining coefficient for user k

            # CVX Optimization to Maximize the Objective Function
            Wc = cp.Variable((M, K), complex=True)  # Communication signal matrix
            objective = 0

            for k in range(K):
                total_power = 0
                for j in range(K):
                    signal_j = hk[:, k].conj().T @ Wc[:, j]  # Signal for each user and radar
                    total_power += cp.square(cp.abs(signal_j))  # Total power calculation

                objective += 2 * np.sqrt(1 + np.abs(c[k])) * cp.real(cp.conj(g[k]) * hk[:, k].conj().T @ Wc[:, k]) - cp.square(cp.abs(g[k])) * total_power

            prob = cp.Problem(cp.Maximize(objective), [cp.sum_squares(cp.vec(Wc)) <= P_total])
            prob.solve(solver=cp.MOSEK)

            W = Wc.value

        # Store results for each power level
        SNR = np.zeros(K)
        for k in range(K):
            rec_signal_k = hk[:, k].conj().T @ W[:, k]  # Received signal for user k
            signal_power = np.abs(rec_signal_k)**2  # Signal power for user k

            other_signal_power = 0  # Initialize interference power
            for j in range(K):
                if j != k:
                    denum_c = hk[:, k].conj().T @ W[:, j]
                    other_signal_power += np.abs(denum_c)**2  # Accumulate interference power

            SNR[k] = signal_power / (other_signal_power + noise_power_linear)  # Calculate SINR for user k

        sum_rate[counter] = np.sum(np.log2(1 + SNR))
        counter += 1
    return sum_rate

power_range = np.linspace(10, 20, 6)  # Power range in dBm from 10 to 20

# Calculate sum rate for the first scenario
sum_rate_scenario_3 = calculate_sum_rate_c(hk, power_range)

# Second scenario: hk = hd_k
hk = hd_k.T

# Calculate sum rate for the second scenario
sum_rate_scenario_4 = calculate_sum_rate_c(hk, power_range)

# Plot both scenarios on the same graph
plt.plot(power_range, sum_rate_scenario_1, label=' With Random RIS ')
plt.plot(power_range, sum_rate_scenario_2, label=' Without RIS ', linestyle='--')
plt.plot(power_range, sum_rate_scenario_3, label=' Comm-only with Random RIS ')
plt.plot(power_range, sum_rate_scenario_4, label=' Comm-only Without RIS ', linestyle='--')
plt.grid(True)
plt.xlabel('Power (dBm)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Sum Rate vs. Power')
plt.xlim(10, 20)
plt.ylim(2, 32)
plt.xticks(range(10, 21, 2))
plt.yticks(range(2, 33, 6))
plt.legend()
plt.show()





















































