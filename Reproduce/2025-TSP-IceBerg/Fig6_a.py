#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 6(a) reproduction for
"Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and
Modulation Design for Random ISAC Signals".

The programming logic follows Fig2.py. The RRC pulse is replaced by the
PSL-designed pulse obtained from (44)--(49), as in Fig. 6(d).
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from Modulations import modulator


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["lines.linestyle"] = "-"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.color"] = "blue"
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["legend.fontsize"] = 14

np.random.seed(42)


def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    for i in range(L):
        for j in range(L):
            mat[i, j] = np.exp(-1j*2*np.pi*i*j/L)/np.sqrt(L)
    return mat


def solve_iceberg_shaping_psl(N, L, alpha, K_s1):
    """PSL iceberg-shaping problem in (44) and (49)."""

    N_alpha = int(alpha*N)
    N_non_rolloff = N-N_alpha
    N_zeros = N_non_rolloff//2
    N_ones = N_non_rolloff//2

    g = cp.Variable(N, nonneg=True)

    constraints = []

    # Reversed FFT-bin ordering used by the numerical implementation:
    # 1 -> monotonically decreasing roll-off -> 0.
    constraints.append(g[0:N_zeros] == 1)
    constraints.append(g[N-N_ones:N] == 0)

    for n in range(N-1):
        constraints.append(g[n+1]-g[n] <= 0)

    constraints.append(cp.sum(g) == N/2)

    psl_terms = []

    for k in K_s1:
        f_k = np.exp(-1j*2*np.pi*k*np.arange(N)/(L*N))
        gk = g+(1-g)*np.exp(-1j*2*np.pi*k/L)
        psl_terms.append(cp.abs(f_k.conj().T @ gk)**2)

    objective = cp.Minimize(cp.max(cp.hstack(psl_terms)))
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("Optimization successful!")
        print(f"Optimal PSL value: {problem.value}")
        return g.value

    raise RuntimeError(f"Optimization failed, status: {problem.status}")


# %% Parameters in Fig. 6(a)
Tsym = 1
pi = np.pi
N = 128
L = 10
alpha = 0.35
kappa = 1.32

FLN = FFTmatrix(L*N)
FN = FFTmatrix(N)


# %% Generate the optimized pulse according to Fig. 6(d)
# The paper interval [5,15] uses symbol-delay units. The discrete ACF is
# sampled at Ts=T/L, and therefore uses k=5L,...,15L.
K_s1 = np.arange(5*L, 15*L + 1)

gN = solve_iceberg_shaping_psl(N, L, alpha, K_s1)

print("Optimal spectrum coefficients:")
print(gN)

g_N = 1-gN
g_design = np.hstack((gN, np.zeros((L-2)*N), g_N))

# g_design=N*|F_{LN}p|^2. Choose zero spectral phase to recover one
# valid pulse with exactly the optimized squared spectrum.
P_spectrum = np.sqrt(np.maximum(np.real(g_design), 0)/N)
p = FLN.conj().T @ P_spectrum
p = p/np.sqrt(np.sum(np.abs(p)**2))

norm2p = np.linalg.norm(p)
g = N*(FLN @ p)*(FLN.conj() @ p.conj())
g = np.real(g)

pulseSpectrumError = np.linalg.norm(g-g_design)/max(
    np.linalg.norm(g_design), np.finfo(float).eps
)
print(
    "Relative squared-spectrum reconstruction error: "
    f"{pulseSpectrumError:.3e}"
)


# %% OFDM theoretical average squared ACF, Eq. (36)
U = FN.conj().T
V = np.eye(N)
tilde_V = V*V.conj()

TheoAveACF_Iceberg = np.zeros(L*N)
TheoAveACF_OFDM_M1 = np.zeros(L*N)
TheoAveACF_OFDM_M10000 = np.zeros(L*N)

for k in range(L*N):
    gk = g[:N]+(1-g[:N])*np.exp(-1j*2*pi*k/L)
    f_k = np.exp(-1j*2*pi*k*np.arange(N)/(L*N))

    r1 = np.abs(gk @ f_k.conj())**2
    TheoAveACF_Iceberg[k] = r1

    M = 1
    r2 = (kappa-1)/M*(N - 2*(1-np.cos(2*pi*k/L))*np.sum(g[:N]*(1-g[:N])))
    TheoAveACF_OFDM_M1[k] = r1+r2

    M = 10000
    r2 = (kappa-1)/M*(N - 2*(1-np.cos(2*pi*k/L))*np.sum(g[:N]*(1-g[:N])))
    TheoAveACF_OFDM_M10000[k] = r1+r2

TheoAveACF_Iceberg = (TheoAveACF_Iceberg/TheoAveACF_Iceberg.max()+1e-14)
TheoAveACF_Iceberg = np.fft.fftshift(TheoAveACF_Iceberg)

TheoAveACF_OFDM_M1 = (TheoAveACF_OFDM_M1/TheoAveACF_OFDM_M1.max()+1e-14)
TheoAveACF_OFDM_M1 = np.fft.fftshift(TheoAveACF_OFDM_M1)

TheoAveACF_OFDM_M10000 = (TheoAveACF_OFDM_M10000/TheoAveACF_OFDM_M10000.max()+1e-14)
TheoAveACF_OFDM_M10000 = np.fft.fftshift(TheoAveACF_OFDM_M10000)


# %% Numerical average squared ACF, inherited from Fig2.py Eq. (26)
MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)

# Keep the full three-dimensional coherent-integration array. Fig. 6(a)
# displays one numerical realization of the 10,000-slot coherent result.
Iter = 100


# %% No coherent integration: M=1
M = 1
SimAveACF_OFDM_M1 = np.zeros((Iter, L*N))

for k in range(L*N):
    fk = FLN[:, k]
    fk_tilde = fk[:N]
    gk = g[:N]+(1-g[:N])*np.exp(-1j*2*pi*k/L)

    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        VHs = np.abs(V.conj().T @ s)**2
        SimAveACF_OFDM_M1[it, k] = np.abs(np.sum(gk*VHs*fk_tilde.conj()))**2


Sim_M1_avg = SimAveACF_OFDM_M1.mean(axis=0)
Sim_M1_avg = Sim_M1_avg/Sim_M1_avg.max()+1e-14
Sim_M1_avg = np.fft.fftshift(Sim_M1_avg)


# %% 10,000 coherent integrations: M=10000
M = 10000
SimAveACF_OFDM_M10000 = np.zeros((M, Iter, L*N), dtype=complex)

for k in range(L*N):
    fk = FLN[:, k]
    fk_tilde = fk[:N]
    gk = g[:N]+(1-g[:N])*np.exp(-1j*2*pi*k/L)

    for m in range(M):
        for it in range(Iter):
            d = np.random.randint(Order, size=N)
            s = Constellation[d]
            VHs = np.abs(V.conj().T @ s)**2

            # Do not square here. Average the complex ACF over M first.
            SimAveACF_OFDM_M10000[m, it, k] = np.sum(gk*VHs*fk_tilde.conj())


# %% Coherent averaging, Eq. (33)
RkBar = SimAveACF_OFDM_M10000.mean(axis=0)
RkBar2 = np.abs(RkBar)**2

Sim_M10000_avg = RkBar2.mean(axis=0)
Sim_M10000_avg = Sim_M10000_avg/Sim_M10000_avg.max()+1e-14
Sim_M10000_avg = np.fft.fftshift(Sim_M10000_avg)


# %% Plot Fig. 6(a)
x = np.arange(-N*L//2, N*L//2)

fig, axs = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

axs.plot( x, 10*np.log10(Sim_M1_avg), color="tab:blue", linestyle="-", linewidth=1.2, label="No Integration, Numerical" )
axs.plot( x, 10*np.log10(TheoAveACF_OFDM_M1), color="tab:blue", linestyle=":", linewidth=1.8, label="No Integration, Theoretical" )

axs.plot(x, 10*np.log10(Sim_M10000_avg), color="tab:orange", linestyle="-", linewidth=1.2, label="10k Coh Integration, Numerical")
axs.plot(x, 10*np.log10(TheoAveACF_OFDM_M10000), color="tab:orange", linestyle=":", linewidth=1.8, label="10k Coh Integration, Theoretical" )

axs.plot(x, 10*np.log10(TheoAveACF_Iceberg), color="black", linestyle="--", linewidth=1.2, label='"Iceberg" of the Designed Pulse')

axs.set_xlabel("Delay Index")
axs.set_ylabel("Ambiguity Level (dB)")
axs.set_xlim([-300, 300])
axs.set_ylim([-120, 0])
axs.set_xticks(np.arange(-300, 301, 100))
axs.set_yticks(np.arange(-120, 1, 20))
axs.grid(True)
axs.legend(loc="lower center", edgecolor="black")

out_fig = plt.gcf()
# out_fig.savefig("Fig6_a.png", dpi=300)
out_fig.savefig("Fig6_a.pdf")
plt.show()
plt.close()




