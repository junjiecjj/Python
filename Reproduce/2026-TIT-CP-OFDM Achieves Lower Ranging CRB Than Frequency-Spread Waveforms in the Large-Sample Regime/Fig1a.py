"""Reproduce Fig. 1(a) or 1(b) of the CP-OFDM ranging-CRB paper.

This file is standalone. The original paper omits the exact 40 target delays,
complex target amplitudes, and Monte Carlo count. The fixed geometry below is
an explicit illustrative choice, not a claim about the authors' random seed.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from DigiCommPy.modem import PSKModem, QAMModem

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 22

#%%>>>>>>>>>>>>>>>>>>>>>>>>  仿真参数
PANEL = "a"                          # "a": 16-QAM; "b": 16-PSK
MOD_TYPE = "qam"
M = 16
N = 128
L = 40
N1, N2 = 32, 4
c1, c2 = 1 / 16, 1 / 8
B = 160e6                          # Hz
c = 3e8                            # m/s
SNR_dB = np.arange(-10, 7, 1)       # SNR = 1 / sigma^2
Iter = 1000                         # Monte Carlo data-symbol realizations
TARGET_SEED = 4
SYMBOL_SEED = 42
MIN_TARGET_SPACING = 0.8             # illustrative; unspecified by the paper

modem_dict = {'psk': PSKModem, 'qam': QAMModem}
modem = modem_dict[MOD_TYPE.lower()](M)
Es = np.mean(np.abs(modem.modulate(np.arange(M)))**2)  # paper: E|s|^2 = 1

def dft_matrix(length):
    index = np.arange(length)
    return np.exp(-2j * np.pi * np.outer(index, index) / length) / np.sqrt(length)

def delay_crb(J, rhs):
    # Equations (37)-(38): trace of the delay block of J^{-1}.
    try:
        solution = cho_solve(cho_factor(J, lower=True, check_finite=False), rhs, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Singular FIM: this realization has no finite delay CRB") from exc
    value = float(np.trace(solution[:L, :]))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Invalid delay CRB; check the target geometry and FIM")
    return value

def delay_crb1(J, L):
    J_inv = np.linalg.inv(J)
    crb = np.trace(J_inv[:L, :L])
    return float(crb)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  目标几何与调制矩阵
if PANEL != "a" or MOD_TYPE.lower() != "qam" or M != 16 or N1 * N2 != N or N <= 3 * L:
    raise ValueError("Check PANEL, N1*N2=N, and the paper's N>3L condition")

# The paper states random target locations in [0,127], but not their realization or minimum spacing. This fixed, non-degenerate realization gives both standalone scripts precisely the same target geometry.
rng_targets = np.random.default_rng(TARGET_SEED)
free_range = (N - 1) - (L - 1) * MIN_TARGET_SPACING
tau = np.sort(rng_targets.uniform(0, free_range, L))
tau += MIN_TARGET_SPACING * np.arange(L)
beta = np.ones(L, dtype = complex)       # target amplitudes unspecified

index = np.arange(N)
a = np.exp(-2j * np.pi * index[:, None] * tau[None, :] / N)  # (29)
a_dot = (-2j * np.pi * index[:, None] / N) * a               # (30)
H = np.column_stack((a_dot * beta[None, :], a, 1j * a))      # (31)
rhs = np.zeros((3 * L, L))
rhs[:L, :] = np.eye(L)

F = dft_matrix(N)
F1 = dft_matrix(N1)
chirp1 = np.exp(2j * np.pi * c1 * index**2)
chirp2 = np.exp(2j * np.pi * c2 * index**2)

# Q^H s = F_N U s in (28)/(32); U comes from (8a)-(8d).
frequency_map = {
    "SC": F,
    "AFDM": F @ ((chirp1[:, None] * F.conj().T) * chirp2[None, :]),
    "OTFS": F @ np.kron(F1.conj().T, np.eye(N2)),
    "OFDM": np.eye(N, dtype=complex),
}
for name, QH in frequency_map.items():
    if not np.allclose(QH.conj().T @ QH, np.eye(N), atol=1e-12):
        raise RuntimeError(f"{name} modulation matrix is not unitary")

# Jensen bound (40)-(43). E|Q^H s|^2 = 1_N for every unitary waveform.
Jbar = 2 * np.real(H.conj().T @ H)   # sigma^2 = 1, before SNR scaling
jensen_normalized = delay_crb(Jbar, rhs)  # delay_crb1(Jbar, L)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  数据符号蒙特卡洛仿真
np.random.seed(SYMBOL_SEED)
samples = {name: np.empty(Iter) for name in frequency_map}
for trial in range(Iter):
    d = np.random.randint(low = 0, high = M, size = N)
    X = modem.modulate(d)
    s = X / np.sqrt(Es)
    for name, QH in frequency_map.items():
        frequency_symbols = QH @ s
        power = np.abs(frequency_symbols)**2
        # Conditional FIM (32) at sigma^2 = 1. No noise sample is needed.
        J0 = 2 * np.real(H.conj().T @ (power[:, None] * H))
        J0 = (J0 + J0.T) / 2
        samples[name][trial] = delay_crb(J0, rhs) # delay_crb1(J0, L)
    if (trial + 1) % 200 == 0:
        print(f"Fig. 1({PANEL}): {trial + 1}/{Iter} symbol realizations")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  整理全部数值与理论曲线
# J(SNR) = SNR*J0, so CRB(SNR) = CRB(1)/SNR.
# The delay-to-range conversion is [c/(2B)]^2, as stated under Fig. 1.
range_scale = (c / (2 * B))**2
snr_scale = 10**(-SNR_dB / 10)
curves = {name: values.mean() * range_scale * snr_scale for name, values in samples.items()}
jensen = jensen_normalized * range_scale * snr_scale

print(f"N={N}, L={L}, Iter={Iter}, TARGET_SEED={TARGET_SEED}, "
      f"MIN_TARGET_SPACING={MIN_TARGET_SPACING}, beta_l=1")
print("SNR=-10 dB: " + ", ".join(
    f"{name}={curve[0]:.3f} m^2" for name, curve in curves.items())
      + f", Jensen={jensen[0]:.3f} m^2")
if PANEL == "b":
    discrepancy = np.max(np.abs(curves["OFDM"] - jensen))
    print(f"PSK+OFDM versus Jensen: max absolute error = {discrepancy:.3e} m^2")
    if discrepancy > 1e-8:
        raise RuntimeError("Corollary 1 check failed: PSK-OFDM must equal Jensen")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  最后统一画图
fig, axs = plt.subplots(1, 1, figsize = (6, 4), constrained_layout = True)
colors = ['#F65314', '#00A1F1', '#77AC30', '#8A2BE2', '#00A8BB', 'k']
linestyles = ['-', ':', (0, (5, 5)), '-.']
for m, name in enumerate(['SC', 'AFDM', 'OTFS', 'OFDM']):
    axs.plot(SNR_dB, curves[name], color = colors[m], ls = linestyles[m], label = name)
if PANEL == 'a':
    axs.plot(SNR_dB, jensen, color = colors[4], ls = '-', label = 'Jensen Bound')
else:
    axs.plot(SNR_dB, jensen, color = colors[4], ls = 'none', marker = 'x', ms = 8, mew = 1.8, label = 'Jensen Bound')

axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5, which = 'both')
axs.set_xlim(-10, 6)
axs.set_ylim(0, 7)
axs.set_xticks(np.arange(-10, 7, 2))
axs.set_yticks(np.arange(0, 8, 1))
axs.set_xlabel('SNR (dB)')
axs.set_ylabel(r'CRB (m$^2$)')

font1 = FontProperties(family = 'Times New Roman', style = 'normal', size = 20)
legend1 = axs.legend(loc = 'upper right', borderaxespad = 0, edgecolor = 'black', labelspacing = 0.1, columnspacing = 0.5, handlelength = 2.2, prop = font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 1
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)
axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 16, width = bw)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]

output = Path(__file__).with_suffix('.pdf')
fig.savefig(output, )
print(f'Saved {output}')
plt.show()
plt.close()
