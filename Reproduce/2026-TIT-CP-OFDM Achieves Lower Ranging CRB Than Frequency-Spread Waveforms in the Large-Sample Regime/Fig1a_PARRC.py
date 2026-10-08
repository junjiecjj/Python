"""Fig. 1(a) 加入插零上采样和 RRC 成形后的 16-QAM 测距 CRB。

模型依据：刘凡等 IT 论文 (8)、(27)–(32)、(36)–(43)，以及项目书的
    x = P A U s,  z = F_M x,  M = L N。
这里采用项目书的 M 点周期卷积和周期分数时延模型；循环前缀及保护间隔
须覆盖成形脉冲的有效记忆和最大目标时延。未另加接收匹配滤波器。

依赖：numpy、scipy、matplotlib、commpy、DigiCommPy。
运行：python Fig1a_PA_RRC.py
"""

#%% 导入与绘图风格
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy.filters
from DigiCommPy.modem import PSKModem, QAMModem

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 18


#%% 上采样矩阵：沿用给定的插零定义
def upsamplingMatrix(M, Q):
    GammaQ = np.zeros((Q * M, M))
    for m in range(M):
        GammaQ[m * Q, m] = 1
    return GammaQ


#%% 从完整的实参数 FIM 直接求逆，提取时延 CRB
def delay_crb_direct(J, n_targets):
    """J 的参数顺序：[tau_1..tau_K, Re(beta_1).., Im(beta_1)..]。"""
    J = 0.5 * (J + J.T)             # 去掉浮点乘法产生的非对称误差
    return float(np.trace(np.linalg.inv(J)[:n_targets, :n_targets]))


#%% 参数：原文 Fig. 1(a) 的 16-QAM、N=128、40 个目标和四种调制
N = 128
K = 40                             # 目标数，原文记 L；与这里的过采样率区分
L = 10                             # 用户给定的过采样率
M = L * N                          # CP 去除后的有效采样点数
Tsym = 1.0
alpha = 0.3
span = 6                          # 供短时域 srrcFunction 接口使用；这里选用全长 commpy RRC
MOD_TYPE = 'qam'
MOD_ORDER = 16
N1, N2 = 32, 4
c1, c2 = 1 / 16, 1 / 8
SNR_dB = np.arange(-10, 7, 1)
N_FRAMES = 1000                    # 对随机数据先求逆再平均；需要快速检查可调小
TARGET_SEED = 4
SYMBOL_SEED = 42
MIN_TARGET_SPACING = 0.8           # 原复现脚本的目标最小间隔，单位：原始符号间隔
B = 160e6                         # 原文用于换算物理距离的采样率
c_light = 3e8


#%% 构造原文的 U；所有 DFT 都是酉变换
def dft_matrix(length):
    idx = np.arange(length)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / length) / np.sqrt(length)


F_N = dft_matrix(N)
F_N1 = dft_matrix(N1)
indices = np.arange(N)
chirp1 = np.exp(-2j * np.pi * c1 * indices**2)
chirp2 = np.exp(-2j * np.pi * c2 * indices**2)

U_dict = {
    'SC': np.eye(N, dtype=complex),
    'AFDM': (chirp1[:, None] * F_N.conj().T) * chirp2[None, :],
    'OTFS': np.kron(F_N1.conj().T, np.eye(N2)),
    'OFDM': F_N.conj().T,
}


#%% x = P A U s：P 为长度 M 的循环卷积矩阵
A = upsamplingMatrix(N, L)

# 用户给定的第二种生成方式；p 是离散采样的全长 RRC，单位能量。
t, p = commpy.filters.rrcosfilter(L * N, alpha, Tsym, L / Tsym)
p = np.asarray(p, dtype=float)
p = p / np.sqrt(np.sum(np.abs(p)**2))
P = scipy.linalg.circulant(p)       # P @ v = np.convolve(p,v) 的 M 点循环版本
PA = P @ A                         # 一次预计算，主循环用 PA @ (U @ s)

# 在每个 U 下预计算 Φ^H = F_M P A U，避免重复构建大矩阵。
# scipy.fft.fft(..., norm='ortho') 与酉 DFT F_M 完全相同。
PhiH_dict = {
    name: scipy.fft.fft(PA @ U, axis=0, norm='ortho')
    for name, U in U_dict.items()
}


#%% 目标几何：保持原 Fig. 1(a) 的物理位置，时延换为过采样点单位
rng_target = np.random.default_rng(TARGET_SEED)
free_range = (N - 1) - (K - 1) * MIN_TARGET_SPACING
if free_range <= 0:
    raise ValueError('目标间距与数量之和超过原 Fig. 1(a) 的时延窗口')
tau_symbol = (np.sort(rng_target.uniform(0.0, free_range, K))
              + MIN_TARGET_SPACING * np.arange(K))
tau = L * tau_symbol               # 项目书约定：tau 以新采样间隔为单位
beta = np.ones(K, dtype=complex)

# 论文 (29)-(32) 的 H 改成 M 个频率点；对 tau 求导时分母必须是 M。
freq_idx = np.arange(M)
a = np.exp(-2j * np.pi * np.outer(freq_idx, tau) / M)
a_dot = (-2j * np.pi * freq_idx[:, None] / M) * a
H = np.column_stack((a_dot * beta[None, :], a, 1j * a))


#%% DigiCommPy 生成 16-QAM 星座，并满足原文 E|s_n|^2 = 1
modem_dict = {'psk': PSKModem, 'qam': QAMModem}
modem = modem_dict[MOD_TYPE](MOD_ORDER)
alphabet = np.asarray(modem.modulate(np.arange(MOD_ORDER)), dtype=complex)
symbol_energy = float(np.mean(np.abs(alphabet)**2))

# E_s[J(s)] 的频谱权重：w_bar[k] = E|[F_M P A U s]_k|^2。
# 当 E[ss^H]=I 且 U 酉时，它与 U 无关，但已经不是原文无脉冲的全 1。
mean_spectrum = np.sum(np.abs(PhiH_dict['SC'])**2, axis=1)
for name, PhiH in PhiH_dict.items():
    np.testing.assert_allclose(
        np.sum(np.abs(PhiH)**2, axis=1), mean_spectrum,
        rtol=1e-10, atol=1e-10, err_msg=f'{name} 的 U 应当为酉矩阵'
    )

J_bar = 2 * np.real(H.conj().T @ (mean_spectrum[:, None] * H))
jensen_tau_trace = delay_crb_direct(J_bar, K)


#%% 给定数据帧：J(s)=2 Re{H^H diag(|Φ^H s|²) H}，直接逆再平均
rng_symbols = np.random.default_rng(SYMBOL_SEED)
crb_sum = {name: 0.0 for name in U_dict}

for frame in range(N_FRAMES):
    d = rng_symbols.integers(low=0, high=MOD_ORDER, size=N)
    s = np.asarray(modem.modulate(d), dtype=complex) / np.sqrt(symbol_energy)

    for name, PhiH in PhiH_dict.items():
        z = PhiH @ s               # z = F_M (P A U s)
        spectrum = np.abs(z)**2
        J = 2 * np.real(H.conj().T @ (spectrum[:, None] * H))
        crb_sum[name] += delay_crb_direct(J, K)

    if (frame + 1) % 100 == 0 or frame + 1 == N_FRAMES:
        print(f'已完成 {frame + 1}/{N_FRAMES} 帧')


#%% SNR=1/sigma_n^2；tau 单位为上采样点，距离换算因子必须同时除以 L
# 原论文每个原始采样点 0.9375 m；这里每个新采样点 0.09375 m。
range_per_sample = c_light / (2 * B * L)
scale = (range_per_sample**2) * 10**(-SNR_dB / 10)
CRB = {name: (crb_sum[name] / N_FRAMES) * scale for name in U_dict}
Jensen = jensen_tau_trace * scale

print(f'上采样后：M={M}, 每个新采样点 {range_per_sample:.5f} m')
print(f'参考脉冲能量：{np.sum(np.abs(p)**2):.8f}')
for name in U_dict:
    print(f'{name:5s}，SNR=0 dB 时 CRB = '
          f'{(crb_sum[name] / N_FRAMES) * range_per_sample**2:.6g} m²')


#%% 统一绘图，风格沿用用户给定的 matplotlib 示例
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
colors = ['#F65314', '#00A1F1', '#77AC30', '#8A2BE2', '#00A8BB', 'k']
linestyles = ['-', ':', (0, (5, 5)), '-.']
for m, name in enumerate(['SC', 'AFDM', 'OTFS', 'OFDM']):
    axs.plot(SNR_dB, CRB[name], color=colors[m], ls=linestyles[m], label=name)
axs.plot(SNR_dB, Jensen, color=colors[4], ls='-', label='Jensen Bound')

axs.set_xlabel('SNR (dB)')
axs.set_ylabel(r'CRB (m$^2$)')
axs.set_xlim(SNR_dB[0], SNR_dB[-1])
axs.set_ylim(bottom=0)
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5, which='both')

font1 = FontProperties(family='Times New Roman', style='normal', size=17)
legend1 = axs.legend(loc='upper right', edgecolor='black',
                     labelspacing=0.2, prop=font1)
legend1.get_frame().set_alpha(1)
legend1.get_frame().set_facecolor('none')

bw = 2
for spine in axs.spines.values():
    spine.set_linewidth(bw)
axs.tick_params(direction='in', axis='both', top=True, right=True,
                labelsize=18, width=bw)
for label in axs.get_xticklabels() + axs.get_yticklabels():
    label.set_fontname('Times New Roman')

out_fig = Path(__file__).with_suffix('.pdf')
fig.savefig(out_fig, bbox_inches='tight')
print(f'已保存：{out_fig}')
plt.show()
plt.close(fig)
