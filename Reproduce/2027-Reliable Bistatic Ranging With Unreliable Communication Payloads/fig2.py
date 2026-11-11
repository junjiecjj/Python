"""Fig.2：Reliable Bistatic Ranging With Unreliable Communication Payloads。
将用户原来的 Modulations.py 放在同一目录；依赖 numpy、matplotlib、scikit-commpy。
参数和主流程均在顶层。公式编号指 arXiv:2611.07803v1。
目标位置由PDF矢量曲线读取；功率和整体尺度由图中估计；相位和MC次数为补充设置。
因此本程序是根据原图重构参数的仿真，不是作者原始数据的逐点重现。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Modulations import modulator

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["lines.linestyle"] = "-"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
np.random.seed(42)

# ==================== 参数：直接修改 ====================
N = 3276  # [原文Table I] 子载波数。
L = 288  # [原文Table I] CIR支撑长度。
SNRdB = 3  # [原文Fig.2]。
sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] SNR=1/sigma^2。
sigma_p2 = sigma2  # [原文Sec.IV] 导频和数据采用相同SNR。
M_values = [30, 300]  # [原文Fig.2] 两种积累长度。
M_max = max(M_values)
MonteCarlo = 64  # [补充] 独立试验次数，原文未公布。
SHOW_FIGURES = True
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig2_results")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "psk"
QAM_mod = 2  # Fig.2采用BPSK，即2-PSK；这里是星座阶数，不是比特数。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
if not np.allclose(np.abs(Constellation), 1):
    raise ValueError("本程序使用BPSK恒模条件下的原文(14)，不能直接改成非恒模QAM。")

# 原图横轴从1开始；模型g[k]和Python数组从0开始。
target_indices = np.array([15, 34, 35, 145, 254, 282])  # [图中读取] 34/35相邻。
target_bins = target_indices - 1
target_power_dB = np.array([-0.277667, 0.0, -31.093638, -38.628508, -37.993070, -19.071398])  # [图中估计] 相对功率。
CIR_PEAK_GAIN = 1.0870961165435937  # [图中反推] 由完美载荷M=300背景和sigma2/(M+1)估计，非原文参数。
target_phase = np.zeros(len(target_indices))  # [补充] 原文未公开相位，先取零；可直接修改。
if np.any(target_bins < 0) or np.any(target_bins >= L):
    raise ValueError("目标时延超出L；修改L时需同步检查target_indices。")
g = np.zeros(L, dtype=complex)
g[target_bins] = np.sqrt(N)*CIR_PEAK_GAIN*10**(target_power_dB/20)*np.exp(1j*target_phase)  # [原文(1)]。
h = np.fft.fft(g, n=N, norm='ortho')  # [原文(2)(3)] h=F_L g；g在尾部补零。
range_axis = np.arange(1, L+1)
non_target_mask = np.ones(L, dtype=bool)
non_target_mask[target_bins] = False

# ==================== 主流程：每次试验共用一个导频和300列数据 ====================
P_pilot = np.zeros(L)
P_hard = np.zeros((len(M_values), L))
P_known = np.zeros((len(M_values), L))
for mc in range(MonteCarlo):
    pilot_bits = np.random.randint(0, 2, N*pilot_bps)
    p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)  # 导频由用户模块生成，|p_n|=1。
    data_bits = np.random.randint(0, 2, N*M_max*bps)
    X = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M_max), order='F')  # BPSK数据由用户模块生成。
    pilot_noise = np.sqrt(sigma_p2/2)*(np.random.randn(N)+1j*np.random.randn(N))
    data_noise = np.sqrt(sigma2/2)*(np.random.randn(N, M_max)+1j*np.random.randn(N, M_max))
    y_p = p*h + pilot_noise  # [原文(2)] 导频回波。
    Y = X*h[:, None] + data_noise  # [原文(3)] 数据回波；第m列对应y_m。
    g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)] F_L^H(p* ⊙ y_p)。
    h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')  # [原文(11)] 均衡所用的导频信道初值。
    P_pilot += np.abs(g_pilot)**2/MonteCarlo  # 平均每次试验的PDP，不先跨MC平均g_pilot。
    for j, M in enumerate(M_values):
        X_part, Y_part = X[:, :M], Y[:, :M]  # M=30方案使用300列中的前30列。
        h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)  # 仅避免数值除零。
        equalized = Y_part/h_safe[:, None]  # [原文(13)] 使用导频初值均衡。
        hard_bits = modem.demodulate((equalized*np.sqrt(Es)).ravel(order='F'), 'hard')  # 硬解调调用用户模块。
        X_hard = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 再调制成参考符号。
        hard_accumulation = np.sum(Y_part*X_hard.conj(), axis=1)
        known_accumulation = np.sum(Y_part*X_part.conj(), axis=1)
        g_hard = (g_pilot + np.fft.ifft(hard_accumulation, norm='ortho')[:L])/(M+1)  # [原文(14)] 等噪声、BPSK形式。
        g_known = (g_pilot + np.fft.ifft(known_accumulation, norm='ortho')[:L])/(M+1)  # [原文(5)] 已知BPSK、等噪声简化。
        P_hard[j] += np.abs(g_hard)**2/MonteCarlo
        P_known[j] += np.abs(g_known)**2/MonteCarlo
    print(f"Fig.2: {mc+1}/{MonteCarlo}", flush=True)

# 原始PDP上的积累增益检查；9.87dB只是理论参照，不用于移动曲线。
hard_reduction_dB = 10*np.log10(np.mean(P_hard[0, non_target_mask])/np.mean(P_hard[1, non_target_mask]))
known_reduction_dB = 10*np.log10(np.mean(P_known[0, non_target_mask])/np.mean(P_known[1, non_target_mask]))
theory_known_reduction_dB = 10*np.log10((M_values[1]+1)/(M_values[0]+1))
print(f"Hard-decision reduction: {hard_reduction_dB:.3f} dB")
print(f"Known-payload reduction: {known_reduction_dB:.3f} dB; theory: {theory_known_reduction_dB:.3f} dB")

# ==================== 绘图 ====================
fig, ax = plt.subplots(num=2, figsize=(8, 6), layout='constrained')
curves = [P_pilot, P_hard[0], P_known[0], P_hard[1], P_known[1]]
labels = ['Pilot-Only', f'M={M_values[0]}, Hard-Decision-Based', f'M={M_values[0]}, Perfect Payload Knowledge', f'M={M_values[1]}, Hard-Decision-Based', f'M={M_values[1]}, Perfect Payload Knowledge']
colors = ['#00A1F1', '#F65314', '#EDB120', '#8A2BE2', '#77AC30']
line_styles = ['--', '-', '-.', '-', '--']
for i, (power, label) in enumerate(zip(curves, labels)):
    range_dB = 10*np.log10(np.maximum(power/np.max(power), 1e-16))  # [图中判读] 各曲线最大值为0dB；原始PDP另存。
    ax.plot(range_axis, range_dB, color=colors[i], ls=line_styles[i], label=label)
for i, k in enumerate(target_indices):
    ax.axvline(k, color='black', ls='--', lw=1.2, alpha=0.4, label='True targets' if i == 0 else None)
ax.set(xlabel='Range index', ylabel='Response (dB)', title=f'SNR={SNRdB} dB, direct coherent integration')
ax.set_xlim(1, L)
ax.set_ylim(-65, 1)
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, prop=FontProperties(family='Times New Roman', size=13))
fig.savefig(os.path.join(output_dir, 'Fig2.pdf'), facecolor='white', bbox_inches='tight')
# fig.savefig(os.path.join(output_dir, 'Fig2.png'), facecolor='white', dpi=300, bbox_inches='tight')
np.savez_compressed(os.path.join(output_dir, 'Fig2.npz'), g=g, target_indices=target_indices, target_phase=target_phase, target_power_dB=target_power_dB, P_pilot=P_pilot, P_hard=P_hard, P_known=P_known, MonteCarlo=MonteCarlo, M_values=M_values, CIR_PEAK_GAIN=CIR_PEAK_GAIN, SNRdB=SNRdB)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)


