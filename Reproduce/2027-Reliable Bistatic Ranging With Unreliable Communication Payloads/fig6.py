"""Fig.6：Reliable Bistatic Ranging With Unreliable Communication Payloads。
同目录放用户的Modulations.py；依赖numpy/scipy/matplotlib/scikit-commpy。
参数、主流程、检测与绘图在顶层；CFAR阈值直接调用用户ca_cfar.py中的cacfar。
仿真采用已确认的补充CIR；原图提取PDP仅在末尾独立核对检测器，不参与回波生成。
原文完整CIR复系数未公开，因此新仿真的检测数不保证等于原图18/5。
"""
import os
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Modulations import modulator
from ca_cfar import cacfar  # 直接使用用户上传的CA-CFAR，原文件保持不变。

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
M = 13  # [原文Table I] 默认数据OFDM符号数；Fig.6未另行注明。
SNRdB = 3  # [原文Fig.6] 导频与数据SNR相同。
sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] SNR=1/sigma^2。
EM_max_iter = 500  # [已确认的补充设置] 最多500轮。
EM_tol = 1e-8  # [已确认的补充设置] 相对CIR变化停止条件。
training_per_side = 16  # [由Fig.6矢量PDP/阈值反推] 每侧16个训练单元。
guard_per_side = 4  # [由原图反推] 每侧4个保护单元。
design_pfa = 0.1  # [由原图反推] 名义设计参数，不等于实际虚警峰概率。
CHECK_PAPER_CFAR = True  # 同目录有附带的原图提取数据时，独立核对检测器。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig6_results_userCFAR")
reference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig6_paper_reference.npz")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "qam"
QAM_mod = 16  # [原文Table I] 16-QAM数据。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Es = float(Es)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)  # 星座只从用户模块读取。
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
pilot_Es = float(pilot_Es)
real_levels = np.unique(Constellation.real)
imag_levels = np.unique(Constellation.imag)
if len(real_levels)*len(imag_levels) != len(Constellation):
    raise ValueError("后验矩函数采用均匀方形QAM的精确分解。")
if not 1 <= L <= N or M < 1 or EM_max_iter < 1 or EM_tol <= 0:
    raise ValueError("要求1<=L<=N、M>=1、EM_max_iter>=1、EM_tol>0。")

# ==================== 信道：位置由原图读取，其余为已确认的补充设置 ====================
target_indices = np.array([40, 103, 159, 164, 234, 252])  # Fig.6原图位置，作图采用1起始索引。
target_bins = target_indices-1
relative_power_dB = np.array([-36.0, -6.0, -28.0, -36.0, 0.0, -6.0])  # [补充] 不是作者公开的抽头功率。
target_phase = np.zeros(len(target_indices))  # [补充] 各径相位暂设0。
if np.any(target_bins < 0) or np.any(target_bins >= L):
    raise ValueError("目标时延超出L。")
alpha_raw = 10**(relative_power_dB/20)*np.exp(1j*target_phase)
g = np.zeros(L, dtype=complex)
g[target_bins] = np.sqrt(N)*alpha_raw/np.linalg.norm(alpha_raw)  # [原文(1)] 稀疏CIR；补充归一化mean(|h|^2)=1。
h = np.fft.fft(g, n=N, norm='ortho')  # [原文(2)(3)] h=F_L g。
range_axis = np.arange(1, L+1)

# ==================== 必要数学功能：加权LS与后验矩 ====================
def weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L):
    w = (1 + np.sum(symbol_second, axis=1))/sigma2  # [原文(24)] 导频和数据噪声方差相同。
    r = (p.conj()*y_p + np.sum(symbol_mean.conj()*Y, axis=1))/sigma2  # [原文(25)]。
    first_column = np.fft.ifft(w)[:L]  # [由原文(5)/(27)推得] F_L^H Diag(w) F_L的首列，此处采用1/N IFFT。
    first_column[0] = first_column[0].real
    rhs = np.fft.ifft(r, norm='ortho')[:L]  # F_L^H r。
    return solve_toeplitz((first_column, first_column.conj()), rhs, check_finite=False)  # 精确求解，不显式求逆。

def posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels):
    # [原文(22)] 均匀方形QAM后验可精确分解成实部和虚部两个PAM后验。
    # levels全部从modem.constellation获得；这里没有自行生成星座。
    sufficient = Y*h_hat[:, None].conj()
    h_power = np.abs(h_hat[:, None, None])**2
    component_moments = []
    for component, levels in [(sufficient.real, real_levels), (sufficient.imag, imag_levels)]:
        log_weights = (2*component[..., None]*levels - h_power*levels**2)/sigma2
        log_weights -= np.max(log_weights, axis=-1, keepdims=True)  # 移除公共常数，避免指数溢出。
        probability = np.exp(log_weights)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        mean = np.sum(probability*levels, axis=-1)
        second = np.sum(probability*levels**2, axis=-1)
        component_moments.append((mean, second))
    symbol_mean = component_moments[0][0] + 1j*component_moments[1][0]  # [原文(23)] E[x|y]。
    symbol_second = component_moments[0][1] + component_moments[1][1]  # [原文(23)] E[|x|^2|y]，不是|E[x|y]|^2。
    return symbol_mean, symbol_second

def ca_cfar_peaks(power, training, guard, pfa):
    # 阈值由用户原始cacfar计算；这里只适配原图的循环边缘及局部峰计数。
    if training < 1 or guard < 0 or 2*(training+guard) >= len(power) or not 0 < pfa < 1:
        raise ValueError("CFAR训练/保护窗口或设计Pfa不合法。")
    padding = training+guard
    padded_power = np.pad(power, (padding, padding), mode='wrap')  # [由Fig.6反推] 两端循环补边。
    positions, threshold, start_cell, stop_cell = cacfar(padded_power, pfa, training, guard)  # 原函数返回1起始位置。
    candidate_indices = positions.astype(int)-padding-1  # 映射回原始PDP的0起始数组索引。
    peak_indices = find_peaks(power)[0]
    detected_indices = np.intersect1d(candidate_indices, peak_indices)  # [由原图反推] 超阈值局部峰。
    multiplier = 2*training*(pfa**(-1/(2*training))-1)
    local_noise = threshold/multiplier  # 仅还原背景估计供保存；不重新计算或替换用户函数的阈值。
    return local_noise, threshold, peak_indices, detected_indices

# ==================== 单次感知帧：HD与EM共享同一组观测 ====================
pilot_bits = np.random.randint(0, 2, N*pilot_bps)
p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)
data_bits = np.random.randint(0, 2, N*M*bps)
X = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 数据调制调用用户模块。
pilot_noise = np.sqrt(sigma2/2)*(np.random.randn(N)+1j*np.random.randn(N))
data_noise = np.sqrt(sigma2/2)*(np.random.randn(N, M)+1j*np.random.randn(N, M))
y_p = p*h+pilot_noise  # [原文(2)]。
Y = X*h[:, None]+data_noise  # [原文(3)]。
g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)] 导频初始化。
h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')
h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)
equalized = Y/h_safe[:, None]
hard_bits = modem.demodulate((equalized*np.sqrt(Es)).ravel(order='F'), 'hard')  # 硬解调调用用户模块。
X_hard = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 重调制调用用户模块。
g_hard = weighted_ls(p, y_p, Y, X_hard, np.abs(X_hard)**2, sigma2, L)  # [原文(5)] 保留16-QAM幅度平方权重。
g_hat = g_pilot.copy()
EM_relative_change = np.zeros(EM_max_iter)
EM_stopped_by_tolerance = False
for iteration in range(1, EM_max_iter+1):
    h_hat = np.fft.fft(g_hat, n=N, norm='ortho')  # [原文(20)]。
    symbol_mean, symbol_second = posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels)  # [原文(22)(23)] E-step。
    g_next = weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L)  # [原文(24)--(27)] M-step。
    EM_relative_change[iteration-1] = np.linalg.norm(g_next-g_hat)/max(np.linalg.norm(g_hat), np.finfo(float).tiny)
    g_hat = g_next
    if EM_relative_change[iteration-1] < EM_tol:
        EM_stopped_by_tolerance = True
        break
EM_iterations_used = iteration
EM_relative_change = EM_relative_change[:EM_iterations_used]
print(f'EM: iterations={EM_iterations_used}, relative change={EM_relative_change[-1]:.3e}, tolerance reached={EM_stopped_by_tolerance}', flush=True)

# ==================== 检测与绘图：线性PDP检测，每幅图独立归一化显示 ====================
method_names = ['Hard-decision-based', 'EM-based']
g_estimates = np.stack([g_hard, g_hat])
PDP = np.abs(g_estimates)**2  # [原文Sec.II-B] 单帧PDP，不做MC平均。
thresholds = np.zeros_like(PDP)
local_noise_power = np.zeros_like(PDP)
local_peak_mask = np.zeros_like(PDP, dtype=bool)
detected_target_mask = np.zeros_like(PDP, dtype=bool)
false_peak_mask = np.zeros_like(PDP, dtype=bool)
figures = []
for method, name in enumerate(method_names):
    power = PDP[method]
    local_noise_power[method], thresholds[method], peak_indices, detected_indices = ca_cfar_peaks(power, training_per_side, guard_per_side, design_pfa)
    local_peak_mask[method, peak_indices] = True
    detected_targets = np.intersect1d(detected_indices, target_bins)  # [补充口径] on-grid精确位置匹配。
    false_peaks = np.setdiff1d(detected_indices, target_bins)
    detected_target_mask[method, detected_targets] = True
    false_peak_mask[method, false_peaks] = True
    reference = np.max(power)  # [原图显示方式] 每幅PDP主峰0dB，阈值共用该幅图的参考，不改变检测。
    spectrum_dB = 10*np.log10(np.maximum(power/reference, np.finfo(float).tiny))
    threshold_dB = 10*np.log10(np.maximum(thresholds[method]/reference, np.finfo(float).tiny))
    suffix = 'a' if method == 0 else 'b'
    fig, ax = plt.subplots(num=f'Fig.6({suffix}) simulated', figsize=(8, 6), layout='constrained')
    ax.plot(range_axis, spectrum_dB, color='#00A1F1', label='Range Profile')
    ax.plot(range_axis, threshold_dB, color='#F65314', ls='--', label='Threshold')
    for k in target_indices:
        ax.axvline(k, color='black', ls=':', lw=1, alpha=0.4)
    ax.plot(target_indices, spectrum_dB[target_bins], ls='none', marker='s', color='#009933', markerfacecolor='none', label='True Targets')
    ax.plot(range_axis[detected_targets], spectrum_dB[detected_targets], ls='none', marker='+', color='#8A2BE2', markersize=9, label='Detected Targets')
    ax.plot(range_axis[false_peaks], spectrum_dB[false_peaks], ls='none', marker='o', color='red', markerfacecolor='none', label='False Alarm')
    ax.set(xlabel='Delay bin', ylabel='Amplitude (dB)', title=f'{name}, SNR={SNRdB:g} dB, false-alarm peaks={len(false_peaks)}')
    ax.set_xlim(1, L)
    ax.set_ylim(-70, 5)
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, prop=FontProperties(family='Times New Roman', size=13))
    fig.savefig(os.path.join(output_dir, f'Fig6{suffix}_simulated.pdf'), facecolor='white', bbox_inches='tight')

    figures.append(fig)

# ==================== 可选核对：仅对原图提取PDP验证CFAR，不参与以上仿真 ====================
if CHECK_PAPER_CFAR and os.path.isfile(reference_path):
    with np.load(reference_path) as ref:
        ref_power, ref_threshold = ref['power'], ref['threshold']
        ref_targets = ref['target_indices']-1
        ref_training, ref_guard, ref_pfa = int(ref['training_per_side']), int(ref['guard_per_side']), float(ref['design_pfa'])
        ref_expected = ref['expected_false_peaks']
    ref_recomputed = np.zeros_like(ref_power)
    ref_false_counts = np.zeros(2, dtype=int)
    ref_threshold_rms_dB = np.zeros(2)
    for method in range(2):
        _, ref_recomputed[method], _, ref_detected = ca_cfar_peaks(ref_power[method], ref_training, ref_guard, ref_pfa)
        ref_false_counts[method] = len(np.setdiff1d(ref_detected, ref_targets))
        ref_threshold_rms_dB[method] = np.sqrt(np.mean((10*np.log10(ref_recomputed[method]/ref_threshold[method]))**2))
    print('Paper-PDP detector check only: false peaks HD/EM=', ref_false_counts, ', expected=', ref_expected, flush=True)
    print('Paper-PDP threshold RMS errors (dB):', ref_threshold_rms_dB, flush=True)

plt.show()
for fig in figures:
    plt.close(fig)
