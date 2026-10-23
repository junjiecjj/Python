#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:02:22 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四份 OFDM 通信/感知程序的单文件合并版。
依赖：numpy、scipy、matplotlib，以及用户现有的Modulations.py。
运行：python OFDM_ISAC_merged.py。
通信：时域多径卷积 -> 去CP -> FFT -> ZF/MMSE -> QAM解调。
感知：直接生成理想的接收频域数据，并非完整时域雷达回波仿真。
所有数据矩阵形状为(N,M)：行是子载波k，列是OFDM符号m。
QAM映射、调制解调和星座平均能量Es均使用用户现有的Modulations模块。
通信SNR以平均单位增益信道为参考，噪声和MMSE均使用返回的Es。
Eb/N0包含CP能量开销；感知SNR=Es/sigma_W^2，目标回波系数为1。
模糊函数为发射波形自身的AF，不是接收回波的目标检测图。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.constants import c as c0
from scipy.integrate import quad
from scipy.special import erfc
from Modulations import modulator



# ==================== 2. 通用函数 ====================
def complex_noise(shape, sigma2, rng):
    return np.sqrt(sigma2 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def awgn(signal, snr_db, rng, measured=True):
    power = np.mean(np.abs(signal)**2) if measured else 1.0
    return signal + complex_noise(np.shape(signal), power * 10**(-snr_db / 10), rng)

def range2time(range_val, c=c0):
    return 2 * range_val / c

def speed2dop(speed, lambda_val):
    return 2 * speed / lambda_val

def power_db(power):
    power = np.asarray(power, dtype=float)
    return 10 * np.log10(np.maximum(power / max(float(np.max(power)), 1e-300), 1e-12))

def ofdm_transmitter(TxData, CPsize):
    if not 0 <= CPsize <= TxData.shape[0]:
        raise ValueError('CP长度必须在0到N之间。')
    TxSignal = np.fft.ifft(TxData, axis=0)
    TxSignal_cp = np.vstack([TxSignal[-CPsize:, :], TxSignal]) if CPsize else TxSignal.copy()
    return TxSignal, TxSignal_cp.T.flatten()

def communication_receiver(TxData, h, CPsize, sigma2, rng, equalizer='MMSE', Es=1.0):
    """sigma2为FFT后噪声方差。NumPy FFT不归一化，因此时域方差为sigma2/N。"""
    n_sub, n_sym = TxData.shape
    if len(h) > n_sub or len(h) - 1 > CPsize:
        raise ValueError('通信信道长度要求L<=N且L-1<=Ncp。')
    _, TxSignal_time = ofdm_transmitter(TxData, CPsize)
    RxSignal = np.convolve(TxSignal_time, h, mode='full')[:TxSignal_time.size]
    RxSignal += complex_noise(RxSignal.shape, sigma2 / n_sub, rng)
    RxSignal_reshaped = RxSignal.reshape(n_sub + CPsize, n_sym, order='F')
    RxData = np.fft.fft(RxSignal_reshaped[CPsize:, :], axis=0)
    H_channel = np.fft.fft(h, n=n_sub)[:, None]
    if equalizer.upper() == 'ZF':
        if np.any(np.abs(H_channel) < 1e-14):
            raise ValueError('ZF遇到零信道响应，请改用MMSE或更换信道。')
        demodRxData = RxData / H_channel
    elif equalizer.upper() == 'MMSE':
        C = np.conj(H_channel) / (np.abs(H_channel)**2 + sigma2 / Es)
        demodRxData = RxData * C
    else:
        raise ValueError('equalizer必须为ZF或MMSE。')
    return demodRxData, TxSignal_time, RxData, RxSignal, H_channel

def radar_channel(TxData, delta_f, Ts, fc, target_pos, target_speed, snr_db, rng, Es=1.0):
    """接收FFT后的单目标理想模型，忽略ICI；不是对时域发射帧施加实际延迟。"""
    n_sub, n_sym = TxData.shape
    target_delay = range2time(target_pos)
    target_dop = speed2dop(target_speed, c0 / fc)
    kSub, mSym = np.meshgrid(np.arange(n_sub), np.arange(n_sym), indexing='ij')
    phase_shift = -1j * 2 * np.pi * (fc * target_delay + kSub * delta_f * target_delay - mSym * Ts * target_dop)
    return TxData * np.exp(phase_shift) + complex_noise(TxData.shape, Es * 10**(-snr_db / 10), rng)

def range_estimation(dividerArray, delta_f, NPer):
    if NPer < dividerArray.shape[0]:
        raise ValueError('距离IFFT点数不能小于子载波数。')
    range_fft = np.fft.ifft(dividerArray, n=NPer, axis=0)
    mean_range_power = np.mean(np.abs(range_fft)**2, axis=1)
    range_axis = np.arange(NPer) * c0 / (2 * NPer * delta_f)
    distanceE = range_axis[np.argmax(mean_range_power)]
    return range_axis, power_db(mean_range_power), float(distanceE)

def velocity_estimation(dividerArray, Ts, fc, MPer):
    if MPer < dividerArray.shape[1]:
        raise ValueError('速度FFT点数不能小于OFDM符号数。')
    velocity_fft = np.fft.fftshift(np.fft.fft(dividerArray, n=MPer, axis=1), axes=1)
    mean_velocity_power = np.mean(np.abs(velocity_fft)**2, axis=0)
    velocityIndex = np.fft.fftshift(np.fft.fftfreq(MPer, d=Ts)) * c0 / (2 * fc)
    velocityE = velocityIndex[np.argmax(mean_velocity_power)]
    return velocityIndex, power_db(mean_velocity_power), float(velocityE)

def music_range(dividerArray, delta_f, points, nTargets=1, TxData=None):
    n_sub, n_sym = dividerArray.shape
    if not 1 <= nTargets < min(n_sub, n_sym):
        raise ValueError('MUSIC目标数必须小于子载波数和快拍数。')
    # 高阶QAM除法导致不等噪声方差，按各行平均条件噪声方差白化；4-QAM时为共同常数。
    noise_scale = np.ones(n_sub) if TxData is None else np.sqrt(np.mean(1 / np.abs(TxData)**2, axis=1))
    whitened = dividerArray / noise_scale[:, None]
    Rxxd = whitened @ whitened.conj().T / n_sym
    _, Vd = np.linalg.eigh((Rxxd + Rxxd.conj().T) / 2)
    distanceEigenMatNoise = Vd[:, :n_sub - nTargets]
    omegaDistance = 2 * np.pi * np.arange(points) / points
    omegaVector = np.exp(-1j * np.arange(n_sub)[:, None] * omegaDistance[None, :]) / noise_scale[:, None]
    denominator = np.sum(np.abs(distanceEigenMatNoise.conj().T @ omegaVector)**2, axis=0)
    SP = np.sum(np.abs(omegaVector)**2, axis=0) / np.maximum(denominator, 1e-15)
    distanceIndex = omegaDistance * c0 / (4 * np.pi * delta_f)
    distanceE = distanceIndex[np.argmax(SP)]
    return distanceIndex, power_db(SP), float(distanceE)

def ambiguity_function(s_t, dt, tau_range, fd_range):
    """归一化自模糊函数幅度 |sum conj(s[n])*s(t[n]-tau)*exp(j2pi*fd*t[n])|。"""
    t_seq = np.arange(len(s_t)) * dt
    ambiguity_func = np.zeros((len(tau_range), len(fd_range)))
    for i, tau in enumerate(tau_range):
        s_tau = np.interp(t_seq - tau, t_seq, s_t, left=0, right=0)
        product = s_t.conj() * s_tau
        for j, fd in enumerate(fd_range):
            ambiguity_func[i, j] = np.abs(np.sum(product * np.exp(1j * 2 * np.pi * fd * t_seq)))
    peak = float(np.max(ambiguity_func))
    return ambiguity_func / peak if peak > 0 else ambiguity_func

def Qfun(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ser_awgn(EbN0dB, MOD_TYPE, order):
    EbN0 = 10**(np.asarray(EbN0dB, dtype=float) / 10)
    EsN0 = np.log2(order) * EbN0
    kind = MOD_TYPE.lower()
    if kind == 'bpsk' or (kind == 'psk' and order == 2):
        return Qfun(np.sqrt(2 * EbN0))
    if kind == 'psk':
        return 2 * Qfun(np.sqrt(2 * EbN0)) - Qfun(np.sqrt(2 * EbN0))**2 if order == 4 else 2 * Qfun(np.sin(np.pi / order) * np.sqrt(2 * EsN0))
    if kind == 'qam':
        tail = 2 * (1 - 1 / np.sqrt(order)) * Qfun(np.sqrt(3 * EsN0 / (order - 1)))
        return 2 * tail - tail**2
    if kind == 'pam':
        return 2 * (1 - 1 / order) * Qfun(np.sqrt(6 * EsN0 / (order**2 - 1)))
    raise ValueError('未知调制类型。')

def ser_rayleigh(EbN0dB, MOD_TYPE, order):
    EbN0dB = np.atleast_1d(np.asarray(EbN0dB, dtype=float))
    EsN0 = np.log2(order) * 10**(EbN0dB / 10)
    kind = MOD_TYPE.lower()
    if kind == 'bpsk' or (kind == 'psk' and order == 2):
        return 0.5 * (1 - np.sqrt(EsN0 / (1 + EsN0)))
    SER = np.zeros(EbN0dB.size)
    for i, snr in enumerate(EsN0):
        if kind == 'psk':
            g = np.sin(np.pi / order)**2
            SER[i] = quad(lambda x: np.sin(x)**2 / (np.sin(x)**2 + g * snr), 0, np.pi * (order - 1) / order)[0] / np.pi
        elif kind == 'qam':
            g = 1.5 / (order - 1)
            fun = lambda x: np.sin(x)**2 / (np.sin(x)**2 + g * snr)
            SER[i] = 4 / np.pi * (1 - 1 / np.sqrt(order)) * quad(fun, 0, np.pi / 2)[0] - 4 / np.pi * (1 - 1 / np.sqrt(order))**2 * quad(fun, 0, np.pi / 4)[0]
        elif kind == 'pam':
            g = 3 / (order**2 - 1)
            SER[i] = 2 * (order - 1) / (order * np.pi) * quad(lambda x: np.sin(x)**2 / (np.sin(x)**2 + g * snr), 0, np.pi / 2)[0]
        else:
            raise ValueError('未知调制类型。')
    return SER

def configure_plots():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': 'Times New Roman' if 'Times New Roman' in installed else 'DejaVu Serif', 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 12, 'lines.linewidth': 2, 'lines.markersize': 8, 'figure.facecolor': 'white', 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def new_figure(number, xlabel, ylabel, title):
    fig, ax = plt.subplots(num=number, figsize=(8, 4), layout='constrained')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, linestyle='--', alpha=0.2)
    return fig, ax

def finish_figure(fig, number, output_dir):
    if SAVE_FIGURES:
        fig.savefig(output_dir / f'OFDM_Fig{number:02d}.png', dpi=PNG_DPI, facecolor='white')
        fig.savefig(output_dir / f'OFDM_Fig{number:02d}.pdf', facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

# ==================== 1. 参数配置 ====================
SEED = 42
N = 256
M = 16
Ncp = N // 4
fc = 30e9
delta_f = 15e3 * 2**6                 # 960 kHz
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
ComSNRdB = 20.0                       # 通信频域Es/N0，单次仿真
RadarSNRdB = 30.0                     # 感知频域Es/sigma_W^2
target_pos = 30.0                     # m，可改为60；理想频域模型仍可生成
target_speed = 20.0                   # m/s，正值对应正多普勒
NPer = 16 * N
MPer = 128 * M
MUSIC_POINTS = NPer
EbN0dBs = np.arange(0, 37, 4)
SNRdBs = np.arange(-20, 10, 5)
nFrameCom = 500
nFrameRadar = 1000
COM_MC_CHANNEL = 'rayleigh'           # 'rayleigh'或'awgn'
COM_EQUALIZER = 'MMSE'                # 'MMSE'或'ZF'
L = 10                               # 蒙特卡洛通信信道长度
PowerdB = np.array([0, -8, -17, -21, -25], dtype=float)
Delay = np.array([0, 3, 5, 6, 8])
RADAR_RANDOM_DATA_EACH_FRAME = False   # False保留原程序的固定数据RMSE实验
AF_POINTS = 101                       # 必须为奇数，使网格精确包含零

SHOW_FIGURES = True
SAVE_FIGURES = False
OUTPUT_DIR = '.'                      # 图和数值结果保存到当前目录
PNG_DPI = 600

# ==================== 3. 主程序 ====================

if N <= 0 or M < 2 or Ncp < 0 or Ncp > N or delta_f <= 0 or fc <= 0:
    raise ValueError('请检查N、M、Ncp、delta_f和fc。')
if AF_POINTS < 3 or AF_POINTS % 2 == 0:
    raise ValueError('AF_POINTS必须为不小于3的奇数。')
if NPer < N or MPer < M or MUSIC_POINTS < 2:
    raise ValueError('FFT点数不能小于对应数据长度，MUSIC_POINTS必须>=2。')
if target_pos < 0 or target_pos >= c0 / (2 * delta_f):
    raise ValueError('目标距离必须位于[0,c/(2*delta_f))内。')
configure_plots()
output_dir = Path(OUTPUT_DIR)
if SAVE_FIGURES:
    output_dir.mkdir(parents=True, exist_ok=True)
rng_com, rng_radar, rng_ser, rng_rmse = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
map_table, demap_table = modem.getMappTable()
Es = float(Es)
if not np.isfinite(Es) or Es <= 0:
    raise ValueError("modulator返回的平均符号能量Es必须为正数。")
T = 1 / delta_f
dt = T / N
Tcp = Ncp * dt
Ts = T + Tcp
lambda_val = c0 / fc
target_delay = range2time(target_pos)
target_dop = speed2dop(target_speed, lambda_val)
range_resolution = c0 / (2 * N * delta_f)
velocity_resolution = lambda_val / (2 * M * Ts)
if abs(target_dop) >= 1 / (2 * Ts):
    raise ValueError('目标多普勒超出符号维FFT无模糊区间。')
print(f'N={N}, M={M}, Ncp={Ncp}, delta_f={delta_f/1e3:.1f} kHz')
print(f'T={T*1e6:.6f} us, Tcp={Tcp*1e6:.6f} us, Ts={Ts*1e6:.6f} us')
print(f'距离分辨尺度={range_resolution:.6f} m，距离网格={c0/(2*NPer*delta_f):.6f} m')
print(f'速度分辨尺度={velocity_resolution:.6f} m/s，速度网格={lambda_val/(2*MPer*Ts):.6f} m/s')
print(f'目标多普勒={target_dop:.6f} Hz，感知采用理想频域模型，忽略ICI。')
if target_delay > Tcp:
    print(f'提示：绝对时延超过CP；按发射时刻对齐的完整时域回波不满足CP条件，当前频域模型未模拟此影响。CP覆盖距离={c0*Tcp/2:.3f} m。')

# 同一份随机QAM数据供单次通信、感知和发射波形分析使用。
bits = rng_com.integers(0, 2, N * M * bps, dtype=np.int8)
TxData = modem.modulate(bits)
sym_int = np.array([demap_table[sym] for sym in TxData])
TxData = TxData.reshape(N, M)
Power = 10**(PowerdB / 10)
Power /= np.sum(Power)             # 归一化平均信道总功率，不归一化每次衰落实现
h = np.zeros(int(np.max(Delay)) + 1, dtype=complex)
h[Delay] = (rng_com.standard_normal(Delay.size) + 1j * rng_com.standard_normal(Delay.size)) * np.sqrt(Power / 2)
sigma2_com = Es * 10**(-ComSNRdB / 10)
demodRxData, TxSignal_time, RxData_com, RxSignal_com, H_channel = communication_receiver(TxData, h, Ncp, sigma2_com, rng_com, COM_EQUALIZER, Es=Es)
demod_bits = modem.demodulate(demodRxData.flatten(), 'hard')
sym_int_hat = []
for j in range(M*N):
    sym_int_hat.append(int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base=2))
sym_int_hat = np.array(sym_int_hat)
bit_errors = int(np.sum(bits != demod_bits))
symbol_errors = int(np.sum(sym_int != sym_int_hat))
print(f'通信：比特错误={bit_errors}/{bits.size}，BER={bit_errors/bits.size:.6g}；符号错误={symbol_errors}/{N*M}，SER={symbol_errors/(N*M):.6g}')

#%% 正确相位：载波时延 + 子载波时延 - 符号间多普勒。
RxData_radar = radar_channel(TxData, delta_f, Ts, fc, target_pos, target_speed, RadarSNRdB, rng_radar, Es=Es)
dividerArray = RxData_radar / TxData
range_axis, range_power_dB, distanceE = range_estimation(dividerArray, delta_f, NPer)
distanceIndex, SP_dB, distanceMUSIC = music_range(dividerArray, delta_f, MUSIC_POINTS, 1, TxData)
velocityIndex, velocity_power_dB, velocityE = velocity_estimation(dividerArray, Ts, fc, MPer)
print(f'距离：真实={target_pos:.6f} m，IFFT={distanceE:.6f} m，MUSIC={distanceMUSIC:.6f} m')
print(f'速度：真实={target_speed:.6f} m/s，FFT={velocityE:.6f} m/s')

fig, ax = new_figure(1, 'Time (us)', 'Amplitude', 'OFDM transmit frame with cyclic prefixes')
t_tx = np.arange(TxSignal_time.size) * dt * 1e6
cp_mask = np.tile(np.arange(N + Ncp) < Ncp, M)
ax.plot(t_tx, TxSignal_time.real, color='#00A1F1', linewidth=1, label='Transmit signal')
ax.plot(t_tx, np.where(cp_mask, TxSignal_time.real, np.nan), color='#F65314', linewidth=1.5, label='Cyclic prefixes')
ax.legend()
# finish_figure(fig, 1, output_dir)

fig, ax = new_figure(3, 'Delay (samples)', 'Magnitude', 'Communication channel impulse response')
ax.stem(np.arange(h.size), np.abs(h), linefmt='C0-', markerfmt='C0o', basefmt='k-')
# finish_figure(fig, 3, output_dir)

fig, ax = new_figure(4, 'Range (m)', 'Normalized spectrum (dB)', 'OFDM range estimation')
ax.plot(range_axis, range_power_dB, color='#00A1F1', label='IFFT')
ax.plot(distanceIndex, SP_dB, color='#F65314', label='MUSIC')
ax.axvline(target_pos, color='black', linestyle='--', linewidth=1.2, label='True range')
ax.axvline(distanceE, color='#8A2BE2', linestyle=':', linewidth=1.2, label='IFFT estimate')
ax.set_xlim(0, min(c0 / (2 * delta_f), max(2 * target_pos, 10)))
ax.legend()
# finish_figure(fig, 4, output_dir)

fig, ax = new_figure(5, 'Velocity (m/s)', 'Normalized power (dB)', 'OFDM Doppler FFT velocity estimation')
ax.plot(velocityIndex, velocity_power_dB, color='#00A1F1', label='FFT')
ax.axvline(target_speed, color='black', linestyle='--', linewidth=1.2, label='True velocity')
ax.axvline(velocityE, color='#F65314', linestyle=':', linewidth=1.2, label='FFT estimate')
span = max(3 * velocity_resolution, 2 * abs(target_speed), 50)
ax.set_xlim(max(velocityIndex[0], -span), min(velocityIndex[-1], span))
ax.legend()
# finish_figure(fig, 5, output_dir)

#%% 发射波形AF：不标注“检测目标”；其峰应位于原点。

RUN_AF = True
if RUN_AF:
    tau_max = max(2 * target_delay, 4 * dt)
    fd_max = max(2 * abs(target_dop), 2 / (M * Ts))
    tau_range = np.linspace(-tau_max, tau_max, AF_POINTS)
    fd_range = np.linspace(-fd_max, fd_max, AF_POINTS)
    ambiguity_func = ambiguity_function(TxSignal_time, dt, tau_range, fd_range)
    zero_idx = AF_POINTS // 2
    af_power_dB = power_db(ambiguity_func**2)
    X, Y = np.meshgrid(fd_range / 1e3, tau_range * 1e9)

    fig = plt.figure(num=6, figsize=(14, 9), layout='constrained')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(X, Y, ambiguity_func, cmap='jet', edgecolor='none')
    ax.set(xlabel='Doppler (kHz)', ylabel='Delay (ns)', zlabel='AF magnitude', title='Transmit waveform ambiguity function')

    ax = fig.add_subplot(2, 2, 2)
    contour = ax.contourf(X, Y, af_power_dB, levels=np.linspace(-60, 0, 31), cmap='jet', extend='min')
    fig.colorbar(contour, ax=ax, label='Normalized power (dB)')
    ax.set(xlabel='Doppler (kHz)', ylabel='Delay (ns)', title='AF contour')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(tau_range * 1e9, af_power_dB[:, zero_idx], color='#00A1F1')
    ax.set(xlabel='Delay (ns)', ylabel='Normalized power (dB)', title='AF slice: zero Doppler')
    ax.grid(True, linestyle='--', alpha=0.2)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(fd_range / 1e3, af_power_dB[zero_idx, :], color='#F65314')
    ax.set(xlabel='Doppler (kHz)', ylabel='Normalized power (dB)', title='AF slice: zero delay')
    ax.grid(True, linestyle='--', alpha=0.2)

    # finish_figure(fig, 6, output_dir)

#%% Eb/N0包含CP开销；FFT后噪声方差sigma2=Es*(1+Ncp/N)/(bps*EbN0)。
# Rayleigh信道E[sum|h|^2]=1，噪声不随每次衰落后的接收功率调整。
RUN_SER = True

if RUN_SER:
    if nFrameCom < 1 or L < 1 or L > N or L - 1 > Ncp:
        raise ValueError('检查nFrameCom和通信信道长度L。')
    if COM_MC_CHANNEL.lower() not in ('rayleigh', 'awgn'):
        raise ValueError('COM_MC_CHANNEL必须为rayleigh或awgn。')
    SER_sim = np.zeros(len(EbN0dBs))
    BER_sim = np.zeros(len(EbN0dBs))
    for idx, ebn0 in enumerate(EbN0dBs):
        errors = 0
        errors_bits = 0
        sigma2 = Es * (1 + Ncp / N) / (bps * 10**(ebn0 / 10))
        for _ in range(nFrameCom):
            frame_bits = rng_ser.integers(0, 2, N * M * bps, dtype=np.int8)
            frame_data = modem.modulate(frame_bits)
            frame_labels = np.array([demap_table[sym] for sym in frame_data])
            frame_data = frame_data.reshape(N, M)
            h_mc = (rng_ser.standard_normal(L) + 1j * rng_ser.standard_normal(L)) / np.sqrt(2 * L) if COM_MC_CHANNEL.lower() == 'rayleigh' else np.array([1.0 + 0j])
            # 保留原蒙特卡洛通信程序的ZF，避免高阶QAM的MMSE幅度收缩影响理论对比。
            equalized, _, _, _, _ = communication_receiver(frame_data, h_mc, Ncp, sigma2, rng_ser, 'ZF', Es=Es)
            demod_bits = modem.demodulate(equalized.flatten(), 'hard')
            sym_int_hat = []
            for j in range(M*N):
                sym_int_hat.append(int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base=2))
            sym_int_hat = np.array(sym_int_hat)
            errors += int(np.sum(frame_labels != sym_int_hat))
            errors_bits += int(np.sum(frame_bits != demod_bits))
        SER_sim[idx] = errors / (nFrameCom * N * M)
        BER_sim[idx] = errors_bits / (nFrameCom * N * M * bps)
        print(f'SER {idx+1}/{len(EbN0dBs)}: Eb/N0={ebn0:g} dB, SER={SER_sim[idx]:.6g}')
    EbN0_effective = EbN0dBs - 10 * np.log10(1 + Ncp / N)
    SER_theory = ser_rayleigh(EbN0_effective, MOD_TYPE, QAM_mod) if COM_MC_CHANNEL.lower() == 'rayleigh' else ser_awgn(EbN0_effective, MOD_TYPE, QAM_mod)
    fig, ax = new_figure(7, 'Eb/N0 including CP energy (dB)', 'SER', f'OFDM {QAM_mod}-QAM over {COM_MC_CHANNEL} channel')
    ax.semilogy(EbN0dBs, np.where(SER_sim > 0, SER_sim, np.nan), 'o-', color='#00A1F1', label='Simulation (ZF)')
    ax.semilogy(EbN0dBs, np.where(SER_theory > 0, SER_theory, np.nan), '*--', color='#F65314', label='Theory with CP overhead')
    ax.legend()

#%% 独立于单次感知的蒙特卡洛实验；不会覆盖单次距离谱和速度谱的数据。
RUN_RMSE = True
if RUN_RMSE:
    if nFrameRadar < 1:
        raise ValueError('nFrameRadar必须>=1。')
    RmseList = np.zeros(len(SNRdBs))
    for idx, radarsnr in enumerate(SNRdBs):
        mse = 0.0
        for _ in range(nFrameRadar):
            if RADAR_RANDOM_DATA_EACH_FRAME:
                frame_bits = rng_rmse.integers(0, 2, N * M * bps, dtype=np.int8)
                frame_data = modem.modulate(frame_bits).reshape(N, M)
            else:
                frame_data = TxData
            rx = radar_channel(frame_data, delta_f, Ts, fc, target_pos, target_speed, radarsnr, rng_rmse, Es=Es)
            _, _, estimate = range_estimation(rx / frame_data, delta_f, NPer)
            mse += (estimate - target_pos)**2
        RmseList[idx] = np.sqrt(mse / nFrameRadar)
        print(f'RMSE {idx+1}/{len(SNRdBs)}: SNR={radarsnr:g} dB, RMSE={RmseList[idx]:.6g} m')
    fig, ax = new_figure(8, 'Radar input SNR (dB)', 'Range RMSE (m)', 'OFDM IFFT range estimation accuracy')
    ax.plot(SNRdBs, RmseList, 'o-', color='#00A1F1', label='IFFT')
    ax.legend()





