#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OFDM通信与感知仿真；矩阵形状为(N,M)，行对应子载波，列对应OFDM符号。
通信采用时域多径信道，感知采用忽略ICI的理想频域模型。
依赖numpy、scipy、matplotlib和用户现有的Modulations.py。
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter
from scipy.constants import c as c0
from scipy.integrate import quad
from scipy.special import erfc
from Modulations import modulator

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置
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

#%%>>>>>>>>>>>>>>>>>>>>>>>>  收发、参数估计与理论SER函数
def complex_noise(shape, sigma2):
    return np.sqrt(sigma2 / 2) * (np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape))

def power_db(power):
    power = np.asarray(power, dtype=float)
    return 10 * np.log10(np.maximum(power / max(float(np.max(power)), 1e-300), 1e-12))

def ofdm_transmitter(TxData, CPsize):
    if not 0 <= CPsize <= TxData.shape[0]:
        raise ValueError('CP长度必须在0到N之间。')
    TxSignal = np.fft.ifft(TxData, axis=0)
    TxSignal_cp = np.vstack([TxSignal[-CPsize:, :], TxSignal]) if CPsize else TxSignal.copy()
    return TxSignal, TxSignal_cp.T.flatten()

def communication_receiver(TxData, h, CPsize, sigma2, equalizer='MMSE', Es=1.0):
    """sigma2为FFT后噪声方差。NumPy FFT不归一化，因此时域方差为sigma2/N。"""
    n_sub, n_sym = TxData.shape
    if len(h) > n_sub or len(h) - 1 > CPsize:
        raise ValueError('通信信道长度要求L<=N且L-1<=Ncp。')
    _, TxSignal_time = ofdm_transmitter(TxData, CPsize)
    RxSignal = np.convolve(TxSignal_time, h, mode='full')[:TxSignal_time.size]
    RxSignal += complex_noise(RxSignal.shape, sigma2 / n_sub)
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
    return demodRxData

def radar_channel(TxData, delta_f, Ts, fc, target_pos, target_speed, snr_db, Es=1.0, target_amplitudes=None):
    """多目标理想频域模型；beta为反射系数，alpha=beta*exp(-j*2*pi*fc*tau)。"""
    positions = np.atleast_1d(np.asarray(target_pos, dtype=float))
    speeds = np.atleast_1d(np.asarray(target_speed, dtype=float))
    beta = np.ones(positions.size, dtype=complex) if target_amplitudes is None else np.atleast_1d(np.asarray(target_amplitudes, dtype=complex))
    if positions.ndim != 1 or positions.shape != speeds.shape or beta.shape != positions.shape or positions.size == 0:
        raise ValueError('距离、速度和反射系数必须为等长的一维数组；单目标也可传标量。')
    if not all(np.all(np.isfinite(a)) for a in (positions, speeds, beta)) or np.any(beta == 0):
        raise ValueError('目标参数必须有限，反射系数不能为零。')
    k = np.arange(TxData.shape[0])[:, None]
    m = np.arange(TxData.shape[1])[None, :]
    H = np.zeros(TxData.shape, dtype=complex)
    for R, v, coefficient in zip(positions, speeds, beta):
        tau, fd = 2 * R / c0, 2 * fc * v / c0
        H += coefficient * np.exp(-2j * np.pi * fc * tau) * np.exp(-2j * np.pi * k * delta_f * tau + 2j * np.pi * m * Ts * fd)
    # 所有目标先相干叠加，再加一次噪声；不随目标个数归一化。
    return TxData * H + complex_noise(TxData.shape, Es * 10**(-snr_db / 10))

def range_estimation(dividerArray, delta_f, NPer, nTargets=1):
    if NPer < dividerArray.shape[0]:
        raise ValueError('距离IFFT点数不能小于子载波数。')
    range_fft = np.fft.ifft(dividerArray, n=NPer, axis=0)
    power = np.mean(np.abs(range_fft)**2, axis=1)
    axis = np.arange(NPer) * c0 / (2 * NPer * delta_f)
    separation = 0.75 * c0 / (2 * dividerArray.shape[0] * delta_f)
    estimates = peak_estimates_1d(power, axis, nTargets, separation)
    return axis, power_db(power), estimates

def velocity_estimation(dividerArray, Ts, fc, MPer, nTargets=1):
    if MPer < dividerArray.shape[1]:
        raise ValueError('速度FFT点数不能小于OFDM符号数。')
    velocity_fft = np.fft.fftshift(np.fft.fft(dividerArray, n=MPer, axis=1), axes=1)
    power = np.mean(np.abs(velocity_fft)**2, axis=0)
    axis = np.fft.fftshift(np.fft.fftfreq(MPer, d=Ts)) * c0 / (2 * fc)
    separation = 0.75 * c0 / (2 * fc * dividerArray.shape[1] * Ts)
    estimates = peak_estimates_1d(power, axis, nTargets, separation)
    return axis, power_db(power), estimates

def music_range(dividerArray, delta_f, points, nTargets=1, TxData=None, Ns=None):
    """沿子载波滑动平滑；同多普勒的多个目标也不直接使用秩一协方差。"""
    n_sub, n_sym = dividerArray.shape
    Ns = n_sub // 2 if Ns is None else int(Ns)
    windows = sliding_window_view(dividerArray, (Ns, 1))
    snapshots = windows[..., 0].reshape(-1, Ns).T
    if not 1 <= nTargets < Ns or snapshots.shape[1] < nTargets:
        raise ValueError('检查一维MUSIC的目标数和子阵长度。')
    scale = np.ones(Ns)
    if TxData is not None:
        scale = np.sqrt(sliding_window_view(1 / np.abs(TxData)**2, (Ns, 1)).mean(axis=(0, 1)))[:, 0]
    snapshots = snapshots / scale[:, None]
    Rxx = snapshots @ snapshots.conj().T / snapshots.shape[1]
    eigvals, V = np.linalg.eigh((Rxx + Rxx.conj().T) / 2)
    order = np.argsort(eigvals)  # 显式升序排序，并同步重排特征向量。
    En = V[:, order[:Ns - nTargets]]
    axis = np.arange(points) * c0 / (2 * points * delta_f)
    steering = np.exp(-2j * np.pi * np.arange(Ns)[:, None] * delta_f * (2 * axis[None, :] / c0)) / scale[:, None]
    denominator = np.sum(np.abs(En.conj().T @ steering)**2, axis=0)
    spectrum = np.sum(np.abs(steering)**2, axis=0) / np.maximum(denominator, 1e-15)
    estimates = peak_estimates_1d(spectrum, axis, nTargets, 3 * (axis[1] - axis[0]))
    return axis, power_db(spectrum), estimates

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

def peak_estimates_1d(power, axis, count, separation):
    local = power == maximum_filter(power, size=3, mode='constant', cval=-np.inf)
    candidates = np.flatnonzero(local)
    selected = []
    for index in candidates[np.argsort(power[candidates])[::-1]]:
        if all(abs(axis[index] - axis[old]) >= separation for old in selected):
            selected.append(int(index))
        if len(selected) == count:
            return axis[selected]
    raise ValueError('一维谱内没有足够的分离峰；请检查目标数、间隔及SNR。')

def joint_peak_estimates(power, delay_axis, doppler_axis, fc, nTargets=1, min_separation=None):
    """返回Q行(tau[s], fD[Hz], R[m], v[m/s])；每行保留距离速度配对。"""
    if not 1 <= nTargets <= power.size:
        raise ValueError('二维谱目标数不合法。')
    if min_separation is None:
        min_separation = (3 * abs(delay_axis[1] - delay_axis[0]), 3 * abs(doppler_axis[1] - doppler_axis[0]))
    local = power == maximum_filter(power, size=3, mode='constant', cval=-np.inf)
    candidates = np.argwhere(local)
    selected = []
    for row, col in candidates[np.argsort(power[local])[::-1]]:
        if all(abs(delay_axis[row] - delay_axis[r]) >= min_separation[0] or abs(doppler_axis[col] - doppler_axis[j]) >= min_separation[1] for r, j in selected):
            selected.append((int(row), int(col)))
        if len(selected) == nTargets:
            return np.array([[delay_axis[r], doppler_axis[j], c0 * delay_axis[r] / 2, c0 * doppler_axis[j] / (2 * fc)] for r, j in selected])
    raise ValueError('搜索区域内没有足够的分离峰；请检查目标数、网格和分辨条件。')

def fft_2d_estimation(dividerArray, delta_f, Ts, fc, NPer, MPer, nTargets=1):
    """G=F_NPer^H Zpad F_MPer；fft2后重排距离索引，统一采用单位化归一化。"""
    if dividerArray.ndim != 2 or not np.all(np.isfinite(dividerArray)):
        raise ValueError('二维FFT输入必须是有限的二维复数矩阵。')
    if NPer < dividerArray.shape[0] or MPer < dividerArray.shape[1]:
        raise ValueError('二维FFT点数不能小于原始数据长度。')
    G = np.fft.ifft(dividerArray, n=NPer, axis=0, norm='ortho')
    G = np.fft.fft(G, n=MPer, axis=1, norm='ortho')
    G = np.fft.fftshift(G, axes=1)
    power = np.abs(G)**2
    delay_axis = np.arange(NPer) / (NPer * delta_f)
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(MPer, d=Ts))
    estimates = joint_peak_estimates(power, delay_axis, doppler_axis, fc, nTargets, (0.75 / (dividerArray.shape[0] * delta_f), 0.75 / (dividerArray.shape[1] * Ts)))
    return delay_axis, doppler_axis, power_db(power), estimates

def music_2d_estimation(dividerArray, TxData, delta_f, Ts, fc, Ns, Ms, delay_axis, doppler_axis, nTargets=1):
    """二维滑动子阵平滑MUSIC。列向量化，a=kron(d(fD), b(tau))。"""
    Z = np.asarray(dividerArray)
    delay_axis, doppler_axis = np.asarray(delay_axis), np.asarray(doppler_axis)
    if Z.ndim != 2 or not np.all(np.isfinite(Z)):
        raise ValueError('二维MUSIC输入必须是有限的二维复数矩阵。')
    if not 2 <= Ns <= Z.shape[0] or not 2 <= Ms <= Z.shape[1]:
        raise ValueError('二维MUSIC子阵尺寸要求2<=Ns<=N、2<=Ms<=M。')
    P = Ns * Ms
    windows = sliding_window_view(Z, (Ns, Ms))
    snapshots = windows.reshape(-1, Ns, Ms).transpose(0, 2, 1).reshape(-1, P).T
    if not 1 <= nTargets < P or snapshots.shape[1] < nTargets:
        raise ValueError('目标数必须小于子阵阵元数，且不大于虚拟快拍数。')
    if delay_axis.ndim != 1 or doppler_axis.ndim != 1 or min(delay_axis.size, doppler_axis.size) < 2:
        raise ValueError('二维MUSIC的时延和多普勒网格必须是一维数组且各至少有2点。')
    if not np.all(np.isfinite(delay_axis)) or not np.all(np.isfinite(doppler_axis)):
        raise ValueError('二维MUSIC搜索网格必须为有限值。')
    noise_scale = np.ones(P)
    if TxData is not None:
        if np.shape(TxData) != Z.shape or not np.all(np.isfinite(TxData)) or np.any(np.abs(TxData) == 0):
            raise ValueError('TxData必须与接收矩阵同形状、为有限值且无零符号。')
        inv_power = sliding_window_view(1 / np.abs(TxData)**2, (Ns, Ms))
        noise_scale = np.sqrt(np.mean(inv_power, axis=(0, 1))).reshape(-1, order='F')
    snapshots = snapshots / noise_scale[:, None]
    Rxx = snapshots @ snapshots.conj().T / snapshots.shape[1]
    eigvals, eigvecs = np.linalg.eigh((Rxx + Rxx.conj().T) / 2)
    order = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    # eigh已经按特征值升序排列；前P-Q列为噪声子空间，后Q列为信号子空间。
    Esignal = eigvecs[:, P - nTargets:]
    b = np.exp(-2j * np.pi * np.arange(Ns)[:, None] * delta_f * delay_axis[None, :])
    d = np.exp(2j * np.pi * np.arange(Ms)[:, None] * Ts * doppler_axis[None, :])
    coefficients = (Esignal.conj().T / noise_scale[None, :]).reshape(nTargets, Ms, Ns)
    projection = np.einsum('qmk,kr,mv->qrv', coefficients, b, d, optimize=True)
    steering_norm2 = float(np.sum(1 / noise_scale**2))
    # ||En^H a_w||^2 = ||a_w||^2 - ||Es^H a_w||^2，避免巨型噪声子空间网格乘法。
    denominator = np.maximum(steering_norm2 - np.sum(np.abs(projection)**2, axis=0), np.finfo(float).eps * steering_norm2)
    spectrum = steering_norm2 / denominator
    estimates = joint_peak_estimates(spectrum, delay_axis, doppler_axis, fc, nTargets)
    return delay_axis, doppler_axis, power_db(spectrum), estimates

#%%>>>>>>>>>>>>>>>>>>>>>>>>  参数配置
N = 256
M = 128
Ncp = N // 4
fc = 30e9
delta_f = 960e3
QAM_mod = 16
MOD_TYPE = 'qam'
ComSNRdB = 20.0
RadarSNRdB = 40.0
# 三个数组逐项对应同一目标；单目标示例：[30.0]、[200.0]、[1.0+0j]。
target_pos = np.array([20.0, 30.0])
target_speed = np.array([200.0, -200.0])
target_amplitudes = np.array([1.0 + 0j, 0.8 * np.exp(0.3j)])
NPer = 16 * N
MPer = 128 * M
MUSIC_POINTS = NPer
MUSIC1D_NS = N // 2
MUSIC2D_NS = min(32, N)
MUSIC2D_MS = min(8, M)
MUSIC2D_RANGE_POINTS = 401
MUSIC2D_VELOCITY_POINTS = 601
JOINT_RANGE_LIMITS = None             # 如(0, 40)，单位m；None自动设置。
JOINT_VELOCITY_LIMITS = None          # 如(-300, 300)，单位m/s；None聚焦目标附近。
JOINT_DB_MIN = -60.0
SURFACE_MAX_POINTS = 151
RUN_COMMUNICATION = True
RUN_SER = False
RUN_AF = True
RUN_RMSE = False
RUN_FFT_2D = True
RUN_MUSIC_2D = True
SHOW_FIGURES = True
OUTPUT_DIR = './fig_MUSIC_FFT/'
output_dir = os.path.abspath(OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

EbN0dBs = np.arange(0, 37, 4)
SNRdBs = np.arange(-20, 10, 5)
nFrameCom = 500
nFrameRadar = 1000
COM_MC_CHANNEL = 'rayleigh'
COM_EQUALIZER = 'MMSE'
L = 10
PowerdB = np.array([0, -8, -17, -21, -25], dtype=float)
Delay = np.array([0, 3, 5, 6, 8])
RADAR_RANDOM_DATA_EACH_FRAME = False
AF_POINTS = 101

#%%>>>>>>>>>>>>>>>>>>>>>>>>  物理尺度与参数检查
if N <= 1 or M < 2 or not 0 <= Ncp <= N or delta_f <= 0 or fc <= 0:
    raise ValueError('请检查N、M、Ncp、delta_f和fc。')
if NPer < N or MPer < M or MUSIC_POINTS < 2:
    raise ValueError('FFT点数不能小于数据长度，MUSIC_POINTS必须>=2。')
positions = np.atleast_1d(np.asarray(target_pos, dtype=float))
speeds = np.atleast_1d(np.asarray(target_speed, dtype=float))
amplitudes = np.atleast_1d(np.asarray(target_amplitudes, dtype=complex))
if positions.ndim != 1 or positions.size == 0 or positions.shape != speeds.shape or positions.shape != amplitudes.shape:
    raise ValueError('target_pos、target_speed和target_amplitudes必须为等长一维数组。')
if not all(np.all(np.isfinite(a)) for a in (positions, speeds, amplitudes)) or np.any(amplitudes == 0):
    raise ValueError('目标参数必须有限，反射系数不能为零。')
Q = positions.size
if np.unique(np.column_stack((positions, speeds)), axis=0).shape[0] != Q:
    raise ValueError('相同距离和速度的目标在此模型中只能合并为一个目标。')
T = 1 / delta_f
dt = T / N
Tcp = Ncp * dt
Ts = T + Tcp
lambda_val = c0 / fc
target_delay = 2 * positions / c0
target_dop = 2 * speeds / lambda_val
range_resolution = c0 / (2 * N * delta_f)
velocity_resolution = lambda_val / (2 * M * Ts)
range_unambiguous = c0 / (2 * delta_f)
velocity_unambiguous = lambda_val / (4 * Ts)
if np.any(positions < 0) or np.any(positions >= range_unambiguous):
    raise ValueError('距离必须位于[0,c/(2*delta_f))内。')
if np.any(np.abs(speeds) >= velocity_unambiguous):
    raise ValueError('速度超出符号维采样的无模糊区间。')
range_padding = max(5.0, 0.2 * np.ptp(positions), 5 * range_resolution)
velocity_padding = max(20.0, 0.15 * np.ptp(speeds), 0.25 * velocity_resolution)
range_limits = (0.0, min(np.nextafter(range_unambiguous, 0), positions.max() + range_padding)) if JOINT_RANGE_LIMITS is None else tuple(JOINT_RANGE_LIMITS)
velocity_limits = (max(-velocity_unambiguous, speeds.min() - velocity_padding), min(np.nextafter(velocity_unambiguous, 0), speeds.max() + velocity_padding)) if JOINT_VELOCITY_LIMITS is None else tuple(JOINT_VELOCITY_LIMITS)
if not 0 <= range_limits[0] < range_limits[1] < range_unambiguous or not -velocity_unambiguous <= velocity_limits[0] < velocity_limits[1] < velocity_unambiguous:
    raise ValueError('检查绘图/搜索范围与无模糊区间。')
if np.any((positions < range_limits[0]) | (positions > range_limits[1])) or np.any((speeds < velocity_limits[0]) | (speeds > velocity_limits[1])):
    raise ValueError('当前仿真目标必须位于设置的二维MUSIC搜索范围内。')

# 保留用户现有的调制器及映射接口，不另写调制/解调函数。
#%%>>>>>>>>>>>>>>>>>>>>>>>>  通信符号与OFDM发射波形
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
_, demap_table = modem.getMappTable()
Es = float(Es)
if not np.isfinite(Es) or Es <= 0:
    raise ValueError('调制器返回的平均符号能量Es必须为正数。')
bits = np.random.randint(0, 2, N * M * bps, dtype=np.int8)
TxData = modem.modulate(bits).reshape(N, M)
if np.any(np.abs(TxData) == 0):
    raise ValueError('逐元素除法要求TxData没有零符号。')
_, TxSignal_time = ofdm_transmitter(TxData, Ncp)
print(f'N={N}, M={M}, Q={Q}, Ts={Ts*1e6:.6f} us')
print(f'距离分辨尺度={range_resolution:.6f} m；距离网格={c0/(2*NPer*delta_f):.6f} m')
print(f'速度分辨尺度={velocity_resolution:.6f} m/s；速度网格={lambda_val/(2*MPer*Ts):.6f} m/s')
print('感知采用忽略ICI的理想频域模型；不模拟完整时域回波。')
if np.any(target_delay > Tcp):
    print(f'存在超出CP覆盖距离{c0*Tcp/2:.3f} m的目标；当前理想模型未模拟相应ISI/ICI。')

#%%>>>>>>>>>>>>>>>>>>>>>>>>  通信链路与BER、SER
if RUN_COMMUNICATION:
    sym_int = np.array([demap_table[sym] for sym in TxData.flatten()])
    Power = 10**(PowerdB / 10)
    Power /= Power.sum()
    h = np.zeros(int(Delay.max()) + 1, dtype=complex)
    h[Delay] = (np.random.standard_normal(Delay.size) + 1j * np.random.standard_normal(Delay.size)) * np.sqrt(Power / 2)
    sigma2_com = Es * 10**(-ComSNRdB / 10)
    demodRxData = communication_receiver(TxData, h, Ncp, sigma2_com, COM_EQUALIZER, Es)
    demod_bits = modem.demodulate(demodRxData.flatten(), 'hard')
    sym_int_hat = []
    for j in range(M*N):
        sym_int_hat.append(int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base=2))
    sym_int_hat = np.array(sym_int_hat)
    print(f'通信BER={np.mean(bits != demod_bits):.6g}，SER={np.mean(sym_int != sym_int_hat):.6g}')
#%%>>>>>>>>>>>>>>>>>>>>>>>>  通信信道冲激响应图
if RUN_COMMUNICATION:
    fig, ax = plt.subplots(num=2, figsize=(8, 6), layout='constrained')
    ax.set(xlabel='Delay (samples)', ylabel='Magnitude', title='Communication channel impulse response')
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.stem(np.arange(h.size), np.abs(h), linefmt='C0-', markerfmt='C0o', basefmt='k-')
    fig.savefig(os.path.join(output_dir, 'OFDM_Fig02.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  感知回波与一维距离、速度估计
RxData_radar = radar_channel(TxData, delta_f, Ts, fc, positions, speeds, RadarSNRdB, Es, amplitudes)
dividerArray = RxData_radar / TxData
# 一维投影只能区分不同距离或不同速度，不能给出目标配对。
true_ranges, true_velocities = np.unique(positions), np.unique(speeds)
range_axis, range_dB, distanceE = range_estimation(dividerArray, delta_f, NPer, len(true_ranges))
music_axis, music_dB, distanceMUSIC = music_range(dividerArray, delta_f, MUSIC_POINTS, len(true_ranges), TxData, MUSIC1D_NS)
velocity_axis, velocity_dB, velocityE = velocity_estimation(dividerArray, Ts, fc, MPer, len(true_velocities))
print('一维IFFT距离峰(m):', np.sort(distanceE))
print('一维MUSIC距离峰(m):', np.sort(distanceMUSIC))
print('一维FFT速度峰(m/s):', np.sort(velocityE))

#%%>>>>>>>>>>>>>>>>>>>>>>>>  含CP的发射时域波形
fig, ax = plt.subplots(num=1, figsize=(8, 6), layout='constrained')
ax.set(xlabel='Time (us)', ylabel='Amplitude', title='OFDM transmit frame with cyclic prefixes')
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
t_tx = np.arange(TxSignal_time.size) * dt * 1e6
cp_mask = np.tile(np.arange(N + Ncp) < Ncp, M)
ax.plot(t_tx, TxSignal_time.real, color='#00A1F1', linewidth=1, label='Transmit signal')
ax.plot(t_tx, np.where(cp_mask, TxSignal_time.real, np.nan), color='#F65314', linewidth=1.5, label='Cyclic prefixes')
ax.legend()
fig.savefig(os.path.join(output_dir, 'OFDM_Fig01.pdf'), facecolor='white')
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
#%%>>>>>>>>>>>>>>>>>>>>>>>>  一维距离谱
fig, ax = plt.subplots(num=3, figsize=(8, 6), layout='constrained')
ax.set(xlabel='Range (m)', ylabel='Normalized spectrum (dB)', title='OFDM range estimation')
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.plot(range_axis, range_dB, color='#00A1F1', label='IFFT')
ax.plot(music_axis, music_dB, color='#F65314', label='MUSIC with smoothing')
for i, R in enumerate(true_ranges):
    ax.axvline(R, color='black', ls='--', lw=1.2, label='True range' if i == 0 else None)
ax.set_xlim(*range_limits)
ax.legend(prop=FontProperties(family='Times New Roman', size=13))
fig.savefig(os.path.join(output_dir, 'OFDM_Fig03.pdf'), facecolor='white')
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
#%%>>>>>>>>>>>>>>>>>>>>>>>>  一维速度谱
fig, ax = plt.subplots(num=4, figsize=(8, 6), layout='constrained')
ax.set(xlabel='Velocity (m/s)', ylabel='Normalized power (dB)', title='OFDM Doppler FFT')
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.plot(velocity_axis, velocity_dB, color='#00A1F1', label='FFT')
for i, v in enumerate(true_velocities):
    ax.axvline(v, color='black', ls='--', lw=1.2, label='True velocity' if i == 0 else None)
ax.set_xlim(*velocity_limits)
ax.legend()
fig.savefig(os.path.join(output_dir, 'OFDM_Fig04.pdf'), facecolor='white')
if SHOW_FIGURES:
    plt.show()
plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维FFT联合估计
if RUN_FFT_2D:
    fft_delay_axis, fft_doppler_axis, fft_power_dB, fft_estimates = fft_2d_estimation(dividerArray, delta_f, Ts, fc, NPer, MPer, Q)
    print('2D FFT联合估计，每行依次为tau(s)、fD(Hz)、R(m)、v(m/s)：')
    print(fft_estimates)
    fft_ranges = c0 * fft_delay_axis / 2
    fft_velocities = c0 * fft_doppler_axis / (2 * fc)
    fft_ir = np.flatnonzero((fft_ranges >= range_limits[0]) & (fft_ranges <= range_limits[1]))
    fft_iv = np.flatnonzero((fft_velocities >= velocity_limits[0]) & (fft_velocities <= velocity_limits[1]))
    fft_Zplot = np.maximum(fft_power_dB[np.ix_(fft_ir, fft_iv)], JOINT_DB_MIN)
    # 三维图仅抽稀显示网格，并保留真值及估计峰附近的网格点。
    peak_rows = [np.argmin(abs(fft_ranges[fft_ir] - value)) for value in np.r_[positions, fft_estimates[:, 2]]]
    peak_cols = [np.argmin(abs(fft_velocities[fft_iv] - value)) for value in np.r_[speeds, fft_estimates[:, 3]]]
    fft_rows = np.unique(np.r_[np.linspace(0, len(fft_ir) - 1, min(len(fft_ir), SURFACE_MAX_POINTS), dtype=int), peak_rows])
    fft_cols = np.unique(np.r_[np.linspace(0, len(fft_iv) - 1, min(len(fft_iv), SURFACE_MAX_POINTS), dtype=int), peak_cols])

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维FFT：Delay_Doppler 3D图
if RUN_FFT_2D:
    x, y = fft_delay_axis[fft_ir] * 1e6, fft_doppler_axis[fft_iv] / 1e3
    tx, ty = target_delay * 1e6, target_dop / 1e3
    ex, ey = fft_estimates[:, 0] * 1e6, fft_estimates[:, 1] / 1e3
    fig = plt.figure(num=9, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(x[fft_rows], y[fft_cols], indexing='xy')
    ZZ = fft_Zplot[np.ix_(fft_rows, fft_cols)].T
    surface_colors = plt.cm.turbo(plt.Normalize(JOINT_DB_MIN, 0)(ZZ))
    surface = ax.plot_surface(XX, YY, ZZ, facecolors=surface_colors, rcount=ZZ.shape[0], ccount=ZZ.shape[1], linewidth=1, shade=False)
    surface.set_facecolor((0, 0, 0, 0))
    for px, py, marker, color, label in ((tx, ty, 'x', 'black', 'True target'), (ex, ey, 'o', '#F65314', 'Estimate')):
        zz = [fft_Zplot[np.argmin(abs(x - a)), np.argmin(abs(y - b))] + 0.5 for a, b in zip(px, py)]
        ax.scatter(px, py, zz, marker=marker, color=color, s=48, linewidths=1.5, label=label, depthshade=False)
    ax.set_zlabel('Normalized spectrum (dB)', labelpad=10, fontsize=14)
    ax.set_zlim(JOINT_DB_MIN, 2)
    ax.tick_params(labelsize=14, width=1, pad=1)
    ax.set_box_aspect((1.25, 1, 0.8))
    ax.set_proj_type('ortho')
    ax.view_init(azim=-120, elev=30)
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
    ax.set(xlabel='Delay (us)', ylabel='Doppler (kHz)', title='2D FFT: Delay-Doppler spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DFFT_Delay_Doppler_3D_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维FFT：Delay_Doppler Top图
if RUN_FFT_2D:
    x, y = fft_delay_axis[fft_ir] * 1e6, fft_doppler_axis[fft_iv] / 1e3
    tx, ty = target_delay * 1e6, target_dop / 1e3
    ex, ey = fft_estimates[:, 0] * 1e6, fft_estimates[:, 1] / 1e3
    fig = plt.figure(num=10, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111)
    surface = ax.pcolormesh(x, y, fft_Zplot.T, shading='auto', cmap='turbo', vmin=JOINT_DB_MIN, vmax=0, rasterized=True)
    ax.scatter(tx, ty, marker='x', color='black', s=75, linewidths=1.8, label='True target')
    ax.scatter(ex, ey, marker='o', facecolors='none', edgecolors='#F65314', s=90, linewidths=1.8, label='Estimate')
    colorbar = fig.colorbar(surface, ax=ax, pad=0.025)
    colorbar.set_label('Normalized spectrum (dB)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.set(xlabel='Delay (us)', ylabel='Doppler (kHz)', title='2D FFT: Delay-Doppler spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DFFT_Delay_Doppler_Top_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维FFT：Range_Velocity 3D图
if RUN_FFT_2D:
    x, y = fft_ranges[fft_ir], fft_velocities[fft_iv]
    tx, ty = positions, speeds
    ex, ey = fft_estimates[:, 2], fft_estimates[:, 3]
    fig = plt.figure(num=11, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(x[fft_rows], y[fft_cols], indexing='xy')
    ZZ = fft_Zplot[np.ix_(fft_rows, fft_cols)].T
    surface_colors = plt.cm.turbo(plt.Normalize(JOINT_DB_MIN, 0)(ZZ))
    surface = ax.plot_surface(XX, YY, ZZ, facecolors=surface_colors, rcount=ZZ.shape[0], ccount=ZZ.shape[1], linewidth=1, shade=False)
    surface.set_facecolor((0, 0, 0, 0))
    for px, py, marker, color, label in ((tx, ty, 'x', 'black', 'True target'), (ex, ey, 'o', '#F65314', 'Estimate')):
        zz = [fft_Zplot[np.argmin(abs(x - a)), np.argmin(abs(y - b))] + 0.5 for a, b in zip(px, py)]
        ax.scatter(px, py, zz, marker=marker, color=color, s=48, linewidths=1.5, label=label, depthshade=False)
    ax.set_zlabel('Normalized spectrum (dB)', labelpad=10, fontsize=14)
    ax.set_zlim(JOINT_DB_MIN, 2)
    ax.tick_params(labelsize=14, width=1, pad=1)
    ax.set_box_aspect((1.25, 1, 0.8))
    ax.set_proj_type('ortho')
    ax.view_init(azim=-120, elev=30)
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
    ax.set(xlabel='Range (m)', ylabel='Velocity (m/s)', title='2D FFT: Range-Velocity spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DFFT_Range_Velocity_3D_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维FFT：Range_Velocity Top图
if RUN_FFT_2D:
    x, y = fft_ranges[fft_ir], fft_velocities[fft_iv]
    tx, ty = positions, speeds
    ex, ey = fft_estimates[:, 2], fft_estimates[:, 3]
    fig = plt.figure(num=12, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111)
    surface = ax.pcolormesh(x, y, fft_Zplot.T, shading='auto', cmap='turbo', vmin=JOINT_DB_MIN, vmax=0, rasterized=True)
    ax.scatter(tx, ty, marker='x', color='black', s=75, linewidths=1.8, label='True target')
    ax.scatter(ex, ey, marker='o', facecolors='none', edgecolors='#F65314', s=90, linewidths=1.8, label='Estimate')
    colorbar = fig.colorbar(surface, ax=ax, pad=0.025)
    colorbar.set_label('Normalized spectrum (dB)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.set(xlabel='Range (m)', ylabel='Velocity (m/s)', title='2D FFT: Range-Velocity spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DFFT_Range_Velocity_Top_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC联合估计
if RUN_MUSIC_2D:
    music_delay_axis = np.linspace(*range_limits, MUSIC2D_RANGE_POINTS) * 2 / c0
    music_doppler_axis = np.linspace(*velocity_limits, MUSIC2D_VELOCITY_POINTS) * 2 * fc / c0
    music_delay_axis, music_doppler_axis, music_power_dB, music_estimates = music_2d_estimation(dividerArray, TxData, delta_f, Ts, fc, MUSIC2D_NS, MUSIC2D_MS, music_delay_axis, music_doppler_axis, Q)
    print('2D MUSIC联合估计，每行依次为tau(s)、fD(Hz)、R(m)、v(m/s)：')
    print(music_estimates)
    music_ranges = c0 * music_delay_axis / 2
    music_velocities = c0 * music_doppler_axis / (2 * fc)
    music_ir = np.flatnonzero((music_ranges >= range_limits[0]) & (music_ranges <= range_limits[1]))
    music_iv = np.flatnonzero((music_velocities >= velocity_limits[0]) & (music_velocities <= velocity_limits[1]))
    music_Zplot = np.maximum(music_power_dB[np.ix_(music_ir, music_iv)], JOINT_DB_MIN)
    # 三维图仅抽稀显示网格，并保留真值及估计峰附近的网格点。
    peak_rows = [np.argmin(abs(music_ranges[music_ir] - value)) for value in np.r_[positions, music_estimates[:, 2]]]
    peak_cols = [np.argmin(abs(music_velocities[music_iv] - value)) for value in np.r_[speeds, music_estimates[:, 3]]]
    music_rows = np.unique(np.r_[np.linspace(0, len(music_ir) - 1, min(len(music_ir), SURFACE_MAX_POINTS), dtype=int), peak_rows])
    music_cols = np.unique(np.r_[np.linspace(0, len(music_iv) - 1, min(len(music_iv), SURFACE_MAX_POINTS), dtype=int), peak_cols])

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC：Delay_Doppler 3D图
if RUN_MUSIC_2D:
    x, y = music_delay_axis[music_ir] * 1e6, music_doppler_axis[music_iv] / 1e3
    tx, ty = target_delay * 1e6, target_dop / 1e3
    ex, ey = music_estimates[:, 0] * 1e6, music_estimates[:, 1] / 1e3
    fig = plt.figure(num=13, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(x[music_rows], y[music_cols], indexing='xy')
    ZZ = music_Zplot[np.ix_(music_rows, music_cols)].T
    surface_colors = plt.cm.turbo(plt.Normalize(JOINT_DB_MIN, 0)(ZZ))
    surface = ax.plot_surface(XX, YY, ZZ, facecolors=surface_colors, rcount=ZZ.shape[0], ccount=ZZ.shape[1], linewidth=1, shade=False)
    surface.set_facecolor((0, 0, 0, 0))
    for px, py, marker, color, label in ((tx, ty, 'x', 'black', 'True target'), (ex, ey, 'o', '#F65314', 'Estimate')):
        zz = [music_Zplot[np.argmin(abs(x - a)), np.argmin(abs(y - b))] + 0.5 for a, b in zip(px, py)]
        ax.scatter(px, py, zz, marker=marker, color=color, s=48, linewidths=1.5, label=label, depthshade=False)
    ax.set_zlabel('Normalized spectrum (dB)', labelpad=10, fontsize=14)
    ax.set_zlim(JOINT_DB_MIN, 2)
    ax.tick_params(labelsize=14, width=1, pad=1)
    ax.set_box_aspect((1.25, 1, 0.8))
    ax.set_proj_type('ortho')
    ax.view_init(azim=-120, elev=30)
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
    ax.set(xlabel='Delay (us)', ylabel='Doppler (kHz)', title='2D MUSIC: Delay-Doppler spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DMUSIC_Delay_Doppler_3D_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC：Delay_Doppler Top图
if RUN_MUSIC_2D:
    x, y = music_delay_axis[music_ir] * 1e6, music_doppler_axis[music_iv] / 1e3
    tx, ty = target_delay * 1e6, target_dop / 1e3
    ex, ey = music_estimates[:, 0] * 1e6, music_estimates[:, 1] / 1e3
    fig = plt.figure(num=14, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111)
    surface = ax.pcolormesh(x, y, music_Zplot.T, shading='auto', cmap='turbo', vmin=JOINT_DB_MIN, vmax=0, rasterized=True)
    ax.scatter(tx, ty, marker='x', color='black', s=75, linewidths=1.8, label='True target')
    ax.scatter(ex, ey, marker='o', facecolors='none', edgecolors='#F65314', s=90, linewidths=1.8, label='Estimate')
    colorbar = fig.colorbar(surface, ax=ax, pad=0.025)
    colorbar.set_label('Normalized spectrum (dB)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.set(xlabel='Delay (us)', ylabel='Doppler (kHz)', title='2D MUSIC: Delay-Doppler spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DMUSIC_Delay_Doppler_Top_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC：Range_Velocity 3D图
if RUN_MUSIC_2D:
    x, y = music_ranges[music_ir], music_velocities[music_iv]
    tx, ty = positions, speeds
    ex, ey = music_estimates[:, 2], music_estimates[:, 3]
    fig = plt.figure(num=15, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(x[music_rows], y[music_cols], indexing='xy')
    ZZ = music_Zplot[np.ix_(music_rows, music_cols)].T
    surface_colors = plt.cm.turbo(plt.Normalize(JOINT_DB_MIN, 0)(ZZ))
    surface = ax.plot_surface(XX, YY, ZZ, facecolors=surface_colors, rcount=ZZ.shape[0], ccount=ZZ.shape[1], linewidth=1, shade=False)
    surface.set_facecolor((0, 0, 0, 0))
    for px, py, marker, color, label in ((tx, ty, 'x', 'black', 'True target'), (ex, ey, 'o', '#F65314', 'Estimate')):
        zz = [music_Zplot[np.argmin(abs(x - a)), np.argmin(abs(y - b))] + 0.5 for a, b in zip(px, py)]
        ax.scatter(px, py, zz, marker=marker, color=color, s=48, linewidths=1.5, label=label, depthshade=False)
    ax.set_zlabel('Normalized spectrum (dB)', labelpad=10, fontsize=14)
    ax.set_zlim(JOINT_DB_MIN, 2)
    ax.tick_params(labelsize=14, width=1, pad=1)
    ax.set_box_aspect((1.25, 1, 0.8))
    ax.set_proj_type('ortho')
    ax.view_init(azim=-120, elev=30)
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
    ax.set(xlabel='Range (m)', ylabel='Velocity (m/s)', title='2D MUSIC: Range-Velocity spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DMUSIC_Range_Velocity_3D_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  二维MUSIC：Range_Velocity Top图
if RUN_MUSIC_2D:
    x, y = music_ranges[music_ir], music_velocities[music_iv]
    tx, ty = positions, speeds
    ex, ey = music_estimates[:, 2], music_estimates[:, 3]
    fig = plt.figure(num=16, figsize=(8, 6), layout='constrained')
    ax = fig.add_subplot(111)
    surface = ax.pcolormesh(x, y, music_Zplot.T, shading='auto', cmap='turbo', vmin=JOINT_DB_MIN, vmax=0, rasterized=True)
    ax.scatter(tx, ty, marker='x', color='black', s=75, linewidths=1.8, label='True target')
    ax.scatter(ex, ey, marker='o', facecolors='none', edgecolors='#F65314', s=90, linewidths=1.8, label='Estimate')
    colorbar = fig.colorbar(surface, ax=ax, pad=0.025)
    colorbar.set_label('Normalized spectrum (dB)', fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.set(xlabel='Range (m)', ylabel='Velocity (m/s)', title='2D MUSIC: Range-Velocity spectrum', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.legend(loc='upper right', prop=FontProperties(family='Times New Roman', size=12), edgecolor='black', framealpha=1)
    fig.savefig(os.path.join(output_dir, f'OFDM_2DMUSIC_Range_Velocity_Top_{QAM_mod}{MOD_TYPE}.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  发射波形模糊函数及其绘图
if RUN_AF:
    if AF_POINTS < 3 or AF_POINTS % 2 == 0:
        raise ValueError('AF_POINTS必须是不小于3的奇数。')
    tau_max = max(2 * target_delay.max(), 4 * dt)
    fd_max = max(2 * np.abs(target_dop).max(), 2 / (M * Ts))
    taus = np.linspace(-tau_max, tau_max, AF_POINTS)
    fds = np.linspace(-fd_max, fd_max, AF_POINTS)
    af = ambiguity_function(TxSignal_time, dt, taus, fds)
    af_db = power_db(af**2)
    X, Y = np.meshgrid(fds / 1e3, taus * 1e9)
    fig = plt.figure(num=6, figsize=(14, 9), layout='constrained')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surface_colors = plt.cm.turbo(plt.Normalize(0, 1)(af))
    surface = ax.plot_surface(X, Y, af, facecolors=surface_colors, rcount=af.shape[0], ccount=af.shape[1], linewidth=1, shade=False)
    surface.set_facecolor((0, 0, 0, 0))
    ax.tick_params(labelsize=14, width=1, pad=1)
    ax.set_box_aspect((1.25, 1, 0.8))
    ax.set_proj_type('ortho')
    ax.view_init(azim=-120, elev=30)
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 1))
    ax.set(xlabel='Doppler (kHz)', ylabel='Delay (ns)', zlabel='AF magnitude', title='Transmit waveform ambiguity function')
    ax = fig.add_subplot(2, 2, 2)
    contour = ax.contourf(X, Y, af_db, levels=np.linspace(-60, 0, 31), cmap='turbo', extend='min')
    fig.colorbar(contour, ax=ax, label='Normalized power (dB)')
    ax.set(xlabel='Doppler (kHz)', ylabel='Delay (ns)', title='AF contour')
    for column, axis, values, xlabel, title in ((3, taus*1e9, af_db[:, AF_POINTS//2], 'Delay (ns)', 'AF: zero Doppler'), (4, fds/1e3, af_db[AF_POINTS//2, :], 'Doppler (kHz)', 'AF: zero delay')):
        ax = fig.add_subplot(2, 2, column)
        ax.plot(axis, values, color='#00A1F1')
        ax.set(xlabel=xlabel, ylabel='Normalized power (dB)', title=title)
        ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    fig.savefig(os.path.join(output_dir, 'OFDM_Fig06.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  通信SER蒙特卡洛仿真
if RUN_SER:
    if nFrameCom < 1 or L < 1 or L > N or L - 1 > Ncp or COM_MC_CHANNEL.lower() not in ('rayleigh', 'awgn'):
        raise ValueError('检查通信蒙特卡洛参数。')
    SER_sim = np.zeros(len(EbN0dBs))
    for idx, ebn0 in enumerate(EbN0dBs):
        errors = 0
        sigma2 = Es * (1 + Ncp / N) / (bps * 10**(ebn0 / 10))
        for _ in range(nFrameCom):
            frame_bits = np.random.randint(0, 2, N * M * bps, dtype=np.int8)
            frame_data = modem.modulate(frame_bits)
            frame_labels = np.array([demap_table[sym] for sym in frame_data])
            frame_data = frame_data.reshape(N, M)
            h_mc = (np.random.standard_normal(L) + 1j * np.random.standard_normal(L)) / np.sqrt(2 * L) if COM_MC_CHANNEL.lower() == 'rayleigh' else np.array([1.0 + 0j])
            equalized = communication_receiver(frame_data, h_mc, Ncp, sigma2, 'ZF', Es)
            demod_bits = modem.demodulate(equalized.flatten(), 'hard')
            sym_int_hat = []
            for j in range(M*N):
                sym_int_hat.append(int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base=2))
            errors += int(np.sum(frame_labels != np.array(sym_int_hat)))
        SER_sim[idx] = errors / (nFrameCom * N * M)
        print(f'SER {idx+1}/{len(EbN0dBs)}: Eb/N0={ebn0:g} dB, SER={SER_sim[idx]:.6g}')
    effective = EbN0dBs - 10 * np.log10(1 + Ncp / N)
    theory = ser_rayleigh(effective, MOD_TYPE, QAM_mod) if COM_MC_CHANNEL.lower() == 'rayleigh' else ser_awgn(effective, MOD_TYPE, QAM_mod)
    #%>>>>>>>>>>>>>>>>>>>>>>>>  通信SER曲线
    fig, ax = plt.subplots(num=7, figsize=(8, 6), layout='constrained')
    ax.set(xlabel='Eb/N0 including CP energy (dB)', ylabel='SER', title=f'OFDM {QAM_mod}-QAM')
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.semilogy(EbN0dBs, np.where(SER_sim > 0, SER_sim, np.nan), 'o-', label='Simulation (ZF)')
    ax.semilogy(EbN0dBs, theory, '*--', label='Theory with CP overhead')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'OFDM_Fig07.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  测距RMSE蒙特卡洛仿真
if RUN_RMSE:
    if nFrameRadar < 1:
        raise ValueError('nFrameRadar必须>=1。')
    RmseList = np.zeros(len(SNRdBs))
    for idx, snr in enumerate(SNRdBs):
        mse = 0.0
        for _ in range(nFrameRadar):
            frame_data = modem.modulate(np.random.randint(0, 2, N * M * bps, dtype=np.int8)).reshape(N, M) if RADAR_RANDOM_DATA_EACH_FRAME else TxData
            rx = radar_channel(frame_data, delta_f, Ts, fc, positions, speeds, snr, Es, amplitudes)
            _, _, estimates = range_estimation(rx / frame_data, delta_f, NPer, len(true_ranges))
            # 一维平方距离下，排序后匹配等价于最小总平方误差的一一匹配。
            mse += np.mean((np.sort(estimates) - true_ranges)**2)
        RmseList[idx] = np.sqrt(mse / nFrameRadar)
        print(f'RMSE {idx+1}/{len(SNRdBs)}: SNR={snr:g} dB, RMSE={RmseList[idx]:.6g} m')
    #%>>>>>>>>>>>>>>>>>>>>>>>>  测距RMSE曲线
    fig, ax = plt.subplots(num=8, figsize=(8, 6), layout='constrained')
    ax.set(xlabel='Radar input SNR (dB)', ylabel='Range RMSE (m)', title='IFFT: matched distinct range peaks')
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.plot(SNRdBs, RmseList, 'o-', color='#00A1F1', label='IFFT')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'OFDM_Fig08.pdf'), facecolor='white')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
