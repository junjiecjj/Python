#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 16:18:48 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247507635&idx=1&sn=eb0d31ff78a7e641785b0e40eb907868&chksm=c2eed7d1df414d437bb02f596b1afb75f791da3c88e78f15863e1d6e75d8d5b5c61ac229ed21&mpshare=1&scene=1&srcid=0925jjW0fD2KWAxqTlaKo2Rk&sharer_shareinfo=bc50b7611b96b98c72744962d8ce41ea&sharer_shareinfo_first=bc50b7611b96b98c72744962d8ce41ea&exportkey=n_ChQIAhIQPXtJNfMj8CzCheChxN23dBKfAgIE97dBBAEAAAAAADUfIn8xL4oAAAAOpnltbLcz9gKNyK89dVj02gUlnKz7gYbh9%2FLXRpd%2BpeoGcsQ60YUp95PnK47ymL21%2FZCfI0%2B2ptIpUhSWyGnUQxmo3ATWqPkGum44XrfCPKu6UhDmO2yzY%2FbCMi8k37mzMJ%2Bahropq2lZkzU8KfaCRUAmzNf82OL8BLT9qYsCU9SGfqym8De6b20vxTOwi2Acas1pt3yvR%2BfdYP7Yf%2B308C8belZzc4FOC55yNfLWa1U7nQECoEGql4cvx4DZNLBZyZhG0%2BqXezIW%2FuAqmTnu9TLgsLKFygziQGVoXcAjNAvkYsiyF9STy3MA55VkCr9KVsNSgjYoyYlgN8WgyBVzCM2ozktcR3%2FF&acctmode=0&pass_ticket=FEuGza9kbp1SRFsAP%2BHoGRrR62%2B%2FWzE%2BM9YG70LPFODlPHXPdKoSnh6KMDazLiOS&wx_header=0&poc_token=HESU1mij8xvlrVogIFgI2_jzu5A0M8fG5pjLPsDb

知识星球
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftshift
from scipy.constants import c
from scipy.interpolate import interp1d
from Modulations import modulator
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

#%%
def awgn(signal, snr_db, measured=True):
    """添加AWGN噪声"""
    if measured:
        signal_power = np.mean(np.abs(signal)**2)
    else:
        signal_power = 1.0  # 假设单位功率

    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

#%%
# ===================== 1. 系统参数配置 =====================
c0 = c
fc = 30e9                        # 载波频率30GHz
lambda_ = c0 / fc                # 波长=0.01m
N = 256                          # 子载波数
M = 16                           # 符号数
delta_f = 15e3 * 2**6            # 子载波间隔=1.536MHz
T = 1 / delta_f                  # 符号时长≈651ns
Tcp = T / 4                      # CP时长≈163ns
Ts = T + Tcp                     # 总符号周期≈814ns
CPsize = int(N / 4)              # CP长度=64点
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
map_table, demap_table = modem.getMappTable()

# ===================== 2. 发射机模块 =====================
# 生成随机数据并进行QAM调制
bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
TxData = modem.modulate(bits)
sym_int = np.array([demap_table[sym] for sym in TxData])
TxData = TxData.reshape(N, M)

# OFDM调制（IFFT+CP）
TxSignal_IFFT = np.fft.ifft(TxData, axis=0)  # 时域信号（N点IFFT）
TxSignal_cp = np.vstack([TxSignal_IFFT[-CPsize:, :], TxSignal_IFFT])  # 添加CP
TxSignal_time = TxSignal_cp.T.flatten()  # 一维时域发射信号

# ===================== 3. 通信信道与接收 =====================
# 多径信道参数
PowerdB = np.array([0, -8, -17, -21, -25])     # 信道抽头功率（dB）
Delay = np.array([0, 3, 5, 6, 8])              # 抽头延迟（采样点）
Power = 10**(PowerdB/10)                       # 功率转线性值
Lch = Delay[-1] + 1                            # 信道长度=9点
h = np.zeros(Lch, dtype=complex)
h[Delay] = (np.random.randn(len(Power)) + 1j*np.random.randn(len(Power))) * np.sqrt(Power/2)

# 通信传输（卷积+加噪）
ComSNRdB = 20
RxSignal_com = np.convolve(TxSignal_time, h, mode='full')
RxSignal_com = RxSignal_com[:len(TxSignal_time)]  # 截断至发射信号长度
RxSignal_com = awgn(RxSignal_com, ComSNRdB, measured=True)

# 通信接收（去CP+FFT+MMSE均衡+解调）
RxSignal_com_reshape = RxSignal_com.reshape(N + CPsize, M, order = 'F')
RxSignal_remove_cp = RxSignal_com_reshape[CPsize:, :]  # 去除CP
RxData_com = np.fft.fft(RxSignal_remove_cp, axis=0)  # FFT变换到频域

# MMSE均衡
H_channel = np.fft.fft(np.hstack([h, np.zeros(N - Lch)]))  # 频域信道响应
H_channel = np.tile(H_channel.reshape(-1, 1), (1, M))  # 扩展到匹配符号数
C = np.conj(H_channel) / (np.abs(H_channel)**2 + 10**(-ComSNRdB/10))  # MMSE均衡器
demodRxData = RxData_com * C

demod_bits = modem.demodulate(demodRxData.flatten(), 'hard')
sym_int_hat = []
for j in range(M*N):
    sym_int_hat.append( int(''.join([str(num) for num in demod_bits[j*bps:(j+1)*bps]]), base = 2) )
sym_int_hat = np.array(sym_int_hat)
# 误码统计
errorCount = np.sum(sym_int_hat != sym_int)
print(f'通信误码数: {errorCount}')

# ===================== 4. 雷达信道与目标探测 =====================
target_pos = 30                  # 目标真实距离=30m
target_speed = 20                # 目标速度=20m/s
target_delay = 2 * target_pos / c0  # 双程时延
target_dop = 2 * target_speed / lambda_  # 目标多普勒频移
RadarSNRdB = 30                  # 雷达信噪比
RadarSNR = 10**(RadarSNRdB/10)

# 生成子载波和符号索引网格
kSub, mSym = np.meshgrid(np.arange(M), np.arange(N))

# 相位偏移公式
phase_shift = -1j * 2 * np.pi * (
    fc * target_delay +
    mSym * delta_f * target_delay -
    kSub * Ts * target_dop
)

noise = (np.random.randn(N, M) + 1j*np.random.randn(N, M)) / np.sqrt(2)
RxData_radar = np.sqrt(RadarSNR) * TxData * np.exp(phase_shift) + noise

# 雷达信号处理（FFT距离估计）
dividerArray = RxData_radar / TxData  # 抵消发射信号
NPer = 16 * N                         # 补零点数
range_fft = np.fft.ifft(dividerArray, n=NPer, axis=0)  # 距离维FFT
range_power = np.abs(range_fft)
mean_range_power = np.mean(range_power, axis=1)  # 沿符号方向平均
mean_range_power = mean_range_power / np.max(mean_range_power)
mean_range_power_dB = 10*np.log10(mean_range_power + 1e-12)

# 距离轴计算
range_axis = np.arange(NPer) * c0 / (2 * NPer * delta_f)
rangeEst_idx = np.argmax(mean_range_power)
distanceE = range_axis[rangeEst_idx]

print(f'目标估计距离: {distanceE:.2f} m (真实距离: {target_pos} m)')

# ===================== 5. 模糊函数计算 =====================
tau_max = 2 * target_delay
fd_max = 2 * target_dop
tau_points = 150
fd_points = 150

tau_range = np.linspace(-tau_max, tau_max, tau_points)
fd_range = np.linspace(-fd_max, fd_max, fd_points)

# 获取发射信号和采样时间
s_t = TxSignal_time.flatten()
t_seq = np.arange(len(s_t)) * (1/(N*delta_f))

# 计算模糊函数
ambiguity_func = np.zeros((len(tau_range), len(fd_range)))

for i, tau in enumerate(tau_range):
    # 时移信号
    t_shifted = t_seq - tau
    valid_idx = (t_shifted >= 0) & (t_shifted <= t_seq[-1])

    if np.any(valid_idx):
        interp_func = interp1d(t_seq, s_t, kind='linear', bounds_error=False, fill_value=0)
        s_tau = interp_func(t_shifted[valid_idx])

        s_tau_padded = np.zeros_like(s_t)
        s_tau_padded[valid_idx] = s_tau

        for j, fd in enumerate(fd_range):
            doppler_shift = np.exp(1j*2*np.pi*fd*t_seq)
            ambiguity_func[i, j] = np.abs( s_t.conj() @( s_tau_padded * doppler_shift))

# 归一化
ambiguity_func = ambiguity_func / np.max(ambiguity_func)

# 提取切片
zero_dop_idx = np.argmin(np.abs(fd_range))
zero_tau_idx = np.argmin(np.abs(tau_range))

range_slice = ambiguity_func[:, zero_dop_idx]
doppler_slice = ambiguity_func[zero_tau_idx, :]

range_slice_dB = 10*np.log10(range_slice + 1e-12)
doppler_slice_dB = 10*np.log10(doppler_slice + 1e-12)

# ===================== 6. 图像绘制 =====================
fig = plt.figure(figsize=(20, 10))

# 子图1：发射信号时域
plt.subplot(2, 4, 1)
t_tx = np.arange(len(TxSignal_time)) * (1/(N*delta_f)) * 1e9
plt.plot(t_tx, np.real(TxSignal_time), 'b-', linewidth=1)
cp_end_idx = CPsize * M
plt.plot(t_tx[:cp_end_idx], np.real(TxSignal_time[:cp_end_idx].flatten()), 'r--', linewidth=1)
plt.xlabel('时间 (ns)'); plt.ylabel('幅度'); plt.title('OFDM发射信号时域（含CP）')
plt.legend(['数据部分', '循环前缀(CP)']); plt.grid(True)

# 子图2：发射信号频域
plt.subplot(2, 4, 2)
freq_tx = np.arange(-N//2, N//2) * delta_f / 1e6
Tx_fft_shift = fftshift(np.abs(TxData[:, 0]))
plt.plot(freq_tx, 10*np.log10(Tx_fft_shift + 1e-12), 'g-', linewidth=1)
plt.xlabel('频率 (MHz)'); plt.ylabel('功率 (dB)'); plt.title('OFDM发射信号频域'); plt.grid(True)

# 子图3：通信信道冲击响应
plt.subplot(2, 4, 3)
plt.stem(np.arange(Lch), np.abs(h), linefmt='m-', markerfmt='mo', basefmt='m-')
plt.xlabel('延迟 (采样点)'); plt.ylabel('幅度'); plt.title('通信信道冲击响应'); plt.grid(True)

# 子图4：雷达距离-功率谱
plt.subplot(2, 4, 4)
plt.plot(range_axis, mean_range_power_dB, 'b-', linewidth=1)
plt.axvline(x=target_pos, color='r', linestyle='--', linewidth=1.2)
plt.axvline(x=distanceE, color='g', linestyle=':', linewidth=1.2)
plt.xlabel('距离 (m)'); plt.ylabel('归一化功率 (dB)'); plt.title('雷达目标距离-功率谱')
plt.legend(['功率谱', '真实距离', '估计距离']); plt.grid(True); plt.xlim(0, 60)

# 子图5：模糊函数3D图
plt.subplot(2, 4, 5)
X, Y = np.meshgrid(fd_range/1000, tau_range*1e9)
ax = fig.add_subplot(2, 4, 5, projection='3d')
ax.plot_surface(X, Y, ambiguity_func, cmap='jet', edgecolor='none')
ax.set_xlabel('多普勒频移 (kHz)'); ax.set_ylabel('时延 (ns)'); ax.set_zlabel('归一化模糊度')
ax.set_title('OFDM信号模糊函数（3D）')

# 子图6：模糊函数等高线
plt.subplot(2, 4, 6)
target_tau_idx = np.argmin(np.abs(tau_range - target_delay))
target_fd_idx = np.argmin(np.abs(fd_range - target_dop))
plt.contourf(X, Y, ambiguity_func, 30, cmap='jet')
plt.plot(fd_range[target_fd_idx]/1000, tau_range[target_tau_idx]*1e9, 'ro', markersize=6, markerfacecolor='r')
plt.xlabel('多普勒频移 (kHz)'); plt.ylabel('时延 (ns)'); plt.title('模糊函数等高线')
plt.colorbar(); plt.grid(True)

# 子图7：距离切片
plt.subplot(2, 4, 7)
plt.plot(tau_range*1e9, range_slice_dB, 'b-', linewidth=1.5)
plt.axvline(x=target_delay*1e9, color='r', linestyle='--', linewidth=1.5)
plt.xlabel('时延 (ns)'); plt.ylabel('归一化幅度 (dB)'); plt.title('距离切片（零多普勒）')
plt.legend(['距离响应', '目标位置']); plt.grid(True)

# 子图8：速度切片
plt.subplot(2, 4, 8)
plt.plot(fd_range/1000, doppler_slice_dB, 'b-', linewidth=1.5)
plt.axvline(x=target_dop/1000, color='r', linestyle='--', linewidth=1.5)
plt.xlabel('多普勒频移 (kHz)'); plt.ylabel('归一化幅度 (dB)'); plt.title('速度切片（零时延）')
plt.legend(['速度响应', '目标位置']); plt.grid(True)

plt.suptitle('OFDM雷达-通信一体化系统仿真结果（含模糊函数分析）', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ===================== 7. 单独绘制模糊函数详细图 =====================
fig2 = plt.figure(figsize=(12, 10))

# 3D模糊函数
ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X, Y, ambiguity_func, cmap='jet', edgecolor='none')
ax1.set_xlabel('多普勒频移 (kHz)'); ax1.set_ylabel('时延 (ns)'); ax1.set_zlabel('归一化模糊度')
ax1.set_title('OFDM信号模糊函数（3D）')

# 等高线图
plt.subplot(2, 2, 2)
plt.contourf(X, Y, ambiguity_func, 30, cmap='jet')
plt.plot(fd_range[target_fd_idx]/1000, tau_range[target_tau_idx]*1e9, 'ro', markersize=8, markerfacecolor='r')
plt.xlabel('多普勒频移 (kHz)'); plt.ylabel('时延 (ns)'); plt.title('模糊函数等高线')
plt.colorbar(); plt.grid(True)

# 距离切片
plt.subplot(2, 2, 3)
plt.plot(tau_range*1e9, range_slice_dB, 'b-', linewidth=2)
plt.axvline(x=target_delay*1e9, color='r', linestyle='--', linewidth=2)
plt.xlabel('时延 (ns)'); plt.ylabel('归一化幅度 (dB)'); plt.title('距离切片（零多普勒）')
plt.legend(['距离响应', '目标位置']); plt.grid(True)

# 速度切片
plt.subplot(2, 2, 4)
plt.plot(fd_range/1000, doppler_slice_dB, 'b-', linewidth=2)
plt.axvline(x=target_dop/1000, color='r', linestyle='--', linewidth=2)
plt.xlabel('多普勒频移 (kHz)'); plt.ylabel('归一化幅度 (dB)'); plt.title('速度切片（零时延）')
plt.legend(['速度响应', '目标位置']); plt.grid(True)

plt.suptitle('OFDM信号模糊函数详细分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()










