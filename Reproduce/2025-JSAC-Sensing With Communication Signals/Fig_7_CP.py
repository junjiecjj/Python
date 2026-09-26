#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:01:24 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OJSP Fig. 2: Sensing range profile with a shared communication transmitter.

Symbols -> OFDM modulation -> add CP -> manual upsampling -> RRC pulse shaping
-> physical target delays -> remove high-rate CP -> periodic matched filtering
-> ensemble average of squared outputs -> range profile.
"""

import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

from tqdm import tqdm
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem


#%% 全局绘图设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 13


#%% 单位能量 SRRC 脉冲，与 rcosdesign(alpha, span, Q, 'sqrt') 对应
def srrcFunction(alpha, Q, span):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1].')
    if (span * Q) % 2 != 0:
        raise ValueError('span * Q must be even.')

    t = np.arange(-(span * Q) // 2, (span * Q) // 2 + 1, dtype=float) / Q

    if alpha == 0:
        p = np.sinc(t)
    else:
        p = np.zeros(t.size)
        zeroIndex = np.isclose(t, 0, rtol=0, atol=1e-12)
        singularIndex = np.isclose(np.abs(4 * alpha * t), 1, rtol=0, atol=1e-12)
        regularIndex = ~(zeroIndex | singularIndex)
        tr = t[regularIndex]

        p[regularIndex] = (np.sin(np.pi * tr * (1-alpha)) + 4 * alpha * tr * np.cos(np.pi * tr * (1+alpha))) / (np.pi * tr * (1-(4 * alpha * tr)**2))
        p[zeroIndex] = 1 + alpha * (4/np.pi - 1)
        p[singularIndex] = alpha / np.sqrt(2) * ((1+2/np.pi) * np.sin(np.pi/(4*alpha)) + (1-2/np.pi) * np.cos(np.pi/(4*alpha)))

    p = p / np.sqrt(np.sum(np.abs(p)**2))
    filtDelay = (p.size - 1) / 2
    return p, t, filtDelay


def add_cyclic_prefix(x, Ncp):
    return np.hstack((x[-Ncp:], x)) if Ncp > 0 else x.copy()


#%% 仿真参数
np.random.seed(42)

MOD_TYPE = "psk"                    # "psk", "qam", "pam"
Order = 16
modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem, 'fsk': FSKModem}

if MOD_TYPE.lower() not in ('psk', 'qam', 'pam'):
    raise ValueError('This simulation supports PSK, QAM, and PAM.')

modem = modem_dict[MOD_TYPE.lower()](Order)

SNRdB = -10
targetRange = np.array([10, 20, 25], dtype=float)
targetAmplitude_dB = np.array([0, -10, -30], dtype=float)
targetPhase = np.array([0, 0, 0], dtype=float)
K = targetRange.size
Iter = 2000

N = 4096
Q = 20
alpha = 0.35
span = 20

rangeSampleSpacing = 0.025          # 距离栅格间隔，单位 m
desiredCPRange = 40                 # CP 总时长对应的预设距离，单位 m
c0 = 299792458.0                    # 光速，单位 m/s
eps = np.finfo(float).eps

# 对应论文中的复反射系数 beta_q
gamma = 10**(targetAmplitude_dB / 20) * np.exp(1j * targetPhase)

#%% 物理距离、传播时延与高采样率时延索引
Fs = c0 / (2 * rangeSampleSpacing)
Ts = 1 / Fs
symbolPeriod = Q * Ts
symbolRangeSpacing = Q * rangeSampleSpacing

targetPropagationDelay = 2 * targetRange / c0
targetDelaySample = np.floor(targetPropagationDelay / Ts + 0.5).astype(int)
targetRangeGrid = targetDelaySample * c0 * Ts / 2
targetRangeError = targetRangeGrid - targetRange
maxTargetDelaySample = int(np.max(targetDelaySample))


#%% 发射脉冲与 CP 长度
p_t, t, filtDelay = srrcFunction(alpha, Q, span)
Lp = p_t.size
pulseMemorySample = Lp - 1

NcpFromDesiredRange = int(np.ceil(desiredCPRange / symbolRangeSpacing))
NcpFromTotalMemory = int(np.ceil((pulseMemorySample + maxTargetDelaySample) / Q))
Ncp = max(NcpFromDesiredRange, NcpFromTotalMemory)

if Ncp > N:
    raise ValueError('Ncp must not exceed N.')

M = N + Ncp
NcpSample = Q * Ncp
Tcp = Ncp * symbolPeriod
cpSupportedRange = NcpSample * c0 * Ts / 2
maxRangeWithPulseMemory = (NcpSample - pulseMemorySample) * c0 * Ts / 2

lengthBlock = Q * N
Kt = Q * M + Lp - 1
Ks = Kt + maxTargetDelaySample

if NcpSample < pulseMemorySample + maxTargetDelaySample:
    raise ValueError('The CP must cover pulse memory plus maximum target delay.')

print(f'Modulation = {Order}-{MOD_TYPE.upper()}')
print(f'N = {N}, Q = {Q}, K = {K}, Lp = {Lp}')
print(f'Ncp = {Ncp}, M = {M}, QN = {lengthBlock}')
print(f'Kt = {Kt}, Ks = {Ks}')
print(f'Sampling rate Fs = {Fs/1e9:.6f} GHz')
print(f'High-rate range grid = {rangeSampleSpacing:.6f} m/sample')
print(f'RRC pulse memory = {pulseMemorySample} high-rate samples')
print(f'Maximum target delay = {maxTargetDelaySample} high-rate samples')
print(f'High-rate CP length = {NcpSample} samples')
print(f'CP duration = {Tcp*1e9:.6f} ns')
print(f'Range corresponding to CP duration = {cpSupportedRange:.6f} m')
print(f'Maximum range after allowing for pulse memory = {maxRangeWithPulseMemory:.6f} m')

for q in range(K):
    print(f'Target {q+1}: range = {targetRange[q]:.6f} m, delay = {targetPropagationDelay[q]*1e9:.6f} ns, delay index = {targetDelaySample[q]}, grid error = {targetRangeError[q]:.3e} m')


#%% 高采样率观察区间与距离坐标
# Python 为零基索引：对应 MATLAB 的 NcpSample+1 : NcpSample+lengthBlock
usefulIndex = slice(NcpSample, NcpSample + lengthBlock)

delaySample = np.arange(lengthBlock)
rangeAxis = delaySample * c0 * Ts / 2
plotIndex = (rangeAxis >= 0) & (rangeAxis <= 35)


#%% 保留完整蒙特卡洛数组
SimRangeProfile = np.zeros((Iter, lengthBlock))
SimTargetProfile = np.zeros((Iter, lengthBlock, K))


#%% 蒙特卡洛仿真
for ii in tqdm(range(Iter), desc='Sensing simulation'):

    # 1. 随机通信符号，调制器使用第一份代码的写法
    symbolIndex = np.random.randint(0, Order, size=N)
    s = np.asarray(modem.modulate(symbolIndex), dtype=complex).reshape(-1)

    # 与 MATLAB pskmod(symbolIndex, Order, pi/Order) 的整体相位对应
    if MOD_TYPE.lower() == 'psk':
        s = s * np.exp(1j * np.pi / Order)

    # 2. OFDM 调制，对应 x = U @ s，U = F_N^H
    x = scipy.fft.ifft(s, N, norm='ortho')

    # 3. 在符号率序列上添加 CP
    xCP = add_cyclic_prefix(x, Ncp)

    # 4. 手动 Q 倍上采样
    xCPUp = np.zeros(Q * M, dtype=complex)
    xCPUp[0::Q] = xCP

    # 5. 物理发射脉冲成形：线性卷积
    xTransmit = np.convolve(xCPUp, p_t, mode='full')

    # 6. 从实际发射波形中提取感知参考 xTilde
    xTilde = xTransmit[usefulIndex]

    # 验证一：有用发射波形等于周期脉冲成形结果
    if ii == 0:
        xUp = np.zeros(lengthBlock, dtype=complex)
        xUp[0::Q] = x

        pPeriodic = np.zeros(lengthBlock, dtype=complex)
        pPeriodic[:Lp] = p_t

        xTildeEquivalent = scipy.fft.ifft(scipy.fft.fft(xUp) * scipy.fft.fft(pPeriodic))
        pulseCircularizationError = np.linalg.norm(xTilde - xTildeEquivalent) / max(np.linalg.norm(xTildeEquivalent), eps)

        print(f'\nPulse-shaping circularization error = {pulseCircularizationError:.3e}')
        assert pulseCircularizationError < 1e-10

    # 7. 物理感知信道：对共同发射波形施加线性时延和复增益
    ysTargetFull = np.zeros((Ks, K), dtype=complex)

    for q in range(K):
        startIndex = targetDelaySample[q]
        endIndex = startIndex + xTransmit.size
        ysTargetFull[startIndex:endIndex, q] = gamma[q] * xTransmit

    # 在复数信号层面叠加所有目标回波
    ysNoiselessFull = np.sum(ysTargetFull, axis=1)

    # 8. 感知接收端提取与发射参考相同的绝对时间区间
    ysTarget = ysTargetFull[usefulIndex, :]
    ysNoiseless = ysNoiselessFull[usefulIndex]

    # 验证二：物理线性时延经观察窗口提取后等于参考波形的循环时移
    if ii == 0:
        for q in range(K):
            ysTargetEquivalent = gamma[q] * np.roll(xTilde, targetDelaySample[q])
            targetCircularizationError = np.linalg.norm(ysTarget[:, q] - ysTargetEquivalent) / max(np.linalg.norm(ysTargetEquivalent), eps)

            print(f'Target {q+1} circularization error = {targetCircularizationError:.3e}')
            assert targetCircularizationError < 1e-10

    # 9. 在有用回波区间添加复高斯白噪声
    # SNR 基于所有目标相干叠加后的总无噪声回波平均功率
    signalPower = np.mean(np.abs(ysNoiseless)**2)
    noisePower = signalPower / 10**(SNRdB / 10)
    noise = np.sqrt(noisePower / 2) * (np.random.randn(lengthBlock) + 1j * np.random.randn(lengthBlock))
    ys = ysNoiseless + noise

    # 10. 周期匹配滤波：回波与参考波形的周期互相关
    referenceSpectrum = scipy.fft.fft(xTilde)

    for q in range(K):
        yMatchedTarget = scipy.fft.ifft(scipy.fft.fft(ysTarget[:, q]) * np.conj(referenceSpectrum))
        SimTargetProfile[ii, :, q] = np.abs(yMatchedTarget)**2

        if ii == 0:
            peakDelay = int(np.argmax(np.abs(yMatchedTarget)**2))
            print(f'Target {q+1}: matched-filter peak index = {peakDelay}, range = {rangeAxis[peakDelay]:.6f} m')
            assert peakDelay == targetDelaySample[q]

    yMatched = scipy.fft.ifft(scipy.fft.fft(ys) * np.conj(referenceSpectrum))
    SimRangeProfile[ii, :] = np.abs(yMatched)**2


#%% 对平方匹配滤波输出进行集合平均
AveRangeProfile = np.mean(SimRangeProfile, axis=0)
AveTargetProfile = np.mean(SimTargetProfile, axis=0)

# 所有曲线使用同一个归一化系数
normalization = np.max(AveRangeProfile)
rangeProfile_dB = 10 * np.log10(AveRangeProfile / normalization + eps)
targetProfile_dB = 10 * np.log10(AveTargetProfile / normalization + eps)


#%% 最后统一绘制三个目标响应和总距离剖面
colors = ['#F65314', '#00A1F1', '#8A2BE2']

fig, axs = plt.subplots(1, 1, figsize=(8, 4))

for q in range(K):
    axs.plot(rangeAxis[plotIndex], targetProfile_dB[plotIndex, q], ls=':', color=colors[q], linewidth=2, label=f'Target {q+1}')

axs.plot(rangeAxis[plotIndex], rangeProfile_dB[plotIndex], ls='-', color='#A9A9A9', linewidth=1.5, label='Range Profile')

axs.set_xlabel('Range [m]')
axs.set_ylabel('Amplitude [dB]')
axs.set_xlim(0, 35)
axs.set_ylim(-80, 0)
axs.set_xticks(np.arange(0, 36, 5))
axs.set_yticks(np.arange(-80, 1, 10))
axs.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
axs.set_axisbelow(True)
axs.legend(loc='upper left', fontsize=13, edgecolor='black')

for spine in axs.spines.values():
    spine.set_linewidth(1)

axs.tick_params(direction='in', axis='both', top=True, right=True)
fig.subplots_adjust(left=0.11, bottom=0.15, right=0.98, top=0.98)

# fig.savefig('OJSP_Fig2_Range_Profile.pdf', bbox_inches='tight')
# fig.savefig('OJSP_Fig2_Range_Profile.png', dpi=300, bbox_inches='tight')

plt.show()
plt.close(fig)
