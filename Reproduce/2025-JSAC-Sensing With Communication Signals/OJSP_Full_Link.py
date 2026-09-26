#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CP-OFDM transmitter, communication SER, and sensing range profile."""

import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from tqdm import tqdm
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.errorRates import ser_rayleigh


#%% 绘图设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['legend.fontsize'] = 14

def srrcFunction(alpha, Q, span):
    if not 0 <= alpha <= 1 or Q < 1 or span < 1 or (span * Q) % 2:
        raise ValueError('Require 0 <= alpha <= 1 and positive integer Q, span with even span*Q.')
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
    p = p / np.linalg.norm(p)
    return p, t, (p.size - 1) / 2

def add_cyclic_prefix(x, Ncp):
    return np.hstack((x[-Ncp:], x)) if Ncp > 0 else x.copy()

def cconv(a, b, n):
    linear_conv = np.convolve(a, b, mode='full')
    result = np.zeros(n, dtype=complex)
    np.add.at(result, np.arange(linear_conv.size) % n, linear_conv)
    return result


#%% 共用发射端参数
np.random.seed(42)
rngSensing = np.random.RandomState(43)       # 独立感知噪声，不影响通信随机数序列

MOD_TYPE = "psk"    ## "pam" "psk",   "fsk" is not suitable.
arrayOfOrder = [2, 8, 16, 32]

# MOD_TYPE = "qam"
# arrayOfOrder = [4, 16, 64,]

modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem, 'fsk': FSKModem}
if MOD_TYPE.lower() not in ('psk', 'qam'):
    raise ValueError('The Rayleigh SER comparison in this script supports PSK and QAM.')

N = 2048
Q = 20                                    # 论文的过采样倍数
alpha = 0.35
span = 20

#%% 通信参数
L = 5                                     # 高采样率物理信道抽头数
n0 = 0                                    # 下采样相位，取 0 <= n0 < Q
nSym = 2000
EbN0dBs = np.arange(-4, 25, 4)

#%% 感知参数
K = 3                                     # 论文的目标数量
targetRange = np.array([10, 20, 25], dtype=float)
targetAmplitude_dB = np.array([0, -10, -30], dtype=float)
targetPhase = np.array([0, 0, 0], dtype=float)
SNRdB = -10                                # sensing SNR
Iter = 2000
sensingSNRIndex = 0                        # 只在这个通信 SNR 循环中统计感知结果
rangeSampleSpacing = 0.025                 # 距离栅格间隔，m；不是距离分辨率
desiredCPRange = 40                        # CP 总时长对应距离，m；不是净目标覆盖距离
c0 = 299792458.0
eps = np.finfo(float).eps

if not 1 <= Iter <= nSym or not 0 <= sensingSNRIndex < EbN0dBs.size:
    raise ValueError('Require 1 <= Iter <= nSym and a valid sensingSNRIndex.')
if not 0 <= n0 < Q:
    raise ValueError('This causal polyphase implementation requires 0 <= n0 < Q.')
if not targetRange.size == targetAmplitude_dB.size == targetPhase.size == K or np.any(targetRange < 0):
    raise ValueError('Target arrays must have length K and nonnegative ranges.')

#%% 物理量与高采样率 bin 的转换
Fs = c0 / (2 * rangeSampleSpacing)
Ts = 1 / Fs
symbolPeriod = Q * Ts
targetPropagationDelay = 2 * targetRange / c0
targetDelaySample = np.floor(targetPropagationDelay / Ts + 0.5).astype(int)
targetRangeGrid = targetDelaySample * c0 * Ts / 2
maxTargetDelaySample = int(np.max(targetDelaySample))
gamma = 10**(targetAmplitude_dB / 20) * np.exp(1j * targetPhase)
channelTapDelay = np.arange(L) * Ts          # 第 ell 个抽头的物理时延为 ell*Ts

#%% 脉冲和通信符号率等效信道长度
p_t, t, filtDelay = srrcFunction(alpha, Q, span)
p_r = np.conj(p_t[::-1])
Lp, Lr = p_t.size, p_r.size
Lc = Lp + L + Lr - 2
Leq = (Lc - 1 - n0) // Q + 1

#%% 共用 CP：同时覆盖通信等效信道记忆和感知总时延
NcpComm = Leq - 1
NcpSensing = int(np.ceil((Lp - 1 + maxTargetDelaySample) / Q))
NcpDesired = int(np.ceil(desiredCPRange / (Q * rangeSampleSpacing)))
Ncp = max(NcpComm, NcpSensing, NcpDesired)
if Ncp > N:
    raise ValueError('Ncp exceeds N; increase N or reduce the required memory/delay.')
assert Ncp >= Leq - 1
assert Q * Ncp >= Lp - 1 + maxTargetDelaySample

M = N + Ncp
NcpSample = Q * Ncp
lengthBlock = Q * N
Kt = Q * M + Lp - 1
Ks = Kt + maxTargetDelaySample
outputLength = M + Leq - 1
usefulIndex = slice(NcpSample, NcpSample + lengthBlock)
rangeAxis = np.arange(lengthBlock) * c0 * Ts / 2
plotIndex = (rangeAxis >= 0) & (rangeAxis <= 35)

print(f'N={N}, Q={Q}, L={L}, Lp={Lp}, Lr={Lr}, Lc={Lc}, Leq={Leq}')
print(f'NcpComm={NcpComm}, NcpSensing={NcpSensing}, NcpDesired={NcpDesired}, Ncp={Ncp}')
print(f'M={M}, Kt={Kt}, Ks={Ks}, sensing block length={lengthBlock}')
print(f'Fs={Fs/1e9:.6f} GHz, Ts={Ts*1e9:.6f} ns, QTs={symbolPeriod*1e9:.6f} ns')
print(f'CP duration={NcpSample*Ts*1e9:.6f} ns, max channel delay={channelTapDelay[-1]*1e9:.6f} ns')
print(f'CP target coverage after pulse memory={(NcpSample-Lp+1)*rangeSampleSpacing:.6f} m')
for q in range(K):
    print(f'Target {q+1}: R={targetRange[q]:.6f} m, tau={targetDelaySample[q]}, grid R={targetRangeGrid[q]:.6f} m')

#%% 通信信道的确定性集合归一化，以及各子载波的平均信道功率
hEqBasis = np.zeros((Leq, L), dtype=complex)
for indexTap in range(L):
    hBasis = np.zeros(L, dtype=complex)
    hBasis[indexTap] = 1
    cBasis = np.convolve(np.convolve(p_t, hBasis), p_r)
    hEqBasis[:, indexTap] = cBasis[n0::Q]
channelNormalization = np.sum(np.abs(hEqBasis)**2)
HEqBasis = scipy.fft.fft(hEqBasis, N, axis=0)
channelPowerSubcarrier = np.sum(np.abs(HEqBasis)**2, axis=1) / channelNormalization

#%% 有限 N 点接收窗口内，匹配滤波噪声的精确子载波方差因子
# FFT 噪声方差 = N * noiseVarianceHighRate * noisePowerSubcarrier[k]
# 权重 (1-lag/N) 来自长度 N 的 Toeplitz 噪声协方差，不假定噪声循环平稳。
noisePowerSubcarrier = np.full(N, np.sum(np.abs(p_r)**2), dtype=float)
for lag in range(1, min((Lr - 1) // Q, N - 1) + 1):
    shift = lag * Q
    correlation = np.sum(p_r[shift:] * np.conj(p_r[:-shift]))
    noisePowerSubcarrier += 2 * (1-lag/N) * np.real(correlation * np.exp(-1j * 2*np.pi * np.arange(N) * lag / N))
assert np.all(noisePowerSubcarrier > 0)
print(f'Ensemble normalization={channelNormalization:.6f}, mean channel power={np.mean(channelPowerSubcarrier):.6f}')

#%% 保存通信和感知结果
SER_sim_all = np.zeros((len(arrayOfOrder), EbN0dBs.size))
SER_theory_all = np.zeros_like(SER_sim_all)
SimRangeProfile_all = []
SimTargetProfile_all = []
rangeProfile_dB_all = []
targetProfile_dB_all = []

#%% 每个数据块仅生成一次发射波形
for indexOrder, Order in enumerate(arrayOfOrder):
    modem = modem_dict[MOD_TYPE.lower()](Order)
    bitsPerSymbol = int(np.log2(Order))
    if 2**bitsPerSymbol != Order:
        raise ValueError('Constellation order must be a power of two.')
    AvgEnergy = np.mean(np.abs(modem.constellation)**2)
    EsN0dBs = EbN0dBs + 10 * np.log10(bitsPerSymbol * N / M)
    errors = np.zeros(EbN0dBs.size)
    SimRangeProfile = np.zeros((Iter, lengthBlock))
    SimTargetProfile = np.zeros((Iter, lengthBlock, K))

    for indexSNR, EsN0dB in enumerate(EsN0dBs):
        EsN0 = 10**(EsN0dB / 10)
        noiseVarianceHighRate = AvgEnergy / (N * EsN0)

        for indexBlock in tqdm(range(nSym), desc=f'{Order}-{MOD_TYPE.upper()}, Eb/N0={EbN0dBs[indexSNR]} dB', leave=False):
            # 共用 ISAC Tx
            d = np.random.randint(0, Order, size=N)
            s = np.asarray(modem.modulate(d), dtype=complex).reshape(-1)
            x = scipy.fft.ifft(s, N)
            x_cp = add_cyclic_prefix(x, Ncp)
            x_up = np.zeros(Q * M, dtype=complex)
            x_up[0::Q] = x_cp
            xTransmit = np.convolve(x_up, p_t, mode='full')

            # 通信物理信道与接收端
            h = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * channelNormalization)
            c = np.convolve(np.convolve(p_t, h), p_r)
            h_eq = c[n0::Q]
            H_eq = scipy.fft.fft(h_eq, N)
            r_noiseless = np.convolve(xTransmit, h, mode='full')
            noise = np.sqrt(noiseVarianceHighRate / 2) * (np.random.randn(r_noiseless.size) + 1j * np.random.randn(r_noiseless.size))
            r = r_noiseless + noise
            z = np.convolve(r, p_r, mode='full')
            r_d = z[n0:n0 + Q * outputLength:Q]
            y = r_d[Ncp:Ncp + N]
            Y = scipy.fft.fft(y, N)
            s_hat = Y / H_eq
            dCap = np.asarray(modem.demodulate(s_hat)).reshape(-1)
            errors[indexSNR] += np.count_nonzero(d != dCap)

            # 首次数据块验证通信链路
            if indexSNR == 0 and indexBlock == 0:
                z0 = np.convolve(r_noiseless, p_r, mode='full')
                rd0 = z0[n0:n0 + Q * outputLength:Q]
                rdEquivalent = np.convolve(x_cp, h_eq, mode='full')
                y0 = rd0[Ncp:Ncp + N]
                yCircular = cconv(h_eq, x, N)
                s0 = scipy.fft.fft(y0, N) / H_eq
                errEquivalent = np.linalg.norm(rd0-rdEquivalent) / max(np.linalg.norm(rdEquivalent), eps)
                errCircular = np.linalg.norm(y0-yCircular) / max(np.linalg.norm(yCircular), eps)
                errRecovery = np.linalg.norm(s0-s) / max(np.linalg.norm(s), eps)
                print(f'\nCommunication: equivalent={errEquivalent:.3e}, circular={errCircular:.3e}, recovery={errRecovery:.3e}')
                assert max(errEquivalent, errCircular) < 1e-10 and errRecovery < 1e-8

            # 感知仅统计指定 SNR 循环中的前 Iter 个共同发射数据块
            if indexSNR == sensingSNRIndex and indexBlock < Iter:
                ii = indexBlock
                xTilde = xTransmit[usefulIndex]
                ysTargetFull = np.zeros((Ks, K), dtype=complex)
                for q in range(K):
                    startIndex = targetDelaySample[q]
                    ysTargetFull[startIndex:startIndex + Kt, q] = gamma[q] * xTransmit
                ysNoiselessFull = np.sum(ysTargetFull, axis=1)
                ysTarget = ysTargetFull[usefulIndex, :]
                ysNoiseless = ysNoiselessFull[usefulIndex]

                if ii == 0:
                    xUp = np.zeros(lengthBlock, dtype=complex)
                    xUp[0::Q] = x
                    xTildeEquivalent = scipy.fft.ifft(scipy.fft.fft(xUp) * scipy.fft.fft(p_t, lengthBlock))
                    errPulse = np.linalg.norm(xTilde-xTildeEquivalent) / max(np.linalg.norm(xTildeEquivalent), eps)
                    print(f'Sensing pulse circularization={errPulse:.3e}')
                    assert errPulse < 1e-10
                    for q in range(K):
                        echoEquivalent = gamma[q] * np.roll(xTilde, targetDelaySample[q])
                        errEcho = np.linalg.norm(ysTarget[:, q]-echoEquivalent) / max(np.linalg.norm(echoEquivalent), eps)
                        print(f'Target {q+1} echo circularization={errEcho:.3e}')
                        assert errEcho < 1e-10

                signalPower = np.mean(np.abs(ysNoiseless)**2)
                noisePower = signalPower / 10**(SNRdB / 10)    # sensing SNR
                noiseSensing = np.sqrt(noisePower / 2) * (rngSensing.randn(lengthBlock) + 1j * rngSensing.randn(lengthBlock))
                ys = ysNoiseless + noiseSensing
                referenceSpectrum = scipy.fft.fft(xTilde)
                for q in range(K):
                    yMatchedTarget = scipy.fft.ifft(scipy.fft.fft(ysTarget[:, q]) * np.conj(referenceSpectrum))
                    SimTargetProfile[ii, :, q] = np.abs(yMatchedTarget)**2
                    if ii == 0:
                        peakDelay = int(np.argmax(np.abs(yMatchedTarget)**2))
                        print(f'Target {q+1}: peak bin={peakDelay}, range={rangeAxis[peakDelay]:.6f} m')
                        assert peakDelay == targetDelaySample[q]
                yMatched = scipy.fft.ifft(scipy.fft.fft(ys) * np.conj(referenceSpectrum))
                SimRangeProfile[ii, :] = np.abs(yMatched)**2

    # 通信 SER 与逐子载波理论结果
    SER_sim_all[indexOrder, :] = errors / (nSym * N)
    EbN0EffectiveDbs = EbN0dBs + 10 * np.log10(N / M)
    for indexSubcarrier in range(N):
        powerRatio = channelPowerSubcarrier[indexSubcarrier] / noisePowerSubcarrier[indexSubcarrier]
        EbN0SubcarrierDbs = EbN0EffectiveDbs + 10 * np.log10(powerRatio)
        SER_theory_all[indexOrder, :] += ser_rayleigh(EbN0SubcarrierDbs, MOD_TYPE, Order) / N

    # 感知：先模平方，再集合平均；所有目标使用总剖面的同一个归一化系数
    AveRangeProfile = np.mean(SimRangeProfile, axis=0)
    AveTargetProfile = np.mean(SimTargetProfile, axis=0)
    normalization = np.max(AveRangeProfile)
    rangeProfile_dB_all.append(10 * np.log10(AveRangeProfile / normalization + eps))
    targetProfile_dB_all.append(10 * np.log10(AveTargetProfile / normalization + eps))
    SimRangeProfile_all.append(SimRangeProfile)
    SimTargetProfile_all.append(SimTargetProfile)

#%% 仿真结束后统一画图
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfOrder)))
figComm, axComm = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
handles_theor, handles_simu, labels_theor = [], [], []
for m, Order in enumerate(arrayOfOrder):
    ht, = axComm.semilogy(EbN0dBs, SER_theory_all[m], color=colors[m], ls='-')
    hs, = axComm.semilogy(EbN0dBs, SER_sim_all[m], color=colors[m], ls='none', marker='o', mfc='none', mew=1.5, ms=8)
    handles_theor.append(ht)
    handles_simu.append(hs)
    labels_theor.append(f'{Order}-{MOD_TYPE.upper()}, Theor')
axComm.set_xlabel(r'$E_b/N_0$ (dB)')
axComm.set_ylabel('SER')
axComm.set_ylim(1e-4, 1)
axComm.grid(linestyle=(0, (5, 10)), linewidth=0.5, which='both', alpha=0.3)
axComm.legend(handles_theor + handles_simu, labels_theor + ['Simu'] * len(arrayOfOrder), ncol=2, loc='lower left', edgecolor='black')
axComm.tick_params(direction='in', top=True, right=True)

# figComm.savefig(f'OJSP_Communication_{MOD_TYPE}.pdf', bbox_inches='tight')
out_fig = plt.gcf()
out_fig.savefig(f'OJSP_Communication_{MOD_TYPE}.pdf', bbox_inches='tight')
plt.show()
plt.close()

targetColors = ['#F65314', '#00A1F1', '#8A2BE2']
for m, Order in enumerate(arrayOfOrder):
    figSensing, axSensing = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
    for q in range(K):
        axSensing.plot(rangeAxis[plotIndex], targetProfile_dB_all[m][plotIndex, q], ls=':', color=targetColors[q % len(targetColors)], label=f'Target {q+1}')
    axSensing.plot(rangeAxis[plotIndex], rangeProfile_dB_all[m][plotIndex], color='#A9A9A9', linewidth=1.5, label='Range Profile')
    axSensing.set_xlabel('Range [m]', fontsize = 22,)
    axSensing.set_ylabel('Normalized power [dB]', fontsize = 22,)
    axSensing.set_xlim(0, 35)
    axSensing.set_ylim(-80, 0)
    axSensing.set_xticks(np.arange(0, 36, 5))
    axSensing.set_yticks(np.arange(-80, 1, 10))
    axSensing.grid(True, linestyle='--', alpha=0.2)
    legend1 = axSensing.legend(loc='best', edgecolor='black', fontsize = 18)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')
    axSensing.tick_params(direction='in', top=True, right=True)
    axSensing.set_title(f'{Order}-{MOD_TYPE.upper()}')
    # figSensing.savefig(f'OJSP_Sensing_{Order}_{MOD_TYPE}.pdf', bbox_inches='tight')
    out_fig = plt.gcf()
    out_fig.savefig(f'OJSP_Sensing_{Order}_{MOD_TYPE}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

# plt.show()











