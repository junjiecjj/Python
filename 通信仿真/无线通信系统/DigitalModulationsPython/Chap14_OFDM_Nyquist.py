#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:45:29 2025

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:29:16 2025

@author: jack
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14                # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16           # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16           # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12          # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12          # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]       # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2                 # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6                # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'    # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'          # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'            # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12

#%%
# 实现与MATLAB cconv完全一致的圆卷积
def cconv(a, b, n=None):
    """
    实现与MATLAB cconv完全一致的圆卷积
    参数:
        a, b: 输入复数数组
        n: 输出长度 (None表示默认长度len(a)+len(b)-1)
    返回:
        圆卷积结果 (复数数组)
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    # 默认输出长度
    if n is None:
        n = len(a) + len(b) - 1
    # 线性卷积
    linear_conv = np.convolve(a, b, mode='full')
    # 处理不同n的情况
    if n <= 0:
        return np.array([], dtype=complex)
    result = np.zeros(n, dtype=complex)
    if n <= len(linear_conv):
        # n <= M+N-1: 重叠相加
        for k in range(n):
            # 收集所有k + m*n位置的元素
            idx = np.arange(k, len(linear_conv), n)
            result[k] = np.sum(linear_conv[idx])
    else:
        # n > M+N-1: 补零
        result[:len(linear_conv)] = linear_conv

    return result

def genH(h, Nx,):
    Nh = h.size
    H = np.zeros((Nx+Nh-1, Nx),  dtype= complex )
    h = np.pad(h, (0, Nx - 1))
    for j in range(Nx):
        H[:,j] = np.roll(h, j)
    return H

def convMatrix(h, N):  #
    """
    Construct the convolution matrix of size (L+N-1)x N from the
    input matrix h of size L. (see chapter 1)
    Parameters:
        h : numpy vector of length L
        N : scalar value
    Returns:
        H : convolution matrix of size (L+N-1)xN
    """
    col = np.hstack((h, np.zeros(N-1)))
    row = np.hstack((h[0], np.zeros(N-1)))

    from scipy.linalg import toeplitz
    H = toeplitz(col, row)
    return H

def add_cyclic_prefix(x, Ncp):
    s = np.hstack((x[-Ncp:], x))
    return s

def remove_cyclic_prefix(r, Ncp, N):
    y = r[Ncp : Ncp+N]
    return y



# Program 7.8: test SRRCPulse.m: Square-root raised-cosine pulse characteristics
def srrcFunction(beta, L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay


#%% <A Dual-Functional Sensing-Communication Waveform Design Based on OFDM, Guanding Yu>

# 下面是OFDM中IFFT -> +cp -> H -> -cp -> FFT的等效过程
h = np.array([-0.4878, -1.5351, 0.2355])
S = np.array([-0.0155, 2.5770, 1.9238, -0.0629, ])
s = np.fft.ifft(S) # IFFT
N = s.size
L = h.size

H = convMatrix(h, N)
y = H @ s

cir_s_h = cconv(h, s, N)

lenCP = L - 1
Acp = np.block([[np.zeros((lenCP, N-lenCP)), np.eye(lenCP)], [np.eye(N)]])

s_cp = Acp @ s                    # add CP

H_cp = convMatrix(h, s_cp.size)
y_cp = H_cp @ s_cp                #  pass freq selected channel

y_remo_cp = y_cp[lenCP:lenCP + N] # receiver, remove cp

H_cp1 = convMatrix(h, s_cp.size)[lenCP:lenCP + N, :]
y_remo_cp1 = H_cp1 @ s_cp        #  pass freq selected channel + remove cp

F = scipy.linalg.dft(N)/np.sqrt(N)
FH = F.conj().T

Diag = F @ H_cp1 @ Acp @ FH  # F@T(h)@A@FH is diagonal such that the data is parallelly transmitted over different subcarriers, and thus the ISI is avoided.

CirH = H_cp1 @ Acp
print(f"h = {h}\nCirH = \n{CirH}") # H --> CirH, 将拓普利兹矩阵变为循环阵, 到这里，从离散信号角度完美的对应OFDM的理论


#%%


#%% Performance of modulations in AWGN
## 使用upfirdn函数
#---------Input Fields------------------------
nSym = 10000
EbN0dBs = np.arange(start = -4, stop = 26, step = 4) # Eb/N0 range in dB for simulation
mod_type = 'PSK' # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
arrayOfM = [2, 8, 32] # array of M values to simulate, [2, 4, 8, 16, 32]
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK

mod_type = 'QAM'
arrayOfM = [4, 16, 256]   # uncomment this line if MOD_TYPE='QAM', [4, 16, 64, 256]

N = 64
Ncp = 16

beta = 0.3
span = 8
L = 4
Tsym = 1
p, t, filtDelay = srrcFunction(beta, L, span)

modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
colors = plt.cm.jet(np.linspace(0, 1, len(arrayOfM))) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6), constrained_layout = True)

for m, M in enumerate(arrayOfM):
    print(f" {M} in {arrayOfM}")
    #----- Initialization of various parameters ----
    k = np.log2(M)
    EsN0dBs = 10*np.log10(k) + EbN0dBs     # EsN0dB calculation
    errors = np.zeros(EsN0dBs.size)
    # SER_sim = np.zeros(EsN0dBs.size)       # simulated Symbol error rates

    if mod_type.lower() == 'fsk':
        modem=modem_dict[mod_type.lower()](M, coherence)#choose modem from dictionary
    else: # for all other modulations
        modem = modem_dict[mod_type.lower()](M)#choose modem from dictionary

    for i, EsN0dB in enumerate(EsN0dBs):
        print(f"    {i+1}/{EsN0dBs.size}")
        for j, sym in enumerate(range(nSym)):
            d = np.random.randint(low = 0, high = M, size = N) # uniform random symbols from 0 to M-1
            u = modem.modulate(d)

            uu = scipy.fft.ifft(u, N)             #  IFFT
            uu_cp = add_cyclic_prefix(uu, Ncp)    #  Add CP

            ##  脉冲成型 + 上变频 -> 基带信号
            s = scipy.signal.upfirdn(p, uu_cp, L)
            ## channel
            r = awgn(s, EsN0dB, L)

            ##  下采样 + 匹配滤波 -> 恢复的基带信号
            s_hat = scipy.signal.upfirdn(p, r, 1, L)  ## 此时默认上采样为1，即不进行上采样

            ## 选取最佳采样点,
            decision_site = int((s_hat.size - uu_cp.size) / 2)
            ## 每个符号选取一个点作为判决
            uu_hat = s_hat[decision_site:decision_site + uu_cp.size] #/ L    ## Note: 当p归一化时，这里千万别用/L，当p没有归一化时，这里需要/L
            # dCap = modem.demodulate(u_hat, outputtype = 'int',)

            y = remove_cyclic_prefix(uu_hat, Ncp, N)  # remove CP
            u_hat = scipy.fft.fft(y, N)               # FFT

            if mod_type.lower()=='fsk': #demodulate (Refer Chapter 3)
                dCap = modem.demodulate(u_hat, coherence)
            else: #demodulate (Refer Chapter 3)
                dCap = modem.demodulate(u_hat)
            ## Error Counter
            numErrors = np.sum(d != dCap)
            errors[i] += numErrors
    SER_sim = errors/(nSym * N)

    SER_theory = ser_awgn(EbN0dBs, mod_type, M, coherence) #theory SER
    ax.semilogy(EbN0dBs, SER_sim, color = colors[m], marker='o', linestyle='', label='Sim '+str(M)+'-'+mod_type.upper())
    ax.semilogy(EbN0dBs, SER_theory, color = colors[m], linestyle='-', label='Theory '+str(M)+'-'+mod_type.upper())

ax.set_ylim(1e-6, 1)
ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Symbol Error Rate for M-'+str(mod_type)+' over AWGN')
ax.legend(fontsize = 12)
out_fig = plt.gcf()
# out_fig.savefig('hh.png',format='png',dpi=1000,)
plt.show()
plt.close()


#%%















































































































