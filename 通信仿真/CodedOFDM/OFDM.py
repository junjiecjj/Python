#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:07:13 2023

@author: jack

https://zhuanlan.zhihu.com/p/438568996
https://cloud.tencent.com/developer/article/2351718
https://blog.csdn.net/qq_34070723/article/details/88926168

https://zhuanlan.zhihu.com/p/678429235#:~:text=%E4%B8%BA%E4%BA%86%E5%85%8B%E6%9C%8D%E9%A2%91%E7%8E%87%E9%80%89%E6%8B%A9,%E6%9C%BA%E5%8F%91%E5%B0%84%E6%9C%BA%E7%9A%84%E8%AE%BE%E8%AE%A1%E3%80%82
## OFDM循环前缀及其作用（矩阵视角解释）
https://blog.csdn.net/weixin_43413559/article/details/125218797
https://blog.csdn.net/weixin_43413559/article/details/125227419

!pip install scikit-commpy

https://commpy.readthedocs.io/en/latest/

https://zhuanlan.zhihu.com/p/434928660
https://github.com/BetterBench/OFDM-simulation-Python
https://github.com/BetterBench/OFDM-simulation-Python/blob/main/ofdm_simulation.py

https://blog.csdn.net/AlphalzZ/article/details/130173613

https://zhuanlan.zhihu.com/p/424962237

https://github.com/berndporr/py_ofdm

## 集成包
https://github.com/darcamo/pyphysim

https://zhuanlan.zhihu.com/p/637862608




## 信道估计---LS、MMSE、LMMSE准则
信道估计主要分为非盲信道估计和盲信道估计。顾名思义，非盲信道估计需要使用基站和接收机均已知的导频序列进行信道估计，并使用不同的时频域插值技术来估计导频之间或者符号之间的子载波上的信道响应。目前主要使用的非盲信道估计包括最小二乘（LS）信道估计、最小均方误差（MMSE）信道估计、基于DFT的信道估计以及基于判决反馈信道估计等；而盲信道估计不需要已经已知的导频序列，主要包括基于最大期望的信道估计、基于子空间的信道估计技术等。
https://blog.csdn.net/ddatalent/article/details/121132095

## 添加循环前缀cp的作用:一方面在于充当保护间隔，对抗频率选择性；另一方面，使得发射信号与信道响应的线性卷积变为循环卷积，接收端使用单抽头均衡器即可恢复发射的符号（详情见书籍《MIMO-OFDM无线通信技术及matlab实现》第4章）。
循环前缀（CP）:OFDM中的循环前缀是指在每个OFDM符号的开头添加一段与该符号结尾相同的信号，以形成一个循环结构。循环前缀的作用是在时域上对信号进行加窗，以减小信号在频域上的泄漏。循环前缀主要充当连续符号之间的保护带，以克服符号间干扰ISI。

导频符号:OFDM中的导频符号是指在OFDM符号中插入的一些特殊符号，用于信道估计和同步。导频符号可以安插在OFDM的时间和频率二维结构内，只要在两个方向的导频密度满足二维Nyquist定理，就可以精确估计信道的时变和衰落特性，因此能够适应快衰落信道。

"""


#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy as cpy
import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
from matplotlib.font_manager import FontProperties
from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# np.random.seed(1)

#%% 1 初始化参数
K = 64            # OFDM子载波数量
channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲击响应
CP = channelResponse.size - 1       # 25%的循环前缀长度, 实际上只需要取 >= len(H) -1  = L-1 就够了，
P = 8             # 导频数
pilotValue = 3 + 3j                 # 导频格式
Modulation_type = 'QAM16'           # 调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type = 'random'              # 信道类型，可选awgn
SNRdb = 10                          # 接收端的信噪比（dB）
allCarriers = np.arange(K)          # 子载波编号 ([0, 1, ... K-1])
pilotCarrier = allCarriers[::K//P]  # 每间隔P个子载波一个导频
# 为了方便信道估计，将最后一个子载波也作为导频
pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
P = P + 1                           # 导频的数量也需要加1


#%%  2 可视化导频插入的方式

# 可视化数据和导频的插入方式
# dataCarriers = np.delete(allCarriers, pilotCarriers)
# fig, axs = plt.subplots(1,1, figsize=(8, 2), constrained_layout=True)
# axs.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
# axs.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')

# axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize = 20, width = 3,  )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # [label.set_fontsize(20) for label in labels] #刻度值字号

# font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1, ncol = 2)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs.set_yticks([])
# axs.set_xlim((-1, K))
# axs.set_ylim((-0.1, 0.3))
# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# plt.xlabel('Carrier index', fontproperties=font)

# axs.grid(True)
# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig .savefig(filepath2+'carrier.eps',   bbox_inches = 'tight')
# plt.close()


##%% 3 定义调制和解调方式
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]
dataCarriers = np.delete(allCarriers, pilotCarriers)
payloadBits_per_OFDM = len(dataCarriers)*mu  # 每个 OFDM 符号的有效载荷位数
# 定义制调制方式
def Modulation(bits):
    if Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol
    elif Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        symbol = PSK4.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol

## 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits


#%% 举例16QAM调制方式，画出星座图
# mapping_table = {
#     (0, 0, 0, 0): -3-3j,
#     (0, 0, 0, 1): -3-1j,
#     (0, 0, 1, 0): -3+3j,
#     (0, 0, 1, 1): -3+1j,
#     (0, 1, 0, 0): -1-3j,
#     (0, 1, 0, 1): -1-1j,
#     (0, 1, 1, 0): -1+3j,
#     (0, 1, 1, 1): -1+1j,
#     (1, 0, 0, 0):  3-3j,
#     (1, 0, 0, 1):  3-1j,
#     (1, 0, 1, 0):  3+3j,
#     (1, 0, 1, 1):  3+1j,
#     (1, 1, 0, 0):  1-3j,
#     (1, 1, 0, 1):  1-1j,
#     (1, 1, 1, 0):  1+3j,
#     (1, 1, 1, 1):  1+1j
# }

# fig, axs = plt.subplots(1,1, figsize=(8, 8), constrained_layout=True)
# for b3 in [0, 1]:
#     for b2 in [0, 1]:
#         for b1 in [0, 1]:
#             for b0 in [0, 1]:
#                 B = (b3, b2, b1, b0)
#                 Q = mapping_table[B]
#                 axs.plot(Q.real, Q.imag, 'bo', markersize = 10)
#                 axs.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha = 'center', fontsize  = 20)
# axs.grid(True)
# axs.set_xlim((-4, 4))
# axs.set_ylim((-4, 4))
# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs.set_xlabel('Real part (I)', fontproperties=font)
# axs.set_ylabel('Imaginary part (Q)', fontproperties=font)
# plt.title('16 QAM Constellation with Gray-Mapping', fontproperties=font)
# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig.savefig(filepath2+'constellation.eps',   bbox_inches = 'tight')
# plt.close()


#%% 可视化信道冲击响应，仿真信道
## the impulse response of the wireless channel
# fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
# channelResponse = np.array([1, 0, 0.3 + 0.3j])
# H_exact = np.fft.fft(channelResponse, K)
# axs.plot(allCarriers, abs(H_exact), linewidth = 3)

# axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=24, width=3,  )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# # [label.set_fontsize(24) for label in labels]  # 刻度值字号

# font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
# axs.set_xlabel('Subcarrier index', fontproperties=font)
# axs.set_ylabel('|H(f)|', fontproperties=font)
# plt.grid(True)
# axs.set_xlim(0, K-1)
# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig .savefig(filepath2 + 'channelresponse.eps',   bbox_inches = 'tight')
# plt.close()

#%% 4 定义信道
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s)**2)
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j * np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr

def channel(in_signal, SNRdb, channel_type="awgn"):
    # channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲击响应
    if channel_type == "random":
        convolved = np.convolve(in_signal, channelResponse)
        out_signal, noise_pwr = add_awgn(convolved, SNRdb)
    elif channel_type == "awgn":
        out_signal, noise_pwr = add_awgn(in_signal, SNRdb)
    return out_signal, noise_pwr

# 插入导频和数据，生成OFDM符号
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # 子载波位置
    symbol[pilotCarriers] = pilotValue  # 在导频位置插入导频
    symbol[dataCarriers] = QAM_payload  # 在数据位置插入数据
    return symbol

## 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

## 添加循环前缀, 频域上消除符号间干扰(ISI)。
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])

## 接收端，去除循环前缀
def removeCP(signal):
    # 这里很关键，原始信号长度为N，cp的长度为 CP =  L - 1 = Len(h) - 1，其中h为信道冲击相应，则过信道后的信号y的长度为N+(L-1)+(L-1)。取y的区间为[L-1, N+L-2]的信号，这样会保证加cp的效果和圆卷积的一样,python是右开的，所以为：[L-1: N+L-1]，区间错了性能会差很多。
    return signal[CP : (CP + K)]

## 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

## 信道估计
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
    Hest_at_pilots = pilots / pilotValue  # LS信道估计s
    # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs( Hest_at_pilots), kind = 'linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle( Hest_at_pilots), kind = 'linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest

## 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
## 获取数据位置的数据
def get_payload(equalized):
    return equalized[dataCarriers]





# 5.1 产生比特流
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
# 5.2 比特信号调制
QAM_s = Modulation(bits)
# print(QAM_s)
# 5.3 插入导频和数据，生成OFDM符号
OFDM_data = OFDM_symbol(QAM_s)
# 5.4 快速傅里叶逆变换
OFDM_time = IDFT(OFDM_data)
# 5.5 添加循环前缀, 频域上消除符号间干扰(ISI)。
OFDM_withCP = addCP(OFDM_time)

# 5.6 经过信道
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX, SNRdb, "random")[0]

# 5.7 接收端，去除循环前缀
OFDM_RX_noCP = removeCP(OFDM_RX)
# 5.8 快速傅里叶变换
OFDM_demod = DFT(OFDM_RX_noCP)
# 5.9 信道估计
Hest = channelEstimate(OFDM_demod)
# 5.10 均衡
equalized_Hest = equalize(OFDM_demod, Hest)
# 5.10 获取数据位置的数据
QAM_est = get_payload(equalized_Hest)
# 5.11 反映射，解调
bits_est = DeModulation(QAM_est)
# print(bits_est)
ber = np.sum(abs(bits-bits_est))/len(bits)
print(f"{SNRdb} 误比特率BER：{ber}", )


#%%  5 OFDM通信仿真
def OFDM_simulation():
    ##
    SNRs = np.arange(0, 21, 0.5)
    BERs = []
    for SNRdb in SNRs:
        # 5.1 产生比特流
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        # 5.2 比特信号调制
        QAM_s = Modulation(bits)
        # print(QAM_s)
        # 5.3 插入导频和数据，生成OFDM符号
        OFDM_data = OFDM_symbol(QAM_s)
        # 5.4 快速傅里叶逆变换
        OFDM_time = IDFT(OFDM_data)
        # 5.5 添加循环前缀, 频域上消除符号间干扰(ISI)。
        OFDM_withCP = addCP(OFDM_time)

        # 5.6 经过信道
        OFDM_TX = OFDM_withCP
        OFDM_RX = channel(OFDM_TX, SNRdb, "awgn")[0]

        # 5.7 接收端，去除循环前缀
        OFDM_RX_noCP = removeCP(OFDM_RX)
        # 5.8 快速傅里叶变换
        OFDM_demod = DFT(OFDM_RX_noCP)
        # 5.9 信道估计
        Hest = channelEstimate(OFDM_demod)
        # 5.10 均衡
        equalized_Hest = equalize(OFDM_demod, Hest)
        # 5.10 获取数据位置的数据
        QAM_est = get_payload(equalized_Hest)
        # 5.11 反映射，解调
        bits_est = DeModulation(QAM_est)
        # print(bits_est)
        ber = np.sum(abs(bits-bits_est))/len(bits)
        # print(f"{SNRdb} 误比特率BER：{ber}", )
        BERs.append(ber)
    return SNRs, np.array(BERs)

# if __name__ == '__main__':
    # OFDM_simulation()



# SNRdb, BERs_QAM64 = OFDM_simulation()

# SNRdb, BERs_QAM16 = OFDM_simulation()



# fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
# axs.plot(SNRdb, BERs_QAM64,'g-', label = 'OFDM, QAM64', )
# axs.plot(SNRdb, BERs_QAM16, 'r-', label = 'OFDM, QAM16', )

# font1 = FontProperties(fname = fontpath1+"Times_New_Roman.ttf", size = 20)
# legend1 = axs.legend(loc = 'best', borderaxespad=0, edgecolor='black', prop=font1,  )
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs.tick_params(direction = 'in', axis='both', top=True, right=True, labelsize=24, width=3, )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# font = FontProperties(fname = fontpath1+"Times_New_Roman.ttf", size = 25)
# axs.set_xlabel('SNR (dB)', fontproperties = font)
# axs.set_ylabel('BER', fontproperties = font)
# # axs.grid(True)
# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig.savefig(filepath2 + 'OFDM_QAM.png', bbox_inches = 'tight')
# plt.close()




### 画出发送信号和过信道后的信号
# fig, axs = plt.subplots(1, 1, figsize = (8, 4), constrained_layout = True)
# axs.plot(abs(OFDM_TX),'g-', label = 'TX signal', )
# axs.plot(abs(OFDM_RX), 'r-', label = 'RX signal', )

# font1 = FontProperties(fname = fontpath1+"Times_New_Roman.ttf", size = 20)
# legend1 = axs.legend(loc = 'best', borderaxespad=0, edgecolor='black', prop=font1, ncol = 2)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# axs.tick_params(direction = 'in', axis='both', top=True, right=True, labelsize=24, width=3, )
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# font = FontProperties(fname = fontpath1+"Times_New_Roman.ttf", size = 25)
# axs.set_xlabel('Time', fontproperties = font)
# axs.set_ylabel('|X(t)|', fontproperties = font)
# axs.grid(True)
# filepath2 = '/home/jack/snap/'
# out_fig = plt.gcf()
# out_fig.savefig(filepath2 + 'TxRx.eps', bbox_inches = 'tight')
# plt.close()






























































































































































































































































































