#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:07:13 2023

@author: jack


!pip install scikit-commpy

https://commpy.readthedocs.io/en/latest/

https://zhuanlan.zhihu.com/p/434928660
https://github.com/BetterBench/OFDM-simulation-Python
https://github.com/BetterBench/OFDM-simulation-Python/blob/main/ofdm_simulation.py

https://blog.csdn.net/AlphalzZ/article/details/130173613

https://zhuanlan.zhihu.com/p/424962237

https://github.com/berndporr/py_ofdm

https://github.com/AnalogArsonist/qpsk_ofdm_system

https://github.com/darcamo/pyphysim

https://zhuanlan.zhihu.com/p/637862608

"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy


## 1 初始化参数
K = 64 # OFDM子载波数量
CP = K//4  #25%的循环前缀长度
P = 8  # 导频数
pilotValue = 3+3j  # 导频格式
Modulation_type = 'QAM16' #调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type ='random' # 信道类型，可选awgn
SNRdb = 25  # 接收端的信噪比（dB）
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1])
pilotCarrier = allCarriers[::K//P]  # 每间隔P个子载波一个导频
# 为了方便信道估计，将最后一个子载波也作为导频
pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
P = P+1 # 导频的数量也需要加1


##  2 可视化导频插入的方式

# 可视化数据和导频的插入方式
dataCarriers = np.delete(allCarriers, pilotCarriers)
plt.figure(figsize=(8, 0.8))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)
plt.xlim((-1, K))
plt.ylim((-0.1, 0.3))
plt.xlabel('Carrier index')
plt.yticks([])
plt.grid(True)
plt.savefig('carrier.png')


## 3 定义调制和解调方式
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]
dataCarriers = np.delete(allCarriers, pilotCarriers)
payloadBits_per_OFDM = len(dataCarriers)*mu  # 每个 OFDM 符号的有效载荷位数
# 定义制调制方式
def Modulation(bits):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        symbol = PSK4.modulate(bits)
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol
# 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits


## 举例16QAM调制方式，画出星座图
mapping_table = {
    (0, 0, 0, 0): -3-3j,
    (0, 0, 0, 1): -3-1j,
    (0, 0, 1, 0): -3+3j,
    (0, 0, 1, 1): -3+1j,
    (0, 1, 0, 0): -1-3j,
    (0, 1, 0, 1): -1-1j,
    (0, 1, 1, 0): -1+3j,
    (0, 1, 1, 1): -1+1j,
    (1, 0, 0, 0):  3-3j,
    (1, 0, 0, 1):  3-1j,
    (1, 0, 1, 0):  3+3j,
    (1, 0, 1, 1):  3+1j,
    (1, 1, 0, 0):  1-3j,
    (1, 1, 0, 1):  1-1j,
    (1, 1, 1, 0):  1+3j,
    (1, 1, 1, 1):  1+1j
}
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x)
                         for x in B), ha='center')
plt.grid(True)
plt.xlim((-4, 4))
plt.ylim((-4, 4))
plt.xlabel('Real part (I)')
plt.ylabel('Imaginary part (Q)')
plt.title('16 QAM Constellation with Gray-Mapping')
plt.savefig('constellation.png')



# 可视化信道冲击响应，仿真信道
# the impulse response of the wireless channel
channelResponse = np.array([1, 0, 0.3+0.3j])
H_exact = np.fft.fft(channelResponse, K)
plt.plot(allCarriers, abs(H_exact))
plt.xlabel('Subcarrier index')
plt.ylabel('|H(f)|')
plt.grid(True)
plt.xlim(0, K-1)
plt.savefig('channelresponse.png')


## 4 定义信道
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                            np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr
def channel(in_signal, SNRdb, channel_type="awgn"):
    channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲击响应
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
# 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
# 添加循环前缀
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])
# 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP+K)]

# 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
# 信道估计
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
    Hest_at_pilots = pilots / pilotValue  # LS信道估计s
    # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest
# 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
#  获取数据位置的数据
def get_payload(equalized):
    return equalized[dataCarriers]

##  5 OFDM通信仿真
def OFDM_simulation():
    # 5.1 产生比特流
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    # 5.2 比特信号调制
    QAM_s = Modulation(bits)
    print(QAM_s)
    # 5.3 插入导频和数据，生成OFDM符号
    OFDM_data = OFDM_symbol(QAM_s)
    # 5.4 快速傅里叶逆变换
    OFDM_time = IDFT(OFDM_data)
    # 5.5 添加循环前缀
    OFDM_withCP = addCP(OFDM_time)

    # 5.6 经过信道
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, SNRdb, "random")[0]
    plt.figure(figsize=(8,2))
    plt.plot(abs(OFDM_TX), label='TX signal')
    plt.plot(abs(OFDM_RX), label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('|X(t)|');
    plt.grid(True);
    # plt.savefig('tran-receiver.png')
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
    print("误比特率BER： ", np.sum(abs(bits-bits_est))/len(bits))
if __name__ == '__main__':
    OFDM_simulation()


















































































































































































































































































































