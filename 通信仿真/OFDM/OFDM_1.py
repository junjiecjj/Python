#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:02:41 2024

@author: jack
https://blog.csdn.net/AlphalzZ/article/details/130173613


"""

import numpy as np

# 16QAM星座点
constellation = np.array([-3-3j, -1-3j, 1-3j, 3-3j, -3-1j, -1-1j, 1-1j, 3-1j, -3+1j, -1+1j, 1+1j, 3+1j, -3+3j, -1+3j, 1+3j, 3+3j])

def qam16_modulation(bits):
    # 将比特流分组为4个比特
    bits_grouped = np.reshape(bits, (-1, 4))
    # 将比特组转换为QAM星座点的索引
    indices = np.packbits(bits_grouped, axis=1, bitorder='little').flatten()
    # 获取对应的星座点
    symbols = constellation[indices]
    return symbols

def ofdm_transmitter(symbols, fft_size = 64, cp_size = 16, vc_size = 16):
    N_used = (fft_size - vc_size)
    # 计算OFDM符号数
    num_symbols = len(symbols) // N_used
    # 将星座点重新分组为OFDM符号
    symbols_grouped = np.reshape(symbols[:num_symbols*N_used], (num_symbols, N_used))
    #添加VC
    symbols_grouped_vc = np.concatenate((np.zeros((num_symbols, vc_size)), symbols_grouped), axis = 1)
    # 进行FFT变换
    freq_symbols = np.fft.ifft(symbols_grouped_vc, axis = 1)
    # 添加循环前缀
    freq_symbols_cp = np.concatenate((freq_symbols[:, -cp_size:], freq_symbols), axis = 1)
    # 将OFDM符号串联起来
    time_signal = freq_symbols_cp.flatten()
    return time_signal


def ofdm_receiver(time_signal, fft_size = 64, cp_size = 16, vc_size = 16):
    # 计算OFDM符号数
    num_symbols = len(time_signal) // (fft_size + cp_size)
    # 将时域信号分组为OFDM符号
    time_symbols = np.reshape(time_signal[:num_symbols*(fft_size+cp_size)], (num_symbols, fft_size+cp_size))
    #去除循环前缀
    time_symbols_cp_removed = time_symbols[:, cp_size:]
    # 进行FFT变换
    freq_symbols = np.fft.fft(time_symbols_cp_removed, axis=1)
    # 去除VC，将频域符号展开成一维数组
    freq_symbols_flat = freq_symbols[:, vc_size:].flatten()
    # 获取星座点的索引
    indices = np.argmin(np.abs(freq_symbols_flat[:, None] - constellation[None, :]), axis=1)
    indices = np.uint8(indices)
    # 将索引转换为比特
    bits = np.unpackbits(indices, bitorder='little').reshape(-1, 8)[:, :4].flatten()
    return bits

#生成随机比特流
bits = np.random.randint(0, 2, 192000)

#进行16QAM调制
symbols = qam16_modulation(bits) #2500

#进行OFDM传输
time_signal = ofdm_transmitter(symbols)

#添加噪声
noise = np.random.normal(0, 0.1, len(time_signal)) + 1j*np.random.normal(0, 0.1, len(time_signal))
received_signal = time_signal + noise

#进行OFDM接收
received_bits = ofdm_receiver(received_signal)

#计算误比特率
ber = np.mean(bits != received_bits)
print("误比特率：", ber)





























