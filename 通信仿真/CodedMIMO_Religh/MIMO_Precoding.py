#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:22:58 2024

https://blog.csdn.net/weixin_52135976/article/details/118893267


https://blog.csdn.net/weixin_43871127/article/details/104593325


@author: jack

可以看到最大比合并MCR的目标是最大化MCR，ZF的目标是最大化SIR，MMSE的目标是最大化SINR


"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy as cpy

##  自己编写的库
from sourcesink import SourceSink
from config import args
from mimo_channel import MIMO_Channel, SignalNorm, SVD_Precoding
import utility



utility.set_random_seed()
SNR = np.arange(0, 21, 2)
source = SourceSink()
source.InitLog(promargs = args,  )


M = args.M
Nr = args.Nr
Nt = args.Nt
Ncl = args.Ncl
Nray = args.Nray
P = args.P
d = args.d
Modulation_type = f"QAM{M}"
if Modulation_type == "QAM16":
    modem = cpy.QAMModem(16)
elif Modulation_type == "QAM64":
    modem = cpy.QAMModem(64)
elif Modulation_type == "QAM256":
    modem = cpy.QAMModem(256)


# 预编码
def main_SVD():
    for snr in SNR:
        # channel = AWGN(snr, polar.coderate)
        source.ClrCnt()
        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            channel = MIMO_Channel(Nr = Nr, Nt = Nt, d = d, P = P)
            channel.mmwave_MIMO_ULA2ULA()

            tx_bits = source.GenerateBitStr(1920)
            # 编码
            # tx_bits = encoder(tx_bits)
            # 调制
            tx_symbol = modem.modulate(tx_bits)
            ## tx_data = tx_symbol.reshape(Nt, -1)

            H = channel.H[:]
            U, D, V = SVD_Precoding(H, P, d, Nt)
            # P_noise = P*(10**(-snr/10))
            total_num = len(tx_symbol)   # 480
            if total_num %  d != 0:
                tx_symbol = np.pad(tx_symbol, (0,  d - total_num % d), constant_values=(0, 0)) # (6668,)
            tx_times = np.ceil(total_num / d).astype(int) # 240
            symbol_group = tx_symbol.reshape(d, tx_times)  # (2, 240)
            ## 符号能量归一化
            tx_data = SignalNorm(symbol_group, M,  denorm = False) # (2, 240)

            ## MIMO信道 y = Hx + noise
            rx_data = channel.forward(V@tx_data, 1, snr )
            # noise = np.sqrt(P_noise / 2) * (np.random.randn( Nr, tx_times) + 1j * np.random.randn( Nr, tx_times))
            # rx_data = H@V@tx_data + noise  # y = HVx+n, (Nr, tx_times)  (6, 240)

            ## 接收方信号检测
            DigD = np.zeros( H.T.shape, dtype = complex)  # (4, 6)
            DigD[np.diag_indices(Nt)] = 1/D               # (4, 6)
            # y_de = DigD.dot(U.conj().T).dot(y) / np.sqrt(self.P)
            y_de = DigD@(U.conj().T)@rx_data / np.sqrt( P)  # (4, 240))
            y_de = y_de[: d]                              # (2, 240)
            rx_symbol = SignalNorm(y_de, M,  denorm = True).flatten()[:total_num]

            rx_bits = modem.demodulate(rx_symbol, 'hard',)

            #%% count
            source.CntErr(tx_bits, rx_bits)
            if source.tot_blk % 1000 == 0:
                source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***\n");
        source.SaveToFile(snr = snr)
    return

def main_precoding_MMSE():
    Nr = 4
    Nt = 4
    for snr in SNR:
        # channel = AWGN(snr, polar.coderate)
        source.ClrCnt()
        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            channel = MIMO_Channel(Nr = Nr, Nt = Nt, d = d, P = P)
            channel.mmwave_MIMO_ULA2ULA()

            tx_bits = source.GenerateBitStr(1920)
            # 编码
            # tx_bits = encoder(tx_bits)
            # 调制
            tx_symbol = modem.modulate(tx_bits)
            ## tx_data = tx_symbol.reshape(Nt, -1)

            H = channel.H[:]

            symbol_group = tx_symbol.reshape(Nt, -1)  # (4, 120)

            ## 符号能量归一化
            tx_data = SignalNorm(symbol_group, M,  denorm = False) # (4, 120)

            ## MMSE预编码
            P_noise = P*(10**(-1*snr/10))
            temp_W = H.T.conjugate() @ np.linalg.inv((H@H.T.conjugate() + P_noise*np.eye(Nr, Nr)))
            beta = np.sqrt(Nt)/np.linalg.norm(temp_W, ord = 'fro', )  # beta = np.sqrt(Nt / np.trace(temp_W@temp_W.conj().T))
            W = beta * temp_W
            tx_data = W @ tx_data

            ## MIMO信道 y = Hx + noise
            rx_data = channel.forward(tx_data, 1, snr )

            y_de = rx_data / beta
            rx_symbol = SignalNorm(y_de, M,  denorm = True).flatten()

            rx_bits = modem.demodulate(rx_symbol, 'hard',)

            #%% count
            source.CntErr(tx_bits, rx_bits)
            if source.tot_blk % 1000 == 0:
                source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***\n")
        source.SaveToFile(snr = snr)

def main_precoding_ZF():
    Nr = 4
    Nt = 4
    for snr in SNR:
        # channel = AWGN(snr, polar.coderate)
        source.ClrCnt()
        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            channel = MIMO_Channel(Nr = Nr, Nt = Nt, d = d, P = P)
            channel.circular_gaussian()

            tx_bits = source.GenerateBitStr(1920)
            # 编码
            # tx_bits = encoder(tx_bits)
            # 调制
            tx_symbol = modem.modulate(tx_bits)
            ## tx_data = tx_symbol.reshape(Nt, -1)
            H = channel.H[:]
            symbol_group = tx_symbol.reshape(Nt, -1)  # (4, 120)

            ## 符号能量归一化
            tx_data = SignalNorm(symbol_group, M,  denorm = False) # (4, 120)
            ## ZF预编码
            P_noise = P*(10**(-1*snr/10))
            temp_W = np.linalg.inv(H)
            beta = np.sqrt(Nt)/np.linalg.norm(temp_W, ord = 'fro', )
            W = beta * temp_W
            tx_data = W @ tx_data
            ## MIMO信道 y = Hx + noise
            rx_data = channel.forward(tx_data, 1, snr )

            y_de = rx_data / beta
            rx_symbol = SignalNorm(y_de, M,  denorm = True).flatten()

            rx_bits = modem.demodulate(rx_symbol, 'hard',)

            #%% count
            source.CntErr(tx_bits, rx_bits)
            if source.tot_blk % 1000 == 0:
                source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***\n");
        # source.SaveToFile(snr = snr)



# main_ZF_MMSE_SIC
# main_SVD()
# main_precoding_MMSE()
# main_precoding_ZF()






