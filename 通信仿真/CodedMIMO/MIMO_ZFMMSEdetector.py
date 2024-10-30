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
Modulation_type = f"{M}QAM"
if Modulation_type=="BPSK":
    modem = cpy.PSKModem(2)
elif Modulation_type=="QPSK":
    modem = cpy.PSKModem(4)
elif Modulation_type=="8PSK":
    modem = cpy.PSKModem(8)
elif Modulation_type=="16QAM":
    modem = cpy.QAMModem(16)
elif Modulation_type=="64QAM":
    modem = cpy.QAMModem(64)
elif Modulation_type == "256QAM":
    modem = cpy.QAMModem(256)
map_table, demap_table = modem.plot_constellation()

# 接收方估计
def main_ZF_MMSE():
    for snr in SNR:
        # channel = AWGN(snr, polar.coderate)
        source.ClrCnt()
        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            channel = MIMO_Channel(Nr = Nr, Nt = Nt, d = d, P = P)
            channel.mmwave_MIMO_ULA2ULA()

            # 编码
            # cc = encoder(uu)
            tx_bits = source.GenerateBitStr(1920)

            # 调制
            tx_symbol = modem.modulate(tx_bits)
            tx_data = tx_symbol.reshape(Nt, -1)
            ## 符号能量归一化
            tx_data = SignalNorm(tx_data, M )

            # 信道
            rx_data = channel.forward(tx_data, 1, snr )

            #%%============================================
            #                  ZF
            ##============================================
            pinvH = scipy.linalg.pinv(channel.H)
            rx_data_zf = pinvH @ rx_data
            rx_symb_zf = rx_data_zf.reshape(-1)
            rx_symb_zf = SignalNorm(rx_symb_zf, M, denorm=True)
            rx_bits = modem.demodulate(rx_symb_zf, 'hard',)

            #%%============================================
            #                 MMSE
            ##============================================
            H = channel.H[:]
            P_noise = 1*(10**(-1*snr/10))
            G_MMSE = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt)) @ H.T.conjugate()
            rx_data_mmse = G_MMSE @ rx_data
            rx_symb_mmse = rx_data_mmse.reshape(-1)
            rx_symb_mmse = SignalNorm(rx_symb_mmse, M, denorm=True)
            rx_bits = modem.demodulate(rx_symb_mmse, 'hard',)


            #%% count
            source.CntErr(tx_bits, rx_bits)
            if source.tot_blk % 1000 == 0:
                source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr)
        print("  *** *** *** *** ***\n");
        source.SaveToFile(snr = snr)
    return



# main_ZF_MMSE()





