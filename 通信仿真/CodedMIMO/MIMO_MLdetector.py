#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:50:37 2024
https://blog.csdn.net/weixin_52135976/article/details/118893267


https://blog.csdn.net/weixin_43871127/article/details/104593325


@author: jack

可以看到最大比合并MCR的目标是最大化MCR，ZF的目标是最大化SIR，MMSE的目标是最大化SINR

"""



import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools


import commpy as cpy
import copy


##  自己编写的库
from sourcesink import SourceSink
from config import args
from mimo_channel import MIMO_Channel, SignalNorm, SVD_Precoding
import utility
import Modulator
utility.set_random_seed()
SNR = np.arange(0, 21, 2)
source = SourceSink()

logf = "ML_BerFer.txt"
source.InitLog(logfile = logf, promargs = args, codeargs={} )

M = args.M
Nr = args.Nr
Nt = args.Nt
P = args.P

Modulation_type = f"BPSK"
if Modulation_type=="BPSK":
    modem = cpy.PSKModem(2)
elif Modulation_type=="QPSK":
    modem = cpy.PSKModem(4)
elif Modulation_type=="8PSK":
    modem = cpy.PSKModem(8)
elif Modulation_type=="4QAM":
    modem = cpy.QAMModem(4)
elif Modulation_type=="16QAM":
    modem = cpy.QAMModem(16)
elif Modulation_type=="64QAM":
    modem = cpy.QAMModem(64)
elif Modulation_type == "QAM256":
    modem = cpy.QAMModem(256)
map_table, demap_table = modem.plot_constellation(Modulation_type)
Es = Modulator.NormFactor(mod_type='bpsk', M = M,)
allposs = np.array(list(itertools.product(modem.constellation, repeat = Nt))).T


# https://blog.csdn.net/weixin_44863193/article/details/124493090
# https://zhuanlan.zhihu.com/p/634499207
# 接收方估计
# def ML_detector():
for snr in SNR:
    # channel = AWGN(snr, polar.coderate)
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, P = P)
        channel.circular_gaussian()
        # 编码
        # cc = encoder(uu)
        tx_bits = source.GenerateBitStr(1024)

        # 调制
        tx_symbols = modem.modulate(tx_bits)
        tx_symbols = tx_symbols.reshape(Nt, -1)
        ## 符号能量归一化
        tx_sig  = tx_symbols / np.sqrt(Es)

        # 信道
        rx_sig = channel.forward(tx_sig, 1, snr)
        P_noise = 1*(10**(-1*snr/10))

        #%%============================================
        #      ML detector
        ##=============================================
        H = copy.deepcopy(channel.H)
        tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)

        for frame in range(rx_sig.shape[1]):
            y = rx_sig[:, frame]
            y_hat = H @ (allposs / np.sqrt(Es))
            delta = y_hat - y[:,None]
            idx = np.sum(np.abs(delta)**2, axis = 0).argmin()
            tx_syms_hat[:, frame] = allposs[:, idx]

        tx_syms_hat = tx_syms_hat.reshape(-1)
        rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        #%% count
        source.CntErr(tx_bits, rx_bits)
        if source.tot_blk % 100 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***");
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n");
    source.SaveToFile(filename = logf, snr = snr)
    # return


















