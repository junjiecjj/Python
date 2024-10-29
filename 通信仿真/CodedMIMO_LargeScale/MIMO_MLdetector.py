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
import argparse
import socket, getpass , os
import commpy as cpy
import copy


##  自己编写的库
from sourcesink import SourceSink
# from config import args
from mimo_channel import MIMO_Channel, SignalNorm, SVD_Precoding
import utility
import Modulator


def parameters():
    home = os.path.expanduser('~')

    Args = {
    "minimum_snr" : 2 ,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 500,
    "maximum_block_number" : 1000000,

    ## LDPC***0***PARAMETERS
    "max_iteration" : 50,
    "encoder_active" : 1,
    "file_name_of_the_H" : "PEG1024regular0.5.txt",

    ## others
    "home" : home,
    "smallprob": 1e-15,

    "Nt" : 4,
    "Nr" : 6,
    "P" : 1,
    "d" : 2,
    ##>>>>>>>  modulation param
    "type" : 'qam',
    "M":  16,

    # "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    }
    args = argparse.Namespace(**Args)
    return args

args = parameters()

utility.set_random_seed()

source = SourceSink()

logf = "ML_BerFer.txt"
source.InitLog(logfile = logf, promargs = args, codeargs={} )

M = args.M
Nr = args.Nr
Nt = args.Nt
P = args.P

modutype = args.type
if modutype == 'qam':
    modem = cpy.QAMModem(M)
elif modutype == 'psk':
    modem =  cpy.PSKModem(M)
Es = Modulator.NormFactor(mod_type = modutype, M = M,)

TxSequence = np.array(list(itertools.product(modem.constellation, repeat = Nt))).T

# https://blog.csdn.net/weixin_44863193/article/details/124493090
# https://zhuanlan.zhihu.com/p/634499207
# 接收方估计
# def ML_detector():
SNR = np.arange(0, 21, 2)
for snr in SNR:
    # channel = AWGN(snr, polar.coderate)
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, P = P)
        channel.circular_gaussian()
        # 编码
        # cc = encoder(uu)
        tx_bits = source.GenerateBitStr(512)

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
            y_hat = H @ (TxSequence / np.sqrt(Es))
            delta = y_hat - y[:,None]
            idx = np.sum(np.abs(delta)**2, axis = 0).argmin()
            tx_syms_hat[:, frame] = TxSequence[:, idx]

        tx_syms_hat = tx_syms_hat.reshape(-1)
        rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        #%% count
        source.CntErr(tx_bits, rx_bits)
        if source.tot_blk % 10 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***");
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n");
    source.SaveToFile(filename = logf, snr = snr)
    # return


















