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
import commpy as cpy
import copy
import argparse
import socket, getpass , os

##  自己编写的库
from sourcesink import SourceSink
# from config import args
from mimo_channel import MIMO_Channel, SignalNorm
from ldpc_coder import LDPC_Coder_llr
import utility
import Modulator

utility.set_random_seed()


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

ldpcCoder =  LDPC_Coder_llr(args)
coderargs = {'codedim' : ldpcCoder.codedim,
             'codelen' : ldpcCoder.codelen,
             'codechk' : ldpcCoder.codechk,
             'coderate' : ldpcCoder.coderate,
             'row' : ldpcCoder.num_row,
             'col' : ldpcCoder.num_col}

source = SourceSink()
logf = "LDPC_MIMO_BerFer_sic4.txt"
source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

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
# map_table, demap_table = modem.plot_constellation(Modulation_type)

# def main_mmseSIC():
SNR = np.arange(12, 21, 2)
for snr in SNR:
    P_noise = 1*(10**(-1*snr/10))
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):")
    bitsPerSym = int(np.log2(M))
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, )
        channel.circular_gaussian()
        # 编码
        uu = source.GenerateBitStr(ldpcCoder.codedim)
        cc = ldpcCoder.encoder(uu)

        # 调制
        yy = modem.modulate(cc)
        tx_symbols = yy.reshape(Nt, -1)
        ## 符号能量归一化
        tx_sig = tx_symbols / np.sqrt(Es)
        # 信道
        rx_sig = channel.forward(tx_sig, 1, snr )

        #%%==========================================================
        ##       (4) wmmse sic 基于列范数排序, 固定H, 只利用检测顺序
        ###==========================================================
        H = copy.deepcopy(channel.H)
        Hnorm = np.linalg.norm(H, ord = 2, axis = 0)
        Order = np.flip(np.argsort(Hnorm,))
        W = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt)) @ H.T.conjugate()
        WH = W@H
        llr_bits = np.zeros((Nt, rx_sig.shape[-1] * bitsPerSym))

        for nt in range(Nt):
            idx = Order[nt]
            xk_est = W[idx] @ rx_sig
            ## hard
            xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
            xk_hat = modem.modulate(xk_bits)
            rx_sig = rx_sig - np.outer(H[:, idx], xk_hat/np.sqrt(Es))

            ## soft
            hk = WH[idx, idx]
            sigmaK = P * (np.sum(np.abs(WH[idx])**2) - np.abs(WH[idx, idx])**2) + P_noise * np.sum(np.abs(W[idx])**2)
            llrK = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'soft', Es = Es, h = hk, noise_var = sigmaK)
            llr_bits[idx] = llrK

        llr_bits = llr_bits.reshape(-1)
        uu_hat, iter_num = ldpcCoder.decoder_spa(llr_bits)
        source.CntErr(uu, uu_hat)
        ##
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = snr)
    # return













































