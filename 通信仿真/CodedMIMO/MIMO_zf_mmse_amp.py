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
import os
import argparse
import copy


##  自己编写的库
from sourcesink import SourceSink
# from config import args
from mimo_channel import MIMO_Channel, SignalNorm, SVD_Precoding
import utility
from mimo_channel import  SignalNorm
from mimo_channel import channelConfig, Generate_hd, PassChannel
import Modulator


def AMP(H, y, sigma2, sigmas2, maxiter):
    M, N = H.shape
    L = y.shape[-1]
    r = np.zeros((M, L))
    xhat = np.zeros((N, L))
    alpha = sigmas2
    for i in range(maxiter):
        r = y - H@xhat + (N/M) * sigmas2 / (sigmas2 + alpha) * r
        alpha = sigma2 + (N/M) * sigmas2*alpha/(sigmas2+alpha)
        xhat = (sigmas2/(sigmas2+alpha))*(H.conjugate().T @ r + xhat)
    return xhat

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

    "Nt" : 16,
    "Nr" : 128,
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
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(Nt)

logfile = "AMP_BerFer.txt"
SNR = np.arange(0, 21, 2)
source = SourceSink()
source.InitLog(logfile = logfile, promargs = args,  codeargs = {} )
# 接收方估计
# def main_ZF_MMSE():
for snr in SNR:
    # channel = AWGN(snr, polar.coderate)
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):\n")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, P = P)
        channel.circular_gaussian()

        # 编码
        # cc = encoder(uu)
        cc = source.GenerateBitStr(1920)

        # 调制
        symbols = modem.modulate(cc)
        symbols = symbols.reshape(Nt, -1)
        ## 符号能量归一化
        yy =  symbols / np.sqrt(Es)

        # 信道
        yy_hat = channel.forward(yy, 1, snr )

        # #%%============================================
        # #                  ZF
        # ##============================================
        pinvH = scipy.linalg.pinv(channel.H)
        symbols_hat = pinvH @ yy_hat
        symbols_hat = symbols_hat.reshape(-1)
        cc_hat = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), symbols_hat, 'hard', Es = Es, )

        # #%%============================================
        # #                 MMSE
        # ##=============================================
        # H = channel.H[:]
        # P_noise = 1*(10**(-1*snr/10))
        # G_MMSE = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt)) @ H.T.conjugate()
        # symbols_hat = G_MMSE @ yy_hat
        # symbols_hat = symbols_hat.reshape(-1)
        # cc_hat = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), symbols_hat, 'hard', Es = Es, ) # modem.demodulate(rx_symb_mmse, 'hard',)

        #%%============================================
        #                 AMP, 此时的H必须除以Nr
        ##============================================
        H = channel.H[:]
        P_noise = 1*(10**(-1*snr/10))
        symbols_hat = AMP(H, yy_hat, P_noise, 1, 6)
        symbols_hat = symbols_hat.reshape(-1)
        cc_hat = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), symbols_hat, 'hard', Es = Es, )

        #%% count
        source.CntErr(cc, cc_hat)
        if source.tot_blk % 200 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***");
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n");
    source.SaveToFile(snr = snr, filename = logfile)
    # return


#
# main_ZF_MMSE()





