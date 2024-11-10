#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:50:37 2024

"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import commpy as comm
import copy
import argparse
import os

##  自己编写的库
from sourcesink import SourceSink
# from Channel import circular_gaussian
from Channel import channelConfig
from Channel import AWGN, QuasiStaticRayleigh, FastFadingRayleigh, LargeRician
from Channel import PassChannel
from ldpc_coder import LDPC_Coder_llr
from SCMA_EncDec import SCMA_SIMO
import utility
# import Modulator

utility.set_random_seed(1)

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
    ##>>>>>>>  modulation param
    "type" : 'qam',
    "M":  16,

    # "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    "Nit" : 6,
    "Nr" : 4,
    ## channel
    'channel_type': 'large + quasi-static rician', # 'AWGN', 'quasi-static rayleigh', 'fast fading rayleigh', 'large + quasi-static rician'
    }
    args = argparse.Namespace(**Args)
    return args
args = parameters()


## SCMA
scma = SCMA_SIMO()
scmaCB = scma.CB
row_valid = np.where(scma.FG != 0)[1].reshape(scma.K, -1)
col_valid = np.where(scma.FG.T != 0)[1].reshape(scma.J, -1).T
J = scma.J # user num
K = scma.K # resource block num
M = scma.M # codeword num
bitsPerSym = int(np.log2(M))

## LDPC
ldpc =  LDPC_Coder_llr(args)
coderargs = {'scma.J' : scma.J ,
             'scma.K' : scma.K,
             'scma.M' : scma.M,
             'codedim' : ldpc.codedim,
             'codelen' : ldpc.codelen,
             'codechk' : ldpc.codechk,
             'coderate' : ldpc.coderate,
             'row' : ldpc.num_row,
             'col' : ldpc.num_col}

frame_len = int(ldpc.codedim/bitsPerSym)

## Source
source = SourceSink()
logf = "SCMA_EPAdetector_SIMO_large.txt"
source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

## 遍历SNR
# sigma2dB = np.arange(0, 16, 1)  # dB
# sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率w
sigma2dB = np.array([-50, -55, -60, -65, -70, -75, -77, -80,])  # dBm
sigma2W = 10**(sigma2dB/10.0)/1000    # 噪声功率
for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        # print(f"{source.tot_blk}")
        if args.channel_type == 'AWGN':
            H = AWGN(K, args.Nr, J, frame_len)
        elif args.channel_type == 'quasi-static rayleigh':
            H = QuasiStaticRayleigh(K, args.Nr, J, frame_len)
        elif args.channel_type == 'fast fading rayleigh':
            H = FastFadingRayleigh(K, args.Nr, J, frame_len)
        elif args.channel_type == 'large + quasi-static rician':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(J, r = 100)
            H = LargeRician(K, args.Nr, J, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        ## 编码
        uu = source.SourceBits(scma.J, ldpc.codedim)
        symbols = scma.mapping(uu, )
        yy = scma.encoder(symbols, H, args.Nr)
        rx_sig = PassChannel(yy, noise_var = 1, )
        # symbols_hat, uu_hat = scma.MPAdetector_SIMO_hard(rx_sig, H, 1, args.Nr, Nit = args.Nit)
        symbols_hat, uu_hat = scma.EPAdetector_SIMO_hard(rx_sig, H, 1, args.Nr, Nit = args.Nit)
        source.CntBerFer(uu, uu_hat)
        source.CntSer(symbols, symbols_hat)

        # if source.tot_blk % 1 == 0:
        source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2db)
    # return













































