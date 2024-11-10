#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:27:55 2024

@author: jack
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
from SCMA_EncDec import SCMA_SISO
import utility
import Modulator

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
    "Nit" : 10,
    ## channel
    'channel_type': 'fast fading rayleigh', # 'AWGN', 'quasi-static rayleigh', 'fast fading rayleigh', 'large + quasi-static rician'
    }
    args = argparse.Namespace(**Args)
    return args
args = parameters()


## SCMA
scma = SCMA_SISO()
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

frame_len = int(ldpc.codelen/bitsPerSym)

## Source
source = SourceSink()
logf = "LDPC_SCMA_BerFer.txt"
source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

## 遍历SNR
sigma2dB = np.arange(6.5, 11, 0.5)  # dB
sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率w
for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN(K, J, frame_len)
        elif args.channel_type == 'quasi-static rayleigh':
            H = QuasiStaticRayleigh(K, J, frame_len)
        elif args.channel_type == 'fast fading rayleigh':
            H = FastFadingRayleigh(K, J, frame_len)
        elif args.channel_type == 'large + quasi-static rician':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(J, r = 100)
            H = LargeRician(K, J, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1)
        # 编码
        uu = source.SourceBits(scma.J, ldpc.codedim)
        cc = np.array([], dtype = np.int8)
        for j in range(scma.J):
            cc = np.hstack((cc, ldpc.encoder(uu[j,:])))
        cc = cc.reshape(scma.J, -1)

        symbols = scma.mapping(cc, )
        yy = scma.encoder(symbols, H, )
        rx_sig = PassChannel(yy, noise_var = sigma2w, )

        symbols_hat, uu_hat, llr_bits = scma.MPAdetector_SISO_soft(rx_sig, H, sigma2w, Nit = args.Nit)
        uu_hat = np.array([], dtype = np.int8)
        for j in range(scma.J):
            uu_hat = np.hstack((uu_hat, ldpc.decoder_spa(llr_bits[j,:])[0] ))
        uu_hat = uu_hat.reshape(scma.J, -1)

        source.CntBerFer(uu, uu_hat)
        # source.CntSer(symbols, symbols_hat)
        ##
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2db)
    # return













































