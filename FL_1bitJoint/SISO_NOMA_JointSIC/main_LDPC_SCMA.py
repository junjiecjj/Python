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
from Channel import AWGN_scma, BlockFading_scma, FastFading_scma, Large_scma
from Channel import PassChannelSCMA
from LDPCcoder import LDPC_Coder
from SCMA_EncDec import SCMA_SISO
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
    ## channel
    'channel_type': 'fast-fading', # 'AWGN', 'block-fading', 'fast-fading', 'large'
    }
    args = argparse.Namespace(**Args)
    return args
args = parameters()


## SCMA
scma = SCMA_SISO()
bitsPerSym = int(np.log2(scma.M))

## LDPC
ldpc =  LDPC_Coder(args)
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
logf = "./resultsTXT/SCMA_MPA_LDPC_fast.txt"

source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

## 遍历SNR
sigma2dB = np.arange(0, 20, 1)  # dB
sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率w

for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'block-fading':
            H = BlockFading_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'fast-fading':
            H = FastFading_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'large':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(scma.J, r = 100)
            H = Large_scma(scma.K, scma.J, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        # 编码
        uu = source.SourceBits(scma.J, ldpc.codedim)
        cc = np.zeros((scma.J, ldpc.codelen), dtype = np.int8)
        for j in range(scma.J):
            cc[j] =  ldpc.encoder(uu[j,:])

        symbols = scma.mapping(cc, )
        yy = scma.encoder(symbols, H, )
        rx_sig = PassChannelSCMA(yy, noise_var = sigma2w, )
        symbols_hat, uu_hard, llr_bits = scma.MPAdetector_SISO_soft(rx_sig, H, sigma2 = sigma2w, Nit = args.Nit)
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













































