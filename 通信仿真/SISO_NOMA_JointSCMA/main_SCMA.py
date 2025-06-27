#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:50:37 2024

"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import commpy as comm
# import copy
# import argparse
# import os

##  自己编写的库
from sourcesink import SourceSink
from Config import parameters
from Channel import channelConfig
from Channel import AWGN_scma, BlockFading_scma, FastFading_scma, Large_scma
from Channel import PassChannelSCMA
from LDPCcoder import LDPC_Coder
from SCMA_EncDec import SCMA_SISO
import utility
# import Modulator

utility.set_random_seed()
args = parameters()


## SCMA
scma = SCMA_SISO()
bps = int(np.log2(scma.M))

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
frame_len = int(ldpc.codelen/bps)

## Source
source = SourceSink()
logf = f"./resultsTXT/{args.channel_type}/BER_onlySCMA_{args.channel_type}_{args.K}U_wo_pa.txt"

source.InitLog(logfile = logf, promargs = args, codeargs = coderargs )

## 遍历SNR
sigma2dB = np.arange(0, 20, 1)  # dB
sigma2W =  args.K / 2 * 10**(-sigma2dB/10.0)  # 噪声功率w
sigma2W /= scma.K

for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'block_fading':
            H = BlockFading_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'fast_fading':
            H = FastFading_scma(scma.K, scma.J, frame_len)
        elif args.channel_type == 'large':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(scma.J, r = 100)
            H = Large_scma(scma.K, scma.J, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        # 编码
        uu = source.SourceBits(scma.J, ldpc.codedim)
        symbols = scma.mapping(uu, )
        yy = scma.encoder(symbols, H, )
        rx_sig = PassChannelSCMA(yy, noise_var = sigma2w, )
        symbols_hat, uu_hat = scma.MPAdetector_SISO_hard(rx_sig, H, sigma2w, Nit = args.Nit)
        # symbols_hat, uu_hat = scma.LogMPAdetector_SISO_hard(rx_sig, H, sigma2w, Nit = args.Nit)
        # symbols_hat, uu_hat = scma.maxLogMPAdetector_SISO_hard(rx_sig, H, sigma2w, Nit = args.Nit)
        source.CntBerFer(uu, uu_hat)
        # source.CntSer(symbols, symbols_hat)
        ##
        # if source.tot_blk % 1 == 0:
        source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2db)
    # return













































