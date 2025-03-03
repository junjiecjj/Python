

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2024.11.12

@author: jack

"""
## system lib
import galois
import numpy  as np
import datetime
import commpy as cpy
import copy
import argparse
import os

##  自己编写的库
import utility
from sourcesink import SourceSink
from Channel import channelConfig
from Channel import Large_rayleigh_fast, Large_rician_fast
from LDPCcoder import LDPC_Coder, BPSK
from SICdetector_LDPC import  inteleaver
from SICdetector_LDPC import  SIC_LDPC_FastFading
import Modulator
from Config import parameters
from CapacityOptimizer import NOMAcapacityOptim
utility.set_random_seed(42)


# def QaryLDPC(args, ):
args = parameters()
ldpc = LDPC_Coder(args)
coderargs = {'codedim':ldpc.codedim,
             'codelen':ldpc.codelen,
             'codechk':ldpc.codechk,
             'coderate':ldpc.coderate,
             'row':ldpc.num_row,
             'col':ldpc.num_col, }

source = SourceSink()
logf = f"./resultsTXT/fast/BER_SIC_{args.channel_type}_{args.K}U_w_pa.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs,)

## modulator
modem, Es, bps = Modulator.modulator(args.type, args.M)
framelen = int(ldpc.codelen/bps)
inteleaverM = inteleaver(args.K, int(ldpc.codelen/bps))

BS_locate, users_locate, beta_Au, PL_Au, d_Au = channelConfig(args.K, r = 100, rmin = args.rmin)

# 遍历SNR
n0   = np.arange(-128.2, -144, -0.2)         # 噪声功率谱密度, dBm/Hz
n00  = 10**(n0/10.0)/1000                # 噪声功率谱密度, Watts/Hz
N0   = n00 * args.B                      # 噪声功率, Watts

for noisePsd, noisepower in zip(n0, N0):
    source.ClrCnt()
    print( f"\n noisePsd = {noisePsd}(dBm/Hz), {noisepower}(w):")

    Htmp = Large_rayleigh_fast(args.K, 100000, PL_Au, noisevar = noisepower)
    Hbar = np.mean(np.abs(Htmp)**2, axis = 1)
    ## (1) Power allocation in NOMA for fast fading.
    P, total_capacity, SINR, Capacity = NOMAcapacityOptim(Hbar, d_Au, args.P_total, args.P_max, noisevar = 1, )
    ## (2) Without Power allocation, equal allocation. for fast fading.
    # P = np.ones(args.K) * args.P_total/args.K
    order = np.argsort(P*Hbar)[::-1]
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        if args.channel_type == 'large_fast':
            H = Large_rayleigh_fast(args.K, framelen, PL_Au, noisevar = noisepower)

        H = H * np.sqrt(P[:,None])
        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)
        uu_sum = ldpc.bits2sum(uu)

        cc = np.zeros((args.K, ldpc.codelen), dtype = np.int8)
        for k in range(args.K):
            cc[k] =  ldpc.encoder(uu[k,:])

        ## Modulate
        symbs = np.zeros((args.K, int(ldpc.codelen/bps)), dtype = complex) # dtype = complex
        for k in range(args.K):
            # symbs[k] = BPSK(cc[k])
            symbs[k] = modem.modulate(cc[k])
            # symbs[k] = symbs[k][inteleaverM[k]]

        ## 符号能量归一化
        symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        yy = ldpc.MACchannel(symbs, H, 1)

        #>>>>>> SIC detecting Then decoding
        if args.channel_type == 'large_block':
            pass
        elif args.channel_type == 'large_fast':
            uu_hat, uu_hat_sum, iter_num = SIC_LDPC_FastFading(H, yy, order, inteleaverM,  Es, modem, ldpc, noisevar = 1, maxiter = 50)
        source.tot_iter += iter_num
        source.CntSumErr(uu_sum, uu_hat_sum)
        # break
        source.CntBerFer(uu, uu_hat)
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = noisePsd)

    print("  *** *** *** *** ***")
    source.PrintScreen(snr = noisePsd)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = noisePsd)
    # return
























































