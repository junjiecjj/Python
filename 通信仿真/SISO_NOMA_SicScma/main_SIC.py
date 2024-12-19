

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack

numpy.nonzero()
numpy.nonzero() 函数返回输入数组中非零元素的索引。

numpy.where()
numpy.where() 函数返回输入数组中满足给定条件的元素的索引。

"""
## system lib
import galois
import numpy  as np
import datetime
import commpy as cpy
import copy
import argparse
import  os

##  自己编写的库
from sourcesink import SourceSink

from Channel import channelConfig
from Channel import AWGN_mac, BlockFading_mac, FastFading_mac, Large_mac
import utility
from LDPCcoder import LDPC_Coder, BPSK
from SICdetector_LDPC import inteleaver, SIC_LDPC_BlockFading_BPSK,SIC_LDPC_BlockFading
from SICdetector_LDPC import  SIC_LDPC_FastFading_BPSK, SIC_LDPC_FastFading_P, SIC_LDPC_FastFading
import Modulator

utility.set_random_seed()

def parameters():
    # 获取当前系统用户目录
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 0,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 300,
    "maximum_block_number" : 1000000,
    "K" : 2,    # User num

    ## LDPC***0***PARAMETERS
    "max_iteration" : 50,
    "encoder_active" : 1,
    "file_name_of_the_H" : "PEG1024regular0.5.txt",

    ## others
    "home" : home,
    "smallprob": 1e-15,

    ##>>>>>>>  modulation param
    # "type" : 'qam',
    # "M":  16,

    "type" : 'psk',
    "M":  2,    # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK

    ## channel
    'channel_type': 'fast-fading', # 'AWGN', 'block-fading', 'fast-fading', 'large'
    }
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()

# def QaryLDPC(args, ):
ldpc = LDPC_Coder(args)
coderargs = {'codedim':ldpc.codedim,
             'codelen':ldpc.codelen,
             'codechk':ldpc.codechk,
             'coderate':ldpc.coderate,
             'row':ldpc.num_row,
             'col':ldpc.num_col, }

source = SourceSink()
logf = "./resultsTXT/BER_SIC_fast_2_w_powerdiv.txt"
# logf = "./resultsTXT/xxxxxx.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs,)

## modulator
M = args.M
bps = int(np.log2(M))
framelen = int(ldpc.codelen/bps)
modutype = args.type
if modutype == 'qam':
    modem = cpy.QAMModem(M)
elif modutype == 'psk':
    modem =  cpy.PSKModem(M)
Es = Modulator.NormFactor(mod_type = modutype, M = M,)

## 遍历SNR
sigma2dB = np.arange(0, 61, 1)  # dB
sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率 w

P = np.sqrt(4**np.arange(args.K)/np.sum(4**np.arange(args.K)))
# P = np.sqrt(np.ones(args.K) / args.K)

for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN_mac(args.K, framelen)
            H = H * P.reshape(-1, 1)
        elif args.channel_type == 'block-fading':
            H = BlockFading_mac(args.K, framelen)
            H = H * P.reshape(-1, 1)
        elif args.channel_type == 'fast-fading':
            H = FastFading_mac(args.K, framelen)
            H = H * P.reshape(-1, 1)
        elif args.channel_type == 'large':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(args.K, r = 100)
            H = Large_mac(args.K, ldpc.codelen, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        inteleaverM = inteleaver(args.K, int(ldpc.codelen/bps))
        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)
        uu_sum = ldpc.bits2sum(uu)

        cc = np.zeros((args.K, ldpc.codelen), dtype = np.int8)
        for k in range(args.K):
            cc[k] =  ldpc.encoder(uu[k,:])

        ## Modulate
        symbs = np.zeros((args.K, int(ldpc.codelen/bps)),) #  dtype = complex
        for k in range(args.K):
            symbs[k] = BPSK(cc[k])
            # symbs[k] = symbs[k][inteleaverM[k]]

        ## 符号能量归一化
        symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        yy = ldpc.MACchannel(symbs, H, sigma2w)

        #>>>>>> SIC detecting Then decoding
        # uu_hat, uu_hat_sum, iter_num = SIC_LDPC_BlockFading_BPSK(H, yy, P, inteleaverM, sigma2w, Es, modem, ldpc, maxiter = 50)
        # uu_hat, uu_hat_sum, iter_num = SIC_LDPC_BlockFading(H, yy, P, inteleaverM, sigma2w, Es, modem, ldpc, maxiter = 50)

        # uu_hat, uu_hat_sum, iter_num = SIC_LDPC_FastFading_BPSK(H, yy, P, inteleaverM, sigma2w, Es, modem, ldpc, maxiter = 50)
        # uu_hat, uu_hat_sum, iter_num = SIC_LDPC_FastFading_P(H, yy, P, inteleaverM, sigma2w, Es, modem, ldpc, maxiter = 50)
        uu_hat, uu_hat_sum, iter_num = SIC_LDPC_FastFading(H, yy, P, inteleaverM, sigma2w, Es, modem, ldpc, maxiter = 50)
        source.tot_iter += iter_num
        source.CntSumErr(uu_sum, uu_hat_sum)
        # break
        source.CntBerFer(uu, uu_hat)
        # if source.tot_blk % 2 == 0:
        source.PrintScreen(snr = sigma2db)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    # break
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2db)
    # return

# QaryLDPC(args)

























































