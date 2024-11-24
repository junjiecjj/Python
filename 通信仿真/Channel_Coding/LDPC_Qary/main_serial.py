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
from Channel import AWGN, QuasiStaticRayleigh, FastFadingRayleigh, LargeRician
import utility
from QaryLDPC import QLDPC_Coding, bpsk
import Modulator

utility.set_random_seed()

def parameters():
    # 获取当前系统用户目录
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 2 ,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 500,
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
    "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK

    ## channel
    'channel_type': 'block-fading', # 'AWGN', 'block-fading', 'fast-fading', 'large-small'
    }
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()
## print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## Rayleigh Fading 信道，BPSK/QAM下，LPDC编码，串行
# def QaryLDPC(args, ):
ldpc =  QLDPC_Coding(args)
coderargs = {'codedim':ldpc.codedim,
             'codelen':ldpc.codelen,
             'codechk':ldpc.codechk,
             'coderate':ldpc.coderate,
             'row':ldpc.num_row,
             'col':ldpc.num_col, }

source = SourceSink()
logf = "BER_QaryLDPC.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs, )

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



l = ldpc.p
q = 2**l
GF = galois.GF(2**l, repr = "int")
# I = GF.Identity(l)
# I[0] = 1
I = [1]*l
I = GF(I)



## 遍历SNR
sigma2dB = np.arange(6, 12, 2)  # dB
sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率 w
# sigma2dB = np.array([-50, -55, -60, -65, -70, -75, -77, -80, -85, -90, -92])  # dBm
# sigma2W = 10**(sigma2dB/10.0)/1000    # 噪声功率w
for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    # channel = Rayleigh(snr)
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN(args.K, framelen)
        elif args.channel_type == 'block-fading':
            H = QuasiStaticRayleigh(args.K, framelen)
        elif args.channel_type == 'fast-fading':
            H = FastFadingRayleigh(args.K, framelen)
        elif args.channel_type == 'large-small':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(args.K, r = 100)
            H = LargeRician(args.K, ldpc.codelen, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)
        uu_fun = np.array(I @ GF(uu), dtype = np.int8)
        cc = np.array([], dtype = np.int8)
        for k in range(args.K):
            cc = np.hstack((cc, ldpc.encoder(uu[k,:])))
        cc = cc.reshape(args.K, -1)
        ## Modulate
        symbs = np.zeros((args.K, int(ldpc.codelen/bps)), dtype = complex)
        for k in range(args.K):
            symbs[k] = bpsk(cc[k])

        ## 符号能量归一化
        symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        yy = ldpc.PassChannel(symbs, H, sigma2w)

        ## llr
        pp = ldpc.post_probability(yy, H, sigma2w)

        ## Decoding
        uu_hat, iter_num = ldpc.decoder_qary_spa(pp, maxiter = 50)
        source.tot_iter += iter_num

        source.CntBerFer(uu, uu_hat)
        # if source.tot_blk % 2 == 0:
        source.PrintScreen(snr = sigma2db)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(snr = sigma2db)
    # return

# QaryLDPC(args)

























































