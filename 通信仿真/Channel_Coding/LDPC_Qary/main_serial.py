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
import numpy  as np
import datetime
import commpy as cpy
import copy
import argparse
import  os

##  自己编写的库
from sourcesink import SourceSink
# from Channel import AWGN
# from Channel import Rayleigh
from Channel import channelConfig
from Channel import AWGN, QuasiStaticRayleigh, FastFadingRayleigh, LargeRician
import utility
from LDPC import LDPC_Coder_llr
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
    "K" : 10,    # User num

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
    'channel_type': 'large-small', # 'AWGN', 'block-fading', 'fast-fading', 'large-small'
    }
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()
## print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## Rayleigh Fading 信道，BPSK/QAM下，LPDC编码，串行
# def QaryLDPC(args, ):
ldpc =  LDPC_Coder_llr(args)
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
modutype = args.type
if modutype == 'qam':
    modem = cpy.QAMModem(M)
elif modutype == 'psk':
    modem =  cpy.PSKModem(M)
Es = Modulator.NormFactor(mod_type = modutype, M = M,)

## 遍历SNR
sigma2dB = np.arange(0, 12, 2)  # dB
sigma2W = 10**(-sigma2dB/10.0)  # 噪声功率 w
# sigma2dB = np.array([-50, -55, -60, -65, -70, -75, -77, -80, -85, -90, -92])  # dBm
# sigma2W = 10**(sigma2dB/10.0)/1000    # 噪声功率w
for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    # channel = Rayleigh(snr)
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN(args.K, ldpc.codelen)
        elif args.channel_type == 'block-fading':
            H = QuasiStaticRayleigh(args.K, ldpc.codelen)
        elif args.channel_type == 'fast-fading':
            H = FastFadingRayleigh(args.K, ldpc.codelen)
        elif args.channel_type == 'large-small':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(args.K, r = 100)
            H = LargeRician(args.K, ldpc.codelen, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)
        cc = np.array([], dtype = np.int8)
        for k in range(args.K):
            cc = np.hstack((cc, ldpc.encoder(uu[k,:])))
        cc = cc.reshape(args.K, -1)

        ## modulate
        syms = modem.modulate(cc)
        ## 符号能量归一化
        syms  = syms / np.sqrt(Es)

        ## channel
        yy, H = channel.forward(syms)

        ## decoding
        llr_yy = Modulator.demod_fading(copy.deepcopy(modem.constellation), yy, "soft", H, Es, noise_var)

        uu_hat, iter_num = ldpcCoder.decoder_spa(llr_yy)
        source.tot_iter += iter_num
        source.CntErr(uu, uu_hat)
        if source.tot_blk % 2 == 0:
            source.PrintScreen(snr = snr)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(snr = snr)
    # return

# QaryLDPC(args)

























































