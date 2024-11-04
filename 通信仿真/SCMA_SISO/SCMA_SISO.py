#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:50:37 2024

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
from Channel import circular_gaussian
from Channel import channelConfig
from Channel import Generate_hd
from Channel import PassChannel
from ldpc_coder import LDPC_Coder_llr
from SCMA_EncDec import SCMA
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

scma = SCMA(args)
scmaCB = scma.CB
ldpc =  LDPC_Coder_llr(args)
coderargs = {'codedim' : ldpc.codedim,
             'codelen' : ldpc.codelen,
             'codechk' : ldpc.codechk,
             'coderate' : ldpc.coderate,
             'row' : ldpc.num_row,
             'col' : ldpc.num_col}

source = SourceSink()
logf = "LDPC_SCMA_BerFer.txt"
source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

M = args.M
Nr = args.Nr
Nt = args.Nt
P = 1

modutype = args.type
if modutype == 'qam':
    modem = cpy.QAMModem(M)
elif modutype == 'psk':
    modem =  cpy.PSKModem(M)
Es = Modulator.NormFactor(mod_type = modutype, M = M,)
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(Nt)

# 接收方估计
sigma2dBm = np.array([-50, -55, -60, -65, -70, -75, -77, -80,])  # dBm
sigma2W = 10**(sigma2dBm/10.0)/1000    # 噪声功率
for sigma2dbm, sigma2w in zip(sigma2dBm, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2dbm}(dBm), {sigma2w}(w):")
    bitsPerSym = int(np.log2(M))
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        H0 = Generate_hd(Nr, Nt, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        # 编码
        uu = source.GenerateBitStr(Nt * ldpc.codedim)
        cc = np.array([], dtype = np.int8)
        for k in range(Nt):
            cc = np.hstack((cc, ldpc.encoder(uu[k*ldpc.codedim : (k+1)*ldpc.codedim])))
        cc = cc.reshape(args.Nt, -1)
        ## scma调制
        yy = {}
        for j in range(args.J):
            yy[j] = scma.encoder(cc[j], scmaCB)

        # 信道
        rx_sig = PassChannel(yy, H0, power = 1, )
        P_noise = 1  # 1*(10**(-1*snr/10))

        llr_bits = scma.MPA_detector(H0, rx_sig, scmaCB )

        llr_bits = llr_bits.reshape(-1)

        uu_hat = np.array([], dtype = np.int8)
        for k in range(Nt):
            uu_hat = np.hstack((uu_hat, ldpc.decoder_spa(llr_bits[k * ldpc.codelen : (k+1) * ldpc.codelen])[0] ))
        source.CntErr(uu, uu_hat)

        ##
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = sigma2dbm)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2dbm)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2dbm)
    # return













































