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
import socket, getpass , os

##  自己编写的库
from sourcesink import SourceSink
from channel import AWGN
from channel import Rayleigh
# from modulation import  BPSK,  demodu_BPSK
import utility
# from argsLDPC import args
from ldpc_coder import LDPC_Coder_llr
import Modulator

utility.set_random_seed()

# Modulation_type = "BPSK"
# if Modulation_type=="BPSK":
#     modem = cpy.PSKModem(2)
# elif Modulation_type=="QPSK":
#     modem = cpy.PSKModem(4)
# elif Modulation_type=="8PSK":
#     modem = cpy.PSKModem(8)
# elif Modulation_type=="4QAM":
#     modem = cpy.QAMModem(4)
# elif Modulation_type=="16QAM":
#     modem = cpy.QAMModem(16)
# elif Modulation_type=="64QAM":
#     modem = cpy.QAMModem(64)
# elif Modulation_type == "QAM256":
#     modem = cpy.QAMModem(256)
# map_table, demap_table = modem.plot_constellation(Modulation_type)
# Es = Modulator.NormFactor(mod_type='bpsk', M = 16,)



def parameters():
    # 获取当前系统主机名
    host_name = socket.gethostname()
    # 获取当前系统用户名
    user_name = getpass.getuser()
    # 获取当前系统用户目录
    user_home = os.path.expanduser('~')
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 5 ,
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

    "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    }
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()
# print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## AWGN信道，BPSK/QAM下，LPDC编码，串行
def AWGN_Simulation(args):
    ldpcCoder =  LDPC_Coder_llr(args)
    coderargs = {'codedim':ldpcCoder.codedim,
                 'codelen':ldpcCoder.codelen,
                 'codechk':ldpcCoder.codechk,
                 'coderate':ldpcCoder.coderate,
                 'row':ldpcCoder.num_row,
                 'col':ldpcCoder.num_col}

    source = SourceSink()
    logf = "BerFer_AWGN.txt"
    source.InitLog(promargs = args, codeargs = coderargs, logfile = logf)

    ## 16QAM + fading
    M = args.M
    modutype = args.type
    if modutype == 'qam':
        modem = cpy.QAMModem(M)
    elif modutype == 'psk':
        modem =  cpy.PSKModem(M)
    Es = Modulator.NormFactor(mod_type = modutype, M = M,)

    for snr in np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr):
        channel = AWGN(snr)
        source.ClrCnt()
        noise_var = 10.0 ** (-snr / 10.0)
        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
            uu = source.GenerateBitStr(ldpcCoder.codedim)
            cc = ldpcCoder.encoder(uu)
            syms = modem.modulate(cc)
            ## 符号能量归一化
            syms  = syms / np.sqrt(Es)

            yy = channel.forward(syms)

            llr_yy = Modulator.demod_awgn(copy.deepcopy(modem.constellation), yy, 'soft', Es, noise_var)

            uu_hat, iter_num = ldpcCoder.decoder_spa(llr_yy)
            source.tot_iter += iter_num
            source.CntErr(uu, uu_hat)
            if source.tot_blk % 2 == 0:
                source.PrintScreen(snr = snr)
                # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr);
        print("  *** *** *** *** ***\n");
        source.SaveToFile(filename = logf, snr = snr,)
    return

AWGN_Simulation(args)














































































































































































































































































































































































































































































