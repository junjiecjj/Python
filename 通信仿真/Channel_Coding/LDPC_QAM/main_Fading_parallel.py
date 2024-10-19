#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack


"""
## system lib
import numpy  as np
import datetime
import commpy as cpy
import copy
import argparse
import socket, getpass , os
import multiprocessing

##  自己编写的库
from sourcesink import SourceSink
from channel import AWGN
from channel import Rayleigh
# from modulation import  BPSK,  demodu_BPSK
import utility
# from argsLDPC import args
from ldpc_coder import LDPC_Coder_llr
import Modulator



def parameters():
    # 获取当前系统主机名
    host_name = socket.gethostname()
    # 获取当前系统用户名
    user_name = getpass.getuser()
    # 获取当前系统用户目录
    user_home = os.path.expanduser('~')
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 0,
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
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()
utility.set_random_seed()

def Fading_Simulation(i, name, args, snr = 2.0, dic_berfer = '',  lock = None):
    np.random.seed(i)
    source = SourceSink()
    source.ClrCnt()

    M = args.M
    modutype = args.type
    if modutype == 'qam':
        modem = cpy.QAMModem(M)
    elif modutype == 'psk':
        modem =  cpy.PSKModem(M)
    Es = Modulator.NormFactor(mod_type = modutype, M = M,)

    channel = Rayleigh(snr)
    noise_var = 10.0 ** (-snr / 10.0)
    print( f"\nsnr = {snr}(dB):\n")
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        uu = source.GenerateBitStr(ldpcCoder.codedim)
        cc = ldpcCoder.encoder(uu)
        syms = modem.modulate(cc)
        ## 符号能量归一化
        syms  = syms / np.sqrt(Es)

        yy, H = channel.forward(syms)

        llr_yy = Modulator.demod_fading(copy.deepcopy(modem.constellation), yy, "soft", H, Es, noise_var)

        uu_hat, iter_num = ldpcCoder.decoder_spa(llr_yy)
        source.tot_iter += iter_num
        source.CntErr(uu, uu_hat)
    dic_berfer[name] = {"ber":source.ber, "fer":source.fer, "ave_iter":source.ave_iter }
    if lock != None:
        lock.acquire()
        source.PrintScreen(snr = snr);
        source.SaveToFile(snr = snr)
        lock.release()
    return

if __name__ == '__main__':
    utility.set_random_seed(1)
    # print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    ldpcCoder =  LDPC_Coder_llr(args)
    coderargs = {'codedim':ldpcCoder.codedim,
                 'codelen':ldpcCoder.codelen,
                 'codechk':ldpcCoder.codechk,
                 'coderate':ldpcCoder.coderate,
                 'row':ldpcCoder.num_row,
                 'col':ldpcCoder.num_col}
    logf = "BerFer_AWGN.txt"
    utility.WrLogHead(logfile = logf, promargs = args, codeargs = coderargs)

    m = multiprocessing.Manager()
    # dict_param = m.dict()
    dict_berfer = m.dict()
    lock = multiprocessing.Lock()  # 这个一定要定义为全局
    jobs = []

    for i, snr in enumerate(np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr)):
        ps = multiprocessing.Process(target = Fading_Simulation, args=(i, f"snr={snr:.2f}(dB)", args, snr, dict_berfer, lock ))
        jobs.append(ps)
        ps.start()

    for p in jobs:
        p.join()

    for snr in  np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr):
        name = f"snr={snr:.2f}(dB)"
        print(f"{name} {dict_berfer[name]['ber']:.8f} {dict_berfer[name]['fer']:.8f} {dict_berfer[name]['ave_iter']:.3f}")











































































































































































































































































































































































































































































