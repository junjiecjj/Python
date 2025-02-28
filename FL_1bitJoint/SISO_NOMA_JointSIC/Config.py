#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:51:00 2024

@author: jack
"""

import argparse
import  os



def parameters():
    # 获取当前系统用户目录
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 0,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 150,
    "maximum_block_number" : 1000000,
    "K" : 4,    # User num

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
    'channel_type': 'large_fast', # 'AWGN', 'block-fading', 'fast-fading', 'large_fast', 'large_block'
    }
    args = argparse.Namespace(**ldpc_args)

    ## system arg
    args.B     = 4e6                           # bandwidth, Hz
    # args.n0     = -140                       # 噪声功率谱密度, dBm/Hz
    # args.n0     = 10**(args.n0/10.0)/1000    # 噪声功率谱密度, Watts/Hz
    # args.N0     = args.n0 * args.B           # 噪声功率, Watts

    args.P_total = args.K
    # args.P_max   = 30                          # 用户发送功率, dBm
    # args.P_max   = 10**(args.P_max/10.0)/1000  # Watts
    args.P_max   = args.P_total / 3              # Watts

    return args





































