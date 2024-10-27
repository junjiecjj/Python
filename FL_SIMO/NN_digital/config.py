

# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""

import argparse
import os
import torch


def args_parser():
    home = os.path.expanduser('~')

    Args = {
    "home" : home,
    "gpu" : 1,
    "gpu_idx" : 'cuda:0',
    "seed" : 9999,

    "dataset" : "MNIST",
    "dir_data": home+'/DigitalFL/Dataset',
    "IID" : False,

    ## 联邦学习相关参数
    "local_up": 1,
    "local_bs": 128,
    "test_bs" : 128,
    "num_of_clients" : 100,
    "active_client" : 10,
    "num_comm" : 1000,
    "save_path" : home + '/DigitalFL/NN/',
    "lr" : 0.01,


    ## Codes
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
    "smallprob": 1e-15,

    "Nt" : 10,
    "Nr" : 16,
    "P" : 1,
    "d" : 2,
    ##>>>>>>>  modulation param
    "type" : 'qam',
    "M":  16,

    # "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    }
    args = argparse.Namespace(**Args)

    # 如果想用GPU且存在GPU, 则用GPU; 否则用CPU;
    args.device = torch.device(args.gpu_idx if torch.cuda.is_available() and args.gpu else "cpu")
    args.Nt = args.active_client
    return args


def ldpc_args():
    home = os.path.expanduser('~')

    Args = {
    "home" : home,
    ## Codes
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
    "smallprob": 1e-15,

    }
    args = argparse.Namespace(**Args)

    return args




# args = args_parser()






















