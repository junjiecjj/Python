

# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

import argparse
import socket, getpass , os
import numpy as np

import torch


def args_parser():
    home = os.path.expanduser('~') # 获取当前系统用户目录

    Args = {
    "home" : home,
    "gpu" : 1,
    "gpu_idx" : 'cuda:0',
    "seed" : 42,

    "dataset" : "MNIST",
    "dir_data": home+'/FL_Sem2026/Data',
    "IID" : False,
    "Quantization" : True,

    ## 联邦学习相关参数
    "local_bs": 50,
    "test_bs" : 128,
    "num_of_clients" : 100,
    "active_client" : 10,
    "local_epoch": 10,
    "epochs" : 1000,
    "lr" : 0.001,
    # "mu" : 0.1, #  FedProx
    "precision":'single',
    "B" : 1,  # quantization bit-width

    ##
    "save": home + '/FL_Sem2026/',
    "optimizer": "adam",
    "loss_type": "MSE",

    ##
    "CompRate": [0.2, 0.5, 0.9],
    "SNRtrain": [2, 10, 20],
    "SNRtest": np.arange(-5, 36, 1),

    }
    args = argparse.Namespace(**Args)

    # 如果想用GPU且存在GPU, 则用GPU; 否则用CPU;
    args.device = torch.device(args.gpu_idx if torch.cuda.is_available() and args.gpu else "cpu")

    return args


args = args_parser()
































