

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
    "dir_minst": home+'/FL_Sem2026/Data',

    ## 学习相关参数
    "batch_size": 128,
    "test_bs" : 128,

    "epochs" : 1000,
    "lr" : 0.001,
    "mu" : 0.1,
    "precision":'single',

    ##
    "save": home + '/FL_Sem2026/',
    "optimizer": "Adam",
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
































