#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:32:56 2024

@author: jack
"""
import scipy
import numpy as np
import torch
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int
from mimo_channel import MIMO_Channel, SignalNorm
from ldpc_coder import LDPC_Coder_llr
import Modulator


def OneBitNR_SIMO(message_lst, args, H):
    D = np.sum([param.numel() for param in message_lst[0].values()])
    # print(f"Dimension = {D}")

    key_lst = []
    info_lst = []
    for key, val in message_lst[0].items():
        key_lst.append(key)
        info_lst.append([key, val.size(), val.numel(), val.dtype])
    ## convert dict to array
    source = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_lst:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        source[k,:] = vec

    # 1bit Quantization
    symbols = np.sign(source)
    sz = np.where(symbols == 0)[0].shape
    symbols[np.where(symbols == 0)] = np.random.choice([-1, 1], size = sz, replace = True, p = [0.5, 0.5] )

    symbols_hat = copy.deepcopy(symbols)
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = symbols_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        mess_recov.append(param_k)

    return mess_recov


def OneBitNR_SIMO_LPDC(message_lst, args, H):

    return



































