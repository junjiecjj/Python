#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:10:28 2024

@author: jack
"""


# import scipy
import numpy as np
import torch
# import seaborn as sns
# import copy
# import matplotlib.pyplot as plt
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int
from Quantizer import QuantizationBbits_NP_int, deQuantizationBbits_NP_int

# B-bit error-free transmission, stochastic rounding (SR), Nearest rounding (NR)
def B_Bit(message_lst, args, rounding = 'nr', err_rate = 0, normfactor = 1):
    D = np.sum([param.numel() for param in message_lst[0].values()])
    # print(f"Dimension = {D}")

    key_lst = []
    info_lst = []
    for key, val in message_lst[0].items():
        key_lst.append(key)
        info_lst.append([key, val.size(), val.numel(), val.dtype])
    ## convert dict to array
    SS = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_lst:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec

    # B-bit Quantization
    B = 8
    uu = QuantizationBbits_NP_int(SS, B = B, rounding = rounding)
    flip_mask = np.random.binomial(n = 1, p = err_rate, size = uu.shape )
    uu_flipped = uu ^ flip_mask
    err_rate = (uu_flipped != uu).sum(axis = 1)/uu.shape[-1]

    ## recv
    uu_hat = deQuantizationBbits_NP_int(uu_flipped, B = B)

    # uu_hat = uu / normfactor
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = uu_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        mess_recov.append(param_k)
    return mess_recov, err_rate















































































































































