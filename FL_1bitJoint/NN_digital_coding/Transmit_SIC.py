#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:53:53 2025

@author: jack
"""


# import scipy
import numpy as np
import torch
import copy
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int




# 1-bit  transmission, same-G, only-Grad
def OneBit_SIC(message_lst, args, rounding = 'nr', err_rate = 0, key_grad = None, G = 2**8):
    key_lst_wo_grad = []
    info_lst = []
    for key, val in message_lst[0].items():
        if key in key_grad:
            info_lst.append([key, val.size(), val.numel(), val.dtype])
        elif key not in key_grad:
            key_lst_wo_grad.append(key)

    ## convert dict to array
    D = np.sum([message_lst[0][key].numel() for key in key_grad])
    SS = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_grad:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec

    # 1bit Quantization
    if rounding == 'nr':
        ## send
        SS[np.where(SS == 0)] = np.random.randint(0, 2, size = np.where(SS == 0)[0].shape)
        uu = np.where(SS <= 0, 0, 1).astype(np.int8)

        ## bit flip, equivalent channel
        flip_mask = np.random.binomial(n = 1, p = err_rate, size = uu.shape )
        uu_flipped = uu ^ flip_mask
        err_rate = (uu_flipped != uu).sum(axis = 1)/uu.shape[-1]

        ## recv
        s_hat = np.where(uu_flipped <= 0, -1, 1) / G

    # stochastic rounding (SR),
    elif rounding == 'sr':
        ## send
        uu = Quantization1bits_NP_int(SS, G)

        ## bit flip, equivalent channel
        flip_mask = np.random.binomial(n = 1, p = err_rate, size = uu.shape )
        uu_flipped = uu ^ flip_mask
        err_rate = (uu_flipped != uu).sum(axis = 1)/uu.shape[-1]

        ## recv
        s_hat = deQuantization1bits_NP_int(uu_flipped, G)

    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = s_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        for key in key_lst_wo_grad:
            param_k[key] = copy.deepcopy(message_lst[k][key])
        mess_recov.append(param_k)
    return mess_recov, err_rate



































