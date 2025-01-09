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
import copy

from Quantizer import QuantizationBbits_NP_int, deQuantizationBbits_NP_int


# B-bit error-free transmission, stochastic rounding (SR), Nearest rounding (NR), do not quantize batch-normalization layer.
def B_Bit(message_lst, args, rounding = 'nr', ber = 0, B = 8, key_grad = None):
    ## D = np.sum([param.numel() for param in message_lst[0].values()])
    key_lst_wo_grad = []
    info_lst = []
    ## 分离可导层和不可导层
    for key, val in message_lst[0].items():
        if key in key_grad:
            info_lst.append([key, val.size(), val.numel(), val.dtype])
        elif key not in key_grad:
            key_lst_wo_grad.append(key)

    ## 将可导层转换为数组
    D = np.sum([message_lst[0][key].numel() for key in key_grad])
    SS = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_grad:
            vec = np.hstack((vec, mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec
    M = np.abs(SS).max(axis = 1)
    G =  2**(B-1)/M

    ## B-bit Quantization
    uu = np.zeros((len(message_lst), D * B), dtype = np.int8)
    for k in range(len(message_lst)):
        uu[k] = QuantizationBbits_NP_int(SS[k], G[k], B = B, rounding = rounding)

    ## flipping
    flip_mask = np.random.binomial(n = 1, p = ber, size = uu.shape )
    uu_flipped = uu ^ flip_mask
    err_rate = (uu_flipped != uu).sum(axis = 1)/uu.shape[-1]

    ## recv
    s_hat = np.zeros((len(message_lst), D), dtype = np.float32)
    for k in range(len(message_lst)):
        s_hat[k] = deQuantizationBbits_NP_int(uu_flipped[k], G[k], B = B )

    ## recover
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = s_hat[k, :]
        param_k = {}
        start = 0
        end = 0
        ## 恢复可导层
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        ## 直接无错拷贝不可导层
        for key in key_lst_wo_grad:
            param_k[key] = copy.deepcopy(message_lst[k][key])
        mess_recov.append(param_k)
    return mess_recov, err_rate




# np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
# B = 8
# rounding = 'nr'
# m = 4
# n = 10
# S = np.random.randn(m, n) * 0.01
# M = np.abs(S).max(axis = 1)
# G =  2**(B-1)/M
# uu = np.zeros((m, n * B), dtype = np.int8)

# for k in range(m):
#     uu[k] = QuantizationBbits_NP_int(S[k], G[k], B = B, rounding = rounding)

# s = np.zeros((m, n ))
# for k in range(m):
#     s[k] = deQuantizationBbits_NP_int(uu[k], G[k], B = B)



































































































































