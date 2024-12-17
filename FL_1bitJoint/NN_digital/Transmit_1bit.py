
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""


# import scipy
import numpy as np
import torch
# import seaborn as sns
# import copy
# import matplotlib.pyplot as plt
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int


# 1-bit  transmission
def OneBit(message_lst, args, rounding = 'nr', err_rate = 0, normfactor = 1):
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

    # 1bit Quantization
    if rounding == 'nr':
        ## send
        SS[np.where(SS == 0)] = np.random.randint(0, 2, size = np.where(SS == 0)[0].shape)
        uu = np.where(SS <= 0, 0, 1).astype(np.int8)

        ## bit flip, equivalent channel
        flip_mask = np.random.binomial(n = 1, p = err_rate, size = uu.shape )
        uu_hat = uu ^ flip_mask
        err_rate = (uu_hat != uu).sum(axis = 1)/uu.shape[-1]

        ## recv
        uu_hat = np.where(uu_hat <= 0, -1, 1) / normfactor

    # stochastic rounding (SR),
    elif rounding == 'sr':
        ## send
        uu = Quantization1bits_NP_int(SS, normfactor)

        ## bit flip, equivalent channel
        flip_mask = np.random.binomial(n = 1, p = err_rate, size = uu.shape )
        uu_hat = uu ^ flip_mask
        err_rate = (uu_hat != uu).sum(axis = 1)/uu.shape[-1]

        ## recv
        uu_hat = deQuantization1bits_NP_int(uu_hat, normfactor)

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


# 1-bit error-free transmission, stochastic rounding (SR),
def OneBitSR(message_lst, args, normfactor = 1):
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

    # Quantization
    uu = Quantization1bits_NP_int(SS, normfactor)
    uu_hat = deQuantization1bits_NP_int(uu, normfactor)

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
    return mess_recov, 0


















































































































