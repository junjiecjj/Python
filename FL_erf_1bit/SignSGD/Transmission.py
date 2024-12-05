
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""


import scipy
import numpy as np
import torch
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int

def FedAvg_grad(w_glob, grad, device):
    ind = 0
    w_return = copy.deepcopy(w_glob)
    for item in w_return.keys():
        a = np.array(w_return[item].size())
        if len(a):
            b = np.prod(a)
            w_return[item] = copy.deepcopy(w_return[item]) - torch.from_numpy(np.reshape(grad[ind:ind+b], a)).float().to(device)
            ind = ind + b
    return w_return

# 1-bit error-free transmission
def OneBitNR(message_lst, args, normfactor):
    D = np.sum([param.numel() for param in message_lst[0].values()])
    # print(f"Dimension = {D}")

    key_lst = []
    info_lst = []
    for key, val in message_lst[0].items():
        key_lst.append(key)
        info_lst.append([key, val.size(), val.numel(), val.dtype])

    source = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_lst:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        source[k,:] = vec

    # Quantization
    symbols = np.sign(source)
    sz = np.where(symbols == 0)[0].shape
    symbols[np.where(symbols == 0)] = np.random.choice([-1, 1], size = sz, replace = True, p = [0.5, 0.5] )

    symbols_hat = copy.deepcopy(symbols) / normfactor
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

# 1-bit error-free transmission, stochastic rounding (SR),
def OneBitSR(message_lst, args, normfactor):
    D = np.sum([param.numel() for param in message_lst[0].values()])
    # print(f"Dimension = {D}")

    key_lst = []
    info_lst = []
    for key, val in message_lst[0].items():
        key_lst.append(key)
        info_lst.append([key, val.size(), val.numel(), val.dtype])

    source = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_lst:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        source[k,:] = vec

    # Quantization
    BG = 7
    bits = Quantization1bits_NP_int(source, BG)

    symbols_hat = deQuantization1bits_NP_int(bits, BG)

    mess_recov = []
    for k in range(len(message_lst)):
        symbols = symbols_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbols[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        mess_recov.append(param_k)

    return mess_recov




















































































































