
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
def OneBitTransmit(message_lst, args, dimension):
    # gradient2 = np.array([[]])
    # for item in copyw.keys():
    #     gradient2 = np.hstack((gradient2, np.reshape((initital_weight[item] - copyw[item]).cpu().numpy(), [1,-1])/libopt.lr))

    source = np.zeros((len(message_lst), dimension))
    for k, mess in enumerate(message_lst):
        vec = np.array([])
        for key, val in mess.items():
            vec = np.hstack((vec,  val.cpu().numpy().flatten()))
        source[k,:] = vec

    # Quantization
    symbols = np.sign(source)
    sz = np.where(symbols == 0)[0].shape
    symbols[np.where(symbols == 0)] = np.random.choice([-1, 1], size = sz, replace = True, p = [0.5, 0.5] )

    symbols_hat = copy.deepcopy(symbols)
    mess_recov = []
    for k in range(symbols_hat.shape[0]):
        ind = 0
        param_k = {}
        for name, val in message_lst[0].items():
            shape = np.array(val.size())
            if len(shape):
                l = np.prod(shape)
                param_k[name] = torch.from_numpy(np.reshape(symbols_hat[k][ind:ind+l], shape)).float().to(args.device)
                ind = ind + l
        mess_recov.append(param_k)

    return mess_recov































































































































