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
from Channel import Large_rayleigh_fast
from Modulator import NormFactor, BPSK, demod_fastfading


def OneBit_proposed(message_lst, args, P, pl_Au, ldpc, modem, order, H = None, noisevar = 1, key_grad = None, G = 2**8):
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

    ## send
    uu = Quantization1bits_NP_int(SS, G)

    uu_hat = transmission_proposed(args, copy.deepcopy(uu), P, pl_Au, ldpc, modem, order, H = H, noisevar = noisevar)
    err_rate = (uu_hat != uu).sum(axis = 1)/uu.shape[-1]

    ## recv
    s_hat = deQuantization1bits_NP_int(uu_hat, G)

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


## 以一帧为单位消去，以P*|h|^2为排序；且考虑其他未消去用户的功率
def SIC_LDPC_FastFading(H, yy, order, Es, modem, ldpc, noisevar = 1, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    # order =  np.argsort(-np.abs(P))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    idx_set = list(np.arange(K))
    for k in order:
        idx_set.remove(k)
        hk = H[k]
        sigmaK = np.sum( np.abs(H[idx_set])**2 , axis = 0) + noisevar
        llrK = demod_fastfading(copy.deepcopy(modem.constellation), yy0, 'soft', H = hk,  Es = Es,  noise_var = sigmaK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    return uu_hat

def transmission_proposed(args, uu, P, pl_Au, ldpc, modem, order, H = None, noisevar = 1):
    K = uu.shape[0]
    D = uu.shape[-1]
    bps = int(np.log2(args.M))
    framelen = int(ldpc.codelen/bps)
    Es = NormFactor(mod_type = args.mod_type, M = args.M,)
    pad_len = int(np.ceil(D/ldpc.codedim) * ldpc.codedim - D)
    uu = np.pad(uu, ((0,0),(0, pad_len)), 'constant', constant_values = 0)

    ## encoder
    cc = np.empty((1, int(uu.shape[1]/ldpc.coderate)), dtype = np.int8)
    num_CB = int(uu.shape[1]/ldpc.codedim)
    for k in range(K):
        vec = np.array([], dtype = np.int8)
        mess = uu[k, :]
        for i in range(num_CB):
            vec = np.hstack((vec, ldpc.encoder(mess[i*ldpc.codedim : (i+1)*ldpc.codedim])))
        cc = np.vstack((cc, vec))
    cc = cc[1:, :]

    yy = np.zeros((cc.shape[0], int(cc.shape[1]/bps)), dtype = complex)
    for k in range(K):
        yy[k] = modem.modulate(cc[k])
    ## 符号能量归一化
    tx_sig = yy / np.sqrt(Es)

    num_CB = int(tx_sig.shape[-1]/ldpc.codelen)
    uu_hat = np.zeros_like(uu)
    for f in range(num_CB):
        print(f"{f}/{num_CB}")
        H = Large_rayleigh_fast(args.active_client, framelen, pl_Au, noisevar = noisevar)
        H = H * np.sqrt(P[:,None])
        symbs = tx_sig[:, int(f*ldpc.codelen):int((f+1)*ldpc.codelen)]
        y = ldpc.MACchannel(symbs, H, 1)
        uu_hat[:, int(f*ldpc.codedim):int((f+1)*ldpc.codedim)] = SIC_LDPC_FastFading(H, y, order, Es, modem, ldpc, noisevar = 1,)

    return uu_hat
































