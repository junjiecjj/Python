#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 18:20:02 2025

@author: jack
"""


import sys
import numpy as np
import torch
import copy
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int
from Quantizer import QuantizationBbits_NP_int, deQuantizationBbits_NP_int
from Channel import Large_rayleigh_fast
from Modulator import NormFactor, BPSK, demod_fastfading


def OneBit_OMA(message_lst, args, P, pl_Au, ldpc, modem, H = None, noisevar = 1, key_grad = None, G = 2**8):
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

    ## channel
    uu_hat = transmission_OMA(args, copy.deepcopy(uu), P, pl_Au, ldpc, modem, H = H, noisevar = noisevar)
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

def transmission_OMA(args, uu, P, pl_Au, ldpc, modem, H = None, noisevar = 1):
    K, D = uu.shape
    codelen = ldpc.codelen
    codedim = ldpc.codedim
    bps = int(np.log2(args.M))
    framelen = int(codelen/bps)
    Es = NormFactor(mod_type = args.mod_type, M = args.M,)
    pad_len = int(np.ceil(D/codedim) * codedim - D)
    uu = np.pad(uu, ((0,0),(0, pad_len)), 'constant', constant_values = 0)

    ## encoder
    cc = np.empty((1, int(uu.shape[1]/ldpc.coderate)), dtype = np.int8)
    num_CB = int(uu.shape[1]/codedim)
    for k in range(K):
        vec = np.array([], dtype = np.int8)
        mess = uu[k, :]
        for i in range(num_CB):
            vec = np.hstack((vec, ldpc.encoder(mess[i*codedim : (i+1)*codedim])))
        cc = np.vstack((cc, vec))
    cc = cc[1:, :]

    yy = np.zeros((cc.shape[0], int(cc.shape[1]/bps)), dtype = complex)
    for k in range(K):
        yy[k] = modem.modulate(cc[k])
    ## 符号能量归一化
    tx_sig = yy / np.sqrt(Es)

    num_CB = int(tx_sig.shape[-1]/codelen)
    uu_hat = np.zeros_like(uu)
    for f in range(num_CB):
        print("\r", end="")
        print("  进度: {:.3f}%: ".format(f*100/(num_CB-1)), "▓" * (f // 4), end="")
        sys.stdout.flush()

        H = Large_rayleigh_fast(args.active_client, framelen, pl_Au, noisevar = noisevar)
        H = H * np.sqrt(P[:,None])
        symbs = tx_sig[:, int(f*codelen):int((f+1)*codelen)]

        ## 符号能量归一化
        # symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        noise = np.sqrt(1/2) * (np.random.randn(*symbs.shape) + 1j*np.random.randn(*symbs.shape))
        yy = H * symbs + noise

        ##>>>>> seperated decoding
        tmp = np.zeros((K, codedim), dtype = np.int8)
        for k in range(K):
            Noise = np.ones(yy.shape[1]) * 1
            llrk = demod_fastfading(copy.deepcopy(modem.constellation), yy[k,:], 'soft', H = H[k,:],  Es = Es,  noise_var = Noise)
            tmp[k,:], iterk = ldpc.decoder_msa(llrk)

        uu_hat[:, int(f*codedim):int((f+1)*codedim)] = tmp
        # y = ldpc.MACchannel(symbs, H, 1)
        ##>>>>> Joint detecting & decoding
        ## llr
        # pp = ldpc.post_probability(y, H, 1)
        # uu_hat[:, int(f*ldpc.codedim):int((f+1)*ldpc.codedim)] = ldpc.decoder_FFTQSPA(pp, maxiter = 50)[0]
    uu_hat = uu_hat[:, :D]
    return uu_hat

def B_Bit_OMA(message_lst, args, P, pl_Au, ldpc, modem, rounding = 'sr', H = None, noisevar = 1, B = 8, key_grad = None):
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

    ## channel
    uu_hat = transmission_OMA(args, copy.deepcopy(uu), P, pl_Au, ldpc, modem, H = H, noisevar = noisevar)
    err_rate = (uu_hat != uu).sum(axis = 1)/uu.shape[-1]

    ## recv
    s_hat = np.zeros((len(message_lst), D), dtype = np.float32)
    for k in range(len(message_lst)):
        s_hat[k] = deQuantizationBbits_NP_int(uu_hat[k], G[k], B = B )

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





























