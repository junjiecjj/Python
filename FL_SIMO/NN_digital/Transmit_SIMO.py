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

import commpy as comm
from Quantizer import Quantization1bits_NP_int, deQuantization1bits_NP_int
from mimo_channel import MIMO_Channel, SignalNorm
from ldpc_coder import LDPC_Coder_llr
import Modulator
from mimo_channel import forward
from config import ldpc_args

def OneBitNR_SIMO(message_lst, args, H, snr_dB = 10, normfactor = 1):
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
    SS[np.where(SS == 0)] = np.random.randint(0, 2, size = np.where(SS == 0)[0].shape)
    uu = np.where(SS <= 0, 0, 1).astype(np.int8)
    bitsPerSym = int(np.log2(args.M))
    pad_len = int(np.ceil(D/bitsPerSym) * bitsPerSym - D)
    uu = np.pad(uu, ((0,0),(0, pad_len)), 'constant', constant_values = 0)

    modutype = args.type
    if modutype == 'qam':
        modem = comm.QAMModem(args.M)
    elif modutype == 'psk':
        modem =  comm.PSKModem(args.M)
    Es = Modulator.NormFactor(mod_type = modutype, M = args.M,)

    yy = modem.modulate(uu.flatten()).reshape(args.Nt, -1)
    ## 符号能量归一化
    tx_sig = yy / np.sqrt(Es)

    ## channel
    P_noise = 1*(10**(-1*snr_dB/10))
    rx_sig = forward(tx_sig, H, Tx_data_power = 1, SNR_dB = snr_dB)

    ## detector
    # H = copy.deepcopy(channel.H)
    G_MMSE = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(args.Nt)) @ H.T.conjugate()
    yy_hat = G_MMSE @ rx_sig
    uu_hat = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), yy_hat.flatten(), 'hard', Es = Es, ).reshape(args.Nt, -1)
    uu_hat = uu_hat[:, :D]

    ##
    SS_hat = np.where(uu_hat < 1, -1, 1) / normfactor
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = SS_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        mess_recov.append(param_k)

    return mess_recov

ldpcargs = ldpc_args()
LDPC =  LDPC_Coder_llr(ldpcargs)

def OneBitNR_SIMO_LPDC(message_lst, args, H, snr_dB = 10, normfactor = 1):
    D = np.sum([param.numel() for param in message_lst[0].values()])
    # print(f"Dimension = {D}")

    key_lst = []
    info_lst = []
    for key, val in message_lst[0].items():
        key_lst.append(key)
        info_lst.append([key, val.size(), val.numel(), val.dtype])
    ## convert dict to array
    SS = np.zeros((len(message_lst), D))
    for k,  mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_lst:
            vec = np.hstack((vec,  mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec

    # 1bit Quantization
    SS[np.where(SS == 0)] = np.random.randint(0, 2,size = np.where(SS == 0)[0].shape)
    uu = np.where(SS <= 0, 0, 1).astype(np.int8)
    pad_len = int(np.ceil(D/LDPC.codedim) * LDPC.codedim - D)
    uu = np.pad(uu, ((0,0),(0, pad_len)), 'constant', constant_values = 0)

    cc = np.empty((1, int(uu.shape[1]/LDPC.coderate)), dtype = np.int8)
    num_block = int(uu.shape[1]/LDPC.codedim)
    for k in range(len(message_lst)):
        vec = np.array([], dtype = np.int8)
        mess = uu[k, :]
        for i in range(num_block):
            vec = np.hstack((vec, LDPC.encoder(mess[i*LDPC.codedim : (i+1)*LDPC.codedim])))
        cc = np.vstack((cc, vec))
    cc = cc[1:, :]

    modutype = args.type
    if modutype == 'qam':
        modem = comm.QAMModem(args.M)
    elif modutype == 'psk':
        modem =  comm.PSKModem(args.M)
    Es = Modulator.NormFactor(mod_type = modutype, M = args.M,)
    bitsPerSym = int(np.log2(args.M))

    yy = modem.modulate(cc.flatten()).reshape(args.Nt, -1)
    ## 符号能量归一化
    tx_sig = yy / np.sqrt(Es)

    ## channel
    P_noise = 1*(10**(-1*snr_dB/10))
    rx_sig = forward(tx_sig, H, Tx_data_power = 1, SNR_dB = snr_dB)

    ## detector
    llr_bits = np.zeros((args.Nt, rx_sig.shape[-1] * bitsPerSym))

    W = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(args.Nt)) @ H.T.conjugate()
    WH = W@H
    for nt in range(args.Nt):
        xk_est = W[nt] @ rx_sig
        ## soft
        hk = WH[nt, nt]
        sigmaK = 1 * (np.sum(np.abs(WH[nt])**2) - np.abs(WH[nt, nt])**2) + P_noise * np.sum(np.abs(W[nt])**2)
        llrK = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'soft', Es = Es, h = hk, noise_var = sigmaK)
        llr_bits[nt] = llrK

    ##
    SS_hat = np.where(uu_hat < 1, -1, 1)
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = SS_hat[k,:]
        param_k = {}
        start = 0
        end = 0
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        mess_recov.append(param_k)

    return



































