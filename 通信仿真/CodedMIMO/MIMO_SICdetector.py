#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:50:37 2024
https://blog.csdn.net/weixin_52135976/article/details/118893267


https://blog.csdn.net/weixin_43871127/article/details/104593325


@author: jack

可以看到最大比合并MCR的目标是最大化MCR，ZF的目标是最大化SIR，MMSE的目标是最大化SINR

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy as cpy
import copy
import argparse
import socket, getpass , os

##  自己编写的库
from sourcesink import SourceSink
# from config import args
from mimo_channel import MIMO_Channel, SignalNorm, SVD_Precoding
import utility
import Modulator


def parameters():
    home = os.path.expanduser('~')

    Args = {
    "minimum_snr" : 2 ,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 500,
    "maximum_block_number" : 1000000,

    ## LDPC***0***PARAMETERS
    "max_iteration" : 50,
    "encoder_active" : 1,
    "file_name_of_the_H" : "PEG1024regular0.5.txt",

    ## others
    "home" : home,
    "smallprob": 1e-15,

    "Nt" : 4,
    "Nr" : 6,
    "P" : 1,
    "d" : 2,
    ##>>>>>>>  modulation param
    "type" : 'qam',
    "M":  16,

    # "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    }
    args = argparse.Namespace(**Args)
    return args

args = parameters()
utility.set_random_seed()


source = SourceSink()
logf = "SIC_BerFer.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = {} )

M = args.M
Nr = args.Nr
Nt = args.Nt
P = args.P

modutype = args.type
if modutype == 'qam':
    modem = cpy.QAMModem(M)
elif modutype == 'psk':
    modem =  cpy.PSKModem(M)
Es = Modulator.NormFactor(mod_type = modutype, M = M,)

# map_table, demap_table = modem.plot_constellation(modutype)


# https://blog.csdn.net/weixin_44863193/article/details/124493090
# https://zhuanlan.zhihu.com/p/634499207
# 接收方估计
# def main_mmseSIC():
SNR = np.arange(0, 27, 2)
for snr in SNR:
    # channel = AWGN(snr, polar.coderate)
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, P = P)
        channel.circular_gaussian()
        # 编码
        tx_bits = source.GenerateBitStr(512)

        # 调制
        tx_symbols = modem.modulate(tx_bits)
        tx_symbols = tx_symbols.reshape(Nt, -1)
        ## 符号能量归一化
        tx_sig =  tx_symbols / np.sqrt(Es)

        # 信道
        rx_sig = channel.forward(tx_sig, 1, snr )
        P_noise = 1*(10**(-1*snr/10))
        #%%============================================
        #              (0)  ZF
        ##============================================
        # H = copy.deepcopy(channel.H)
        # pinvH = scipy.linalg.pinv(H)
        # rx_data_zf = pinvH @ rx_sig
        # rx_symb_zf = rx_data_zf.reshape(-1)
        # rx_bits = Modulator.demod(copy.deepcopy(modem.constellation), rx_symb_zf, 'hard', Es = Es, )

        # # #%%============================================
        # # #          (0)  WMMSE
        # # ##============================================
        # H = copy.deepcopy(channel.H)
        # G_MMSE = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt)) @ H.T.conjugate()
        # rx_data_mmse = G_MMSE @ rx_sig
        # rx_symb_mmse = rx_data_mmse.reshape(-1)
        # rx_bits = Modulator.demod(copy.deepcopy(modem.constellation), rx_symb_mmse, 'hard', Es = Es, )

        #%%============================================
        ##       (一) mmse sic 基于SINR排序, 书上(11.15)
        ###============================================
        # H = copy.deepcopy(channel.H)

        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)
        # Order = []
        # idx_ary = list(np.arange(Nt))
        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     Wmmse = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt - nt)) @ H.T.conjugate()
        #     WH = Wmmse @ H
        #     SINR = []
        #     for i in range(Nt - nt):
        #         tmp = P * (np.sum(np.abs(WH[i])**2) - np.abs(WH[i, i])**2) + P_noise * np.sum(np.abs(Wmmse[i])**2)
        #         SINR.append(P * np.abs(WH[i, i])**2 / tmp)
        #     maxidx = np.argmax(SINR)
        #     Order.append(idx_ary[maxidx])
        #     idx_ary.remove(idx_ary[maxidx])
        #     xk_est = Wmmse[maxidx] @ rx_sig
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[-1]] = xk_hat
        #     rx_sig = rx_sig -  np.outer(H[:, maxidx], xk_hat/np.sqrt(Es))
        #     H = np.delete(H, [maxidx], axis = 1)
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        #%%============================================
        ##       (二) wmmse sic 基于SNR排序 ,书上(11.17)
        ###============================================
        # H = copy.deepcopy(channel.H)

        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)
        # Order = []
        # idx_ary = list(np.arange(Nt))
        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     W = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt - nt)) @ H.T.conjugate()
        #     SNRo = np.linalg.norm(W, ord = 2, axis = 1)
        #     minidx = np.argmin(SNRo)
        #     Order.append(idx_ary[minidx])
        #     idx_ary.remove(idx_ary[minidx])
        #     xk_est = W[minidx] @ rx_sig
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[-1]] = xk_hat
        #     rx_sig = rx_sig -  np.outer(H[:, minidx], xk_hat/np.sqrt(Es))
        #     H = np.delete(H, [minidx], axis = 1)
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        # #%%============================================
        # ##       (二) zf sic 基于SNR排序, 书上(11.17)
        # ###============================================
        # H = copy.deepcopy(channel.H)

        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)
        # Order = []
        # idx_ary = list(np.arange(Nt))
        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     W = scipy.linalg.pinv(H)
        #     SNRo = np.linalg.norm(W, ord = 2, axis = 1)
        #     minidx = np.argmin(SNRo)
        #     Order.append(idx_ary[minidx])
        #     idx_ary.remove(idx_ary[minidx])
        #     xk_est = W[minidx] @ rx_sig
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[-1]] = xk_hat
        #     rx_sig = rx_sig -  np.outer(H[:, minidx], xk_hat/np.sqrt(Es))
        #     H = np.delete(H, [minidx], axis = 1)
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        # # #%%============================================
        # # ##       (三) wmmse sic 基于列范数排序,每次更新H
        # # ###============================================
        H = copy.deepcopy(channel.H)
        Hnorm = np.linalg.norm(H, ord = 2, axis = 0)
        Order = np.flip(np.argsort(Hnorm,))
        tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)

        # print(f"0: idx_ary = {idx_ary}")
        for nt in range(Nt):
            H_bar = H[:, Order[nt:]]
            G = scipy.linalg.pinv(H_bar.T.conjugate()@H_bar + P_noise*np.eye(Nt - nt)) @ H_bar.T.conjugate()
            xk_est = (G @ rx_sig)[0,:]
            xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
            xk_hat = modem.modulate(xk_bits)
            tx_syms_hat[Order[nt]] = xk_hat
            rx_sig = rx_sig - np.outer(H[:, Order[nt]], xk_hat/np.sqrt(Es))
        tx_syms_hat = tx_syms_hat.reshape(-1)
        rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        # #%%============================================
        # ##       (三) zf sic 基于列范数排序,每次更新H
        # ###============================================
        # H = copy.deepcopy(channel.H)
        # Hnorm = np.linalg.norm(H, ord = 2, axis = 0)
        # Order = np.flip(np.argsort(Hnorm,))
        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)

        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     H_bar = H[:, Order[nt:]]
        #     G = scipy.linalg.pinv(H_bar)
        #     xk_est = (G @ rx_sig)[0,:]
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[nt]] = xk_hat
        #     rx_sig = rx_sig - np.outer(H[:, Order[nt]], xk_hat/np.sqrt(Es))
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        # # #%%================================================================
        # # ##       (三) wmmse sic 基于列范数排序,固定H，只利用检测顺序, 书上(11.18)
        # # ###================================================================
        # H = copy.deepcopy(channel.H)
        # Hnorm = np.linalg.norm(H, ord = 2, axis = 0)
        # Order = np.flip(np.argsort(Hnorm,))
        # W = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt)) @ H.T.conjugate()
        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)

        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     xk_est = W[Order[nt]] @ rx_sig
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[nt]] = xk_hat
        #     rx_sig = rx_sig -  np.outer(H[:, Order[nt]], xk_hat/np.sqrt(Es))
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        # # #%%================================================================
        # # ##       (三) zf sic 基于列范数排序,固定H，只利用检测顺序, 书上(11.18)
        # # ###================================================================
        # H = copy.deepcopy(channel.H)
        # Hnorm = np.linalg.norm(H, ord = 2, axis = 0)
        # Order = np.flip(np.argsort(Hnorm,))
        # W = scipy.linalg.pinv(H)
        # tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)

        # # print(f"0: idx_ary = {idx_ary}")
        # for nt in range(Nt):
        #     xk_est = W[Order[nt]] @ rx_sig
        #     xk_bits = Modulator.demod_MIMO(copy.deepcopy(modem.constellation), xk_est, 'hard', Es = Es, )
        #     xk_hat = modem.modulate(xk_bits)
        #     tx_syms_hat[Order[nt]] = xk_hat
        #     rx_sig = rx_sig -  np.outer(H[:, Order[nt]], xk_hat/np.sqrt(Es))
        # tx_syms_hat = tx_syms_hat.reshape(-1)
        # rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        #%% count
        source.CntErr(tx_bits, rx_bits)
        if source.tot_blk % 100 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***");
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n");
    source.SaveToFile(filename = logf, snr = snr)
    # return


















































