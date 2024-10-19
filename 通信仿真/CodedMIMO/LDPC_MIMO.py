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


##  自己编写的库
from sourcesink import SourceSink
from config import args
from mimo_channel import MIMO_Channel, SignalNorm
from ldpc_coder import LDPC_Coder_llr
import utility
import Modulator

def SIC_detecor(y, H, Nt, M, Es):
    print(f"0: y = {y}")
    x = np.zeros((Nt, 1), dtype = complex)
    Order = []
    idx_ary = list(np.arange(Nt))
    y = y[:,None]
    for nt in range(Nt):
        Wmmse = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt - nt)) @ H.T.conjugate()
        WH = Wmmse @ H
        SINR = []
        for i in range(Nt - nt):
            tmp = P * (np.sum(np.abs(WH[i])**2) - np.abs(WH[i, i])**2) + P_noise * np.sum(np.abs(Wmmse[i])**2)
            SINR.append(P * np.abs(WH[i, i])**2 / tmp)
        maxidx = np.argmax(SINR)
        Order.append(idx_ary[maxidx])
        idx_ary.remove(idx_ary[maxidx])
        xk_est = Wmmse[maxidx] @ y
        xk_bits = Modulator.demod(modem.constellation, xk_est, 'hard', Es = Es, )
        xk_hat = modem.modulate(xk_bits)
        x[Order[-1]] = xk_hat
        y = y -  np.outer(H[:, maxidx], xk_hat/np.sqrt(Es))
        H = np.delete(H, [maxidx], axis = 1)
    return x

utility.set_random_seed()

ldpcCoder =  LDPC_Coder_llr(args)
coderargs = {'codedim' : ldpcCoder.codedim,
             'codelen' : ldpcCoder.codelen,
             'codechk' : ldpcCoder.codechk,
             'coderate' : ldpcCoder.coderate,
             'row' : ldpcCoder.num_row,
             'col' : ldpcCoder.num_col}

source = SourceSink()
logf = "LDPC_MIMO_BerFer.txt"
source.InitLog(logfile = logf, promargs = args,  codeargs = coderargs )

M = args.M
Nr = args.Nr
Nt = args.Nt
P = args.P

Modulation_type = f"{M}QAM"
print(f"Use {Modulation_type}")
if Modulation_type=="BPSK":
    modem = cpy.PSKModem(2)
elif Modulation_type=="QPSK":
    modem = cpy.PSKModem(4)
elif Modulation_type=="8PSK":
    modem = cpy.PSKModem(8)
elif Modulation_type=="16QAM":
    modem = cpy.QAMModem(16)
elif Modulation_type=="64QAM":
    modem = cpy.QAMModem(64)
elif Modulation_type == "QAM256":
    modem = cpy.QAMModem(256)
Es = Modulator.NormFactor(mod_type='qam', M = M,)
# map_table, demap_table = modem.plot_constellation(Modulation_type)

# def main_mmseSIC():
SNR = np.arange(0, 21, 2)
for snr in SNR:
    P_noise = 1*(10**(-1*snr/10))
    source.ClrCnt()
    print( f"\nsnr = {snr}(dB):")
    while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
        channel = MIMO_Channel(Nr = Nr, Nt = Nt, )
        channel.circular_gaussian()
        # 编码
        uu = source.GenerateBitStr(ldpcCoder.codedim)
        cc = ldpcCoder.encoder(uu)

        # 调制
        yy = modem.modulate(cc)
        tx_symbols = yy.reshape(Nt, -1)
        ## 符号能量归一化
        tx_sig = tx_symbols / np.sqrt(Es)
        # 信道
        rx_sig = channel.forward(tx_sig, 1, snr )


        #%%============================================
        ##   LDPC  MIMO
        ###============================================
        H = copy.deepcopy(channel.H)

        tx_syms_hat = np.zeros((Nt, rx_sig.shape[-1]), dtype = complex)
        # for j in range(rx_sig.shape[-1]):
        #     tx_syms_hat[:,j] = SIC_detecor(rx_sig[:,j], H, Nt, M, Es)

        Order = []
        idx_ary = list(np.arange(Nt))
        # print(f"0: idx_ary = {idx_ary}")
        for nt in range(Nt):
            Wmmse = scipy.linalg.pinv(H.T.conjugate()@H + P_noise*np.eye(Nt - nt)) @ H.T.conjugate()
            WH = Wmmse @ H
            SINR = []
            for i in range(Nt - nt):
                tmp = P * (np.sum(np.abs(WH[i])**2) - np.abs(WH[i, i])**2) + P_noise * np.sum(np.abs(Wmmse[i])**2)
                SINR.append(P * np.abs(WH[i, i])**2 / tmp)
            maxidx = np.argmax(SINR)
            Order.append(idx_ary[maxidx])
            idx_ary.remove(idx_ary[maxidx])
            xk_est = Wmmse[maxidx] @ rx_sig
            xk_bits = Modulator.demod(modem.constellation, xk_est, 'hard', Es = Es, )
            xk_hat = modem.modulate(xk_bits)
            tx_syms_hat[Order[-1]] = xk_hat
            rx_sig = rx_sig -  np.outer(H[:, maxidx], xk_hat/np.sqrt(Es))
            H = np.delete(H, [maxidx], axis = 1)
        tx_syms_hat = tx_syms_hat.reshape(-1)
        rx_bits = modem.demodulate(tx_syms_hat, 'hard',)

        #%% count
        source.CntErr(cc, rx_bits)
        if source.tot_blk % 100 == 0:
            source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = snr)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = snr)
    # return













































