#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:13:05 2024

@author: jack
"""

# import galois
import numpy as np
import copy
import sys, os
import itertools
# from functools import reduce
import commpy as comm
import Modulator

def BPSK(bins):
    c = 1 - 2*bins
    return c

def inteleaver(K, n):
    M = np.zeros((K, n), dtype = np.int32)
    for k in range(K):
        vec = np.arange(n)
        M[k] = np.random.permutation(vec)
    return M


# ##  每一个时刻分别排序，硬消除，其他用户当噪声
# def SIC_LDPC_FastFading1(H, yy, P, noise_var, Es, modem, ldpc, maxiter = 50):
#     YY = copy.deepcopy(yy)
#     K, frameLen = H.shape
#     uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
#     llr = np.zeros((K, ldpc.codelen))

#     for f in range(frameLen):
#         H0 = H[:, f]
#         yy0 = np.array([YY[f]])
#         order =  np.argsort(-np.abs(H0))   # 降序排列
#         idx_set = list(np.arange(K))
#         for k in order:
#             # print(k)
#             idx_set.remove(k)
#             hk = H0[k]
#             sigmaK = np.sum(np.abs(H0[idx_set])**2 * np.array(P)[idx_set]) + noise_var
#             llr[k, f] = Modulator.demod_blockfading(copy.deepcopy(modem.constellation), yy0, 'soft', Es = Es, h = hk,  noise_var = sigmaK)[0]
#             xk_bits = Modulator.demod_blockfading(copy.deepcopy(modem.constellation), yy0, 'hard', Es = Es, h = hk,)
#             sym_k = modem.modulate(xk_bits)
#             yy0 = yy0 -  H0[k] * sym_k/np.sqrt(Es)
#     llr[np.isinf(llr)] = 2 * np.sign(llr[np.isinf(llr)]) / noise_var
#     tot_iter = 0
#     for k in range(K):
#         uu_hat[k], iterk =  ldpc.decoder_msa(llr[k])
#         tot_iter += iterk
#     uu_sum = ldpc.bits2sum(uu_hat)

#     return uu_hat, uu_sum, tot_iter

# ##  每一个时刻分别排序，硬消除，其他用户忽略，只有高斯噪声
# def SIC_LDPC_FastFading2(H, yy, P, noise_var, Es, modem, ldpc, maxiter = 50):
#     YY = copy.deepcopy(yy)
#     K, frameLen = H.shape
#     uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
#     llr = np.zeros((K, ldpc.codelen))

#     for f in range(frameLen):
#         H0 = H[:, f]
#         yy0 = np.array([YY[f]])
#         order =  np.argsort(-np.abs(H0))   # 降序排列
#         idx_set = list(np.arange(K))
#         for k in order:
#             # print(k)
#             idx_set.remove(k)
#             y_tmp = yy0 / H0[k]

#             llrK = 2 * np.real(y_tmp) * np.abs(H0[k])**2 / noise_var
#             llr[k, f] = llrK
#             xk_bits = 0 if llrK >= 0 else 1
#             sym_k = BPSK(xk_bits)
#             yy0 = yy0 -  H0[k] * sym_k/np.sqrt(Es)

#     tot_iter = 0
#     for k in range(K):
#         uu_hat[k], iterk =  ldpc.decoder_msa(llr[k])
#         tot_iter += iterk
#     uu_sum = ldpc.bits2sum(uu_hat)

#     return uu_hat, uu_sum, tot_iter


## 以一帧为单位消去，按每个用户的h的模的平方和为排序；且不考虑其他未消去用户的功率
def SIC_LDPC_FastFading_BPSK(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    H0 = np.linalg.norm(H, ord = 2, axis = 1)
    order =  np.argsort(-np.abs(H0))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    tot_iter = 0
    for k in order:
        y_tmp = yy0 / H[k]
        idx_set.remove(k)
        llrK = 2 * np.real(y_tmp) * np.abs(H[k])**2 / noise_var
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter


## 以一帧为单位消去，以P为排序；且不考虑其他未消去用户的功率
def SIC_LDPC_FastFading_P(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    order =  np.argsort(-np.abs(P))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    tot_iter = 0
    for k in order:
        y_tmp = yy0 / H[k]
        idx_set.remove(k)
        llrK = 2 * np.real(y_tmp) * np.abs(H[k])**2 / noise_var
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter


## 以一帧为单位消去，按P*h大小确定消去顺序；且不考虑其他未消去用户的功率
def SIC_LDPC_BlockFading_BPSK(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    H0 = H[:,0]
    order =  np.argsort(-np.abs(H0))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    tot_iter = 0
    for k in order:
        y_tmp = yy0 / H[k]
        idx_set.remove(k)
        llrK = 2 * np.real(y_tmp) * np.abs(H[k])**2 / noise_var
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter

## 以一帧为单位消去，按P*h大小确定消去顺序；考虑其他未消去用户的功率
def SIC_LDPC_BlockFading(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    H0 = H[:,0]
    order =  np.argsort(-np.abs(H0))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)

    # llr = np.zeros((K, ldpc.codelen))
    tot_iter = 0
    idx_set = list(np.arange(K))
    for k in order:
        idx_set.remove(k)
        hk = H0[k]
        sigmaK = np.sum(np.abs(H0[idx_set])**2) + noise_var
        llrK = Modulator.demod_blockfading(copy.deepcopy(modem.constellation), yy0, 'soft', Es = Es, h = hk,  noise_var = sigmaK)
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter

## 以一帧为单位消去，以P为排序；且考虑其他未消去用户的功率
def SIC_LDPC_FastFading(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    order =  np.argsort(-np.abs(P))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    tot_iter = 0
    idx_set = list(np.arange(K))
    for k in order:
        idx_set.remove(k)
        hk = H[k]
        sigmaK = np.sum(np.abs(H[idx_set])**2, axis = 0) + noise_var
        llrK = Modulator.demod_fastfading(copy.deepcopy(modem.constellation), yy0, 'soft', H = hk,  Es = Es,  noise_var = sigmaK)
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter





























