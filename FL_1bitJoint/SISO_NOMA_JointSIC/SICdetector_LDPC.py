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

## 以一帧为单位消去，以P*|h|^2为排序；且考虑其他未消去用户的功率
def SIC_LDPC_FastFading(H, yy, order, inteleaverM, Es, modem, ldpc, noisevar = 1, maxiter = 50):
    yy0 = copy.deepcopy(yy)
    K, frameLen = H.shape
    # order =  np.argsort(-np.abs(P))   #降序排列
    uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
    idx_set = list(np.arange(K))

    tot_iter = 0
    idx_set = list(np.arange(K))
    for k in order:
        idx_set.remove(k)
        hk = H[k]
        sigmaK = np.sum( np.abs(H[idx_set])**2 , axis = 0) + noisevar
        llrK = Modulator.demod_fastfading(copy.deepcopy(modem.constellation), yy0, 'soft', H = hk,  Es = Es,  noise_var = sigmaK)
        # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
        uu_hat[k], iterk = ldpc.decoder_msa(llrK)
        tot_iter += iterk
        sym_k = BPSK(ldpc.encoder(uu_hat[k]))
        # sym_k = sym_k[inteleaverM[k]]
        yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

    uu_sum = ldpc.bits2sum(uu_hat)
    return uu_hat, uu_sum, tot_iter



# ## 以一帧为单位消去，以P为排序；且考虑其他未消去用户的功率
# def SIC_LDPC_FastFading(H, yy, P, inteleaverM, noise_var, Es, modem, ldpc, maxiter = 50):
#     yy0 = copy.deepcopy(yy)
#     K, frameLen = H.shape
#     order =  np.argsort(-np.abs(P))   #降序排列
#     uu_hat = np.zeros((K, ldpc.codedim), dtype = np.int8)
#     idx_set = list(np.arange(K))

#     tot_iter = 0
#     idx_set = list(np.arange(K))
#     for k in order:
#         idx_set.remove(k)
#         hk = H[k]
#         sigmaK = np.sum(np.abs(H[idx_set])**2, axis = 0) + noise_var
#         llrK = Modulator.demod_fastfading(copy.deepcopy(modem.constellation), yy0, 'soft', H = hk,  Es = Es,  noise_var = sigmaK)
#         # llrK[inteleaverM[k]] = copy.deepcopy(llrK)
#         uu_hat[k], iterk = ldpc.decoder_msa(llrK)
#         tot_iter += iterk
#         sym_k = BPSK(ldpc.encoder(uu_hat[k]))
#         # sym_k = sym_k[inteleaverM[k]]
#         yy0 = yy0 -  H[k] * sym_k/np.sqrt(Es)

#     uu_sum = ldpc.bits2sum(uu_hat)
#     return uu_hat, uu_sum, tot_iter





























