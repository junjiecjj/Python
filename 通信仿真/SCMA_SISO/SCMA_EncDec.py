#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:16 2024
@author: Junjie Chen
@email: 2716705056@qq.com
"""

import numpy as np
import copy
import itertools
import scipy.io as sio
import commpy as comm


class SCMA(object):
    ## code parameters
    def __init__(self, codebookfile = 'DE_rayleigh.mat'):
        # self.args = args
        self.J = 0    # user num
        self.K = 0    # resource block num
        self.M = 0    # codeword num
        self.bits_weigh = None
        self.bps = 0  # bits per symbol
        self.Init(codebookfile)
        self.RowColValid()

    ## init & normalize CB;
    def Init(self, codebookfile = 'DE_rayleigh.mat'):
        CB = sio.loadmat(codebookfile)['CB']
        (self.K, self.M, self.J) = CB.shape
        self.bps = int(np.log2(self.M))
        ## normlized CodeBook
        for k in range(CB.shape[0]):
            tmp = np.sum(np.abs(CB[k])**2) / self.M
            CB[k] /= np.sqrt(tmp)
        self.CB = CB
        ## factor Graph
        F = np.zeros((self.K, self.J), dtype = np.int8)
        for j in range(self.J):
            F[np.where(CB[:,1,j] != 0)[0], j] = 1
        self.FG = F
        self.df = np.sum(self.FG, axis = 1)[0]
        self.dv = np.sum(self.FG, axis = 0)[0]
        self.bits_weigh = 2**np.arange(int(np.log2(self.M)) - 1, -1, -1).reshape(-1,1)
        self.combination = np.array(list(itertools.product(np.arange(self.M), repeat = self.df - 1)))
        return

    def RowColValid(self):
        ## 字典  {行号: {不为 0 的列号}}
        self.SetRows  = {i: list(np.nonzero(self.FG[i,:])[0].astype(int)) for i in range(self.FG.shape[0])}
        ## 字典  {列号: {不为 0 的行号}}
        self.SetCols = {j: list(np.nonzero(self.FG[:,j])[0].astype(int)) for j in range(self.FG.shape[1])}
        return

    def mapping(self, uu, ):
        frame_len =  int(uu.shape[-1]/self.bps)
        symbols = np.zeros((uu.shape[0], frame_len), dtype = np.int32)
        for c in range(frame_len):
            symbols[:, c] = (uu[:, c*self.bps:(c+1)*self.bps] @ self.bits_weigh).flatten()
        return symbols

    def encoder(self, symbols, H, ):
        yy = np.zeros((self.K, symbols.shape[-1]), dtype = complex)
        for c in range(symbols.shape[-1]):
            yy[:, c] = np.array([self.CB[:, symbols[j, c], j] * H[:, j, c]  for j in range(self.J)]).sum(axis = 0).flatten()
        return yy

    def MPAdetector_soft(self, yy, H, sigma2, Nit = 10):
        return

    def MPAdetector_hard(self, yy, H, sigma2, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
        decision = np.zeros((self.J, 1))
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]

        for f in range(frame_len):
            MR2U = np.zeros((self.K, self.J, self.M))
            MU2R = np.ones((self.J, self.K, self.M))/self.M
            ## channel reverse
            for j in range(self.J):
                CB_temp[:,:,j] = self.CB[:,:,j] * (H[:, j, f].reshape(-1,1))

            for it in range(Nit):
                ## update FN to VN
                MR2U = np.zeros((self.K, self.J, self.M))
                for k in range(self.K):
                    for j in self.SetRows[k]:
                        col_in = copy.deepcopy(self.SetRows[k])
                        col_in.remove(j)
                        for m in range(self.M):
                            for comb in self.combination:
                                tmp = yy[k, f] - CB_temp[k, m, j]
                                for idx, u in enumerate(col_in):
                                    tmp -= CB_temp[k, comb[idx], u]
                                tmp = np.exp(-np.abs(tmp)**2/N0)
                                for idx, u in enumerate(col_in):
                                    tmp *= MU2R[u, k, comb[idx]]
                                MR2U[k, j, m] += tmp
                ## update VN to FN
                MU2R = np.ones((self.J, self.K, self.M))/self.M
                for j in range(self.J):
                    for k in self.SetCols[j]:
                        row_in = copy.deepcopy(self.SetCols[j])
                        row_in.remove(k)
                        for m in range(self.M):
                            for r in row_in:
                                MU2R[j, k, m] *= MR2U[r, j, m]
                        MU2R[j, k, :] = MU2R[j, k, :] / np.sum(MU2R[j, k, :])
            ## decision
            result = np.ones((self.J, self.M))/self.M
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for r in row_in:
                        result[j, m] *= MR2U[r, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat




scma = SCMA()
CB = scma.CB











































































































































































