#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:16 2024
@author: Junjie Chen
@email: 2716705056@qq.com
"""

import numpy as np
import scipy.io as sio
# import commpy as comm


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
        self.bits_weigh = 2**np.arange(int(np.log2(self.M)) - 1, -1, -1).reshape(-1,1)
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

    def MPAdetector_soft(self, yy, H, sigma2, Nit = 6):
        return

    def MPAdetector_hard(self, yy, H, sigma2, Nit = 6):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
        decision = np.zeros((self.J, 1))
        decoded_symbols = np.zeros((self.J, yy.shape[-1]))

        for f in range(yy.shape[-1]):
            for j in range(self.J):
                CB_temp[:,:,j] = self.CB[:,:,j] * (H[:, j, f].reshape(-1,1))

            for _ in range(Nit):
                ## update FN to VN
                for k in range(self.K):
                    for j in self.SetRows[k]:

                        for m in range(self.M):
                            pass

        return




scma = SCMA()
CB = scma.CB











































































































































































