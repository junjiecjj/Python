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


class SCMA_SISO(object):
    ## code parameters
    def __init__(self, codebookfile = 'DE_rayleigh.mat'):
        self.J = 0    # user num
        self.K = 0    # resource block num
        self.M = 0    # codeword num
        self.bits_weigh = None
        self.bps = 0  # bits per symbol
        self.Init(codebookfile)
        self.RowColValid()

    ## init & normalize CB
    def Init(self, codebookfile = 'DE_rayleigh.mat'):
        CB = sio.loadmat(codebookfile)['CB']
        (self.K, self.M, self.J) = CB.shape
        self.bps = int(np.log2(self.M))
        # ## normlized CodeBook
        # for k in range(CB.shape[0]):
        #     tmp = np.sum(np.abs(CB[k])**2) / self.M
        #     CB[k] /= np.sqrt(2*tmp)

        tmp = np.sum(np.abs(CB)**2) / (self.M * self.K)
        CB /= np.sqrt(tmp)
        CB *= np.sqrt(3)
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
        self.bits = comm.utilities.dec2bitarray(np.arange(self.M), self.bps).reshape(-1, self.bps)
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

    def MPAdetector_SISO_hard(self, yy, H, sigma2, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
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
            ## hard decision
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

    def MPAdetector_SISO_soft(self, yy, H, sigma2 = 1, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
        ## decision = np.zeros((self.J, 1))
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        llr_bits = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.float32)
        frame_len = yy.shape[-1]

        for f in range(frame_len):
            pro_bit = np.zeros((2, self.bps, self.J))  # 存储每个用户所有码字比特0/1的概率
            MR2U = np.zeros((self.K, self.J, self.M))
            MU2R = np.ones((self.J, self.K, self.M))/self.M
            ## channel reverse
            for j in range(self.J):
                CB_temp[:,:,j] = self.CB[:,:,j] * (H[:, j, f].reshape(-1,1))
            for it in range(Nit):
                ## update resource to user
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
                ## update user to resource
                MU2R = np.ones((self.J, self.K, self.M))/self.M
                for j in range(self.J):
                    for k in self.SetCols[j]:
                        row_in = copy.deepcopy(self.SetCols[j])
                        row_in.remove(k)
                        for m in range(self.M):
                            for r in row_in:
                                MU2R[j, k, m] *= MR2U[r, j, m]
                        MU2R[j, k, :] = MU2R[j, k, :] / np.sum(MU2R[j, k, :])
            ## hard decision
            result = np.ones((self.J, self.M))/self.M
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for r in row_in:
                        result[j, m] *= MR2U[r, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
            ## get LLR
            result1 = result / result.sum(axis = 1).reshape(-1,1)
            for j in range(self.J):
                for b in range(self.bps):
                    for m, bits in enumerate(self.bits):
                        pro_bit[bits[b], b, j] += result1[j, m]
            llr_tmp = np.log(pro_bit[0]/pro_bit[1]).reshape(self.bps, self.J)
            ## print(f"  llr_tmp = {llr_tmp}")
            llr_tmp[np.isinf(llr_tmp)] = np.sign(llr_tmp[np.isinf(llr_tmp)])/N0
            ## llr_tmp[np.where(llr_tmp == np.inf)] = np.sign(llr_tmp[np.where(llr_tmp == np.inf)])/N0
            llr_tmp[np.isnan(llr_tmp)] = 1/N0
            llr_bits[:, f*self.bps:(f+1)*self.bps]  = llr_tmp.T
        ## hard decoded bits
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)

        return decoded_symbols, uu_hat, llr_bits

    def LogMPAdetector_SISO_hard(self, yy, H, sigma2, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]
        F2V = np.zeros(self.combination.shape[0])
        for f in range(frame_len):
            # MR2U = np.zeros((self.K, self.J, self.M))
            MU2R = np.log(np.ones((self.J, self.K, self.M))/self.M)
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
                            for i, comb in enumerate(self.combination):
                                tmp = yy[k, f] - CB_temp[k, m, j]
                                for idx, u in enumerate(col_in):
                                    tmp -= CB_temp[k, comb[idx], u]
                                tmp = (-np.abs(tmp)**2/N0)
                                for idx, u in enumerate(col_in):
                                    tmp += MU2R[u, k, comb[idx]]
                                F2V[i] = tmp
                            MR2U[k, j, m] = np.log(np.sum(np.exp(F2V)))
                ## update VN to FN
                MU2R = np.log(np.ones((self.J, self.K, self.M))/self.M)
                for j in range(self.J):
                    for k in self.SetCols[j]:
                        row_in = copy.deepcopy(self.SetCols[j])
                        row_in.remove(k)
                        for m in range(self.M):
                            for r in row_in:
                                MU2R[j, k, m] += MR2U[r, j, m]
            ## hard decision
            result = np.log(np.ones((self.J, self.M))/self.M)
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for r in row_in:
                        result[j, m] += MR2U[r, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat


    def maxLogMPAdetector_SISO_hard(self, yy, H, sigma2, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros_like(self.CB)
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]
        F2V = np.zeros(self.combination.shape[0])
        for f in range(frame_len):
            MR2U = np.zeros((self.K, self.J, self.M))
            MU2R = np.log(np.ones((self.J, self.K, self.M))/self.M)
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
                            for i, comb in enumerate(self.combination):
                                tmp = yy[k, f] - CB_temp[k, m, j]
                                for idx, u in enumerate(col_in):
                                    tmp -= CB_temp[k, comb[idx], u]
                                tmp = (-np.abs(tmp)**2/N0)
                                for idx, u in enumerate(col_in):
                                    tmp += MU2R[u, k, comb[idx]]
                                F2V[i] = tmp
                            MR2U[k, j, m] = np.max(F2V)
                ## update VN to FN
                MU2R = np.log(np.ones((self.J, self.K, self.M))/self.M)
                for j in range(self.J):
                    for k in self.SetCols[j]:
                        row_in = copy.deepcopy(self.SetCols[j])
                        row_in.remove(k)
                        for m in range(self.M):
                            for r in row_in:
                                MU2R[j, k, m] += MR2U[r, j, m]
                        # MU2R[j, k, :] = MU2R[j, k, :] / np.sum(MU2R[j, k, :])
            ## hard decision
            result = np.log(np.ones((self.J, self.M))/self.M)
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for r in row_in:
                        result[j, m] += MR2U[r, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat




































































































































































