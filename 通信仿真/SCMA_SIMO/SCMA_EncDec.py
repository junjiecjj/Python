


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

class SCMA_SIMO(object):
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

    def encoder(self, symbols, H, Nr):
        yy = np.zeros((self.K, Nr, symbols.shape[-1]), dtype = complex)
        for nr in range(Nr):
            for c in range(symbols.shape[-1]):
                yy[:, nr, c] = np.array([self.CB[:, symbols[j, c], j] * H[:,nr, j, c]  for j in range(self.J)]).sum(axis = 0).flatten()
        return yy

    def MPAdetector_SIMO_hard(self, yy, H, sigma2, Nr, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros((self.K, self.M, self.J, Nr), dtype = complex)
        # decision = np.zeros((self.J, 1))
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]

        for f in range(frame_len):
            MR2U = np.zeros((self.K, Nr, self.J, self.M))
            MU2R = np.ones((self.J, self.K, Nr, self.M))/self.M
            ## channel reverse
            for nr in range(Nr):
                for j in range(self.J):
                    CB_temp[:, :, j, nr] = self.CB[:, :, j] * (H[:, nr, j, f].reshape(-1,1))
            for it in range(Nit):
                ## update Resource to User
                MR2U = np.zeros((self.K, Nr, self.J, self.M))
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            col_in = copy.deepcopy(self.SetRows[k])
                            col_in.remove(j)
                            for m in range(self.M):
                                for comb in self.combination:
                                    tmp = yy[k, nr, f] - CB_temp[k, m, j, nr]
                                    # print(f"tmp = {tmp}")
                                    for idx, u in enumerate(col_in):
                                        tmp -= CB_temp[k, comb[idx], u, nr]
                                    tmp = np.exp(-np.abs(tmp)**2/N0)
                                    for idx, u in enumerate(col_in):
                                        tmp *= MU2R[u, k, nr, comb[idx]]
                                    MR2U[k, nr, j, m] += tmp
                ## update User to Resource
                MU2R = np.ones((self.J, self.K, Nr, self.M))/self.M
                for nr in range(Nr):
                    nr_remain = [i for i in range(Nr)]
                    nr_remain.remove(nr)
                    for j in range(self.J):
                        for k in self.SetCols[j]:
                            row_in = copy.deepcopy(self.SetCols[j])
                            row_in.remove(k)
                            for m in range(self.M):
                                for r in row_in:
                                    MU2R[j, k, nr, m] *= MR2U[r, nr, j, m]
                            for ne in nr_remain:
                                for m in range(self.M):
                                    for r in copy.deepcopy(self.SetCols[j]):
                                        MU2R[j, k, nr, m] *= MR2U[r, ne, j, m]
                            MU2R[j, k, nr, :] = MU2R[j, k, nr, :] / np.sum(MU2R[j, k, nr, :])
            ## decision
            result = np.ones((self.J, self.M))/self.M
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for nr in range(Nr):
                        for r in row_in:
                            result[j, m] *= MR2U[r, nr, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat

    def MPAdetector_SIMO_soft(self, yy, H, sigma2, Nr, Nit = 10):
        N0 = sigma2
        CB_temp = np.zeros((self.K, self.M, self.J, Nr), dtype = complex)
        # decision = np.zeros((self.J, 1))
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]
        llr_bits = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.float32)

        for f in range(frame_len):
            pro_bit = np.zeros((2, self.bps, self.J))  # 存储每个用户所有码字比特0/1的概率
            MR2U = np.zeros((self.K, Nr, self.J, self.M))
            MU2R = np.ones((self.J, self.K, Nr, self.M))/self.M
            ## channel reverse
            for nr in range(Nr):
                for j in range(self.J):
                    CB_temp[:, :, j, nr] = self.CB[:, :, j] * (H[:, nr, j, f].reshape(-1,1))
            for it in range(Nit):
                ## update Resource to User
                MR2U = np.zeros((self.K, Nr, self.J, self.M))
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            col_in = copy.deepcopy(self.SetRows[k])
                            col_in.remove(j)
                            for m in range(self.M):
                                for comb in self.combination:
                                    tmp = yy[k, nr, f] - CB_temp[k, m, j, nr]
                                    # print(f"tmp = {tmp}")
                                    for idx, u in enumerate(col_in):
                                        tmp -= CB_temp[k, comb[idx], u, nr]
                                    tmp = np.exp(-np.abs(tmp)**2/N0)
                                    for idx, u in enumerate(col_in):
                                        tmp *= MU2R[u, k, nr, comb[idx]]
                                    MR2U[k, nr, j, m] += tmp
                ## update User to Resource
                MU2R = np.ones((self.J, self.K, Nr, self.M))/self.M
                for nr in range(Nr):
                    nr_remain = [i for i in range(Nr)]
                    nr_remain.remove(nr)
                    for j in range(self.J):
                        for k in self.SetCols[j]:
                            row_in = copy.deepcopy(self.SetCols[j])
                            row_in.remove(k)
                            for m in range(self.M):
                                for r in row_in:
                                    MU2R[j, k, nr, m] *= MR2U[r, nr, j, m]
                            for ne in nr_remain:
                                for m in range(self.M):
                                    for r in copy.deepcopy(self.SetCols[j]):
                                        MU2R[j, k, nr, m] *= MR2U[r, ne, j, m]
                            MU2R[j, k, nr, :] = MU2R[j, k, nr, :] / np.sum(MU2R[j, k, nr, :])
            ## hard decision
            result = np.ones((self.J, self.M))/self.M
            for j in range(self.J):
                for m in range(self.M):
                    for nr in range(Nr):
                        for r in copy.deepcopy(self.SetCols[j]):
                            result[j, m] *= MR2U[r, nr, j, m]
            decoded_symbols[:, f] = np.argmax(result, axis = 1)
            ## get LLR
            result1 = result / result.sum(axis = 1).reshape(-1,1)
            for j in range(self.J):
                for b in range(self.bps):
                    for m, bits in enumerate(self.bits):
                        pro_bit[bits[b], b, j] += result1[j, m]
            llr_tmp = np.log(pro_bit[0]/pro_bit[1]).reshape(self.bps, self.J)
            llr_tmp[np.where(llr_tmp == np.inf)] = np.sign(llr_tmp[np.where(llr_tmp == np.inf)])/N0
            llr_tmp[np.isnan(llr_tmp)]  = 1/N0
            llr_bits[:, f*self.bps:(f+1)*self.bps]  = llr_tmp.T

        ## hard decoded bits
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat, llr_bits

    def EPAdetector_SIMO_hard(self, yy, H, sigma2, Nr, Nit = 10):
        N0 = sigma2
        CB = copy.deepcopy(self.CB)
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        frame_len = yy.shape[-1]

        for f in range(frame_len):
            meanK2J = np.zeros((self.K, self.J, Nr), dtype = complex)
            varK2J = np.ones((self.K, self.J, Nr)) * 1000
            for _ in range(Nit):
                # print(f"frame = {f}, it = {_}")
                ## (1.1) 计算每个用户每个码字的后验概率（根据从资源节点返回的均值和方差计算）
                post_prob = np.ones((self.J, self.M)) / self.M
                for j in range(self.J):
                    for m in range(self.M):
                        for nr in range(Nr):
                            for k in copy.deepcopy(self.SetCols[j]):
                                post_prob[j, m] *= np.exp(-np.abs(CB[k, m, j] - meanK2J[k,j,nr])**2/varK2J[k,j,nr])
                post_prob /= post_prob.sum(axis = 1).reshape(-1,1)
                if True in np.isnan(post_prob):
                    post_prob[:] = 1/self.M
                    # print(f"True, it = {_}, post_prob ={post_prob}")
                ## (1.2) 计算后验均值
                mean_post = np.zeros((self.K, self.J), dtype = complex)
                for k in range(self.K):
                    for j in self.SetRows[k]:
                        for m in range(self.M):
                            mean_post[k, j] += post_prob[j, m] * CB[k, m, j]
                ## (1.3) 计算后验方差
                var_post = np.zeros((self.K, self.J))
                for k in range(self.K):
                    for j in self.SetRows[k]:
                        for m in range(self.M):
                            var_post[k,j] += post_prob[j, m] * np.abs(CB[k, m, j] - mean_post[k, j])**2
                # print(f"it = {_}, mean_post = {mean_post}, var_post = {var_post}")
                ## (1.4) 计算用户节点到资源节点传递的均值和方差
                meanJ2K = np.zeros((self.K, self.J, Nr), dtype = complex)
                varJ2K = np.zeros((self.K, self.J, Nr), )
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            varJ2K[k, j, nr] = varK2J[k, j, nr] * var_post[k,j] / (varK2J[k, j, nr] - var_post[k,j])
                            meanJ2K[k,j, nr] = varJ2K[k, j, nr] * (mean_post[k, j] / var_post[k,j] - meanK2J[k,j,nr]/varK2J[k,j,nr])
                ## (2) 资源节点更新:计算资源节点向用户节点传递的均值和方差
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            col_in = copy.deepcopy(self.SetRows[k])
                            col_in.remove(j)
                            tmp1 = 0
                            tmp2 = 0
                            for idx, u in enumerate(col_in):
                                tmp1 += H[k, nr, u, f] * meanJ2K[k, u, nr]
                                tmp2 += np.abs(H[k, nr, u, f])**2 * varJ2K[k, u, nr]
                            meanK2J[k, j, nr] = (yy[k, nr, f] - tmp1) / H[k, nr, j, f]
                            varK2J[k, j, nr] = (N0 + tmp2)/np.abs(H[k, nr, j, f])**2
            ## (3) 判决:
            post_prob = np.ones((self.J, self.M)) / self.M
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for nr in range(Nr):
                        for k in row_in:
                            post_prob[j, m] *= np.exp(-np.abs(CB[k, m, j] - meanK2J[k,j,nr])**2/varK2J[k,j,nr])
            post_prob /= post_prob.sum(axis = 1).reshape(-1,1)
            if True in np.isnan(post_prob):
                post_prob[:] = 1/self.M
            decoded_symbols[:, f] = np.argmax(post_prob, axis = 1)
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat

    def EPAdetector_SIMO_soft(self, yy, H, sigma2, Nr, Nit = 10):
        N0 = sigma2
        CB = copy.deepcopy(self.CB)
        decoded_symbols = np.zeros((self.J, yy.shape[-1]), dtype = np.int32)
        uu_hat = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.int8)
        llr_bits = np.zeros((self.J, yy.shape[-1]*self.bps), dtype = np.float32)
        frame_len = yy.shape[-1]

        for f in range(frame_len):
            pro_bit = np.zeros((2, self.bps, self.J))  # 存储每个用户所有码字比特0/1的概率
            meanK2J = np.zeros((self.K, self.J, Nr), dtype = complex)
            varK2J = np.ones((self.K, self.J, Nr)) * 1000
            for _ in range(Nit):
                # print(f"frame = {f}, it = {_}")
                ## (1.1) 计算每个用户每个码字的后验概率（根据从资源节点返回的均值和方差计算）
                post_prob = np.ones((self.J, self.M)) / self.M
                for j in range(self.J):
                    for m in range(self.M):
                        for nr in range(Nr):
                            for k in copy.deepcopy(self.SetCols[j]):
                                post_prob[j, m] *= np.exp(-np.abs(CB[k, m, j] - meanK2J[k,j,nr])**2/varK2J[k,j,nr])
                post_prob /= post_prob.sum(axis = 1).reshape(-1,1)
                if True in np.isnan(post_prob):
                    post_prob[:] = 1/self.M
                    # print(f"True, it = {_}, post_prob ={post_prob}")
                ## (1.2) 计算后验均值
                mean_post = np.zeros((self.K, self.J), dtype = complex)
                for k in range(self.K):
                    for j in self.SetRows[k]:
                        for m in range(self.M):
                            mean_post[k, j] += post_prob[j, m] * CB[k, m, j]
                ## (1.3) 计算后验方差
                var_post = np.zeros((self.K, self.J))
                for k in range(self.K):
                    for j in self.SetRows[k]:
                        for m in range(self.M):
                            var_post[k,j] += post_prob[j, m] * np.abs(CB[k, m, j] - mean_post[k, j])**2
                # print(f"it = {_}, mean_post = {mean_post}, var_post = {var_post}")
                ## (1.4) 计算用户节点到资源节点传递的均值和方差
                meanJ2K = np.zeros((self.K, self.J, Nr), dtype = complex)
                varJ2K = np.zeros((self.K, self.J, Nr), )
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            varJ2K[k, j, nr] = varK2J[k, j, nr] * var_post[k,j] / (varK2J[k, j, nr] - var_post[k,j])
                            meanJ2K[k,j, nr] = varJ2K[k, j, nr] * (mean_post[k, j] / var_post[k,j] - meanK2J[k,j,nr]/varK2J[k,j,nr])
                ## (2) 资源节点更新:计算资源节点向用户节点传递的均值和方差
                for nr in range(Nr):
                    for k in range(self.K):
                        for j in self.SetRows[k]:
                            col_in = copy.deepcopy(self.SetRows[k])
                            col_in.remove(j)
                            tmp1 = 0
                            tmp2 = 0
                            for idx, u in enumerate(col_in):
                                tmp1 += H[k, nr, u, f] * meanJ2K[k, u, nr]
                                tmp2 += np.abs(H[k, nr, u, f])**2 * varJ2K[k, u, nr]
                            meanK2J[k, j, nr] = (yy[k, nr, f] - tmp1) / H[k, nr, j, f]
                            varK2J[k, j, nr] = (N0 + tmp2)/np.abs(H[k, nr, j, f])**2
            ## (3) 判决:
            ## hard decode
            post_prob = np.ones((self.J, self.M)) / self.M
            for j in range(self.J):
                row_in = copy.deepcopy(self.SetCols[j])
                for m in range(self.M):
                    for nr in range(Nr):
                        for k in row_in:
                            post_prob[j, m] *= np.exp(-np.abs(CB[k, m, j] - meanK2J[k,j,nr])**2/varK2J[k,j,nr])
            post_prob /= post_prob.sum(axis = 1).reshape(-1,1)
            if True in np.isnan(post_prob):
                post_prob[:] = 1/self.M
            decoded_symbols[:, f] = np.argmax(post_prob, axis = 1)

            ## get LLR
            result1 = post_prob / post_prob.sum(axis = 1).reshape(-1,1)
            for j in range(self.J):
                for b in range(self.bps):
                    for m, bits in enumerate(self.bits):
                        pro_bit[bits[b], b, j] += result1[j, m]
            llr_tmp = np.log(pro_bit[0]/pro_bit[1]).reshape(self.bps, self.J)
            llr_tmp[np.where(llr_tmp == np.inf)] = np.sign(llr_tmp[np.where(llr_tmp == np.inf)])/N0
            llr_tmp[np.isnan(llr_tmp)]  = 1/N0
            llr_bits[:, f*self.bps:(f+1)*self.bps]  = llr_tmp.T

        ## hard decoded bits
        for j in range(self.J):
            uu_hat[j, :] = comm.utilities.dec2bitarray(decoded_symbols[j, :], self.bps)
        return decoded_symbols, uu_hat, llr_bits





































































































































































