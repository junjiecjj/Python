

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen

pip install galois

"""

import galois
import numpy as np
import copy
import sys, os
import itertools
from functools import reduce
import commpy as comm

from utility import Gauss_Elimination

def bpsk(bins):
    bits = copy.deepcopy(bins)
    bits[np.where(bits == 1)] = -1
    bits[np.where(bits == 0)] = 1
    return bits

class QLDPC_Coding(object):
    def __init__(self, args):
        ## code parameters
        self.args = args
        self.p = args.K          # q = 2^p
        self.q = 2**self.p       # q-ary LDPC
        self.codedim = 0         # 码的维数，编码前长度
        self.codelen = 0         # 码的长度，编码后长度，码字长度
        self.codechk = 0         # 校验位的个数
        self.coderate = 0.0      # 码率

        ## parity-check matrix
        self.num_row = 0
        self.num_col = 0
        self.encH = None        # 用于编码校验矩阵
        self.decH = None        # 用于译码的校验矩阵

        ## 译码相关参数
        self.max_iter = args.max_iteration  # 最大迭代次数
        self.smallprob = args.smallprob
        self.SetRows = {}                   # 每行不为0的列号
        self.SetCols = {}                   # 每列不为0的行号
        self.MV2C = None                    # 变量节点 到 校验节点 的消息
        self.MC2V = None                    # 校验节点 到 变量节点 的消息
        self.qcomb = None
        self.readH()
        ## print("读取H完成...\n")
        self.systemH()
        ## print("高斯消元完成, encH，decH已获取...\n")
        self.NoneZeros()
        ## print("保存行列索引完成...")
        return

    def readH(self):
        current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(current_dir+self.args.file_name_of_the_H, 'r', encoding='utf-8') as f:
            tmp = f.readline()
            ## print(tmp)
            tmp = f.readline()
            # print(tmp)
            self.num_row, self.num_col, self.codechk = [int(i) for i in tmp.strip().split()]
            self.decH = np.zeros( (self.num_row, self.num_col), dtype = np.int8 )
            tmp = f.readline()
            # print(tmp)
            while 1:
                tmp = f.readline()
                if not tmp:
                    break
                row_dt = [int(i) for i in tmp.strip().split()]
                for i in range(row_dt[1]):
                    self.decH[row_dt[0], row_dt[i+2]] = 1
        self.codelen = self.num_col
        self.codedim = self.codelen - self.codechk
        self.coderate = self.codedim / self.codelen
        return

    # 相对较快
    def systemH(self):
        tmpH = copy.deepcopy(self.decH)
        self.encH = copy.deepcopy(self.decH)
        col_exchange = np.arange(self.num_col)
        ##=======================================================
        ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
        ##=======================================================
        self.encH, col_exchange = Gauss_Elimination(copy.deepcopy(self.encH), self.num_row, self.num_col)
        ##====================== Gauss 消元 end =================================

        ##======================================================
        ##  根据列交换结果交换原 decH 矩阵的列
        ##======================================================
        for j in range(self.num_col):
            self.decH[:, j] = tmpH[:, col_exchange[j]]
        return

    def encoder(self, uu):
        cc = np.zeros(self.codelen, dtype = np.int8)
        cc[self.codechk:] = uu
        ## 2：相对快
        for i in range(self.codechk):
            cc[i] = np.logical_xor.reduce(np.logical_and(uu[:], self.encH[i, self.codechk:]))
        return cc

    def NoneZeros(self):
        ## 字典  {行号: {不为 0 的列号}}
        self.SetRows  = {i: list(np.nonzero(self.decH[i,:])[0].astype(int)) for i in range(self.decH.shape[0])}
        ## 字典  {列号: {不为 0 的行号}}
        self.SetCols = {j: list(np.nonzero(self.decH[:,j])[0].astype(int)) for j in range(self.decH.shape[1])}
        self.qbits = comm.utilities.dec2bitarray(np.arange(self.q), self.p).reshape(-1, self.p)
        row_weigh = self.decH.sum(axis = 1)[0]
        GF = galois.GF(2**self.p, repr = "int")
        self.qcomb = {}
        for i in range(self.q):
            self.qcomb[i] = []
        for comb in itertools.product(GF.elements, repeat = row_weigh - 1):
            self.qcomb[int(np.sum(GF([int(g) for g in comb])))].append([int(g) for g in comb])
        return

    def PassChannel(self, symbs, H, noise_var):
        yy = symbs * H
        noise = np.sqrt(noise_var/2) * (np.random.normal(0, 1, size = symbs.shape[-1] ) + 1j*np.random.normal(0, 1, size = symbs.shape[-1] ))
        yy = yy.sum(axis  = 0) + noise
        return yy

    def post_probability(self, yy, H, noise_var):
        frame_len = yy.shape[-1]
        pp = np.zeros((self.q, frame_len))
        for f in range(frame_len):
            for i, vec in enumerate(self.qbits):
                pp[i, f] = np.exp(np.abs(yy[f] - H[:,f] @ bpsk(vec))**2/(2 * noise_var)) # / (np.sqrt(2 * np.pi * noise_var))
        pp = pp/ pp.sum(axis = 0)
        pp = np.clip(pp, self.smallprob, 1-self.smallprob)
        return pp

# ##  yy --> 概率域
# def yyToProb(yy, noise_var = 1.0):
#     prob = np.zeros_like(yy)
#     prob = 1.0 / (1.0 + np.exp(-2.0*yy/noise_var))
#     return prob

    def decoder_qary_spa_fun(self, pp, GF, I, maxiter = 50):
        MV2C = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        MC2V = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        # uu_hat = np.zeros(self.codedim, dtype = np.int8)
        ##===========================================
        ## (初始化) V 到 C 的初始化信息
        ##===========================================
        for col in self.SetCols.keys():
            for row in self.SetCols[col]:
                for q in range(self.q):
                    MV2C[row, col, q] = pp[q, col]
        ## 开始迭代，概率域的消息传播,
        for iter_num in range(maxiter):
            print(f"  {iter_num}")
            ##==========================================================
            ## (一) 更新 C 到 V 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[row]:
                    col_in = copy.deepcopy(self.SetRows[row])
                    col_in.remove(col)
                    for q in range(self.q):
                        Sum = 0
                        for comb in self.qcomb[q]:
                            # print(comb)
                            tmp = 1
                            for v, q in zip(col_in, comb):
                                tmp *= MV2C[row, v, q]
                            Sum += tmp
                        MC2V[row, col, q] = Sum
            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            PQ = np.zeros_like(pp, dtype = np.float64)
            for col in self.SetCols.keys():
                for q in range(self.q):
                    tmp = pp[q, col]
                    for row in self.SetCols[col]:
                        tmp *= MC2V[row, col, q]
                    PQ[q, col] = tmp
            Pdecision = PQ.argmax(axis = 0)
            cc_hat = comm.utilities.dec2bitarray(Pdecision, self.p).reshape(-1, self.p).T
            cc_hat_fun = np.array(I@GF(cc_hat), dtype = np.int8)
            uu_hat_fun = cc_hat_fun[ self.codechk:]

            success = 1
            # parity checking，校验
            # for k in range(cc_hat_fun.shape[0]):
            for i in range(self.num_row):
                # parity_check = np.logical_xor.reduce(np.logical_and(cc_hat, self.decH[i,:]))
                parity_check = np.bitwise_xor.reduce(cc_hat_fun & self.decH[i,:])
                if parity_check != 0:
                    success = 0
                    break
                # if success == 0:
                    # break
            if success == 1:
                return uu_hat_fun, iter_num + 1
            #========================================================================
            ## (三) 更新 v 到 c 的消息，半边输入信息也要考虑进去
            #========================================================================
            for col in self.SetCols.keys():
                for row in self.SetCols[col]:
                    row_in = copy.deepcopy(self.SetCols[col])
                    row_in.remove(row)
                    for q in range(self.q):
                        tmp = pp[q, col]
                        for c in row_in:
                            tmp *= MC2V[c, col, q]
                        MV2C[row, col, q] = tmp
        return uu_hat_fun, iter_num + 1

    def decoder_qary_spa(self, pp, maxiter = 50):
        MV2C = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        MC2V = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        # uu_hat = np.zeros(self.codedim, dtype = np.int8)
        ##===========================================
        ## (初始化) V 到 C 的初始化信息
        ##===========================================
        for col in self.SetCols:
            for row in self.SetCols[col]:
                for q in range(self.q):
                    MV2C[row, col, q] = pp[q, col]
        ## 开始迭代，概率域的消息传播,
        for iter_num in range(maxiter):
            print(f"  {iter_num}")
            ##==========================================================
            ## (一) 更新 C 到 V 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[row]:
                    col_in = copy.deepcopy(self.SetRows[row])
                    col_in.remove(col)
                    for q in range(self.q):
                        Sum = 0
                        for comb in self.qcomb[q]:
                            # print(comb)
                            tmp = 1
                            for v, q in zip(col_in, comb):
                                tmp *= MV2C[row, v, q]
                            Sum += tmp
                        MC2V[row, col, q] = Sum
                # print(f"  {iter_num}:MC2V = {MC2V}")
            # MC2V = MC2V/MC2V.sum(axis = -1, keepdims=1)
            # MC2V = np.clip(MC2V, self.smallprob, 1-self.smallprob)
            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            PQ = np.zeros_like(pp, dtype = np.float64)
            for col in self.SetCols:
                row_in = copy.deepcopy(self.SetCols[col])
                for q in range(self.q):
                    tmp = pp[q, col]
                    for row in row_in:
                        tmp *= MC2V[row, col, q]
                    PQ[q, col] = tmp
            Pdecision = PQ.argmax(axis = 0)
            print(f"  {iter_num}:{Pdecision}")
            cc_hat = comm.utilities.dec2bitarray(Pdecision, self.p).reshape(-1, self.p).T
            uu_hat  = cc_hat[:, self.codechk:]

            success = 1
            # parity checking，校验
            for k in range(cc_hat.shape[0]):
                for i in range(self.num_row):
                    parity_check = np.bitwise_xor.reduce(cc_hat[k] & self.decH[i,:])
                    if parity_check != 0:
                        success = 0
                        break
                if success == 0:
                    break
            if success == 1:
                return uu_hat, iter_num + 1
            #========================================================================
            ## (三) 更新 v 到 c 的消息，半边输入信息也要考虑进去
            #========================================================================
            for col in self.SetCols:
                for row in self.SetCols[col]:
                    row_in = copy.deepcopy(self.SetCols[col])
                    row_in.remove(row)
                    for q in range(self.q):
                        tmp = pp[q, col]
                        for c in row_in:
                            tmp *= MC2V[c, col, q]
                        MV2C[row, col, q] = tmp
            # MV2C = MV2C/MV2C.sum(axis = -1, keepdims=1)
            # MV2C = np.clip(MV2C, self.smallprob, 1-self.smallprob)
        return uu_hat, iter_num + 1














































































































































































































































