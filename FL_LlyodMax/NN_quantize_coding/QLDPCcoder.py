

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/11/22

@author: JunJie Chen

pip install galois

"""

# import galois
import numpy as np
import copy
import sys, os
# import itertools
# from functools import reduce
import commpy as comm


def Gauss_Elimination(encH, num_row, num_col):
    codechk = 0
    col_exchange = np.arange(num_col)
    ##======================================================================
    ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
    ##======================================================================
    for i in range(num_row):
        # 获取当前对角线位置 [i, i] 右下角元素中的不为0的元素的索引;
        flag = 0
        for jj in range(i, num_col):
            for ii in range(i, num_row):
                if encH[ii, jj] != 0:
                    flag = 1
                    break
            if flag == 1:
                codechk += 1
                break
        if flag == 0:
            print("I am break")
            break
        else:     # 如果右下角有非零元素,则找出第一个非零元素的行和列;
            ## 交换 i 行和 ii 行;
            if ii != i:
                # print(f"{i} 行交换")
                encH[[i, ii], :] = encH[[ii, i], :]
            if jj != i:
                # print("1: 列交换")
                ## 记录列交换
                temp = col_exchange[i]
                col_exchange[i] = col_exchange[jj]
                col_exchange[jj] = temp
                ## 交换 i 列和 jj 列;
                encH[:, [i, jj]] = encH[:, [jj, i]]
            ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
            for m in range(num_row):
                if m != i and (encH[m, i] == 1):
                    # encH[m, :] = encH[m, :] ^ encH[i, :]
                    encH[m, :] = np.logical_xor(encH[m, :], encH[i, :])
                    # encH[m, :] = np.bitwise_xor(encH[m, :], encH[i, :])
                    # for n in range(num_col):
                        # encH[m, n] ^= encH[i, n]
    ##====================== Gauss 消元 end =================================
    return encH, col_exchange

def bpsk(bins):
    bits = copy.deepcopy(bins)
    bits[np.where(bits == 1)] = -1
    bits[np.where(bits == 0)] = 1
    return bits

def BPSK(bins):
    c = 1 - 2*bins
    return c


class QLDPC_Coding(object):
    def __init__(self, args):
        ## code parameters
        self.args = args
        self.p = args.active_client          # q = 2^p
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
        self.qbits = None
        self.qcomb = None
        self.Hadm = None
        self.readH()
        ## print("读取H完成...\n")
        self.systemH()
        ## print("高斯消元完成, encH，decH已获取...\n")
        self.NoneZeros()
        ## print("保存行列索引完成...")
        self.Hadamard()
        self.Init()
        return

    def Init(self, ):
        real_ary = BPSK(self.qbits) # .astype(np.float32)
        rowsum = np.unique(real_ary.sum(axis = 1))
        self.ordered_sum = sorted(rowsum)

        self.sum2idx = {}
        for i, bins in enumerate(real_ary):
            if int(np.sum(bins)) in self.sum2idx.keys():
                self.sum2idx[int(np.sum(bins))].append(i)
            else:
                self.sum2idx[int(np.sum(bins))] = []
                self.sum2idx[int(np.sum(bins))].append(i)
        return

    def readH(self):
        current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(current_dir + self.args.file_name_of_the_H, 'r', encoding = 'utf-8') as f:
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
        # row_weigh = self.decH.sum(axis = 1)[0]
        # GF = galois.GF(2**self.p, repr = "int")
        # self.qcomb = {}
        # for i in range(self.q):
        #     self.qcomb[i] = []
        # for comb in itertools.product(GF.elements, repeat = row_weigh - 1):
        #     self.qcomb[int(np.sum(GF([int(g) for g in comb])))].append([int(g) for g in comb])
        return

    def MACchannel(self, symbs, H, noise_var):
        yy = symbs * H
        noise = np.sqrt(noise_var/2) * (np.random.normal(0, 1, size = symbs.shape[-1] ) + 1j*np.random.normal(0, 1, size = symbs.shape[-1] ))
        yy = yy.sum(axis  = 0) + noise
        return yy

    def post_probability_mess(self, yy, H, noise_var):
        pp = np.exp(-np.abs(yy - BPSK(self.qbits) @ H)**2 /(2 * noise_var))
        pp = pp/ pp.sum(axis = 0)

        tmp = copy.deepcopy(pp[1,:])
        pp[1,:] = pp[2,:]
        pp[2,:] = tmp

        pp = np.clip(pp, self.smallprob, 1 - self.smallprob)
        return pp

    def post_probability(self, yy, H, noise_var):
        pp = np.exp(-np.abs(yy - BPSK(self.qbits) @ H)**2 /(2 * noise_var))
        pp = pp/ pp.sum(axis = 0)
        pp = np.clip(pp, self.smallprob, 1 - self.smallprob)
        return pp

    def Hadamard(self,):
        H0 = np.array([[1,1], [1,-1]])
        for i in range(self.p - 1):
            H0 = np.block([[H0, H0], [H0, -1*H0]])
        self.Hadm = H0
        return

    def bits2sum(self, bits):
        real_ary = BPSK(bits)
        real_sum = real_ary.sum(axis = 0)
        return real_sum

    ## hadmard变换后的算法
    def decoder_FFTQSPA(self, pp, maxiter = 50):
        self.MV2C = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        self.MC2V = np.zeros((self.num_row, self.num_col, self.q), dtype = np.float64 )
        # uu_hat = np.zeros(self.codedim, dtype = np.int8)
        ##===========================================
        ## (初始化) V 到 C 的初始化信息
        ##===========================================
        for col in self.SetCols:
            for row in self.SetCols[col]:
                for q in range(self.q):
                    self.MV2C[row, col, q] = pp[q, col]
        ## 开始迭代，概率域的消息传播,
        for iter_num in range(maxiter):
            # print(f"  {iter_num}")
            ##==========================================================
            ## (一) 更新 C 到 V 的消息,
            ##==========================================================
            # print(f"    {iter_num} -> C2V")
            for row in self.SetRows:
                for col in self.SetRows[row]:
                    col_in = copy.deepcopy(self.SetRows[row])
                    col_in.remove(col)
                    tmp = np.ones(self.q)
                    for v in col_in:
                        tmp *= (self.Hadm @ self.MV2C[row, v, :])
                    self.MC2V[row, col, :] = self.Hadm @ tmp / self.q
                    # self.MC2V[row, col, :] = self.MC2V[row, col, :]/self.MC2V[row, col, :].sum()
                    # self.MC2V[row, col, :] = np.clip(self.MC2V[row, col, :], self.smallprob, 1-self.smallprob)
            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出,
            ##=============================================================================
            # print(f"    {iter_num} -> decision")
            PQ = np.zeros_like(pp, dtype = np.float64)
            for col in self.SetCols:
                row_in = copy.deepcopy(self.SetCols[col])
                for q in range(self.q):
                    tmp = pp[q, col]
                    for c in row_in:
                        tmp *= self.MC2V[c, col, q]
                    PQ[q, col] = tmp
            Pdecision = PQ.argmax(axis = 0)
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
            ## (三) 更新 v 到 c 的消息, 半边输入信息也要考虑进去
            #========================================================================
            # print(f"    {iter_num} -> V2C")
            for col in self.SetCols:
                for row in self.SetCols[col]:
                    row_in = copy.deepcopy(self.SetCols[col])
                    row_in.remove(row)
                    for q in range(self.q):
                        tmp = pp[q, col]
                        for c in row_in:
                            tmp *= self.MC2V[c, col, q]
                        self.MV2C[row, col, q] = tmp
                    self.MV2C[row, col, :] = self.MV2C[row, col, :]/self.MV2C[row, col, :].sum()
                    self.MV2C[row, col, :] = np.clip(self.MV2C[row, col, :], self.smallprob, 1 - self.smallprob)
        return uu_hat, iter_num + 1








































































































































































































