

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen
"""


import numpy as np
import copy
import sys, os
import functools


from Utility  import Gauss_Elimination

class LDPC_Coder_llr(object):
    def __init__(self, args):
        ## code parameters
        self.args = args
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

        self.readH()
        # print("读取H完成...\n")
        self.systemH()
        # print("高斯消元完成, encH，decH已获取...\n")
        self.NoneZeros()
        # print("保存行列索引完成...")
        return

    def readH(self):
        current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        with open(current_dir+self.args.file_name_of_the_H, 'r', encoding='utf-8') as f:
            tmp = f.readline()
            # print(tmp)
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
        # np.savetxt('orig_decH.txt', self.decH, fmt='%d', delimiter=' ')

        self.codelen = self.num_col
        self.codedim = self.codelen - self.codechk
        self.coderate = self.codedim / self.codelen

        self.MV2C = np.zeros((self.num_row, self.num_col), dtype = np.float64 )
        self.MC2V = np.zeros((self.num_row, self.num_col), dtype = np.float64 )

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
        ## 根据列交换结果交换原 decH 矩阵的列
        ##=======================================================
        for j in range(self.num_col):
            self.decH[:, j] = tmpH[:, col_exchange[j]]
        return

    def NoneZeros(self):
        ## 字典  {行号: {不为 0 的列号}}
        self.SetRows  = {f'{i}': set(np.nonzero(self.decH[i,:])[0].astype(int)) for i in range(self.decH.shape[0])}
        ## 字典  {列号: {不为 0 的行号}}
        self.SetCols = {f'{j}': set(np.nonzero(self.decH[:,j])[0].astype(int)) for j in range(self.decH.shape[1])}
        return

    def encoder(self, uu):
        cc = np.zeros(self.codelen, dtype = np.int8)
        cc[self.codechk:] = uu

        ## 1：可能出错
        # cc[:self.codechk] = np.mod(np.matmul(uu, self.encH[:,self.codechk:].T), 2)

        ## 2：相对快
        for i in range(self.codechk):
            cc[i] = np.logical_xor.reduce(np.logical_and(uu[:], self.encH[i, self.codechk:]))
            # cc[i] = np.logical_xor.reduce(uu[:] & self.encH[j, self.codechk:])
            # cc[i] = np.bitwise_xor.reduce(uu[:] & self.encH[j, self.codechk:])

        ## 3：慢
        # for i in range(self.codechk):
        #     for j in range(self.codedim):
        #         cc[i] ^=  (uu[j]&self.encH[i, self.codechk:][j])
        return cc


    ## 对数域的和积算法
    def decoder_spa(self, yy_llr):
        iter_num = 0
        uu_hat = np.zeros(self.codedim, dtype = np.int8)
        ##===========================================
        ## (初始化) 变量节点 到 校验节点 的初始化信息
        ##===========================================
        for col in self.SetCols.keys():
            for row in self.SetCols[f'{col}']:
                self.MV2C[int(row), int(col)] = yy_llr[int(col)]

        ## 开始迭代，对数域的消息传播,
        for iter_num in range(self.max_iter):
            ##==========================================================
            ## (一) 更新 [校验节点] 到 [变量节点] 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[f'{row}']:
                    Mes = 1.0
                    for cin in self.SetRows[f'{row}']:
                        if cin != col:
                            Mes *= np.tanh(self.MV2C[int(row), int(cin)]/2)
                    # Mes = np.sign(Mes) * min(abs(Mes), 1-1e-15 )  ## 解决数值不稳定性问题
                    Mes = np.clip(Mes, self.smallprob - 1, 1 - self.smallprob)  ## 解决数值不稳定性问题
                    self.MC2V[int(row), int(col)] = np.log((1 + Mes)/(1 - Mes)) # (白老师书上3.43)
            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            dec_llr = np.zeros(self.codelen, dtype = np.float64)
            for col in self.SetCols.keys():
                Mes = 0
                for row in self.SetCols[f'{col}']:
                    Mes += self.MC2V[int(row), int(col)]
                dec_llr[int(col)] = Mes + yy_llr[int(col)] # (白老师书上3.49)
            # 对等号节点判决
            cc_hat = np.zeros(self.codelen, dtype = np.int8 )
            cc_hat[np.where(dec_llr < 0)] = 1

            uu_hat = cc_hat[self.codechk:]
            success = 1
            # parity checking，校验
            for i in range(self.num_row):
                parity_check = np.bitwise_xor.reduce(cc_hat & self.decH[i,:])
                if parity_check != 0:
                    success = 0
                    break
            if success == 1:
                return uu_hat, iter_num + 1
            #==========================================================
            ## (三) 更新 [变量节点] 到 [校验节点] 的消息，半边输入信息也要考虑进去
            #==========================================================
            for col in self.SetCols.keys():
                for row in self.SetCols[f'{col}']:
                    Mes = 0
                    for cout in self.SetCols[f'{col}']:
                        if cout != row:
                            Mes += self.MC2V[int(cout),int(col)] # (白老师书上3.48)
                    self.MV2C[int(row),int(col)] = Mes +  yy_llr[int(col)]
        return uu_hat, iter_num + 1

    ## 对数域的最小和算法
    def decoder_msa(self, yy_llr, alpha = 0.75):
        iter_num = 0
        uu_hat = np.zeros(self.codedim, dtype = np.int8)
        cc_hat = np.zeros(self.codelen, dtype = np.int8 )
        ##===========================================
        ## (初始化) 变量节点 到 校验节点 的初始化信息
        ##===========================================
        for col in self.SetCols.keys():
            for row in self.SetCols[f'{col}']:
                self.MV2C[int(row), int(col)] = yy_llr[int(col)]

        ## 开始迭代，对数域的消息传播,
        for iter_num in range(self.max_iter):
            ##==========================================================
            ## (一) 更新 [校验节点] 到 [变量节点] 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[f'{row}']:
                    Sign = 1.0
                    Min = min([abs(self.MV2C[int(row), int(i)]) for i in self.SetRows[f'{row}'] if i != col])
                    sign_list = [np.sign(self.MV2C[int(row), int(i)]) for i in self.SetRows[f'{row}'] if i != col]
                    Sign = functools.reduce(lambda a, b: a*b, sign_list)
                    self.MC2V[int(row), int(col)] = Sign * Min * alpha

            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            dec_llr = np.zeros(self.codelen, dtype = np.float64)
            for col in self.SetCols.keys():
                Mes = 0
                for row in self.SetCols[f'{col}']:
                    Mes += self.MC2V[int(row), int(col)]
                dec_llr[int(col)] = Mes + yy_llr[int(col)]

            # 对等号节点判决
            cc_hat.fill(0)
            cc_hat[np.where(dec_llr < 0)] = 1
            uu_hat = cc_hat[self.codechk:]

            success = 1
            # parity checking，校验
            for i in range(self.num_row):
                parity_check = np.bitwise_xor.reduce(cc_hat & self.decH[i,:])
                if parity_check != 0:
                    success = 0
                    break
            if success == 1:
                return uu_hat, iter_num + 1
            #==========================================================
            ## (三) 更新 [变量节点] 到 [校验节点] 的消息，半边输入信息也要考虑进去
            #==========================================================
            for col in self.SetCols.keys():
                for row in self.SetCols[f'{col}']:
                    Mes = 0
                    for cout in self.SetCols[f'{col}']:
                        if cout != row:
                            Mes += self.MC2V[int(cout),int(col)]
                    self.MV2C[int(row),int(col)] = Mes +  yy_llr[int(col)]

        return uu_hat, iter_num + 1















































































































































































































































