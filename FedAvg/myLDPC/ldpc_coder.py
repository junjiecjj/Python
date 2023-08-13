

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen
"""


import numpy as np
import copy



class LDPC_Coder_Llr(object):
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
        self.m_max_iter = args.max_iteration  # 最大迭代次数

        self.cc_hat = None
        return

    def readH(self):
        with open(self.args.file_name_of_the_H, 'r', encoding='utf-8') as f:
            tmp = f.readline()
            # print(tmp)
            tmp = f.readline()
            # print(tmp)
            self.num_row, self.num_col, self.codechk = [int(i) for i in tmp.strip().split()]
            self.encH = np.zeros( (self.num_row, self.num_col), dtype = np.int64 )
            tmp = f.readline()
            # print(tmp)
            while 1:
                tmp = f.readline()
                if not tmp:
                    break
                row_dt = [int(i) for i in tmp.strip().split()]
                for i in range(row_dt[1]):
                    self.encH[row_dt[0], row_dt[i+2]] = 1

        self.codelen = self.num_col
        self.codedim = self.codelen - self.codechk
        self.coderate = self.codedim / self.codelen
        return

    def systemH(self):
        codechk = 0
        tmpH = copy.deepcopy(self.encH)
        self.decH = copy.deepcopy(self.encH)
        col_exchange = np.arange(self.num_col)

        ##=======================================================
        ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
        ##=======================================================
        for i in range(self.num_row):
            flag = 0
            for jj in range(i, self.num_col):
                for ii in range(i, self.num_row):
                    if self.encH[ii, jj] != 0:
                        flag = 1
                        break
                if flag == 1:
                    codechk += 1
                    break
            if flag == 0:
                break
            else:
                ## 交换 i 行和 ii 行;
                if ii != i:
                    for n in range(self.num_col):
                        tmpdt = self.encH[i, n]
                        self.encH[i, n] = self.encH[ii, n]
                        self.encH[ii, n] = tmpdt
                if jj != i:
                    ## 记录列交换
                    tmpdt = col_exchange[i]
                    col_exchange[i] = col_exchange[jj]
                    col_exchange[jj] = tmpdt

                    ## 交换 i 列和 jj 列;
                    for m in range(self.num_row):
                        tmpdt = self.encH[m, i]
                        self.encH[m, i] = self.encH[m, jj]
                        self.encH[m, jj] = tmpdt
                ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
                for m in range(self.num_row):
                    if m != i and self.encH[m, i] == 1:
                        for n in range(self.num_col):
                            self.encH[m, n] ^= self.encH[i, n]
        ## Gauss 消元结束

        ## 计算码的参数
        self.codechk = codechk
        self.codedim = self.codelen -  self.codechk
        self.coderate = self.codedim / self.codelen

        ##======================================================
        ## 根据列交换结果交换原encH矩阵的列
        ##=======================================================
        for j in range(self.num_col):
            self.decH[:, j] = tmpH[:, col_exchange[j]]

        return


    def encoder(self, uu):
        cc = np.zeros(self.num_col)
        cc[self.codechk:] = uu

        ## 1
        cc[:self.codechk] = np.mod(np.matmul(uu, self.encH[:,self.codechk:].T), 2)

        ## 2
        # for i in range(self.codechk):
        #     cc[i] = np.logical_xor.reduce(np.logical_and(uu[:], encH[i, self.codechk:]))

        ## 3
        # for i in range(self.codechk):
        #     for j in range(self.codedim):
        #         cc[i] ^=  (uu[j]&encH[i, codechk:][j])
        return cc

    def NoneZeros(self):
        ## 字典  {行号: {不为 0 的列号}}
        self.SetRows  = {f'{i}': set(np.nonzero(self.decH[i,:])[0].astype(int)) for i in range(self.decH.shape[0])}
        ## 字典  {列号: {不为 0 的行号}}
        self.SetCols = {f'{j}': set(np.nonzero(self.decH[:,j])[0].astype(int)) for j in range(self.decH.shape[1])}
        return

    def decoder(self, yy_llr):
        iter_num = 0
        uu_hat = np.zeros(self.codedim, dtype = np.int64)



        return uu_hat, iter_num



# def square(A):
#     A_2 = np.zeros((A.shape[0], A.shape[1]), dtype=int)
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             A_2[i][j] = np.logical_xor.reduce(np.logical_and(A[i], A[:, j]))
#     return A_2



# def Multipul(A, B):  # 二元域上的矩阵乘法
#     A_B = np.zeros((A.shape[0], B.shape[1]), dtype=int)
#     for i in range(A.shape[0]):
#         for j in range(B.shape[1]):
#             A_B[i][j] = np.logical_xor.reduce(np.logical_and(A[i], B[:, j]))
#     return A_B


# codechk = 3
# uu = np.array([1, 0, 0, 1, 1, 1, 0])
# encH = np.random.randint(low = 0, high = 2, size = (3, 10 ), dtype = np.int64)
# cc1 = np.zeros(10, dtype = np.int64)
# cc2 = np.zeros(10, dtype = np.int64)
# cc3 = np.zeros(10, dtype = np.int64)


# cc1[codechk:] = uu
# cc1[: codechk] = np.mod(np.matmul(uu,  encH[:, codechk:].T), 2)

# cc2[codechk:] = uu
# for j in range(codechk):
#     cc2[j] = np.logical_xor.reduce(np.logical_and(uu[:], encH[j, codechk:]))


# cc3[codechk:] = uu
# for i in range(codechk):
#     for j in range(7):
#         cc3[i] ^=  (uu[j]&encH[i, codechk:][j])






















































































































































































































































