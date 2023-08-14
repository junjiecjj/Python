

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen
"""


import numpy as np
import copy
import sys



from utility import  Gauss_Elimination

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
        self.max_iter = args.max_iteration  # 最大迭代次数
        self.SetRows = {}                   # 每行不为0的列号
        self.SetCols = {}                   # 每列不为0的行号
        self.MV2C = None                    # 变量节点 到 校验节点 的消息
        self.MC2V = None                    # 校验节点 到 变量节点 的消息
        self.cc_hat = None

        self.readH()
        self.systemH1()
        print("高斯消元完成，encH，decH完成\n")
        self.NoneZeros()
        return

    def readH(self):
        with open(self.args.file_name_of_the_H, 'r', encoding='utf-8') as f:
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

    def systemH1(self):
        codechk = 0
        tmpH = copy.deepcopy(self.decH)
        self.encH = copy.deepcopy(self.decH)
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
                        # print("0: I am break")
                        break
                if flag == 1:
                    codechk += 1
                    # print("1: I am break")
                    break
            if flag == 0:
                print("I am break")
                break
            else:
                ## 交换 i 行和 ii 行;
                if ii != i:
                    # print(f"{i} 行交换")
                    for n in range(self.num_col):
                        temp = self.encH[i, n]
                        self.encH[i, n] = self.encH[ii, n]
                        self.encH[ii, n] = temp
                if jj != i:
                    print("列交换")
                    ## 记录列交换
                    temp = col_exchange[i]
                    col_exchange[i] = col_exchange[jj]
                    col_exchange[jj] = temp

                    ## 交换 i 列和 jj 列;
                    for m in range(self.num_row):
                        temp = self.encH[m, i]
                        self.encH[m, i] = self.encH[m, jj]
                        self.encH[m, jj] = temp
                ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
                for m in range(self.num_row):
                    if m != i and (self.encH[m, i] == 1):
                        for n in range(self.num_col):
                            self.encH[m, n] ^= self.encH[i, n]
        ##====================== Gauss 消元 end =================================
        ## 计算码的参数
        self.codechk = codechk
        self.codedim = self.codelen -  self.codechk
        self.coderate = self.codedim / self.codelen
        ##======================================================
        ## 根据列交换结果交换原 decH 矩阵的列
        ##=======================================================
        for j in range(self.num_col):
            self.decH[:, j] = tmpH[:, col_exchange[j]]
        return


    def encoder(self, uu):
        cc = np.zeros(self.num_col, dtype = np.int8)
        cc[self.codechk:] = uu

        ## 1
        # cc[:self.codechk] = np.mod(np.matmul(uu, self.encH[:,self.codechk:].T), 2)

        ## 2
        for i in range(self.codechk):
            cc[i] = np.logical_xor.reduce(np.logical_and(uu[:], self.encH[i, self.codechk:]))
            # cc[i] = np.logical_xor.reduce(uu[:] & self.encH[j, self.codechk:])
            # cc[i] = np.bitwise_xor.reduce(uu[:] & self.encH[j, self.codechk:])

        ## 3
        # for i in range(self.codechk):
        #     for j in range(self.codedim):
        #         cc[i] ^=  (uu[j]&self.encH[i, self.codechk:][j])
        return cc

    def NoneZeros(self):
        ## 字典  {行号: {不为 0 的列号}}
        self.SetRows  = {f'{i}': set(np.nonzero(self.decH[i,:])[0].astype(int)) for i in range(self.decH.shape[0])}
        ## 字典  {列号: {不为 0 的行号}}
        self.SetCols = {f'{j}': set(np.nonzero(self.decH[:,j])[0].astype(int)) for j in range(self.decH.shape[1])}
        return

    def decoder(self, yy_llr):
        iter_num = 0
        uu_hat = np.zeros(self.codedim, dtype = np.int8)

        ## 变量节点 到 校验节点 的初始化信息
        for col in self.SetCols.keys():
            for row in self.SetCols[f'{col}']:
                self.MV2C[int(row), int(col)] = yy_llr[int(col)]

        ## 开始迭代，消息传播
        for iter_num in range(self.max_iter):
            #==========================================================
            ## 对数域的消息传播, 更新 [校验节点] 到 [变量节点] 的消息
            #==========================================================
            for row in self.SetRows:
                for col in self.SetRows[f'{row}']:
                    pt = 1.0
                    for cin in self.SetRows[f'{row}']:
                        if cin != col:
                            pt *= np.tanh(self.MV2C[int(row), int(cin)]/2)
                    self.MC2V[int(row), int(col)] = np.log((1 + pt)/(1 - pt))
            #=============================================================================
            ## 计算变量节点的那半边的输出信息, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            #=============================================================================
            dec_llr = np.zeros(self.codelen, dtype = np.float64)
            for col in self.SetCols.keys():
                tllr = 0
                for row in self.SetCols[f'{col}']:
                    tllr += self.MC2V[int(row), int(col)]
                dec_llr[int(col)] = tllr + yy_llr[int(col)]

            # 对等号节点判决
            self.cc_hat = np.zeros(self.codelen, dtype = np.int8 )
            sgn = np.sign(dec_llr)
            for i in range(0, dec_llr.shape[-1]):
                if sgn[i] < 0:
                    self.cc_hat[i] = 1
            uu_hat = self.cc_hat[self.codechk:]

            success = 1
            # # parity checking，校验
            # for row in self.SetRows:
            #     parity_check = 0
            #     for col in self.SetRows[f'{row}']:
            #         parity_check ^= self.cc_hat[int(col)]
            #     if parity_check != 0:
            #         success = 0
            #         break
            # parity checking，校验
            for i in range(self.num_row):
                parity_check = np.logical_xor.reduce(np.logical_and(self.cc_hat, self.decH[i,:]))
                if parity_check != 0:
                    success = 0
                    break
            # # parity checking，校验
            # if np.all(np.mod(np.matmul(self.cc_hat, self.decH.T), 2) != 0):
            #     success = 0

            if success == 1:
                return uu_hat, iter_num
            #==========================================================
            ## 更新 [变量节点] 到 [校验节点] 的消息，半边输入信息也要考虑进去
            #==========================================================
            for col in self.SetCols.keys():
                for row in self.SetCols[f'{col}']:
                    Mes = 0
                    for cout in self.SetCols[f'{col}']:
                        if cout != row:
                            Mes += self.MC2V[int(cout),int(col)]
                    self.MV2C[int(row),int(col)] = Mes +  yy_llr[int(col)]

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


codechk = 128
codedim = 128
codelen = 256

# uu = np.ones(codedim, dtype = np.int8)
uu = np.random.randint(low = 0, high = 2, size = (codedim,), dtype = np.int8)
encH = np.random.randint(low = 0, high = 2, size = (codechk, codelen ), dtype = np.int8)
cc1 = np.zeros(codelen, dtype = np.int8)
cc2 = np.zeros(codelen, dtype = np.int8)
cc3 = np.zeros(codelen, dtype = np.int8)
cc4 = np.zeros(codelen, dtype = np.int8)


cc1[codechk:] = uu
cc1[: codechk] = np.mod(np.matmul(uu,  encH[:, codechk:].T), 2)

cc2[codechk:] = uu
for j in range(codechk):
    # cc2[j] = np.logical_xor.reduce(np.logical_and(uu[:], encH[j, codechk:]))
    cc2[j] = np.logical_xor.reduce(uu[:] & encH[j, codechk:])


cc3[codechk:] = uu
for i in range(codechk):
    for j in range(codedim):
        cc3[i] ^=  (uu[j]&encH[i, codechk:][j])


cc4[codechk:] = uu
for j in range(codechk):
    cc4[j] = np.bitwise_xor.reduce( uu[:] & encH[j, codechk:])




print((cc1-cc2).min(), (cc1-cc2).max())
print((cc1-cc3).min(), (cc1-cc3).max())
print((cc1-cc4).min(), (cc1-cc4).max())



# import copy

# tmpH = np.arange(20).reshape(4,5)
# encH = copy.deepcopy(tmpH)
# decH = copy.deepcopy(tmpH)



# exchange = [4,3,1,0,2]

# for j in range(5):
#     for i in range(4):
#         encH[i, j] = tmpH[i, exchange[j]]


# for j in range(5):
#     decH[:, j] = tmpH[:, exchange[j]]




















































































































































































































































