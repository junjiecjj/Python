

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen
"""


import numpy as np
import copy



class LDPC_Coder(object):
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

        ##
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
        # self.codedim = self.codelen - self.codechk
        # self.coderate = self.codedim / self.codelen
        return

    def systemH(self):
        codechk = 0
        tmpH = copy.deepcopy(self.encH)
        self.decH = copy.deepcopy(self.encH)
        col_exchange = np.arange(self.num_col)

        ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
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

        ## 根据列交换结果交换原encH矩阵的列
        for j in range(self.num_col):
            self.decH[:, j] = tmpH[:, col_exchange[j]]

        return


    def encode(self, uu):
        cc = np.zeros()
        return cc

    def decoder(self, yy):
        iter_num = 0
        uu_hat = np.array([1,2])
        return uu_hat, iter_num




































































































































































































































































