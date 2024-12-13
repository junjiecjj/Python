

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:45:57 2023

@author: JunJie Chen
"""


import numpy as np
import copy
import sys, os
from functools import reduce
import random


from utility  import Gauss_Elimination

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
        return

    ## 对数域的和积算法, 二元域
    def decoder_spa(self, yy_llr):
        self.MV2C = np.zeros((self.num_row, self.num_col), dtype = np.float64 )
        self.MC2V = np.zeros((self.num_row, self.num_col), dtype = np.float64 )
        iter_num = 0
        # uu_hat = np.zeros(self.codedim, dtype = np.int8)
        ##===========================================
        ## (初始化) 变量节点 到 校验节点 的初始化信息
        ##===========================================
        for col in self.SetCols:
            for row in self.SetCols[col]:
                self.MV2C[row, col] = yy_llr[col]
        ## 开始迭代，对数域的消息传播,
        for iter_num in range(self.max_iter):
            ##==========================================================
            ## (一) 更新 [校验节点] 到 [变量节点] 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[row]:
                    col_in = copy.deepcopy(self.SetRows[row])
                    col_in.remove(col)
                    Mes = 1.0
                    for cin in col_in:
                        Mes *= np.tanh(self.MV2C[row, cin]/2)
                    # Mes = np.sign(Mes) * min(abs(Mes), 1-1e-15 )  ## 解决数值不稳定性问题
                    Mes = np.clip(Mes, self.smallprob - 1, 1 - self.smallprob)  ## 解决数值不稳定性问题
                    self.MC2V[row, col] = np.log((1 + Mes)/(1 - Mes))
            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            dec_llr = np.zeros(self.codelen, dtype = np.float64)
            for col in self.SetCols:
                Mes = 0
                for row in self.SetCols[col]:
                    Mes += self.MC2V[row, col]
                dec_llr[col] = Mes + yy_llr[col]
            # 对等号节点判决
            cc_hat = np.zeros(self.codelen, dtype = np.int8 )
            cc_hat[np.where(dec_llr < 0)] = 1
            # uu_hat = cc_hat[self.codechk:]

            success = 1
            unpassNum = 0
            # parity checking，校验
            for i in range(self.num_row):
                parity_check = np.bitwise_xor.reduce(cc_hat & self.decH[i,:])
                if parity_check != 0:
                    success = 0
                    unpassNum += 1
                    # break
            if success == 1:
                return cc_hat, iter_num + 1, unpassNum / self.codechk, success
            #==========================================================
            ## (三) 更新 [变量节点] 到 [校验节点] 的消息，半边输入信息也要考虑进去
            #==========================================================
            for col in self.SetCols.keys():
                for row in self.SetCols[col]:
                    Mes = 0
                    row_in = copy.deepcopy(self.SetCols[col])
                    row_in.remove(row)
                    for r in row_in:
                        Mes += self.MC2V[r, col]
                    self.MV2C[row, col] = Mes +  yy_llr[col]
        return cc_hat, iter_num + 1, unpassNum / self.codechk, success

    ## 对数域的最小和算法, 二元域
    def decoder_msa(self, yy_llr, alpha = 0.75):
        self.MV2C = np.zeros((self.num_row, self.num_col), dtype = np.float64 )
        self.MC2V = np.zeros((self.num_row, self.num_col), dtype = np.float64 )
        iter_num = 0
        uu_hat = np.zeros(self.codedim, dtype = np.int8)
        cc_hat = np.zeros(self.codelen, dtype = np.int8 )
        ##===========================================
        ## (初始化) 变量节点 到 校验节点 的初始化信息
        ##===========================================
        for col in self.SetCols.keys():
            for row in self.SetCols[col]:
                self.MV2C[row, col] = yy_llr[col]

        ## 开始迭代，对数域的消息传播,
        for iter_num in range(self.max_iter):
            ##==========================================================
            ## (一) 更新 [校验节点] 到 [变量节点] 的消息,
            ##==========================================================
            for row in self.SetRows:
                for col in self.SetRows[row]:
                    Sign = 1.0
                    Min = min([abs(self.MV2C[row, i]) for i in self.SetRows[row] if i != col])
                    sign_list = [np.sign(self.MV2C[row, i]) for i in self.SetRows[row] if i != col]
                    Sign = reduce(lambda a,b: a*b, sign_list)
                    self.MC2V[row, col] = Sign * Min * alpha

            ##=============================================================================
            ## (二) 合并, 判决,校验, 输出, 在计算半边的输出的时候, 半边输入信息也要考虑进去
            ##=============================================================================
            dec_llr = np.zeros(self.codelen, dtype = np.float64)
            for col in self.SetCols.keys():
                Mes = 0
                for row in self.SetCols[col]:
                    Mes += self.MC2V[row, col]
                dec_llr[col] = Mes + yy_llr[col]

            # 对等号节点判决
            cc_hat.fill(0)
            cc_hat[np.where(dec_llr < 0)] = 1
            uu_hat = cc_hat[self.codechk:]

            success = 1
            # parity checking，校验
            for i in range(self.num_row):
                # parity_check = np.logical_xor.reduce(np.logical_and(cc_hat, self.decH[i,:]))
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
                for row in self.SetCols[col]:
                    Mes = 0
                    row_in = copy.deepcopy(self.SetCols[col])
                    row_in.remove(row)
                    for r in row_in:
                        Mes += self.MC2V[r, col]
                    self.MV2C[row, col] = Mes +  yy_llr[col]

        return uu_hat, iter_num + 1

    def find(self, uu, p):
        uu_hat = copy.deepcopy(uu)
        success = 1
        ## 全局校验
        for i in range(self.num_row):
            parity_check = np.bitwise_xor.reduce(uu_hat & self.decH[i,:])
            # print(f"{i}: {parity_check}")
            if parity_check != 0:
                success = 0
                break
        if success == 1:
            # print("    globe pass check 0")
            return uu_hat, 0, 0, success, 0
            # sys.exit(1)

        tot_num_flip = 0
        flippedCOL = []
        # ROWS = np.arange(self.num_row)
        # ROWS = np.random.permutation(ROWS)
        for r, row in enumerate(self.SetRows):
            cols = copy.deepcopy(self.SetRows[row])
            restcol = list(set(cols).difference(set(flippedCOL)))
            restcol.sort()
            # print(f"  row = {r}, restcol = {restcol}")
            flippedCOL = list(set(flippedCOL).union(set(cols)))
            flippedCOL.sort()
            L = len(restcol)
            pass_check_row = np.bitwise_xor.reduce(uu_hat[cols] & self.decH[row, cols])

            if pass_check_row == 1:
                # print(f"   #{row} check node NEED flip, restcol = {restcol}  ")
                if  len(restcol) > 0:
                    # print(f"   #{row} check node need flip, restcol = {restcol}  ")
                    local_filp_num = 0
                    while pass_check_row != 0:
                        flipping = np.random.choice([0, 1], size = L, p = [1 - p, p], ).astype(np.int8)
                        uu_hat[restcol] = uu_hat[restcol] ^ flipping
                        tot_num_flip += 1
                        local_filp_num += 1
                        pass_check_row = np.bitwise_xor.reduce(uu_hat[cols] & self.decH[row, cols])
                    # print(f"      {row} check node pass check by flip {local_filp_num} times, tot_filp = {tot_num_flip}")
                    success = 1
                    ## 全局校验
                    for i in range(r+1, self.num_row):
                        parity_check = np.bitwise_xor.reduce(uu_hat & self.decH[i,:])
                        if parity_check != 0:
                            success = 0
                            break
                    if success == 1:
                        # print("    globe pass check 1")
                        return uu_hat, r, tot_num_flip, success, 0
                else:
                    print(f"   #{r} check node NEED flip, but has {len(restcol)} rest ")
                    ## 全局校验
                    # success = 1
                    # start = 0
                    count = 0
                    rest_num = self.num_row - r
                    # flag = 0
                    for i in range(r, self.num_row):
                        parity_check = np.bitwise_xor.reduce(uu_hat & self.decH[i,:])
                        if parity_check != 0:
                            count += 1

                    # print(f"   Start = {start}")
                    return uu_hat, r, tot_num_flip, success, count/rest_num
            # else:
                # pass
                # print(f"   #{row} check node pass check 0")
        return uu_hat, r, tot_num_flip, success, 0
















































































































































































































































