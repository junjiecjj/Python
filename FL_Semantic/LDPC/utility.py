#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:10 2023
@author: JunJie Chen

np.mod(np.matmul(KbinRe, G), 2)


"""

import datetime
import os
import numpy as np
import random
import numpy as np
import torch


def WrLogHead(logfile = "SNR_BerFer.txt", promargs = '', codeargs = ''):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    logfile = current_dir + logfile
    # print(logfile)
    with open(logfile, 'a+') as f:
        print("#=====================================================================================",  file = f)
        print("                      " +  now,  file = f)
        print("#=====================================================================================\n",  file = f)

        f.write("######### [program config] #########\n")
        for k, v in promargs.__dict__.items():
            f.write(f"{k: <25}: {v: <40}\n")
        f.write("######### [code config] ##########\n")
        for k, v in codeargs.items():
            f.write(f"{k: <25}: {v: <40}\n")
        f.write("\n#=============================== args end  ===============================\n")
    return


## complex method
def Gauss_Elimination_complex(encH, num_row, num_col):
    codechk = 0
    col_exchange = np.arange(num_col)
    ##=======================================================
    ##  开始 Gauss 消元，建立系统阵(生成矩阵G )，化简为: [I, P]的形式
    ##=======================================================
    for i in range(num_row):
        flag = 0
        for jj in range(i, num_col):
            for ii in range(i, num_row):
                if encH[ii, jj] != 0:
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
                for n in range( num_col):
                    temp =  encH[i, n]
                    encH[i, n] = encH[ii, n]
                    encH[ii, n] = temp
            if jj != i:
                print("列交换")
                ## 记录列交换
                temp = col_exchange[i]
                col_exchange[i] = col_exchange[jj]
                col_exchange[jj] = temp

                ## 交换 i 列和 jj 列;
                for m in range(num_row):
                    temp = encH[m, i]
                    encH[m, i] = encH[m, jj]
                    encH[m, jj] = temp
            ## 消去 [I, P] 形式的前半部分 mxm 矩阵的第 i 列主对角线外的其他元素
            for m in range(num_row):
                if m != i and (encH[m, i] == 1):
                    for n in range(num_col):
                        encH[m, n] ^= encH[i, n]
    ##====================== Gauss 消元 end =================================
    return encH, col_exchange


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
                print("1: 列交换")
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


# 初始化随机数种子
def set_random_seed(seed = 1, deterministic = False, benchmark = False):
    np.set_printoptions(linewidth = 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return

##  yy --> 对数域LLR
def yyToLLR(yy, noise_var = 1.0):
    LLr = np.zeros_like(yy)
    LLr =  2.0*yy/noise_var
    return LLr

##  yy --> 概率域
def yyToProb(yy, noise_var = 1.0):
    prob = np.zeros_like(yy)
    prob = 1.0 / (1.0 + np.exp(-2.0*yy/noise_var))
    return prob


"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的全0的码字，bits，如果yy[i]<0,则bits[i] = 1
"""
def bit_judge(data_in):
    bits = np.zeros(data_in.size, dtype = 'int')
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits



def bit_judge1(TLLRs):
    CodeWord = np.zeros(np.shape(TLLRs)).astype(int)
    sgn = np.sign(TLLRs)
    for i in range(0,np.shape(TLLRs)[1]):
        if sgn[0,i] == -1:
            CodeWord[0,i] = 1

    return CodeWord


def CodeWordValidation(ParityCheckMatrix, CodeWord):
    if np.all((np.dot(CodeWord, np.transpose(ParityCheckMatrix))%2 == 0)):
        return True
    else:
        return False





"""
输入：两个等长的比特串。
输出：两个比特串的汉明距离，即不同位的长度。
"""
def err_count(bits0, bits1):
    err = 0
    assert bits0.shape == bits1.shape

    total = np.size(bits0)
    for i in range(total):
        if bits0[i] != bits1[i]:
            err += 1
    err_rate = err/total
    return err, err_rate


class  EdgeLDPC(object):
    def __init__(self):
        self.m_row_no = 0
        self.m_col_no = 0
        self.m_alpha  = [0, 0]
        self.m_beta   = [0, 0]
        self.m_v2c    = [0, 0]
        self.m_c2v    = [0, 0]

        self.left     = None
        self.right    = None
        self.up       = None
        self.down     = None
        return



# a = EdgeLDPC()
# a.m_row_no = 6
# a.m_col_no = 7
# aa = EdgeLDPC()
# aa.m_row_no = 12
# aa.m_col_no = 3

# a.right = aa


# b = EdgeLDPC()
# b.m_row_no = 4
# b.m_col_no = 9

# L = []

# L.append(a)
# L.append(b)














































































































































































