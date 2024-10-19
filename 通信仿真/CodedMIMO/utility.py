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
# import numpy as np
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

##  yy --> 概率域: P(+1|y)
def yyToProb(yy, noise_var = 1.0):
    prob = np.zeros_like(yy)
    prob = 1.0 / (1.0 + np.exp(-2.0*yy/noise_var))
    return prob


"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的码字bits，如果yy[i]<0,则bits[i] = 1
"""
def bit_judge(data_in):
    bits = np.zeros(data_in.size, dtype = 'int')
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits


"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的码字bits，如果yy[i]<0,则bits[i] = 1
"""
def hard_decision(data_in):
    bits = np.zeros(data_in.size, dtype = np.int8)
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits




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















































































































































































