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














































































































































































