#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:10 2023

@author: jack

np.mod(np.matmul(KbinRe, G), 2)


"""

import numpy as np

def HardDecision(TLLRs):
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
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的全0的码字，bits，如果yy[i]<0,则bits[i] = 1
"""
def bit_judge(data_in):
    bits = np.zeros(len(data_in), dtype = 'int')
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits


"""
输入：两个等长的比特串。
输出：两个比特串的汉明距离，即不同位的长度。
"""
def err_bit_count(bits0, bits1):
    number = 0
    for i in range(len(bits0)):
        if bits0[i] != bits1[i]:
            number += 1
    return number


def errorRate(c,y1):
    err=0
    total=np.size(c)
    i=0
    for i in range(total):
        if c[i] != y1[i]:
            err+=1
    r = err/total
    return r









































