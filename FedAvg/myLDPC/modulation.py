#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:44 2023

@author: jack
"""

import numpy as np



def QPSK(cc):
    for i in range(len(cc)):
        # print(f"int(cc[{i}] = {int(cc[i])}")
        if int(cc[i]) == 1:
            cc[i] = -1.0
        elif int(cc[i]) == 0:
            cc[i] = 1.0
    return cc

# BPSK(c for codeword)
def modulate(c):
    for i in range(np.size(c)):
        if c[i] == 0:
            c[i] = 1
        else:
            c[i] = -1
    return c

def demodulate(y):
    for i in range(np.size(y)):
        if y[i] > 0:
            y[i] = 0
        else:
            y[i] = 1
    return y
