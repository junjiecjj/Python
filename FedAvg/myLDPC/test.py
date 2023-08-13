#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:32:51 2023

@author: jack
"""

import sys, os
import numpy as np




enc_H = None

with open('PEG1024regular0.5.txt', 'r', encoding='utf-8') as f:
    tmp = f.readline()
    print(tmp)
    tmp = f.readline()
    print(tmp)
    rows, cols, chk = [int(i) for i in tmp.strip().split()]
    enc_H = np.zeros((rows, cols), dtype = np.int64)
    tmp = f.readline()
    print(tmp)
    while 1:
        tmp = f.readline()
        if not tmp:
            break
        row_dt = [int(i) for i in tmp.strip().split()]
        for i in range(row_dt[1]):
            enc_H[row_dt[0], row_dt[i+2]] = 1








































































































































































