#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:44 2023
@author: JunJie Chen
"""

import numpy as np


def BPSK(c):
    # for i in range(cc.shape[-1]):
    #     if cc[i] == 0:
    #         cc[i] = 1
    #     else:
    #         cc[i] = -1
    c = 1 - 2*c
    return c



def demodu_BPSK(y):
    for i in range(y.shape[-1]):
        if y[i] > 0:
            y[i] = 0
        else:
            y[i] = 1
    return y






















































































































































