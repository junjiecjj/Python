#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:12:50 2024

@author: jack
"""

import numpy as np




def Quantize_1bit(mess_lst):
    mes_return = []
    mess_lst1 = [np.sign(mess) for mess in mess_lst]
    for mes in mess_lst1:
        mes[np.where(mes == 0)] = np.random.choice([-1, 1], size = np.where(mes == 0)[0].shape, replace = True, p = [0.5, 0.5] )
        mes_return.append(mes / 2)
    return mes_return







