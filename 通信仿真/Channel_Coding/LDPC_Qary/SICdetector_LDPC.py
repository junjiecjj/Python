#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:13:05 2024

@author: jack
"""

# import galois
import numpy as np
import copy
import sys, os
import itertools
# from functools import reduce
import commpy as comm


def bpsk(bins):
    bits = copy.deepcopy(bins)
    bits[np.where(bits == 1)] = -1
    bits[np.where(bits == 0)] = 1
    return bits


def SeparatedDetectDecoding(H, yy, ldpc,  maxiter = 50):



    return




















































