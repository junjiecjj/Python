#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:52:42 2025

@author: jack
"""

import numpy as np


def add_awgn_noise(x, snrdB, L = 1):
    snr = 10**(snrdB/10)
    P = L * np.sum(np.abs(x)**2)/x.size
    N0 = P/snr
    if x.dtype == 'complex':
        n = np.sqrt(N0/2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    elif x.dtype == 'float' or x.dtype == 'int' :
        n = np.sqrt(N0/2) * np.random.randn(1, x.shape)
    r = x + n
    return r, N0














