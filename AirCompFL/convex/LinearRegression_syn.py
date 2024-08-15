#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:11:55 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt


m = 25;                     # workers
d = 100;
iter = 1002;               # iteration
lr = 0.05;
count = 1;
sigma2 = 0.25;

D = 500 * np.ones((m,1))
pi = D / sum(D)
batchsize = 500
SNR = 0 # dB















































