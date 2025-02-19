#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:21 2025

@author: jack
"""

import numpy as np

from Channel import channelConfig
from Channel import AWGN_mac, BlockFading_mac, FastFading_mac, Large_mac







B = 4e6 # Hz
sigma2 = -60 # dBm/Hz
sigma2 = 10**(sigma2/10.0)/1000 # w/Hz
N0 = sigma2 * B # w

pmax = 30 # dBm
pmax = 10**(pmax/10.0)/1000 # Watts
pmax = 0.1      # Watts


K = 100
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(K, r = 100)














































































