#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:41:16 2023

@author: jack
"""

import numpy as np
from bitstring import BitArray
from numpy.random import shuffle


noise_var = lambda snr_in_db: 10 ** (-snr_in_db / 10)


class Channel:
    def __init__(self, snr_in_db):
        self.std_dev = np.sqrt(noise_var(snr_in_db))

    def send(self, x):  # incoming cw {0,1}, outgoing cw {-1,+1}
        return (2 * x - 1) + np.random.normal(0, self.std_dev, x.shape)

def lines_to_array(lines):
    return [list(map(int, x.split(' '))) for x in lines]













