#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:41:16 2023
@author: JunJie Chen
"""

import numpy as np
# from bitstring import BitArray
from numpy.random import shuffle
from typing import List, Sequence, TypeVar, Union, Dict, Tuple

_list_float = List[float]
_array = Union[List[float], List[int], np.array]


noise_var = lambda snr_in_db: 10.0 ** (-snr_in_db / 10.0)

class AWGN(object):
    def __init__(self, snr_in_db:float) -> float:
        self.noise_var = noise_var(snr_in_db)
        self.noise_std = np.sqrt(noise_var(snr_in_db))

    def forward(self, cc:_array) -> _array:
        # if type(cc) == list:
        #     print("cc is list")
        #     shape = len(cc)
        # elif type(cc) == np.ndarray:
        #     shape = cc.shape
        return cc + np.random.normal(0, self.noise_std, size = cc.shape )


def lines_to_array(lines):
    return [list(map(int, x.split(' '))) for x in lines]




class Rayleigh(object):
    def __init__(self, snr_in_db):
        self.noise_var = noise_var(snr_in_db)
        self.noise_std = np.sqrt(noise_var(snr_in_db))

    def forward(self, cc):
        shape = cc.shape
        sigma = math.sqrt(1 / 2)
        H = np.random.normal(0.0, sigma, size=shape) + 1j * np.random.normal(0.0, sigma, size=shape)
        Tx_sig = tx_signal * H
        Rx_sig = channel_Awgn(Tx_sig, snr, output_power=output_power)
        # Channel estimation
        Rx_sig = Rx_sig / H
        return Rx_sig














































































































































































































































