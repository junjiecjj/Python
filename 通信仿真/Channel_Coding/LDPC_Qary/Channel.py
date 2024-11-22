#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:41:16 2023
@author: JunJie Chen
"""

import numpy as np
# from bitstring import BitArray
# from numpy.random import shuffle
from sklearn.metrics import pairwise_distances




def AWGN(K,  frame_len):
    H = np.ones((K, frame_len))
    return H


def QuasiStaticRayleigh(K, frame_len):
    H0 = (np.random.randn(K, ) + 1j * np.random.randn(K, ))/np.sqrt(2)
    H = np.expand_dims(H0, -1).repeat(frame_len, axis = -1)
    return H

def FastFadingRayleigh(K,  frame_len):
    H = (np.random.randn(K, frame_len) + 1j * np.random.randn(K, frame_len))/np.sqrt(2)
    return H

def LargeRician(K, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    hdLos = np.ones((K,))
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, ) + 1j * np.random.randn(K, ))
    h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
    h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    H = np.expand_dims(h_d, -1).repeat(frame_len, axis = -1)
    return H


def channelConfig(K, r = 100):
    C0 = -30                             # dB
    C0 = 10**(C0/10.0)                   # 参考距离的路损
    d0 = 1

    ## path loss exponents
    alpha_Au = 3.6

    ## Rician factor
    beta_Au = 3   # dB
    beta_Au = 10**(beta_Au/10)

    sigmaK2 = -60                        # dBm
    sigmaK2 = 10**(sigmaK2/10.0)/1000    # 噪声功率
    P0 = 30 # dBm
    P0 = 10**(P0/10.0)/1000

    # Location, Case II
    BS_locate = np.array([[0, 0, 10]])
    # radius = np.random.rand(K, 1) * r
    radius = (np.linspace(0.05, 1, K) * r).reshape(-1,1)
    angle = np.random.rand(K, 1) * 2 * np.pi
    users_locate_x = radius * np.cos(angle)
    users_locate_y = radius * np.sin(angle)
    users_locate_z = np.zeros((K, 1))
    users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

    ## Distance
    d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)

    ## generate path-loss
    PL_Au = C0 * (d_Au/d0)**(-alpha_Au)

    return BS_locate, users_locate, beta_Au, PL_Au
























































































































































































































































