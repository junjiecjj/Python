# -*- coding:utf-8 -*-
# @Time: 2023/5/28 23:29



"""
https://www.cnblogs.com/MayeZhang/p/12374196.html
https://www.zhihu.com/question/28698472#
https://blog.csdn.net/weixin_39274659/article/details/111477860
https://zhuyulab.blog.csdn.net/article/details/104434934
https://blog.csdn.net/UncleWa/article/details/123780502
https://zhuanlan.zhihu.com/p/627524436
"""


# import math
import numpy as np
from sklearn.metrics import pairwise_distances


def channelConfig(K, r = 100, rmin = 0.1):
    C0 = -30                             # dB
    C0 = 10**(C0/10.0)                   # 参考距离的路损
    d0 = 1

    ## path loss exponents
    alpha_Au = 3 # 3.76

    ## Rician factor
    beta_Au = 3.0   # dB
    beta_Au =  10**(beta_Au/10)

    # Location, Case II
    BS_locate = np.array([[0, 0, 10]])
    radius = np.random.rand(K, 1) * r
    # radius = (np.linspace(0.2, 1, K) * r).reshape(-1, 1)
    radius = np.random.uniform(rmin * r, r, size = (K, 1))
    # theta = (np.log10(r) - np.log10(r*0.1))/(K-1)
    # radius = np.log10(r*0.1) + np.linspace(0, (K-1)*theta,K)[:,None]
    # radius = 10**radius

    angle = np.random.rand(K, 1) * 2 * np.pi
    users_locate_x = radius * np.cos(angle)
    users_locate_y = radius * np.sin(angle)
    users_locate_z = np.zeros((K, 1))
    users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

    ## Distance
    d_Au = pairwise_distances(users_locate, BS_locate, metric = 'euclidean',)

    ## generate path-loss
    PL_Au = C0 * (d_Au/d0)**(-alpha_Au)

    return BS_locate, users_locate, beta_Au, PL_Au, d_Au

def Large_rician_block(K, frame_len, beta_Au, PL_Au, noisevar = 1):
    hdLos = np.sqrt(1/2) * (np.ones((K,)) + 1j * np.ones((K,)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, ) + 1j * np.random.randn(K, ))
    h_ds = np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos
    h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/noisevar))
    H = np.expand_dims(h_d, -1).repeat(frame_len, axis = -1)
    return H

def Large_rician_fast(K, frame_len, beta_Au, PL_Au, noisevar = 1):
    hdLos = np.sqrt(1/2) * (np.ones((frame_len, K)) + 1j * np.ones((frame_len, K)))
    hdNLos = np.sqrt(1/2) * (np.random.randn(frame_len, K) + 1j * np.random.randn(frame_len, K))
    h_ds = np.sqrt(beta_Au/(1 + beta_Au)) * hdLos + np.sqrt(1/(1 + beta_Au)) * hdNLos
    H = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/noisevar))
    return H.T

def Large_rayleigh_block(K, frame_len, PL_Au, noisevar = 1):
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, ) + 1j * np.random.randn(K, ))
    h_d = hdNLos @ np.diag(np.sqrt(PL_Au.flatten()/noisevar))
    H = np.expand_dims(h_d, -1).repeat(frame_len, axis = -1)
    return H

def Large_rayleigh_fast(K, frame_len, PL_Au, noisevar = 1):
    hdNLos = np.sqrt(1/2) * (np.random.randn(frame_len, K) + 1j * np.random.randn(frame_len, K))
    H = hdNLos @ np.diag(np.sqrt(PL_Au.flatten()/noisevar))
    return H.T









































































































































