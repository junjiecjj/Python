# -*- coding:utf-8 -*-
# @Time: 2023/5/28 23:29



import math
import numpy as np
from sklearn.metrics import pairwise_distances

def AWGN(K, Nr, J, frame_len):
    H = np.ones((K, Nr, J, frame_len))
    return H

def QuasiStaticRayleigh(K, Nr, J, frame_len):
    H0 = (np.random.randn(K, Nr, J ) + 1j * np.random.randn(K, Nr, J ))/np.sqrt(2)
    H = np.expand_dims(H0, -1).repeat(frame_len, axis = -1)
    return H

def FastFadingRayleigh(K, Nr, J, frame_len):
    H = (np.random.randn(K, Nr, J, frame_len) + 1j * np.random.randn(K, Nr, J, frame_len))/np.sqrt(2)
    return H


def PassChannel(Tx_sig, noise_var = 1, ):
    noise = np.sqrt(noise_var/2.0) * ( np.random.randn(*Tx_sig.shape) + 1j * np.random.randn(*Tx_sig.shape) )
    Rx_sig =  Tx_sig + noise
    return Rx_sig

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
    P0 = 30                              # dBm
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

def Point2ULASteerVec(N, K, BS_locate, users_locate):
    XY = (users_locate - BS_locate)[:,:2]
    x = XY[:,0]
    y = XY[:,1]
    theta = -np.arctan2(y, x)
    d = np.arange(N)
    stevec = np.exp(1j * np.pi * np.outer(d, np.sin(theta)))
    return stevec

def Generate_hd(N, K, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    # User to AP/RIS channel
    hdLos = Point2ULASteerVec(N, K, BS_locate, users_locate)
    hdNLos = np.sqrt(1/2) * ( np.random.randn(N, K) + 1j * np.random.randn(N, K))
    h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
    h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    return h_d

def LargeRician(K, Nr, J, frame_len, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = 1):
    hdLos = Point2ULASteerVec(Nr, J, BS_locate, users_locate)
    hdLos = np.expand_dims(hdLos, 0).repeat(K, axis = 0)
    hdNLos = np.sqrt(1/2) * (np.random.randn(K, Nr, J) + 1j * np.random.randn(K, Nr, J))
    h_ds = (np.sqrt(beta_Au/(1+beta_Au)) * hdLos + np.sqrt(1/(1+beta_Au)) * hdNLos )
    for j in range(J):
        h_ds[:,:,j] *= np.sqrt(PL_Au[j,0]/sigma2)
    # h_d = h_ds @ np.diag(np.sqrt(PL_Au.flatten()/sigma2))
    H = np.expand_dims(h_ds, -1).repeat(frame_len, axis = -1)
    return H




































