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


import math
import numpy as np
from sklearn.metrics import pairwise_distances


def SVD_Precoding(hmat, power, d, Nt):
    """
        SVD precoding.

        Parameters
        ----------
        hmat: array(Nr, Nt). MIMO channel.
        power: float. Transmitting power constraint.
        d: int. data streams, d <= min(Nt/K, Nr).
        Returns
        ----------
        U: array(Nr, Nr). SVD decoding matrix.
        D: array(*, ). Singular value of hmat.
        V: array(Nt, d). SVD precoding matrix.
    """
    U, D, VH = np.linalg.svd(hmat, full_matrices = True)
    V = VH.conj().T[:, :d]
    V_norm = np.linalg.norm(V, ord = 'fro', ) # np.sqrt(np.trace(V.dot(V.conj().T)))
    # print(V_norm)
    V = V * math.sqrt(power) * np.sqrt(Nt) / V_norm  # power normalization
    return U, D, V


def SignalNorm(signal, M, mod_type='qam', denorm = False):
    """
        Signal power normalization and de-normalization.
        Parameters
            signal: array(*, ). Signal to be transmitted or received.
            M: int. Modulation order.
            mod_type: str, default 'qam'. Type of modulation technique.
            denorm: bool, default False. 0: Power normalization. 1: Power de-normalization.
        Returns
    """
    if mod_type == 'bpsk' or mod_type == 'qpsk' or mod_type == '8psk':
        Es = 1
    if mod_type == 'qam':
        if M == 8:
            Es = 6
        elif M == 32:
            Es = 25.875
        else: ##  https://blog.csdn.net/qq_41839588/article/details/135202875
            Es = 2 * (M - 1) / 3
    if not denorm:
        signal = signal / math.sqrt(Es)
    else:
        signal = signal * math.sqrt(Es)
    return signal, Es


def PassChannel(Tx_sig, H, power = None, SNR_dB = None, ):
    """
    Parameters
    ----------
    Tx_sig : 二维数组：Nt X 长度L
        DESCRIPTION.
    Tx_data_power : 发送功率
    SNR_dB :

    Returns
    -------
    Rx_sig : 接收信号

    """
    Nr = H.shape[0]
    if power == None:
        Tx_data_power = np.mean(abs(Tx_sig)**2)
    if SNR_dB != None:
        noise_pwr = Tx_data_power*(10**(-1*SNR_dB/10))
    elif SNR_dB == None:
        # sigmaK2 = -60                        # dBm
        noise_pwr = 1  # 10**(sigma2_dBm/10.0)/1000    # 噪声功率

    # noise = np.sqrt(noise_pwr/2.0) * ( np.random.randn((self.Nr, Tx_sig.shape[-1])) + 1j * np.random.randn((self.Nr, Tx_sig.shape[-1])) )
    noise = np.sqrt(noise_pwr/2) * (np.random.normal(loc=0.0, scale=1.0, size = ( Nr, Tx_sig.shape[-1])) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (Nr, Tx_sig.shape[-1])))
    Rx_sig =  H @ Tx_sig + noise
    return Rx_sig

def channelConfig(K):
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
    radius = np.random.rand(K, 1) * 100
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

























