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



def forward(Tx_sig, H, power = None, SNR_dB = None, ):
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
    # else:
    #     raise ValueError("信噪比和噪声方差同时只能给定一个")
    # noise = np.sqrt(noise_pwr/2.0) * ( np.random.randn((self.Nr, Tx_sig.shape[-1])) + 1j * np.random.randn((self.Nr, Tx_sig.shape[-1])) )
    noise = np.sqrt(noise_pwr/2) * (np.random.normal(loc=0.0, scale=1.0, size = ( Nr, Tx_sig.shape[-1])) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (Nr, Tx_sig.shape[-1])))
    Rx_sig =  H @ Tx_sig + noise
    return Rx_sig

def AWGN_scma(K, J, frame_len):
    H = np.ones((K, J, frame_len))
    return H

def BlockFading_scma(K, J, frame_len):
    H0 = (np.random.randn(K, J ) + 1j * np.random.randn(K, J ))/np.sqrt(2)
    H = np.expand_dims(H0, 2).repeat(frame_len, axis = 2)
    return H

def FastFading_scma(K, J, frame_len):
    H = (np.random.randn(K, J, frame_len) + 1j * np.random.randn(K, J, frame_len))/np.sqrt(2)
    return H

def FastFading_Mac(K, frame_len):
    H = (np.random.randn(K, frame_len) + 1j * np.random.randn(K, frame_len))/np.sqrt(2)
    return H














