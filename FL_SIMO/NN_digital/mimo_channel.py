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



class MIMO_Channel():
    def __init__(self, Nr = 2, Nt = 4, d = 2, P = 1, Tw = None, Th = None, Rw = None, Rh = None, mod_type='qam', ):
        # Base configs
        self.Nt = Nt  # transmit antenna
        # self.K = K  # users
        self.Nr = Nr  # receive antenna
        self.d = d  # data streams, d <= min(Nt/K, Nr)
        self.P = P  # power
        # self.M = M  # modulation order
        self.mod_type = mod_type  # modulation type

        # mmWave configs, 发射和接收为ULA
        # 假设有 N_cl 个散射簇，每个散射簇中包含 N_ray 条传播路径
        self.Ncl = 4  # clusters, 族群数目
        self.Nray = 6  # ray, 每个族中的路径数
        self.sigma_h = 0.3  # gain
        self.Tao = 0.001  # delay
        self.fd = 3  # maximum Doppler shift

        # mmWave configs, 发射和接收为UPA
        ##  Nt == Tw x Th
        self.Tw = Tw    # 发射阵面的天线长度
        self.Th = Th    # 发射阵面的天线宽度
        ##  Nr == Rw x Rh
        self.Rw = Rw    # 接收阵面的天线长度
        self.Rh = Rh    # 接收阵面的天线宽度
        self.H = None
        return

    ## 普通的瑞丽衰落信道模型，
    ## 当为ULA到ULA时，self.Nr为接收天线数，self.Nt为发射天线数，
    ## 当为UPA到UPA时，self.Nr为接收阵面的天线总数，self.Nt为发射阵面的总天线数，
    def circular_gaussian(self, ):
        """
            Circular gaussian MIMO channel.

            Parameters
            ----------
            Tx_sig: array(num_symbol, ). Modulated symbols.
            snr: int. SNR at the receiver side.
            Returns
            ----------
            Rx_sig: array(num_symbol, ). Decoded symbol at the receiver side.
        """
        self.H =  (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / math.sqrt(2)

        return

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

def channelConfig(args):
    K = args.num_of_clients

    C0 = -30                             # dB
    C0 = 10**(C0/10.0)                   # 参考距离的路损
    d0 = 1

    ## path loss exponents
    alpha_Au = 3.3

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























