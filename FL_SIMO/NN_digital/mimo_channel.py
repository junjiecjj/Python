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

def forward(Tx_sig, H, Tx_data_power = None, SNR_dB = 5,):
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
    if Tx_data_power == None:
        Tx_data_power = np.mean(abs(Tx_sig)**2)
    noise_pwr = Tx_data_power*(10**(-1*SNR_dB/10))
    # noise = np.sqrt(noise_pwr/2.0) * ( np.random.randn((self.Nr, Tx_sig.shape[-1])) + 1j * np.random.randn((self.Nr, Tx_sig.shape[-1])) )
    noise = np.sqrt(noise_pwr/2) * (np.random.normal(loc=0.0, scale=1.0, size = ( Nr, Tx_sig.shape[-1])) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (Nr, Tx_sig.shape[-1])))
    Rx_sig =  H @ Tx_sig + noise
    return Rx_sig





























