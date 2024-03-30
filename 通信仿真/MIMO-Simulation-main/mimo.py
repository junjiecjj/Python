# -*- coding:utf-8 -*-
# @Time: 2023/5/28 23:29



"""

https://www.zhihu.com/question/28698472#!
https://blog.csdn.net/weixin_39274659/article/details/111477860
https://zhuyulab.blog.csdn.net/article/details/104434934
https://blog.csdn.net/UncleWa/article/details/123780502


线性天线阵列（Uniform Linear Array，ULA）
方形天线阵列（Uniform Planar Array，UPA）：


2D MIMO 通信系统发射天线是线性天线，它形成的波束较宽，只有水平维度的方向，没有垂直维度的方向。这样每条子径包含发射端的出发角AoD（Angle of Departure），接收端的到达角AoA（Angle of Arrival）以及时延三个特征变量。

"""


import math
import numpy as np


def SVD_Precoding(hmat, power, d):
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
        W_svd: array(Nt, d). SVD precoding matrix.
    """
    U, D, V = np.linalg.svd(hmat, full_matrices=True)
    W_svd = V.conj().T[:, :d]
    W_svd_norm = np.sqrt(np.trace(W_svd.dot(W_svd.conj().T)))
    print(W_svd_norm)
    W_svd = W_svd * math.sqrt(power) / W_svd_norm  # power normalization
    return U, D, W_svd


def SignalNorm(signal, M, mod_type='qam', denorm=False):
    """
        Signal power normalization and de-normalization.

        Parameters
        ----------
        signal: array(*, ). Signal to be transmitted or received.
        M: int. Modulation order.
        mod_type: str, default 'qam'. Type of modulation technique.
        denorm: bool, default False. 0: Power normalization. 1: Power de-normalization.
        Returns
        ----------
        W_svd: array(Nt, Nt). SVD precoding matrix.
        D: array(*, ). Singular value of hmat.
        M_svd: array(Nr, Nr). SVD decoding matrix.
    """
    if mod_type == 'qam':
        if M == 8:
            Es = 6
        elif M == 32:
            Es = 25.875
        else:
            Es = 2 * (M - 1) / 3
    if not denorm:
        signal = signal / math.sqrt(Es) * math.sqrt(2)
    else:
        signal = signal * math.sqrt(Es)
    return signal


class MIMO_Channel():
    def __init__(self, Nr=2, Nt=4, d=2, K=1, P=1, M=16, mod_type='qam'):
        # Base configs
        self.Nt = Nt  # transmit antenna
        self.K = K  # users
        self.Nr = Nr  # receive antenna
        self.d = d  # data streams, d <= min(Nt/K, Nr)
        self.P = P  # power
        self.M = M  # modulation order
        self.mod_type = mod_type  # modulation type

        # mmWave configs
        # Nt = 32         # T antennas
        # Nr = 16         # R antennas
        # self.NtRF = 4  # RF chains at the transmitter
        # self.NrRF = 4  # RF chains at the receiver
        # 假设有 N_cl 个散射簇，每个散射簇中包含 N_ray 条传播路径
        self.Ncl = 4  # clusters
        self.Nray = 6  # ray
        self.sigma_h = 0.3  # gain
        self.Tao = 0.001  # delay
        self.fd = 3  # maximum Doppler shift

    def trans_procedure(self, Tx_sig, H, V, D, U, snr=20):
        """
            MIMO transmission procedure.

            Parameters
            ----------
            Tx_sig: array(num_symbol, ). Modulated symbols.
            H: array(Nr, Nt). MIMO Channel matrix.
            V: array(Nt, d). Precoding matrix.
            D: array(*, ). Singular value of H.
            U: array(Nr, Nr). decoding matrix.
            snr: int. SNR at the receiver side.
            Returns
            ----------
            symbol_y: array(num_symbol, ). Decoded symbol at the receiver side.
        """
        sigma2 = self.P * 10 ** (-snr / 10)
        total_num = len(Tx_sig)
        if total_num % self.d != 0:
            Tx_sig = np.pad(Tx_sig, (0, self.d - total_num % self.d), constant_values=(0, 0))
        tx_times = np.ceil(total_num / self.d).astype(int)
        symbol_group = Tx_sig.reshape(self.d, tx_times)
        symbol_x = SignalNorm(symbol_group, self.M, mod_type=self.mod_type, denorm=False)

        noise = np.sqrt(sigma2 / 2) * (np.random.randn(self.Nr, tx_times) + 1j * np.random.randn(self.Nr, tx_times))
        y = H.dot(V).dot(symbol_x) + noise  # y = HVx+n, (Nr, tx_times)
        y_de = np.diag(1 / D).dot(U.conj().T).dot(y) / np.sqrt(self.P)
        y_de = y_de[:self.d]
        symbol_y = SignalNorm(y_de, self.M, mod_type=self.mod_type, denorm=True).flatten()[:total_num]
        return symbol_y

    def circular_gaussian(self, Tx_sig, snr=10):
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
        H = 1 / math.sqrt(2) * (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt))
        U, D, V = SVD_Precoding(H, self.P, self.d)
        Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        return Rx_sig

    def theta(self, N, Seed=100):
        phi = np.zeros(self.Ncl * self.Nray)  # azimuth AoA and AoD
        a = np.zeros((self.Ncl * self.Nray, N, 1), dtype=complex)

        for i in range(self.Ncl * self.Nray):
            phi[i] = np.random.uniform(-np.pi / 3, np.pi / 3)
        f = 0
        for j in range(self.Ncl * self.Nray):
            f += 1
            for z in range(N):
                a[j][z] = np.exp(1j * np.pi * z * np.sin(phi[f - 1]))
        PHI = phi.reshape(self.Ncl * self.Nray)
        return a / np.sqrt(N), PHI

    # https://www.cnblogs.com/MayeZhang/p/12374196.html
    def H_gen(self, Seed=100):
        # complex gain
        alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray)) + 1j * np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray))
        # receive and transmit array response vectors
        ar, ThetaR = self.theta(self.Nr, Seed + 10000)
        at, ThetaT = self.theta(self.Nt, Seed)
        H = np.zeros((self.Nr, self.Nt), dtype=complex)
        fff = 0
        for i in range(self.Ncl):
            for j in range(self.Nray):
                # H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)
                H += alpha_h[fff] * np.dot(ar[fff], np.conjugate(at[fff]).T)*np.exp(1j*2*np.pi*self.Tao*self.fd*np.cos(ThetaR[fff]))    # channel with delay
                fff += 1
        H = np.sqrt(self.Nt * self.Nr / self.Ncl * self.Nray) * H
        return H

    def mmwave_MIMO(self, Tx_sig, snr=10):
        """
            MIMO transmission procedure.

            Parameters
            ----------
            Tx_sig: array(num_symbol, ). Modulated symbols.
            snr: int. SNR at the receiver side.
            Returns
            ----------
            symbol_y: array(num_symbol, ). Decoded symbol at the receiver side.
        """
        H = self.H_gen()  # Nr * Nt
        U, D, V = SVD_Precoding(H, self.P, self.d)
        Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        return Rx_sig















