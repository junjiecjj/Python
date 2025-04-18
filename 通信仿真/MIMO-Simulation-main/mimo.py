# -*- coding:utf-8 -*-
# @Time: 2023/5/28 23:29



"""
https://github.com/ZJU-IICNS-AICOMM/MIMO-Simulation

https://www.cnblogs.com/MayeZhang/p/12374196.html
https://www.zhihu.com/question/28698472#!
https://blog.csdn.net/weixin_39274659/article/details/111477860
https://zhuyulab.blog.csdn.net/article/details/104434934
https://blog.csdn.net/UncleWa/article/details/123780502


线性天线阵列（Uniform Linear Array，ULA）
方形天线阵列（Uniform Planar Array，UPA）：
方位角(azimuth angle)，仰角 (elevation angle)

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
    U, D, VH = np.linalg.svd(hmat, full_matrices = True)
    V = VH.conj().T[:, :d]
    V_norm = np.linalg.norm(V, ord = 'fro', ) # np.sqrt(np.trace(V.dot(V.conj().T)))
    print(V_norm)
    V = V * math.sqrt(power) # / V_norm  # power normalization
    return U, D, V


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
    """
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
    return signal


class MIMO_Channel():
    def __init__(self, Nr = 2, Nt = 4, d = 2, K = 1, P = 1, M = 16, mod_type='qam', Tw = None, Th = None, Rw = None, Rh = None, ):
        # Base configs
        self.Nt = Nt  # transmit antenna
        self.K = K  # users
        self.Nr = Nr  # receive antenna
        self.d = d  # data streams, d <= min(Nt/K, Nr)
        self.P = P  # power
        self.M = M  # modulation order
        self.mod_type = mod_type  # modulation type

        # mmWave configs, 发射和接收为ULA
        # Nt = 32         # T antennas
        # Nr = 16         # R antennas
        # self.NtRF = 4  # RF chains at the transmitter
        # self.NrRF = 4  # RF chains at the receiver
        # 假设有 N_cl 个散射簇，每个散射簇中包含 N_ray 条传播路径
        self.Ncl = 4  # clusters, 族群数目
        self.Nray = 6  # ray, 每个族中的路径数
        self.sigma_h = 0.3  # gain
        self.Tao = 0.001  # delay
        self.fd = 3  # maximum Doppler shift

        # mmWave configs, 发射和接收为UPA
        # 假设有 N_cl 个散射簇，每个散射簇中包含 N_ray 条传播路径
        # self.Ncl = 4   # clusters, 族群数目
        # self.Nray = 6  # ray, 每个族中的路径数
        ##  Nt == Tw x Th
        self.Tw = Tw    # 发射阵面的天线长度
        self.Th = Th    # 发射阵面的天线宽度
        ##  Nr == Rw x Rh
        self.Rw = Rw    # 接收阵面的天线长度
        self.Rh = Rh    # 接收阵面的天线宽度
        return

    def trans_procedure(self, Tx_sig, H, V, D, U, snr = 20):
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
            Tx_sig = np.pad(Tx_sig, (0, self.d - total_num % self.d), constant_values=(0, 0)) # (6668,)
        tx_times = np.ceil(total_num / self.d).astype(int) # 3334
        symbol_group = Tx_sig.reshape(self.d, tx_times)  # (2, 3334)
        print(f"symbol_group power = {np.mean(np.abs(symbol_group)**2)}")
        symbol_x = SignalNorm(symbol_group, self.M, mod_type=self.mod_type, denorm = False) # (2, 3334)
        print(f"symbol_x power = {np.mean(np.abs(symbol_x)**2)}")

        noise = np.sqrt(sigma2 / 2) * (np.random.randn(self.Nr, tx_times) + 1j * np.random.randn(self.Nr, tx_times))
        print(f"noise power = {np.mean(np.abs(noise)**2)}, {sigma2}")
        y = H.dot(V).dot(symbol_x) + noise  # y = HVx+n, (Nr, tx_times)
        print(f"y power = {np.mean(np.abs(y)**2)}")
        y_de = np.diag(1 / D).dot(U.conj().T).dot(y) / np.sqrt(self.P)
        # print(f"{np.diag(1 / D).shape}, {U.conj().T.shape}")
        y_de = y_de[:self.d]
        print(f"y_de power = {np.mean(np.abs(y_de)**2)}")
        symbol_y = SignalNorm(y_de, self.M, mod_type=self.mod_type, denorm=True).flatten()[:total_num]
        print(f"symbol_y power = {np.mean(np.abs(symbol_y)**2)}")
        return symbol_y


    ## 5G 毫米波通信的信道模型, 发射和接收为 ULA
    def mmwave_MIMO_ULA2ULA(self, Tx_sig, snr = 10):
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
        def theta(N, Seed=100):
            phi = np.zeros(self.Ncl * self.Nray)   # 一共有L = Ncl*Nray条路径, (24,)
            a = np.zeros((self.Ncl * self.Nray, N, 1), dtype = complex)  # (24, 8, 1)

            for i in range(self.Ncl * self.Nray):
                phi[i] = np.random.uniform(-np.pi / 2, np.pi / 2)  # 为每条路径产生随机的到达角 AoA 或出发角 AoD

            for j in range(self.Ncl * self.Nray):
                for z in range(N):  # N为发射天线数或者接收天线数
                    ## λ 是波长，d是天线间隔， 由于一般都设置有d = 0.5λ，因此这里没有出现d和λ； 这是默认天线以半波长为间隔。
                    ## 如果你看到的指数项是类似于e^{j*2*pi*d/λ}之类的形式，其实是一样的，因为这里d = λ/2。
                    a[j][z]  = np.exp(1j * np.pi * z * np.sin(phi[j]))
            PHI = phi.reshape(self.Ncl * self.Nray)
            return a / np.sqrt(N), PHI

        # https://www.cnblogs.com/MayeZhang/p/12374196.html
        def H_gen( Seed = 100):
            # complex gain, 第i个族中第l条路径的复合增益，看成复高斯分布
            alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray)) + 1j * np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray))
            # receive and transmit array response vectors
            ar, ThetaR =  theta(self.Nr, Seed + 10000)
            at, ThetaT =  theta(self.Nt, Seed)
            H = np.zeros((self.Nr, self.Nt), dtype=complex)
            l = 0
            for i in range(self.Ncl):
                for j in range(self.Nray):
                    H += alpha_h[l] * ( ar[l] @ (at[l].T.conjugate()))
                    ## channel with delay
                    # H += alpha_h[l] * ( ar[l]@ (at[l].T.conjugate())) * np.exp(1j*2*np.pi*self.Tao*self.fd*np.cos(ThetaR[l]))
                    l += 1
            H = np.sqrt(self.Nt * self.Nr / self.Ncl * self.Nray) * H
            return H
        H =  H_gen()  # Nr * Nt
        U, D, V = SVD_Precoding(H, self.P, self.d)
        print(f"V power = {np.mean(np.abs(V)**2)}")
        Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        return Rx_sig


    ## 普通的瑞丽衰落信道模型，
    ## 当为ULA到ULA时，self.Nr为接收天线数，self.Nt为发射天线数，
    ## 当为UPA到UPA时，self.Nr为接收阵面的天线总数，self.Nt为发射阵面的总天线数，
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


    def mmwave_MIMO_UPA2UPA(self, Tx_sig, snr = 10):
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
        def theta(W, H, Seed = 100):
            """
            Parameters
            ----------
            W : int
                阵面的天线长度.
            H : int
                阵面的天线宽度，阵面天线总数 = W*H.
            Seed : int, optional
                DESCRIPTION. The default is 100.

            Returns
            -------
            a: 不同传播路径的空间特征。L x (W*H)
            PHI :

            """
            ## 方位角 (azimuth angle) 和 仰角 (elevation angle)
            azimuth   = np.random.uniform(-np.pi / 2, np.pi / 2, size = (int(self.Ncl * self.Nray), ))   # 一共有L = Ncl*Nray条路径, (24,)
            elevation = np.random.uniform(-np.pi / 8, np.pi / 8, size = (int(self.Ncl * self.Nray), ))   # 一共有L = Ncl*Nray条路径, (24,)

            a = np.zeros((self.Ncl * self.Nray, int(W*H), 1), dtype=complex)  # (24, 8, 1)


            for i in range(self.Ncl * self.Nray):
                for w in range(W):
                    for h in range(H):
                        k = w*H + h
                        ## 如果你看到的指数项是类似于e^{j*2*pi*d/λ}之类的形式，其实是一样的，因为这里d = λ/2。
                        a[i][k] = np.exp(1j * np.pi * (w * np.sin(azimuth[i]) * np.cos(elevation[i]) + h*np.sin(elevation[i])))
            Azimuth = azimuth.reshape(self.Ncl * self.Nray)
            Elevation = elevation.reshape(self.Ncl * self.Nray)
            return a / np.sqrt(W*H), Azimuth, Elevation

        # https://www.cnblogs.com/MayeZhang/p/12374196.html
        def H_gen( Seed = 100):
            # complex gain, 第i个族中第l条路径的复合增益，看成复高斯分布
            alpha_h = np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray)) + 1j * np.random.normal(0, self.sigma_h, (self.Ncl * self.Nray))
            # receive and transmit array response vectors
            ar, phiR, thetaR =  theta(self.Rw, self.Rh, Seed + 10000)
            at, phiT, thetaT =  theta(self.Tw, self.Th, Seed)
            H = np.zeros((self.Rw*self.Rh, self.Tw*self.Th), dtype=complex)
            print(ar.shape, at.shape)
            l = 0
            for i in range(self.Ncl):
                for j in range(self.Nray):
                    H += alpha_h[l] * np.dot(ar[l], np.conjugate(at[l]).T)
                    ## channel with delay
                    # H += alpha_h[l] * np.dot(ar[l], np.conjugate(at[l]).T)*np.exp(1j * 2 * np.pi * self.Tao * self.fd * np.cos(phiR[l]))
                    l += 1
            H = np.sqrt(self.Tw * self.Th * self.Rw * self.Rh / self.Ncl * self.Nray) * H
            return H


        H =  H_gen()  # Nr * Nt
        U, D, V = SVD_Precoding(H, self.P, self.d)
        Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        return Rx_sig
        return








