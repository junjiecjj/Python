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

    ## 5G 毫米波通信的信道模型, 发射和接收为 ULA
    def mmwave_MIMO_ULA2ULA(self, ):
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
            # phi = np.zeros(self.Ncl * self.Nray)   # 一共有L = Ncl*Nray条路径, (24,)
            a = np.zeros((self.Ncl * self.Nray, N, 1), dtype = complex)  # (24, 8, 1)

            # for i in range(self.Ncl * self.Nray):
            phi = np.random.uniform(-np.pi / 2, np.pi / 2, self.Ncl * self.Nray)  # 为每条路径产生随机的到达角 AoA 或出发角 AoD

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
        self.H =  H_gen()  # Nr * Nt
        # U, D, V = SVD_Precoding(H, self.P, self.d)
        # Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        # return self.H

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
        self.H =  (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt)) / math.sqrt(2 * self.Nr)

        return

    def mmwave_MIMO_UPA2UPA(self, ):
        """
            MIMO transmission procedure.
            Parameters
            Tx_sig: array(num_symbol, ). Modulated symbols.
            snr: int. SNR at the receiver side.
            Returns
            symbol_y: array(num_symbol, ). Decoded symbol at the receiver side.
        """
        def theta(W, H, Seed = 100):
            """
            Parameters
            W : int
                阵面的天线长度.
            H : int
                阵面的天线宽度，阵面天线总数 = W*H.
            Seed : int, optional
                DESCRIPTION. The default is 100.

            Returns
            a: 不同传播路径的空间特征。L x (W*H)
            PHI :
            """
            ## 方位角 (azimuth angle) 和 仰角 (elevation angle)
            azimuth   = np.random.uniform(-np.pi / 2, np.pi / 2, size = (int(self.Ncl * self.Nray), ))   # 一共有L = Ncl*Nray条路径, (24,)
            elevation = np.random.uniform(-np.pi / 8, np.pi / 8, size = (int(self.Ncl * self.Nray), ))   # 一共有L = Ncl*Nray条路径, (24,)
            a = np.zeros((self.Ncl * self.Nray, int(W*H), 1), dtype=complex)  # (24, 8, 1)

            for i in range(self.Ncl * self.Nray):
                for h in range(H):
                    for w in range(W):
                        k = h*W + w
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

        self.H =  H_gen()  # Nr * Nt
        # U, D, V = SVD_Precoding(H, self.P, self.d)
        # Rx_sig = self.trans_procedure(Tx_sig, H, V, D, U, snr)
        # return self.H


    def forward(self, Tx_sig, Tx_data_power = None, SNR_dB = 5,):
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
        if Tx_data_power == None:
            Tx_data_power = np.mean(abs(Tx_sig)**2)
        noise_pwr = Tx_data_power*(10**(-1*SNR_dB/10))
        # noise = np.sqrt(noise_pwr/2.0) * ( np.random.randn((self.Nr, Tx_sig.shape[-1])) + 1j * np.random.randn((self.Nr, Tx_sig.shape[-1])) )
        noise = np.sqrt(noise_pwr/2) * (np.random.normal(loc=0.0, scale=1.0, size = (self.Nr, Tx_sig.shape[-1])) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (self.Nr, Tx_sig.shape[-1])))
        Rx_sig = self.H @ Tx_sig + noise
        return Rx_sig

    def SVD_Precoding_transceiver(self, Tx_sig, P, snr = 3):
        def trans_procedure( Tx_sig, H, V, D, U, snr = 20):
            """
                MIMO transmission procedure.
                Parameters
                    Tx_sig: array(num_symbol, ). Modulated symbols.
                    H: array(Nr, Nt). MIMO Channel matrix.
                    V: array(Nt, d). Precoding matrix.
                    D: array(*, ). Singular value of H.
                    U: array(Nr, Nr). decoding matrix.
                    snr: int. SNR at the receiver side.
                Returns
                    symbol_y: array(num_symbol, ). Decoded symbol at the receiver side.
            """
            sigma2 = self.P * 10 ** (-snr / 10)
            total_num = len(Tx_sig)   # 480
            if total_num % self.d != 0:
                Tx_sig = np.pad(Tx_sig, (0, self.d - total_num % self.d), constant_values=(0, 0)) # (6668,)
            tx_times = np.ceil(total_num / self.d).astype(int) # 240
            symbol_group = Tx_sig.reshape(self.d, tx_times)  # (2, 240)
            symbol_x = SignalNorm(symbol_group, self.M, mod_type=self.mod_type, denorm = False) # (2, 240)

            noise = np.sqrt(sigma2 / 2) * (np.random.randn(self.Nr, tx_times) + 1j * np.random.randn(self.Nr, tx_times))
            y = H@V@symbol_x + noise  # y = HVx+n, (Nr, tx_times)  (6, 240)

            DigD = np.zeros(self.H.T.shape, dtype = complex)  # (4, 6)
            DigD[np.diag_indices(self.Nr)] = 1/D    # (4, 6)
            # y_de = DigD.dot(U.conj().T).dot(y) / np.sqrt(self.P)
            y_de = DigD@(U.conj().T)@y / np.sqrt(self.P)   # (4, 240))
            y_de = y_de[:self.d]                           # (2, 240)
            symbol_y = SignalNorm(y_de, self.M, mod_type=self.mod_type, denorm = True).flatten()[:total_num]
            return symbol_y

        U, D, V = SVD_Precoding(self.H, self.P, self.d)
        Rx_sig =  trans_procedure(Tx_sig, self.H, V, D, U, snr)
        return Rx_sig


# channel = MIMO_Channel(Nr = 24, Nt = 24, d = 2, P = 1, Tw = 4, Th = 6, Rw = 6, Rh = 4)
# # ula = channel.mmwave_MIMO_ULA2ULA()
# upa = channel.mmwave_MIMO_UPA2UPA()


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





























