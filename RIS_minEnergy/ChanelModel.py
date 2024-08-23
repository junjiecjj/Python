#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:58:07 2024

@author: jack

https://www.cnblogs.com/MayeZhang/p/12374196.html
https://www.zhihu.com/question/28698472#
https://blog.csdn.net/weixin_39274659/article/details/111477860
https://zhuyulab.blog.csdn.net/article/details/104434934
https://blog.csdn.net/UncleWa/article/details/123780502
https://zhuanlan.zhihu.com/p/627524436
"""


import math
import numpy as np



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
def mmwave_MIMO_ULA2ULA(Nr = 2, Nt = 4, Ncl = 4, Nray = 6):
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
        a = np.zeros((Ncl * Nray, N, 1), dtype = complex)  # (24, 8, 1)

        # for i in range(self.Ncl * self.Nray):
        phi = np.random.uniform(-np.pi / 2, np.pi / 2, Ncl * Nray)  # 为每条路径产生随机的到达角 AoA 或出发角 AoD

        for j in range( Ncl * Nray):
            for z in range(N):  # N为发射天线数或者接收天线数
                ## λ 是波长，d是天线间隔， 由于一般都设置有d = 0.5λ，因此这里没有出现d和λ； 这是默认天线以半波长为间隔。
                ## 如果你看到的指数项是类似于e^{j*2*pi*d/λ}之类的形式，其实是一样的，因为这里d = λ/2。
                a[j][z]  = np.exp(1j * np.pi * z * np.sin(phi[j]))
        PHI = phi.reshape( Ncl * Nray)
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
    self.H = 1 / math.sqrt(2) * (np.random.randn(self.Nr, self.Nt) + 1j * np.random.randn(self.Nr, self.Nt))

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


# channel = MIMO_Channel(Nr = 24, Nt = 24, d = 2, P = 1, Tw = 4, Th = 6, Rw = 6, Rh = 4)
# # ula = channel.mmwave_MIMO_ULA2ULA()
# upa = channel.mmwave_MIMO_UPA2UPA()






