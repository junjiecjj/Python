#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:25:18 2024

@author: jack

信号在传输过程中由于多径效应，接收信号是直射信号（主信号）和多径信号的叠加，此时接收信号的包络服从莱斯分布。事实上，在考虑多径效应的时候，原始信号与呈现瑞利分布的多径分量的和常常被描述为莱斯分布。


https://blog.csdn.net/weixin_43935696/article/details/114648768
https://github.com/Jonathan-Browning/Rician-Fading-Python

"""


import numpy as np
from scipy.stats import gaussian_kde as kdf
from scipy import special as sp

#%%
# 2.1 Rician衰落类
class Rice:
    # numSamples = 2*(10**6)  # the number of samples used in the simulation
    # numSamples = 64 #表示要产生的随机数的长度
    r = np.linspace(0, 6, 6000) # theoretical envelope PDF x axes
    theta = np.linspace(-np.pi, np.pi, 6000)    # theoretical phase PDF x axes

    def __init__(self, K, r_hat_2, phi, numSamples):
        # # user input checks and assigns value
        self.K = self.input_Check(K, "K", 0, 50)
        self.r_hat_2 = self.input_Check(r_hat_2, "\hat{r}^2", 0.5, 2.5)
        self.phi = self.input_Check(phi, "\phi", -np.pi, np.pi)
        self.numSamples = numSamples
        # user input checks and assigns value

        # simulating and theri densities
        self.multipathFading = self.complex_Multipath_Fading()
        self.xdataEnv, self.ydataEnv = self.envelope_Density()
        self.xdataPh, self.ydataPh = self.phase_Density()

        # theoretical PDFs calculated
        self.envelopeProbability = self.envelope_PDF()
        self.phaseProbability = self.phase_PDF()

	# 输入格式检查
    def input_Check(self, data, inputName, lower, upper):
        # input_Check checks the user inputs
        # has a value been entered
        if data == "":
            raise ValueError(" ".join((inputName, "must have a numeric value")))
        # incase of an non-numeric input
        try:
            data = float(data)
        except:
            raise ValueError(" ".join((inputName, "must have a numeric value")))
        # data must be within the range
        if data < lower or data > upper:
            raise ValueError(" ".join((inputName, f"must be in the range [{lower:.2f}, {upper:.2f}]")))
        return data

    def calculate_Means(self):
        # calculate_means calculates the means of the complex Gaussians representing the
        # in-phase and quadrature components
        p = np.sqrt(self.K * self.r_hat_2 / (1+self.K)) * np.cos(self.phi)
        q = np.sqrt(self.K * self.r_hat_2 / (1+self.K)) * np.sin(self.phi)
        return p, q

    def scattered_Component(self):
        # scattered_Component calculates the power of the scattered signal component
        sigma = np.sqrt(self.r_hat_2 / ( 2 * (1+self.K) ) )
        return sigma

    def generate_Gaussians(self, mean, sigma):
        # generate_Gaussians generates the Gaussian random variables
        gaussians = np.random.default_rng().normal(mean, sigma, self.numSamples)
        return gaussians

    def complex_Multipath_Fading(self):
        # complex_Multipath_Fading generates the complex fading random variables
        p, q = self.calculate_Means()
        sigma = self.scattered_Component()
        multipathFading = self.generate_Gaussians(p, sigma) + (1j*self.generate_Gaussians(q, sigma))

        return multipathFading

    def envelope_PDF(self):
        # envelope_PDF calculates the theoretical envelope PDF
        PDF = 2 * (1+self.K) * self.r / self.r_hat_2 * np.exp(- self.K - ((1+self.K)*self.r**2)/self.r_hat_2) * np.i0(2 * self.r * np.sqrt(self.K*(1+self.K)/self.r_hat_2))
        return PDF

    def phase_PDF(self):
        # phase_PDF calculates the theoretical phase PDF
        def q_func(x):
        # Q-function
            return 0.5-0.5*sp.erf(x/np.sqrt(2))
        PDF = (1/(2*np.pi))* np.exp(- self.K) * (1 + (np.sqrt(4*np.pi*self.K) * np.exp(self.K * (np.cos(self.theta-self.phi))**2) *np.cos(self.theta-self.phi)) * (1 - q_func(np.sqrt(2*self.K) * np.cos(self.theta-self.phi))))
        return PDF


    def envelope_Density(self):
        # envelope_Density finds the envelope PDF of the simulated random variables
        R = np.sqrt((np.real(self.multipathFading))**2 + (np.imag(self.multipathFading))**2)
        kde = kdf(R)
        x = np.linspace(R.min(), R.max(), 100)
        p = kde(x)

        return x, p

    def phase_Density(self):
        # phase_Density finds the phase PDF of the simulated random variables
        R = np.angle(self.multipathFading)
        kde = kdf(R)
        x = np.linspace(R.min(), R.max(), 100)
        p = kde(x)
        return x, p



# from rice import *
K = 1 # K因子
hat = 1 # 振幅
phi = 1 # 相位
numSamples = 16	# 随机样本的长度
s = Rice(1,1,1,16)
# rician衰落随机数
print(s.multipathFading)



# 2.2 Ricican信道

# signal输入的信号
# channelResponse rician衰落随机数,就是以上代码求的s.multipathFading
# SNRdb,信噪比
def rician_channel(signal, channelResponse, SNRdb):
	# 卷积
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    # 噪声
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise














