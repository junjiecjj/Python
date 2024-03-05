#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:23:50 2024

@author: jack

"""


#%% https://blog.csdn.net/qq_45889056/article/details/135660662

import numpy as np

def generate_rayleigh_channels(num_channels):
# 生成实部和虚部
    real_parts = np.random.normal(0, 1/np.sqrt(2), num_channels)
    imag_parts = np.random.normal(0, 1/np.sqrt(2), num_channels)
    # 合成复数衰落系数数组
    h = real_parts + 1j * imag_parts
    return h
# 生成多个瑞利信道衰落系数
num_channels = 10
h_array = generate_rayleigh_channels(num_channels)
# print("Generated Rayleigh channel fading coefficients:", h_array)


#%%   https://blog.csdn.net/qq_45889056/article/details/134133182
#其中p为第K个用户的发射功率，g为第n个用户的大尺度信道增益，包括路径损耗和阴影，
#h~CN( 0 , 1)为第K个用户的瑞利衰落系数，N0为噪声功率谱密度。
def large_small(K,d,p,W): #
    N0_dBm = 3.981e-21  # -174dBm==3.981e-21
    path_loss = 128.1 + 37.6*np.log10(d/1000) #dB    小区半径为500m
    # the shadowing factor is set as 6 dB.
    shadow_factor = 6 #dB
    h_large=10** (-(path_loss + shadow_factor) / 10)

    sigma = np.sqrt(1 / 2)
    h_small=sigma * np.random.randn(K) + 1j * sigma * np.random.randn(K)

    h = abs(h_small)**2
    snr = h_large*h*p / (N0_dBm*W)
    return snr

if __name__ == '__main__':
    snr  = large_small(1,200, 0.1, 10000000) # W设置为180KHz
    snr_dB  = 10*np.log10(snr/10)
    print(snr)
    print( snr_dB)










