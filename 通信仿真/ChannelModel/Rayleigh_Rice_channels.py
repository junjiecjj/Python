#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:56:49 2024

@author: jack
"""

#%%
# https://blog.csdn.net/weixin_41608328/article/details/88833665?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170859540616800226523213%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170859540616800226523213&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-88833665-null-null.142^v99^pc_search_result_base2&utm_term=%E4%BF%A1%E5%8F%B7%E9%80%9A%E8%BF%87rayleigh%20ricean%E4%BF%A1%E9%81%93python&spm=1018.2226.3001.4187
# 莱斯信道衰落下的QPSK误码率分析


# https://blog.csdn.net/weixin_44703913/article/details/128982361?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-128982361-blog-130114353.235%5Ev43%5Epc_blog_bottom_relevance_base8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-128982361-blog-130114353.235%5Ev43%5Epc_blog_bottom_relevance_base8&utm_relevant_index=5
## 信道状态由大尺度衰落和小尺度衰落共同决定，时域卷积，频域相乘（平时大多数论文所写的y=hx+n }是复频域表达式，信道模型是信道的频域冲激响应，不是时域表达式）[1]

# https://zhuanlan.zhihu.com/p/372500189

#%% 瑞丽和莱斯都是小尺度衰落信道，瑞丽衰落：多径反射链路； 莱斯信道：直达链路+多径反射链路
"""
(一)大尺度衰落由收发两端的距离决定；
(二)小尺度衰落由收发两端的环境决定，比如是否有遮挡，场景有室内、室外、平原、山村、城镇等，这些环境影响到收发两端是否有直达链路，多径效应严重与否。通常根据直达链路与等效多径链路的强度占比情况，表示为为莱斯衰落和瑞利衰落。
    在无线信道中，莱斯分布是一种最常见的用于描述接收信号包络统计时变特性的分布类型。其中莱斯因子是反映信道质量的重要参数，在计算信道质量和链路预算、移动台移动速度以及测向性能分析等都发挥着重要的作用。信号在传输过程中由于多径效应，接收信号是直射信号（主信号， 直射信号(LoS，Line of Sight)）和多径信号的叠加，此时接收信号的包络服从莱斯分布。事实上，在考虑多径效应的时候，原始信号与呈现瑞利分布的多径分量的和常常被描述为莱斯分布。

(三)信道状态由大尺度衰落和小尺度衰落共同决定，时域卷积，频域相乘（平时大多数论文所写的 y=hx+n 是复频域表达式，信道模型是信道的频域冲激响应，不是时域表达式）。

    小尺度衰落，根据直达链路与多径反射链路的情况，通常用莱斯衰落和瑞利衰落来表示。

    莱斯衰落表示一类有直达链路和多条反射链路的信道衰落情况:
    \sqrt(k/(K+1)) * H^{Los} + \sqrt(1/(K+1))H^{NLos}
        其中，H^{Los}表示直射链路，与天线摆放角度有关（决定了信号的来去角），单天线场景下可令h^{LoS}=1；多天线通信场景下，则需要考虑发送端信号离去角和接收端信号的到达角，此时h^{LoS}=a_{T}a_{R}，a_T和a_R为表示角度的矢量向量，仿真时可利用一下代码块简单生成:
        def H_Los(phi, N)
        h_LoS = exp^{1j*pi*sin(phi)*(0:N-1)}
        return h_LoS
    K 为莱斯因子，表示直达链路与等效多条反射链路的强度比：
    K 取值越大表示直达链路强度大，占直达链路占主要的比重, 这时K/(K+1)约等于1，1/(K+1)约等于0，H^{Los} = 1, 所以退化为高斯信道。
    K 取值越小，表示直达链路强度弱，反射链路占主要的比重，多径效应明显。当收发两端没有直达路径，只有多条反射链路连接时，这时莱斯衰落退化为瑞利衰落。
    下面的代码是单天线的，所以 h_Los = 1
"""
import numpy as np

def generate_Rayleigh_model(num_channels):
# 生成实部和虚部
    real_parts = np.random.normal(0, 1, num_channels)
    imag_parts = np.random.normal(0, 1, num_channels)
    # 合成复数衰落系数数组
    h = np.sqrt(1.0/2.0) * (real_parts + 1j * imag_parts)
    return h

def generate_Rice_model(K_dB, L):
    K = 10**(K_dB/10.0)
    # LoS分量：sqrt(K / (K + 1))；NLoS分量：sqrt(1 / (K + 1)) * Ray_model(N)
    H = (np.sqrt(K) + generate_Rayleigh_model(L))/(np.sqrt(K+1))
    return H


def Rayleigh_channel( L, Tx_data, Tx_data_power = None, SNR_dB = 5, ):
    if Tx_data_power == None:
        Tx_data_power = np.mean(abs(Tx_data)**2)
    noise_pwr = Tx_data_power/(10**(SNR_dB/10))
    noise = np.sqrt(noise_pwr/2.0) * (np.random.randn(L) + 1j * np.random.randn(L))
    # noise = np.sqrt(1/2) * (np.random.normal(loc=0.0, scale=1.0,  size = (L, )) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (L, )))
    Rx_data = generate_Rayleigh_model( L ) * Tx_data + noise
    return Rx_data


def Rice_channel( L, Tx_data, Tx_data_power = None, SNR_dB = 5, K_dB = 30):
    if Tx_data_power == None:
        Tx_data_power = np.mean(abs(Tx_data)**2)
    noise_pwr = Tx_data_power/(10**(SNR_dB/10))
    noise = np.sqrt(noise_pwr/2.0) * (np.random.randn(L) + 1j * np.random.randn(L))
    # noise = np.sqrt(1/2) * (np.random.normal(loc=0.0, scale=1.0,  size = (L, )) + 1j * np.random.normal(loc=0.0, scale=1.0,  size = (L, )))
    Rx_data = generate_Rice_model(K_dB, L) * Tx_data + noise
    return Rx_data





















