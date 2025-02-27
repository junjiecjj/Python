#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:33:21 2025

@author: jack
"""

#%%
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


def NormFactor(mod_type = 'qam', M = 16,):
    """
        Signal power normalization and de-normalization.
        Parameters
            signal: array(*, ). Signal to be transmitted or received.
            M: int. Modulation order.
            mod_type: str, default 'qam'. Type of modulation technique.
            denorm: bool, default False. 0: Power normalization. 1: Power de-normalization.
        Returns
    """
    if mod_type == 'psk':
        Es = 1
    # if mod_type == 'qpsk':
    #     Es = 1
    # if mod_type == '8psk':
    #     Es = 1
    if mod_type == 'qam':
        if M == 8:
            Es = 6
        elif M == 32:
            Es = 25.875
        else: ##  https://blog.csdn.net/qq_41839588/article/details/135202875
            Es = 2 * (M - 1) / 3
    return Es


def modulator(modutype, M, ):
    # M = args.M
    bps = int(np.log2(M))
    # framelen = int(ldpc.codelen/bps)
    # modutype = args.type
    if modutype == 'qam':
        modem = commpy.QAMModem(M)
        Es = NormFactor(mod_type = modutype, M = M,)
    elif modutype == 'psk':
        modem =  commpy.PSKModem(M)
        Es = NormFactor(mod_type = modutype, M = M,)
    elif modutype == 'pam':
        pass
    return modem, Es, bps



class PAM_modulator(object):
    def __init__(self, M):
        self.M = M
        self.constellation = None
        self.Es = self.init(M)

    def init(self, M):
        m = np.arange(1, M + 1, 1)
        self.constellation = np.complex64(2*m - 1 - M)
        Es = np.mean(np.abs(self.constellation)**2)
        return Es


pam = PAM_modulator(4)
































































































































































































































































































































































































