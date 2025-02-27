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

from numpy import  vectorize
from commpy.utilities import bitarray2dec, dec2bitarray, signal_power

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
    bps = int(np.log2(M))
    if modutype == 'qam':
        modem = commpy.QAMModem(M)
        Es = NormFactor(mod_type = modutype, M = M,)
    elif modutype == 'psk':
        modem =  commpy.PSKModem(M)
        Es = NormFactor(mod_type = modutype, M = M,)
    elif modutype == 'pam':
        modem = PAM_modulator(M)
        Es = modem.Es
    return modem, Es, bps


class PAM_modulator(object):
    def __init__(self, M):
        self.M = M
        self.bps = int(np.log2(self.M))
        self.constellation = None
        self.Es = self.init(M)
        self.map_table = {}
        self.demap_table = {}
        self.getMappTable()
        return

    def init(self, M):
        m = np.arange(1, M + 1, 1)
        self.constellation = np.complex128(2*m - 1 - M)
        Es = np.mean(np.abs(self.constellation)**2)
        return Es

    def getMappTable(self):
        for idx, symb in enumerate(self.constellation):
            self.map_table[idx] = symb
            self.demap_table[symb] = idx
        return self.map_table, self.demap_table

    def modulate(self, x, inputtype = 'bit'):
        """ Modulate (map) an array of bits to constellation symbols.
        Parameters
        ----------
        x : 1D ndarray of ints
            Inputs bits to be modulated (mapped).
        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.
        """
        if inputtype == 'bit':
            mapfunc = np.vectorize(lambda i: self.constellation[commpy.utilities.bitarray2dec(x[i:i + self.bps])])
            baseband_symbols = mapfunc(np.arange(0, len(x), self.bps))
        if inputtype == 'int':
            baseband_symbols = self.constellation[x]
        return baseband_symbols

    def demodulate(self, input_symbols, demod_type = 'hard', outputtype = 'bit', noise_var = 0):
        """ Demodulate (map) a set of constellation symbols to corresponding bits.
        Parameters
        ----------
        input_symbols : 1D ndarray of complex floats Input symbols to be demodulated.
        demod_type : string
            'hard': for hard decision output (bits).
            'soft': for soft decision output (LLRs).
        noise_var : float
            AWGN variance. Needs to be specified only if demod_type is 'soft'
        Returns
        ----------
        demod_bits : 1D ndarray of ints Corresponding demodulated bits.
        """
        if outputtype == 'bit':
            if demod_type == 'hard':
                index_list = np.abs(input_symbols - self.constellation[:, None]).argmin(0)
                demod_bits = commpy.utilities.dec2bitarray(index_list, self.bps)
            elif demod_type == 'soft':
                demod_bits = np.zeros(len(input_symbols) * self.bps)
                for i in np.arange(len(input_symbols)):
                    current_symbol = input_symbols[i]
                    for bit_index in np.arange(self.bps):
                        llr_num = 0
                        llr_den = 0
                        for bit_value, symbol in enumerate(self.constellation):
                            if (bit_value >> bit_index) & 1:
                                llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                            else:
                                llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                        demod_bits[i * self.bps + self.bps - 1 - bit_index] = np.log(llr_num / llr_den)
            else:
                raise ValueError('demod_type must be "hard" or "soft"')
        elif outputtype == 'int':
            tmp = input_symbols.reshape(1,-1) - self.constellation[:,None]
            tmp = np.abs(tmp)
            demod_bits = tmp.argmin(axis = 0)
        return demod_bits

    def plot_constellation(self, Modulation_type = 'PAM'):
        import math
        M = len(self.constellation)
        nbits = int(math.log2(M))
        # map_table = {}
        # demap_table = {}

        fig, axs = plt.subplots(1,1, figsize=(8, 8), constrained_layout=True)
        for idx, symb in enumerate(self.constellation):
            # map_table[idx] = symb
            # demap_table[symb] = idx
            axs.scatter(symb.real, symb.imag, s = 40, c = 'b')
            # axs.text(symb.real-0.4, symb.imag + 0.1, str(self.demodulate(symb, 'hard')) + ":" + str(idx), fontsize=18, color='black', )
            axs.text(symb.real-0.4, symb.imag + 0.1, bin(idx)[2:].rjust(nbits, '0') + ":" + str(idx), fontsize = 18, color = 'black', )
        ##
        font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
        axs.set_title(f"{Modulation_type} Mapping Table", fontproperties=font2,)

        axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        [label.set_fontsize(24) for label in labels]  # 刻度值字号

        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        axs.spines['bottom'].set_linewidth(2)    #### 设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2)     #### 设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

        axs.set_xlim([self.constellation.real.min() - 1, self.constellation.real.max() + 1])
        axs.set_ylim([self.constellation.imag.min() - 1, self.constellation.imag.max() + 1])
        plt.show()

        return

# M = 4
# pam = PAM_modulator(M)

# bits   = np.random.randint(0, 2, pam.bps*20)

# syms   = pam.modulate(bits)
# bits_1 = pam.demodulate(syms,)

# ints  = np.random.randint(0, M, 20)
# syms1 = pam.modulate(ints, inputtype = 'int')
# syms2 = syms1 + 1.1 * np.random.randn(*syms1.shape)
# ints_1 = pam.demodulate(syms2, outputtype = "int")

# pam.plot_constellation()



























































































































































































































































































































































































