#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:44 2023
@author: JunJie Chen
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import commpy

def BPSK(c):
    # for i in range(cc.shape[-1]):
    #     if cc[i] == 0:
    #         cc[i] = 1
    #     else:
    #         cc[i] = -1
    c = 1 - 2*c
    return c

def demodu_BPSK(y):
    for i in range(y.shape[-1]):
        if y[i] > 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

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


## BPSK, QPSK, 8PSK, 16QAM, 64 QAM, 256QAM + fast Fading
def demod_fading(constellation, input_symbols, demod_type, H, Es = None, noise_var=0):
    M = len(constellation)
    bitsPerSym = int(np.log2(M))
    if Es != None:
        constellation = constellation / np.sqrt(Es)
    ##
    if demod_type == 'hard':
        idx = np.abs(input_symbols.reshape(-1,1) - H[:,None] @ constellation.reshape(1, -1)).argmin(1)
        # index_list = np.abs(input_symbols - constellation[:, None]).argmin(0)
        demod_bits = commpy.utilities.dec2bitarray(idx, bitsPerSym)
    elif demod_type == 'soft':
        demod_bits = np.zeros(len(input_symbols) * bitsPerSym)
        for i in np.arange(len(input_symbols)):
            current_symbol = input_symbols[i]
            h = H[i]
            for bit_index in np.arange(bitsPerSym):
                llr_num = 0
                llr_den = 0
                for bit_value, symbol in enumerate(h * constellation):
                    if (bit_value >> bit_index) & 1:
                        llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    else:
                        llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                demod_bits[i * bitsPerSym + bitsPerSym - 1 - bit_index] = np.log(llr_num / llr_den)
    else:
        raise ValueError('demod_type must be "hard" or "soft"')
    return demod_bits

## BPSK, QPSK, 8PSK, 16QAM, 64 QAM, 256QAM + AWGN
def demod_awgn(constellation, input_symbols, demod_type, Es = None, noise_var = 0):
    M = len(constellation)
    bitsPerSym = int(np.log2(M))
    if Es != None:
        constellation = constellation / np.sqrt(Es)
    if demod_type == 'hard':
        index_list = np.abs(input_symbols - constellation[:, None]).argmin(0)
        demod_bits = commpy.utilities.dec2bitarray(index_list, bitsPerSym)
    elif demod_type == 'soft':
        demod_bits = np.zeros(len(input_symbols) * bitsPerSym)
        for i in np.arange(len(input_symbols)):
            current_symbol = input_symbols[i]
            for bit_index in np.arange(bitsPerSym):
                llr_num = 0
                llr_den = 0
                for bit_value, symbol in enumerate(constellation):
                    if (bit_value >> bit_index) & 1:
                        llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                    else:
                        llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                demod_bits[i * bitsPerSym + bitsPerSym - 1 - bit_index] = np.log(llr_num / llr_den)
    else:
        raise ValueError('demod_type must be "hard" or "soft"')

    return demod_bits


def plot_constellation(constellation = "None", map_table = "None", Modulation_type = "Constellation"):
    import math
    if type(constellation) == str and type(map_table) == str:
        raise Exception("Both constellation and map_table are Empty!")
    if type(constellation) == str:
        M = len(map_table)
        constellation = np.zeros(M, dtype = complex)
        for i in range(M):
            constellation[i] = map_table[i]
    if type(map_table) == str:
        map_table = {}
        M = len(constellation)
        for i in range(M):
            map_table[i] = constellation[i]

    nbits = int(math.log2(M))

    fig, axs = plt.subplots(1,1, figsize=(8, 8), constrained_layout=True)
    for idx, symb in map_table.items():
        axs.scatter(symb.real, symb.imag, s = 40, c = 'b')
        # axs.text(symb.real-0.4, symb.imag + 0.1, str(self.demodulate(symb, 'hard')) + ":" + str(idx), fontsize=18, color='black', )
        axs.text(symb.real-0.4, symb.imag + 0.1, bin(idx)[2:].rjust(nbits, '0') + ":" + str(idx), fontsize=18, color='black', )
    ##
    font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30}
    axs.set_title(f"{Modulation_type} Mapping Table", fontproperties=font2,)

    axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]  # 刻度值字号

    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
    axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

    axs.set_xlim([constellation.real.min() - 1, constellation.real.max() + 1])
    axs.set_ylim([constellation.imag.min() - 1, constellation.imag.max() + 1])
    plt.show()

    return


def qam8_32_mod(M):
    """
        Generate M-QAM mapping table and demapping table.

        Parameters
        ----------
        M: int. Modulation order, must be a positive integer power of 2 and a perfect square number, or one of 8 and 32.
        Returns
        -------
        map_table: dict. M-QAM mapping table.
        demap_table: dict.M-QAM demapping table
    """
    sqrtM = int(math.sqrt(M))
    assert (sqrtM ** 2 == M and M & (M-1) == 0) or (M == 32) or (M == 8), "M must be a positive integer power of 2 and a perfect square number, or one of 8 and 32."
    if M == 8:
        graycode = np.array([0, 1, 3, 7, 5, 4, 6, 2])
        constellation = [(-2-2j), (-2+0j), (-2+2j), (0+2j), (2+2j), (2+0j), (2-2j), (0-2j)]
    elif M == 32:
        temp1 = np.bitwise_xor(np.arange(8), np.right_shift(np.arange(8), 1))
        temp2 = np.bitwise_xor(np.arange(4), np.right_shift(np.arange(4), 1))
        graycode = np.zeros(M, dtype=int)
        num = 0
        for i in temp1:
            for j in temp2:
                graycode[num] = 4 * i + j
                num += 1
        constellation = [(-7 - 3j) + 2 * (x + y * 1j) for x, y in np.ndindex(8, 4)]
    map_table = dict(zip(graycode, constellation))
    demap_table = {v: k for k, v in map_table.items()}
    return map_table, demap_table
























































































































