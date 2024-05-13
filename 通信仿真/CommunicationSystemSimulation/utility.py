#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:31:10 2023
@author: JunJie Chen

np.mod(np.matmul(KbinRe, G), 2)


"""

import datetime
import os
import numpy as np
import random
import math
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


def WrLogHead(logfile = "SNR_BerFer.txt", promargs = '', codeargs = ''):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    logfile = current_dir + logfile
    # print(logfile)
    with open(logfile, 'a+') as f:
        print("#=====================================================================================",  file = f)
        print("                      " +  now,  file = f)
        print("#=====================================================================================\n",  file = f)

        f.write("######### [program config] #########\n")
        for k, v in promargs.__dict__.items():
            f.write(f"{k: <25}: {v: <40}\n")
        f.write("######### [code config] ##########\n")
        for k, v in codeargs.items():
            f.write(f"{k: <25}: {v: <40}\n")
        f.write("\n#=============================== args end  ===============================\n")
    return



# 初始化随机数种子
def set_random_seed(seed = 1, deterministic = False, benchmark = False):
    np.set_printoptions(linewidth = 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return

##  yy --> 对数域LLR
def yyToLLR(yy, noise_var = 1.0):
    LLr = np.zeros_like(yy)
    LLr =  2.0*yy/noise_var
    return LLr

##  yy --> 概率域: P(+1|y)
def yyToProb(yy, noise_var = 1.0):
    prob = np.zeros_like(yy)
    prob = 1.0 / (1.0 + np.exp(-2.0*yy/noise_var))
    return prob

"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的码字bits，如果yy[i]<0,则bits[i] = 1
"""
def bit_judge(data_in):
    bits = np.zeros(data_in.size, dtype = 'int')
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits

"""
输入：信道编码后以及过信道后接受到的信号,yy。
输出：生成与yy等长的码字bits，如果yy[i]<0,则bits[i] = 1
"""
def hard_decision(data_in):
    bits = np.zeros(data_in.size, dtype = np.int8)
    for i in range(len(bits)):
        if data_in[i] < 0 :
            bits[i] = 1
    return bits



#%%==========================================================================================
##                  星座图
#============================================================================================


def draw_mod_constellation(map_table, modu_type = '16QAM'):
    """
        Draw constellation of M-QAM.
        Parameters
        ----------
        map_table: int. dict. M-QAM mapping table.
    """
    M = len(map_table)
    nbits = int(math.log2(M))

    width = 10
    high = 10
    horvizen = 1
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 16)
    for i in range(M):
        Q = map_table[i]
        axs.plot(Q.real, Q.imag, color = 'r', marker = 'o', markersize = 13,)
        axs.text(Q.real, Q.imag + 0.15, bin(i)[2:].rjust(nbits, '0'), ha = 'center', fontproperties=font)

    ## axis
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ##
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴 指定左边的边为 y 轴
    ## axis pos
    axs.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    axs.spines['left'].set_position(('data', 0))
    ## axis linewidth
    bw = 3
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细

    axs.tick_params(direction='in',axis='both',  labelsize=16, width=6,  )
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(25) for label in labels] #刻度值字号

    ## x/ylabel, title
    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 25)
    axs.set_xlabel('I', loc='right', labelpad = 0.5, fontproperties=font)
    axs.set_ylabel('Q', loc='top', labelpad = 0.5, fontproperties=font)  # 设置坐标轴的文字标签
    axs.set_title(f"{M}-QAM Mapping Table", fontproperties=font)

    axs.set_xlim([-nbits - 1, nbits + 1]) ## 设置坐标的数字范围
    axs.set_ylim([-nbits - 1, nbits + 1]) ## 设置坐标的数字范围

    out_fig = plt.gcf()
    out_fig.savefig(f'./figures/{modu_type}/constellation_{modu_type}.eps', )
    out_fig.savefig(f'./figures/{modu_type}/constellation_{modu_type}.png', dpi = 1000,)
    plt.show()
    return


def draw_trx_constellation(syms, map_table, tx = True, snr = None, channel = None, modu_type = 'QAM16'):
    """
        Draw constellation of transmitted or received signal.

        Parameters
        ----------
        syms: array(num_symbol, ). Modulated symbols to be transmitted or received symbols.
        tx: bool, default True. 1: Draw constellation of transmitted signal. 2: Draw constellation of received signal.
        snr: int. SNR at the receiver side.
        channel: str. Type of wireless channel.
    """
    # if tx:
    #     plt.title(f"Constellation of Transmitted Signal")
    # else:
    #     assert snr is not None, "SNR is required."
    #     assert channel, "Channel type is required."
    #     plt.title(f"Constellation of Received Signal ({channel.upper()}, SNR={snr}dB)")
    # for sym in syms:
    #     plt.plot(sym.real, sym.imag, 'r*')
    # plt.show()

    """
        Draw constellation of M-QAM.
        Parameters
        ----------
        map_table: int. dict. M-QAM mapping table.
    """
    M = len(map_table)

    nbits = int(math.log2(M))

    width = 10
    high = 10
    horvizen = 1
    vertical = 1
    fig, axs = plt.subplots(horvizen, vertical, figsize = (horvizen*width, vertical*high), constrained_layout = True)
    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 16)
    for i in range(M):
        Q = map_table[i]
        axs.plot(Q.real, Q.imag, color = 'r', marker = 'o', markersize = 13,)

    for sym in syms:
        axs.plot(sym.real, sym.imag, color = 'b', marker = '*', markersize = 12,)
        # axs.scatter(sym.real, sym.imag, s = 43, marker = '*', color = 'b')

    axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

    axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize = 25, width=6,  )
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # [label.set_fontsize(25) for label in labels] #刻度值字号
    # axs.grid( linestyle = '--', linewidth = 0.5, )

    ## x/ylabel, title
    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
    axs.set_xlabel('I',  labelpad = 0.5, fontproperties=font)
    axs.set_ylabel('Q',  labelpad = 0.5, fontproperties=font)  # 设置坐标轴的文字标签
    # axs.set_title(f"{M}-QAM Mapping Table", fontproperties=font)
    if tx:
        axs.set_title("Constellation of Transmitted Signal", fontproperties=font)
    else:
        assert snr is not None, "SNR is required."
        assert channel, "Channel type is required."
        axs.set_title(f"Constellation of Received Signal ({channel.upper()}, SNR={snr}dB)", fontproperties=font)

    # axs.set_xlim([np.real(syms).min() - 1, np.real(syms).max() + 1]) ## 设置坐标的数字范围
    # axs.set_ylim([np.imag(syms).min() - 1, np.imag(syms).max() + 1]) ## 设置坐标的数字范围

    axs.set_xlim([-nbits-1, nbits+1]) ## 设置坐标的数字范围
    axs.set_ylim([-nbits-1, nbits+1]) ## 设置坐标的数字范围

    out_fig = plt.gcf()
    if tx:
        name = f"transmit_constellation_{modu_type}"
    else:
        name = f"recv_constellation_{modu_type}_snr={snr}_{channel}"
    # out_fig.savefig(f'./figures/{modu_type}/{name}.eps', )
    # out_fig.savefig(f'./figures/{modu_type}/{name}.png', dpi = 1000,)
    plt.show()

    return










































































































































































