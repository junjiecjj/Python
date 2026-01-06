#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:10:28 2024

@author: jack
"""

import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties

from matplotlib.pyplot import MultipleLocator
# 使用Savitzky-Golay 滤波器后得到平滑图线
from scipy.signal import savgol_filter
# import scipy
import numpy as np
import torch
# import seaborn as sns
import copy
from Quantizer import QuantizationBbits_NP_int, deQuantizationBbits_NP_int


# B-bit error-free transmission, stochastic rounding (SR), Nearest rounding (NR), do not quantize batch-normalization layer.
def B_Bit(message_lst, args, rounding = 'nr', ber = 0, B = 8, key_grad = None):
    ## D = np.sum([param.numel() for param in message_lst[0].values()])
    key_lst_wo_grad = []
    info_lst = []
    ## 分离可导层和不可导层
    for key, val in message_lst[0].items():
        if key in key_grad:
            info_lst.append([key, val.size(), val.numel(), val.dtype])
        elif key not in key_grad:
            key_lst_wo_grad.append(key)

    ## 将可导层转换为数组
    D = np.sum([message_lst[0][key].numel() for key in key_grad])
    SS = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_grad:
            vec = np.hstack((vec, mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec
    M = np.abs(SS).max(axis = 1)
    G =  2**(B-1)/M

    ## B-bit Quantization
    uu = np.zeros((len(message_lst), D * B), dtype = np.int8)
    for k in range(len(message_lst)):
        uu[k] = QuantizationBbits_NP_int(SS[k], G[k], B = B, rounding = rounding)

    ## flipping
    flip_mask = np.random.binomial(n = 1, p = ber, size = uu.shape )
    uu_flipped = uu ^ flip_mask
    err_rate = (uu_flipped != uu).sum(axis = 1)/uu.shape[-1]

    ## recv
    s_hat = np.zeros((len(message_lst), D), dtype = np.float32)
    for k in range(len(message_lst)):
        s_hat[k] = deQuantizationBbits_NP_int(uu_flipped[k], G[k], B = B )

    ## recover
    mess_recov = []
    for k in range(len(message_lst)):
        symbolsK = s_hat[k, :]
        param_k = {}
        start = 0
        end = 0
        ## 恢复可导层
        for info in info_lst:
            end += info[2]
            param_k[info[0]] = torch.tensor(symbolsK[start:end].reshape(info[1]), dtype = info[3]).to(args.device)
            start += info[2]
        ## 直接无错拷贝不可导层
        for key in key_lst_wo_grad:
            param_k[key] = copy.deepcopy(message_lst[k][key])
        mess_recov.append(param_k)
    return mess_recov, err_rate

# np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
# B = 8
# rounding = 'nr'
# m = 4
# n = 10
# S = np.random.randn(m, n) * 0.01
# M = np.abs(S).max(axis = 1)
# G =  2**(B-1)/M
# uu = np.zeros((m, n * B), dtype = np.int8)

# for k in range(m):
#     uu[k] = QuantizationBbits_NP_int(S[k], G[k], B = B, rounding = rounding)

# s = np.zeros((m, n ))
# for k in range(m):
#     s[k] = deQuantizationBbits_NP_int(uu[k], G[k], B = B)

def mess_stastic(message_lst, args, savename, key_grad ):
    # D = np.sum([param.numel() for param in message_lst[0].values()])
    key_lst_wo_grad = []
    info_lst = []
    ## 分离可导层和不可导层
    for key, val in message_lst[0].items():
        if key in key_grad:
            info_lst.append([key, val.size(), val.numel(), val.dtype])
        elif key not in key_grad:
            key_lst_wo_grad.append(key)

    ## 将可导层转换为数组
    D = np.sum([message_lst[0][key].numel() for key in key_grad])
    SS = np.zeros((len(message_lst), D))
    for k, mess in enumerate(message_lst):
        vec = np.array([], dtype = np.float32)
        for key in key_grad:
            vec = np.hstack((vec, mess[key].detach().cpu().numpy().flatten()))
        SS[k,:] = vec

    fig, axs = plt.subplots( figsize = (6, 4), constrained_layout = True)

    sns.kdeplot(SS[0], fill = True, label = f"Round = {0}", color='blue', alpha=0.2, common_norm = True)
    sns.kdeplot(SS[1], fill = True, label = f"Round = {50}", color='orange', alpha=0.2, common_norm = True)
    sns.kdeplot(SS[2], fill = True, label = f"Round = {100}", color='green', alpha=0.2, common_norm = True)

    # axs.hist(SS[0], bins=30, density = True, alpha=0.5, label=f"Round = {0}", color='orange')
    # axs.hist(SS[1], bins=30, density = True, alpha=0.5, label=f"Round = {50}", color='blue')
    # axs.hist(SS[2], bins=30, density = True, alpha=0.5, label=f"Round = {100}", color='blue')

    # count, bins, ignored = axs.hist(SS.flatten(), density=False, bins='auto', histtype='stepfilled', alpha=0.5, facecolor = "#0099FF", label= lb, zorder = 4)

    bw = 2
    axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 19,}
    legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none')  # 设置图例legend背景透明

    font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
    axs.set_xlabel('Value', fontdict = font, )
    axs.set_ylabel('Density', fontdict = font, )

    axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
    axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

    axs.set_xlim(-0.1, 0.1)  #拉开坐标轴范围显示投影
    # 显示图形
    out_fig = plt.gcf()
    out_fig.savefig(savename, pad_inches = 0,)
    plt.show()
    return

# message_lst = statistics3
# mess_stastic(statistics3, args, savename, key_grad )


def ParamRange(message_lst, key_want, k = 0):

    K = len(message_lst)
    N = len(key_want)

    res = np.zeros((K, N))
    for k, mess in enumerate(message_lst):
        for j, key in enumerate(key_want):
            tmp = mess[key].detach().cpu().numpy().flatten()
            res[k, j] = tmp.max() - tmp.min()

    return res











































































































