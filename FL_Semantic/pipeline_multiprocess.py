
# -*- coding: utf-8 -*-
"""
Created on 2023/08/17

@author: Junjie Chen

"""


## system lib
import numpy  as np
# import datetime
# import copy
import math
import torch


##  自己编写的库
# from LDPC.sourcesink import SourceSink
# from LDPC.channel import AWGN
# from LDPC.modulation import  BPSK
# from LDPC.modulation import demodu_BPSK
# from LDPC import utility
# from LDPC.argsLDPC import arg as topargs
# from LDPC.ldpc_coder import LDPC_Coder_llr
# from LDPC.quantiation import   QuantizationNP_uint, deQuantizationNP_uint
from LDPC.quantiation import  QuantizationBbits_NP_int, deQuantizationBbits_NP_int
from LDPC.quantiation import Quantization1bits_NP_int, Quantization1bits_NP_int_NR,  deQuantization1bits_NP_int
# from LDPC.quantiation import  QuantizationTorch_int

# utility.set_random_seed()
# print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## LDPC初始化
# ldpcCoder =  LDPC_Coder_llr(topargs)
# coderargs = {'codedim':ldpcCoder.codedim,
#              'codelen':ldpcCoder.codelen,
#              'codechk':ldpcCoder.codechk,
#              'coderate':ldpcCoder.coderate,
#              'row':ldpcCoder.num_row,
#              'col':ldpcCoder.num_col}

# utility.WrLogHead(promargs = topargs, codeargs = coderargs)





## 将联邦学习得到的浮点数依次：量化、以指定概率 bit flipping、反量化;
def Quant_BbitFlipping(com_round = 1, client = "", param_W = '', err_rate = 0.0001, quantBits = 8, dic_parm = "", dict_berfer = ""):
    # np.random.seed()
    np.random.seed(int(client[6:]) + com_round)
    # print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    ##================================================= 将参数字典序列化为向量 =============================================
    pam_size_len = {}
    pam_order = []
    params_float = np.empty((0, 0))
    # params_float = torch.Tensor()
    num_sum = 0

    for key, val in param_W.items():
        if 'float' in str(val.dtype):
            pam_order.append(key)
            tmp_list = []
            tmp_list.append(val.shape)
            tmp_list.append(val.size)
            num_sum += val.size
            pam_size_len[key] = tmp_list
            params_float = np.append(params_float, val)
        # print(key, val.shape)

    ##================================================= 量化 ===========================================================
    # print(f"   {client}")
    # if quantBits > 1:
    binary_send = QuantizationBbits_NP_int(params_float, B = quantBits, rounding = "nr")
    # elif quantBits == 1:
        # binary_send = Quantization1bits_NP_int(params_float,  BG = 8,)

    assert binary_send.size == num_sum * quantBits
    # ##==========================================  Bit Flipping  ==================================================

    # binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size )
    # # print(f"{binary_recv.mean()}, {binary_recv.min()}, {binary_recv.max()}")
    # # binary_recv = rdm.binomial(n = 1, p = err_rate, size = binary_send.size )
    # binary_recv = binary_recv ^ binary_send

    # err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================
    # if quantBits > 1:
    param_recv = deQuantizationBbits_NP_int(binary_send, B = quantBits)
    # elif quantBits == 1:
        # param_recv = deQuantization1bits_NP_int(binary_recv, BG = 8)
    ##============================================= 将反量化后的实数序列变成字典形式 =======================================
    param_recover = {}
    start = 0
    end = 0
    for key in pam_order:
        end += pam_size_len[key][1]
        param_recover[key] = param_recv[start:end].reshape(pam_size_len[key][0])
        start += pam_size_len[key][1]

    for key, val in param_W.items():
        if 'int' in str(val.dtype):
            param_recover[key] = val

    ##=================================================== 结果打印和记录 ================================================
    dic_parm[client] = param_recover
    dict_berfer[client] = {"ber": 0 }
    return

## 将联邦学习得到的浮点数依次：量化、以指定概率 bit flipping、反量化;
def Quant_1bitFlipping(com_round = 1, client = "", param_W = '', err_rate = 0, quantBits = 8, dic_parm = "", dict_berfer = ""):
    np.random.seed()
    # np.random.seed(int(client[6:]) + com_round)
    # print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    ##================================================= 将参数字典序列化为向量 =============================================
    pam_size_len = {}
    pam_order = []
    params_float = np.empty((0, 0))
    # params_float = torch.Tensor()
    num_sum = 0

    for key, val in param_W.items():
        if 'float' in str(val.dtype):
            pam_order.append(key)
            tmp_list = []
            tmp_list.append(val.shape)
            tmp_list.append(val.size)
            num_sum += val.size
            pam_size_len[key] = tmp_list
            params_float = np.append(params_float, val)
        # print(key, val.shape)
    ##================================================= 量化 ===========================================================
    # print(f"   {client}")
    # if quantBits > 1:
    BG = 7
    binary_send = Quantization1bits_NP_int_NR(params_float, BG = BG )
    # elif quantBits == 1:
        # binary_send = Quantization1bits_NP_int(params_float,  BG = 8,)

    assert binary_send.size == num_sum * quantBits
    # ##==========================================  Bit Flipping  ==================================================

    # binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size )
    # # print(f"{binary_recv.mean()}, {binary_recv.min()}, {binary_recv.max()}")
    # # binary_recv = rdm.binomial(n = 1, p = err_rate, size = binary_send.size )
    # binary_recv = binary_recv ^ binary_send

    # err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================
    # if quantBits > 1:
    param_recv = deQuantization1bits_NP_int(binary_send, BG = BG )
    # elif quantBits == 1:
        # param_recv = deQuantization1bits_NP_int(binary_recv, BG = 8)
    ##============================================= 将反量化后的实数序列变成字典形式 =======================================
    param_recover = {}
    start = 0
    end = 0
    for key in pam_order:
        end += pam_size_len[key][1]
        param_recover[key] = param_recv[start:end].reshape(pam_size_len[key][0])
        start += pam_size_len[key][1]

    for key, val in param_W.items():
        if 'int' in str(val.dtype):
            param_recover[key] = val

    ##=================================================== 结果打印和记录 ================================================
    dic_parm[client] = param_recover
    dict_berfer[client] = {"ber":0 }
    return





























































































































































































