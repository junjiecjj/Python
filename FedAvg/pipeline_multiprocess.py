
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
from LDPC.quantiation import Quantization1bits_NP_int,  deQuantization1bits_NP_int
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
def Quant_BbitFlipping(param_W = '', err_rate = 0.0001, quantBits = 8, com_round = 1, client = "",  dic_parm = "", dic_berfer="", rdm = ""):
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
    ##==========================================  Bit Flipping  ==================================================

    binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size )
    # print(f"{binary_recv.mean()}, {binary_recv.min()}, {binary_recv.max()}")
    # binary_recv = rdm.binomial(n = 1, p = err_rate, size = binary_send.size )
    binary_recv = binary_recv ^ binary_send

    err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================
    # if quantBits > 1:
    param_recv = deQuantizationBbits_NP_int(binary_recv, B = quantBits)
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

    ##=================================================== 结果打印和记录 ================================================
    dic_parm[client] = param_recover
    dic_berfer[client] = {"ber":err_rate }
    return

## 将联邦学习得到的浮点数依次：量化、以指定概率 bit flipping、反量化;
def Quant_1bitFlipping(param_W = '', err_rate = 0.0001, quantBits = 8, com_round = 1, client = "",  dic_parm = "", dic_berfer="", rdm = ""):
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
        # binary_send = QuantizationBbits_NP_int(params_float, B = quantBits, rounding = "nr")
    # elif quantBits == 1:
    BG = 8
    binary_send = Quantization1bits_NP_int(params_float,  BG = BG,)

    assert binary_send.size == num_sum * quantBits
    ##==========================================  Bit Flipping  ==================================================

    binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size )
    # print(f"{binary_recv.mean()}, {binary_recv.min()}, {binary_recv.max()}")
    # binary_recv = rdm.binomial(n = 1, p = err_rate, size = binary_send.size )
    binary_recv = binary_recv ^ binary_send

    err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================
    # if quantBits > 1:
        # param_recv = deQuantizationBbits_NP_int(binary_recv, B = quantBits)
    # elif quantBits == 1:
    param_recv = deQuantization1bits_NP_int(binary_recv, BG = BG)
    ##============================================= 将反量化后的实数序列变成字典形式 =======================================
    param_recover = {}
    start = 0
    end = 0
    for key in pam_order:
        end += pam_size_len[key][1]
        param_recover[key] = param_recv[start:end].reshape(pam_size_len[key][0])
        start += pam_size_len[key][1]

    ##=================================================== 结果打印和记录 ================================================
    dic_parm[client] = param_recover
    dic_berfer[client] = {"ber":err_rate }
    return



err_dist = np.loadtxt("/home/jack/公共的/Python/FedAvg/LDPC/Err_Distribution_5GLDPC.txt", delimiter = ' ')
lastrow = np.zeros(err_dist.shape[1], )
lastrow[0] = 3
err_dist = np.vstack([err_dist, lastrow])
berfer = np.loadtxt("/home/jack/公共的/Python/FedAvg/LDPC/SNR_BerFer_5GLDPC.txt", delimiter = ' ')
l5gfr2BitExtraBerFer = np.array([[0.000000,  0.0473476854,  0.0703531729 ],
                                 [0.250000,  0.0179997841,  0.0269861831 ],
                                 [0.500000,  0.0039655110,  0.0060634724 ],
                                 [0.750000,  0.0006277281,  0.0009383081 ],
                                 [1.000000,  0.0000550000,  0.0000770000 ],
                                 [1.250000,  0.0000014500,  0.0000020500 ]])


# array([[ 0.0000000,  0.1611016,  0.9956173,  49.9532517],
#        [ 0.2500000,  0.1332203,  0.9480830,  49.3041984],
#        [ 0.5000000,  0.0893237,  0.7431130,  45.5720836],
#        [ 0.7500000,  0.0408117,  0.3847760,  36.5045025],
#        [ 1.0000000,  0.0110951,  0.1134080,  25.8912509],
#        [ 1.2500000,  0.0016331,  0.0175140,  18.6358786],
#        [ 1.5000000,  0.0001316,  0.0014770,  14.7466427],
#        [ 1.7500000,  0.0000055,  0.0000850,  12.5056155],
#        [ 2.0000000,  0.0000002,  0.0000070,  10.9944520],
#        [ 2.2500000,  0.0000000,  0.0000010,  9.8766209],
#        [ 2.5000000,  0.0000000,  0.0000010,  9.0037550],
#        [ 2.7500000,  0.0000000,  0.0000010,  8.2981793],
#        [ 3.0000000,  0.0000000,  0.0000000,  7.7118247]])

## 将联邦学习得到的浮点数依次：量化int、模拟编解码、反量化;
def  Quant_LDPC_BPSK_AWGN_equa(com_round = 1, client = '', param_W = '', snr = 2.0 , quantBits = 8, dic_parm = " ", dic_berfer='', lock = None):
    # np.random.seed(int(client[6:]) + com_round)
    np.random.seed()
    # print(f"  CommRound {com_round}, {client}")
    ## 信源、统计初始化
    # source = SourceSink()
    # source.InitLog(promargs = topargs, codeargs = coderargs)
    # print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    codedim = 960
    ##================================================= 将参数字典序列化为向量 =============================================
    pam_size_len = {}
    pam_order = []
    params_float = np.empty((0, 0))
    num_sum = 0

    for key, val in param_W.items():
        pam_order.append(key)
        tmp_list = []
        tmp_list.append(val.shape)
        tmp_list.append(val.size)
        num_sum += val.size
        pam_size_len[key] = tmp_list
        params_float = np.append(params_float, val)
        # print(key, val.shape)

    BG = 7
    ##================================================= 量化 ===========================================================
    if quantBits > 1:
        binary_send = QuantizationBbits_NP_int(params_float, B = quantBits, rounding = "nr")
    elif quantBits == 1:
        binary_send = Quantization1bits_NP_int(params_float,  BG = BG,)

    len_af_quant = binary_send.size
    assert binary_send.size == num_sum * quantBits

    ##================== 将发送信息补齐为信息位的整数倍 ====================
    total_frames = int(math.ceil(binary_send.size / codedim))
    patch_len = total_frames * codedim - binary_send.size
    if patch_len != 0:
        binary_send = np.append(binary_send, np.zeros((patch_len, ), dtype = np.int8 ))

    ##==========================================  编码、调制、信道、译码 ==================================================
    raw = np.abs(berfer[:,0] - snr).argmin()
    wer = berfer[raw, 2]
    frame_err = np.random.binomial(n = 1, p = wer, size = total_frames)

    binary_recv = np.empty((0, 0), dtype = np.int8)
    for fidx in range(total_frames):
        # print("\r   " + "▇"*int(fidx/total_frames*30) + f"{fidx/total_frames*100:.5f}%", end="")
        ##========== 帧切块 ===========
        uu = binary_send[fidx * codedim : (fidx + 1) * codedim]

        if frame_err[fidx] == 1:
            num_err_bits = np.random.choice(np.arange(codedim), 1, p= err_dist[raw, 1:]/err_dist[raw, 1:].sum())[0]
            bits_flip = np.zeros(codedim, dtype = np.int8 )
            where = np.random.choice(np.arange(codedim), num_err_bits ,replace = False )
            bits_flip[where] = 1
            uu_hat = uu ^ bits_flip
            binary_recv = np.append(binary_recv, uu_hat)
        elif frame_err[fidx] == 0:
            binary_recv = np.append(binary_recv, uu)

    err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================
    if quantBits > 1:
        param_recv = deQuantizationBbits_NP_int(binary_recv[:len_af_quant], B = quantBits)
    elif quantBits == 1:
        param_recv = deQuantization1bits_NP_int(binary_recv[:len_af_quant], BG = BG)

    ##============================================= 将反量化后的实数序列 变成字典形式 ======================================
    param_recover = {}
    start = 0
    end = 0
    for key in pam_order:
        end += pam_size_len[key][1]
        param_recover[key] =  param_recv[start:end].reshape(pam_size_len[key][0])
        start += pam_size_len[key][1]

    ##=================================================== 结果打印和记录 ================================================
    dic_parm[client] = param_recover
    dic_berfer[client] = {"ber":err_rate }
    return


# raw = 2
# codedim = 960
# err_dis = np.zeros(codedim)
# for i in range(100000):
#     num_err_bits = np.random.choice(np.arange(codedim), 1, p = err_dist[raw, 1:]/err_dist[raw, 1:].sum())[0]
#     err_dis[num_err_bits] += 1

##=======================================================================================

def acc2Qbits1(acc):
    if acc <= 0.8:
        bits = 8
        lr = 0.001
    elif 0.8 < acc <= 0.9:
        bits = 6
        lr = 0.002
    elif 0.9 < acc <=0.95:
        bits = 4
        lr = 0.003
    elif 0.95 < acc:
        bits = 1
        lr = 0.001
    return bits, lr

# def acc2Qbits(acc):
#     if acc <= 0.8:
#         bits = 8
#     elif 0.8 < acc <= 0.9:
#         bits = 6
#     elif 0.9 < acc:
#         bits = 4
#     else:
#         bits = 8
#     return bits

# array([[ 0.0000000,  0.1611016,  0.9956173,  49.9532517],
#        [ 0.2500000,  0.1332203,  0.9480830,  49.3041984],
#        [ 0.5000000,  0.0893237,  0.7431130,  45.5720836],
#        [ 0.7500000,  0.0408117,  0.3847760,  36.5045025],
#        [ 1.0000000,  0.0110951,  0.1134080,  25.8912509],
#        [ 1.2500000,  0.0016331,  0.0175140,  18.6358786],
#        [ 1.5000000,  0.0001316,  0.0014770,  14.7466427],
#        [ 1.7500000,  0.0000055,  0.0000850,  12.5056155],
#        [ 2.0000000,  0.0000002,  0.0000070,  10.9944520],
#        [ 2.2500000,  0.0000000,  0.0000010,  9.8766209],
#        [ 2.5000000,  0.0000000,  0.0000010,  9.0037550],
#        [ 2.7500000,  0.0000000,  0.0000010,  8.2981793],
#        [ 3.0000000,  0.0000000,  0.0000000,  7.7118247]])
def acc2Qbits(acc, snr = 2.0 ):
    if 1.2 <= snr:
        if acc <= 0.8:
            bits = 8
            lr = 0.001
        elif 0.8 < acc <= 0.9:
            bits = 6
            lr = 0.002
        elif 0.9 < acc <=0.95:
            bits = 4
            lr = 0.003
        elif 0.95 < acc:
            bits = 1
            lr = 0.001
        else:
            bits = 8
            lr = 0.001
    elif 1 <= snr < 1.2:
        if acc <= 0.8:
            bits = 8
            lr = 0.001
        elif 0.8 < acc:
            bits = 1
            lr = 0.002
    elif  snr < 1:
            bits = 1
            lr = 0.002

    return bits, lr

# ##     SNR        BER            FER
# ([[0.000000, 0.1838549728 , 0.9952159842],
# [  0.250000, 0.1418919170,  0.9472420121],
# [  0.500000, 0.0914873711,  0.7425752780],
# [  0.750000, 0.0412932836,  0.3859083010],
# [  1.000000, 0.0111733911,  0.1136940000],
# [  1.250000, 0.0016431222,  0.0176500000],
#   ## 1.250000 0.0016257687  0.0175260000
# [  1.500000, 0.0001304365,  0.0014870000],
# [  1.750000, 0.0000052276,  0.0000810000],
# [  2.000000, 0.0000002755,  0.0000090000]])
def err2Qbits(acc, err = 0.2):
    if err < 0.01:
        if acc <= 0.8:
            bits = 8
            lr = 0.001
        elif 0.8 < acc <= 0.9:
            bits = 6
            lr = 0.002
        elif 0.9 < acc <=0.95:
            bits = 4
            lr = 0.003
        elif 0.95 < acc:
            bits = 1
            lr = 0.002
        else:
            bits = 8
            lr = 0.001
    elif 0.01 <= err <= 0.05 :
        if acc <= 0.7:
            bits = 8
            lr = 0.001
        elif 0.7 < acc:
            bits = 1
            lr = 0.002
    elif  0.05 < err:
            bits = 1
            lr = 0.003
    return bits, lr




def acc2Qbits_origin(acc, snr = 2.0 ):
    if snr >= 1.2:
        if acc <= 0.8:
            bits = 8
            lr = 0.001
        elif 0.8 < acc <= 0.9:
            bits = 6
            lr = 0.002
        elif 0.9 < acc <=0.95:
            bits = 4
            lr = 0.003
        elif 0.95 < acc:
            bits = 1
            lr = 0.004
        else:
            bits = 8
            lr = 0.001
    elif 1 <= snr < 1.2:
        if acc <= 0.8:
            bits = 8
            lr = 0.001
        elif 0.8 < acc:
            bits = 1
            lr = 0.004
    elif  snr < 1:
            bits = 1
            lr = 0.004

    return bits, lr


def Qbits2Lr(num_bits):
    if num_bits  <= 4:
        Lr = 0.005
    elif 4 < num_bits:
        Lr = 0.001
    return Lr




def Qbits2Lr_1(num_bits):
    if num_bits  <= 4:
        Lr = 0.004
    elif 4 < num_bits <= 6:
        Lr = 0.002
    elif 6 < num_bits:
        Lr = 0.001

    return Lr





































































































































































































