#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:26:19 2023

@author: jack
"""


##============================================================================================================
##============================================================================================================
##   将网络的参数转为实数序列，然后 flipping, 然后反序列化
##============================================================================================================
##============================================================================================================

import torch
import numpy as np

def Quantilize_end2end1(params,  B = 8):
    print(f"1  B = {B}")
    G =  2**(B - 1)
    params = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )/G
    return params

def QuantizationNP_int(params,  B = 8):
    print(f"2  B = {B}")
    # print(f"{B} Bit quantization..")
    G =  2**(B - 1)
    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)
    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1,)
    # Shift = Clip
    Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B)]

    return binary_send


def QuantizationTorch_int(params, G = None, B = 8):
    print(f"3  B = {B}")
    # print(f"{B} Bit quantization..")
    if G ==None:
        G =  2**(B - 1)

    Clip = torch.clamp(torch.round(params * G), min = -1*G, max = G - 1, )

    Int = Clip.type(torch.int32)

    # Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.numel() * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        # print(f"{num} {num.item()}")
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num.item(), width = B)]

    return binary_send

def signed_bin2dec(bin_str: str) -> int:
    '''
    函数功能：2进制补码字符串 -> 10进制整数\n
    输入：2进制补码字符串，不可带正负号，前后可加任意个 \\n 和 空格，数字间可加下划线\n
    输出：10进制整数，只保留负号，正号不保留
    '''
    if (bin_str[:2] == '0b'):
        bin_str = bin_str[2:]
    elif (bin_str[0] == '0'):
        return int(bin_str, base = 2)
    elif (bin_str[0] == '1'):
        a = int(bin_str, base = 2) # 此语句可检查输入是否合法
        return a - 2**len(bin_str)


def deQuantizationNP_int(bin_recv, B = 8):
    print(f"4  B = {B}")
    G = 2**(B-1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv.astype(np.float32)



data = torch.randn(100, )


B = 7
##=======
data_end2end_np = Quantilize_end2end1(np.array(data), B = B)


##=======
binary_send = QuantizationTorch_int(data,  B = B)
err_rate = 0
binary_recv = np.random.binomial(n = 1, p = err_rate, size = binary_send.size  )
binary_recv = binary_recv ^ binary_send
dt = deQuantizationNP_int(binary_recv, B = B)


##=======
binary_send1 = QuantizationNP_int(np.array(data),  B = B)
err_rate = 0
binary_recv1 = np.random.binomial(n = 1, p = err_rate, size = binary_send1.size  )
binary_recv1 = binary_recv1 ^ binary_send1
dt1 = deQuantizationNP_int(binary_recv1, B = B)

