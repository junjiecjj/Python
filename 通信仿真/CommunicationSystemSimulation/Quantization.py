#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:10:33 2024

@author: jack
"""
import numpy as np





## 用在并行中的np的多比特的量化
def QuantizationBbits_NP_int(params, B = 8, ):
    G =  2**(B - 1)

    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1,)
    # Shift = Clip
    Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        binary_send[idx*B : (idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B)]
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


## 用在并行中的np的多比特的反量化
def deQuantizationBbits_NP_int(bin_recv, B = 8):
    G = 2**(B-1)
    padlen = np.ceil(bin_recv.size/B) * B - bin_recv.size
    bin_recv = np.pad(bin_recv, (0, int(padlen)), constant_values=(0, 0))
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv.astype(np.float32)



















