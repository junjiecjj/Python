






# import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
# import copy
# import torch


"""
## 功能: 将浮点数序列量化为 0/1 比特序列
## Input:
##  params:
##        格式 : np.array(size = (n, ), dtype = np.float)
##        含义 : 需要被量化的 n长 实数向量;
##      B :
##         Number of bits for quantization, 量化比特数
## Output:
##     binary_send:
##         长度为 n*B 的0，1整数序列, 量化后的结果;
"""
def Quantization(params,  B = 8):
    # print(f"{B} Bit quantization..")
    G =  2**(B - 1)

    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)

    Clip = np.clip(np.round(params * G), a_min = -1*G, a_max = G - 1, )

    Shift = Clip + G

    Uint =  np.array(Shift, dtype = np.uint32 )

    bin_len = int(Uint.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Uint):
        binary_send[idx*B:(idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B+1)[-B:]]

    return binary_send


"""
## 功能: 将译码后的01序列 bin_recv 反量化为实数
## Input:
##  bin_recv:
##        格式 : np.array(size = (n*B, ), dtype = np.int8)
##        含义 : 需要被量化的 n*B 长的0/1整数向量;
##      B :
##         Number of bits for quantization, 量化比特数
## Output:
##     binary_send:
##         长度为 n 的实数序列;
"""
def deQuantization(bin_recv, B = 8):
    G =  2**(B - 1)
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.uint32 )
    for idx in range(num_dig):
        param_recv[idx] = int(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]), 2)
    param_recv = (param_recv*1.0 - G)/G
    return param_recv






































































































































































































