





import numpy as np

import torch



##======================================================================================================
##============  端到端的量化函数
##======================================================================================================

## B比特的 stochastic rounding (SR), torch
def SR_torch(param):
    p = param - torch.floor(param)
    param = torch.floor(param) + torch.bernoulli(p)
    return param

## B比特的量化全流程, torch
def QuantilizeBbits_torch(params, G = None, B = 8, rounding = 'sr'):
    if G == None:
        G =  2**(B - 1)
    if rounding == 'nr':
        ## nearest rounding (NR)
        params = torch.clamp(torch.round(params * G), min = -G, max = G - 1, )/G
    elif rounding == 'sr':
        ### stochastic rounding (SR)
        params = torch.clamp(SR_torch(params * G), min = -G, max = G - 1, )/G
    return params

## 1比特的全流程，nearest rounding (NR), torch
def NR1Bit_torch(params, G = None, B = 8):
    if G == None:
        G =  2**(B - 1)
    params = torch.where(params < 0, -1, 1) / G
    return params


## 1比特的全流程，stochastic rounding (SR), torch
def SR1Bit_torch(params, G = None, BG = 8):
    if G == None:
        G =  2**BG
    p = (params*G + 1)/2
    p = torch.clamp(p, min = 0, max = 1, )
    param = torch.bernoulli(p)
    param = torch.where(param < 1, -1, param)/G
    return param


##=========================================================================================
##  发送方的量化函数
##=========================================================================================
## B比特的 stochastic rounding (SR), np
def SR_np(param):
    p = param - np.floor(param)

    f1 = np.frompyfunc(lambda x : int(np.random.binomial(1, x, 1)[0]), 1, 1)
    # a = torch.bernoulli(torch.tensor(p)).numpy()
    # print(f"1:    client ")
    # print(p)
    # param = np.floor(param) + a
    # print(a)
    param = np.floor(param) + f1(p).astype('float32')
    return  param

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
##         长度为 n*B 的0，1整数序列, 量化后的结果, np.array(np.int8);
"""
## 用在并行中的np的多比特的量化
def QuantizationBbits_NP_int(params, G, B = 8, rounding = "nr"):
    Ub =  2**(B - 1)
    if rounding == "sr":
        Clip = np.clip(SR_np(params * G), a_min = -1*Ub, a_max = Ub - 1,)
    elif rounding == "nr":
        Clip = np.clip(np.round(params * G), a_min = -1*Ub, a_max = Ub - 1,)
    Int =  np.array(Clip, dtype = np.int32)

    bin_len = int(Int.size * B)
    binary_send = np.zeros(bin_len, dtype = np.int8 )
    for idx, num in enumerate(Int):
        binary_send[idx*B : (idx+1)*B] = [int(b) for b in  np.binary_repr(num, width = B)]
    return binary_send

## 用在并行中的np的1比特的量化, stochastic rounding (SR),
# def Quantization1bits_NP_int(params,  BG = 8,):
#     G =  2**BG
#     p = (params * G + 1)/2
#     p = np.clip(p, a_min = 0, a_max = 1, )
#     f1 = np.frompyfunc(lambda x : int(np.random.binomial(1, x, 1)[0]), 1, 1)
#     Int = f1(p).astype(np.int8)
#     return Int
def Quantization1bits_NP_int(params,  G = 256,):
    # G =  2**BG
    p = (params * G + 1)/2
    p = np.clip(p, a_min = 0, a_max = 1, )
    # print(p)
    Int =  np.random.binomial(1, p).astype(np.int8)
    return Int

##=========================================================================================
##                             接收方的量化函数
##=========================================================================================

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
##         长度为 n 的实数序列, np.array(np.float32);
"""
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
def deQuantizationBbits_NP_int(bin_recv, G, B = 8):
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G

    return param_recv.astype(np.float32)

## 用在并行中的np的1比特的反量化
def deQuantization1bits_NP_int(bin_recv, G = 1, ):
    param_recv = np.where(bin_recv < 1, -1, bin_recv).astype(np.float32)/G
    return param_recv


##=========================================================================================
##=========================================================================================



























































































































































































