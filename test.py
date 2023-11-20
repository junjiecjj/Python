

import math
import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params
import copy
import torch
import torchvision
from torchvision import transforms as transforms

err_dist = np.loadtxt("/home/jack/公共的/Python/FedAvg/LDPC/Err_Distribution_5GLDPC.txt", delimiter = ' ')
lastrow = np.zeros(err_dist.shape[1], )
lastrow[0] = 3
err_dist = np.vstack([err_dist, lastrow])
berfer = np.loadtxt("/home/jack/公共的/Python/FedAvg/LDPC/SNR_BerFer_5GLDPC.txt", delimiter = ' ')


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
    return   param

## 用在并行中的np的多比特的量化
def QuantizationBbits_NP_int(params,  B = 8, rounding = "nr"):
    # print("      QuantizationBbits_NP_int\n")

    # print(f"{B} Bit quantization..")
    G =  2**(B - 1)
    # Scale_up = params * G
    # Round = np.round(Scale_up)
    # Clip = np.clip(Round, a_min = -1*G, a_max = G - 1,)
    if rounding == "sr":
        Clip = np.clip(SR_np(params * G), a_min = -1*G, a_max = G - 1,)

    elif rounding == "nr":
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
    num_dig = int(bin_recv.size/B)
    param_recv = np.zeros(num_dig, dtype = np.int32 )
    for idx in range(num_dig):
        param_recv[idx] = signed_bin2dec(''.join([str(num) for num in bin_recv[idx*B:(idx+1)*B]]))
    param_recv = (param_recv*1.0 )/G
    return param_recv.astype(np.float32)

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

    binary_send = QuantizationBbits_NP_int(params_float, B = quantBits, rounding = "nr")


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
            where = np.random.choice(np.arange(codedim), num_err_bits ,replace=False )
            bits_flip[where] = 1
            uu_hat = uu ^ bits_flip
            binary_recv = np.append(binary_recv, uu_hat)
        elif frame_err[fidx] == 0:
            binary_recv = np.append(binary_recv, uu)

    err_rate = (binary_recv != binary_send).mean()
    ##================================================= 反量化 =========================================================

    param_recv = deQuantizationBbits_NP_int(binary_recv[:len_af_quant], B = quantBits)

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







