
# -*- coding: utf-8 -*-
"""
Created on 2023/08/17

@author: Junjie Chen

"""


## system lib
import numpy  as np
import datetime
import copy
import math

import torch


##  自己编写的库
from LDPC.sourcesink import SourceSink
from LDPC.channel import AWGN
from LDPC.modulation import  BPSK, demodu_BPSK
from LDPC import utility
from LDPC.argsLDPC import arg as topargs
from LDPC.ldpc_coder import LDPC_Coder_llr
from LDPC.quantiation import  Quantization, deQuantization

# utility.set_random_seed()
# print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## LDPC初始化
ldpcCoder =  LDPC_Coder_llr(topargs)
coderargs = {'codedim':ldpcCoder.codedim,
             'codelen':ldpcCoder.codelen,
             'codechk':ldpcCoder.codechk,
             'coderate':ldpcCoder.coderate,
             'row':ldpcCoder.num_row,
             'col':ldpcCoder.num_col}


# print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

## 将联邦学习得到的浮点数依次：量化、编码、信道、解码、反量化;
def  Quant_BPSK_AWGN_Pipe(param_W, snr = 2.0 , quantBits = 8, com_round = 1, client = ''):
    ## 信源、统计初始化
    source = SourceSink()
    source.InitLog(promargs = topargs, codeargs = coderargs)
    ##================================================= 将参数字典序列化为向量 =============================================
    pam_size_len = {}
    pam_order = []
    params_float = np.empty((0, 0))
    num_sum = 0

    for key, val in param_W.items():
        pam_order.append(key)
        tmp = {}
        tmp_list = []
        tmp_list.append(val.shape)
        tmp_list.append(val.numel())
        num_sum += val.numel()
        pam_size_len[key] = tmp_list
        params_float = np.append(params_float, np.array(val.cpu().clone()))
        # print(key, val.shape)

    ##================================================= 量化 ===========================================================
    binary_send = Quantization(params_float, B = quantBits)
    assert binary_send.size == num_sum * quantBits

    ##================== 将发送信息补齐为信息位的整数倍 ====================
    total_frames = int(math.ceil(binary_send.size / ldpcCoder.codedim))
    patch_len = total_frames * ldpcCoder.codedim - binary_send.size
    binary_send = np.append(binary_send, np.zeros((patch_len, ), dtype = np.int8 ))

    ##==========================================  编码、调制、信道、译码 ==================================================
    source.ClrCnt()
    channel = AWGN(snr)
    binary_recv = np.empty((0, 0), dtype = np.int8)
    for fidx in range(total_frames):
        print("\r   " + "▇"*int(fidx/total_frames*100) + f"{fidx/total_frames*100:.5f}%", end="")

        ##========== 帧切块 ===========
        uu = binary_send[fidx * ldpcCoder.codedim : (fidx + 1) * ldpcCoder.codedim]
        ##=== 编码 ===
        cc = ldpcCoder.encoder(uu)
        ##=== 调制 ===
        yy = BPSK(cc)
        ##=== 信道 ===
        yy = channel.forward(yy)
        ##=== soft ===
        yy_recv = utility.yyToLLR(yy, channel.noise_var)
        ##=== 译码 ===
        uu_hat, iter_num = ldpcCoder.decoder_msa(yy_recv)
        ##=== 信息合并 ===
        binary_recv = np.append(binary_recv, uu_hat)
        ##=== 统计 ===
        source.tot_iter += iter_num
        source.CntErr(uu, uu_hat)
        # if source.tot_blk % 2 == 0:
            # source.PrintScreen(snr = snr)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    # print("  *** *** *** *** ***");
    # source.PrintScreen(snr = snr);
    # print("  *** *** *** *** ***\n");
    source.FLPerformance(snr = snr,  Cround = com_round, client = client)

    ##================================================= 反量化 =========================================================
    param_recv = deQuantization(binary_recv[:-patch_len], B = quantBits)

    ##============================================= 将反量化后的实数序列 变成字典形式 =======================================
    param_recover = {}
    start = 0
    end = 0
    for key in pam_order:
        end += pam_size_len[key][1]
        param_recover[key] = torch.tensor(param_recv[start:end].reshape(pam_size_len[key][0]))
        start += pam_size_len[key][1]

    return param_recover, source.ber, source.fer, source.ave_iter




















































































































































































































