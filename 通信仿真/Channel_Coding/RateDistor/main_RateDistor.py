#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack


"""
## system lib
import numpy  as np
import datetime
import commpy as cpy
import copy
import argparse
import socket, getpass, os, sys

##  自己编写的库
from sourcesink import SourceSink
from sourcesink_flip import SourceSink_flip
from channel import AWGN
from channel import Rayleigh
# from modulation import  BPSK,  demodu_BPSK
import utility
# from argsLDPC import args
from ldpc_coder import LDPC_Coder_llr
import Modulator

utility.set_random_seed(1)

def parameters():
    # 获取当前系统主机名
    host_name = socket.gethostname()
    # 获取当前系统用户名
    user_name = getpass.getuser()
    # 获取当前系统用户目录
    user_home = os.path.expanduser('~')
    home = os.path.expanduser('~')

    ldpc_args = {
    "minimum_snr" : 2 ,
    "maximum_snr" : 13,
    "increment_snr" : 1,
    "maximum_error_number" : 500,
    "maximum_block_number" : 1000000,

    ## LDPC***0***PARAMETERS
    "max_iteration" : 50,
    "encoder_active" : 1,
    "file_name_of_the_H" : "PEG1024regular0.5.txt",

    ## others
    "home" : home,
    "smallprob": 1e-15,

    ##>>>>>>>  modulation param
    "type" : 'qam',
    "M":  16,

    # "type" : 'psk',
    # "M":  2,  # BPSK
    # "M":  4,  # QPSK
    # "M":  8,  # 8PSK
    }
    args = argparse.Namespace(**ldpc_args)
    return args

args = parameters()

def main(args,  ):
    ldpc =  LDPC_Coder_llr(args)
    coderargs = {'codedim':ldpc.codedim,
                 'codelen':ldpc.codelen,
                 'codechk':ldpc.codechk,
                 'coderate':ldpc.coderate,
                 'row':ldpc.num_row,
                 'col':ldpc.num_col}

    source = SourceSink()

    P = np.arange(0.3, 0.51, 0.01)

    for p in P:
        logf = f"BerFer_RateDistor_{p}.txt"
        source.InitLog(logfile = logf, promargs = args, codeargs = coderargs, )
        source.ClrCnt()
        print( f"\n p = {p}:\n")
        while source.tot_blk < args.maximum_block_number: # and source.err_blk < args.maximum_error_number:
            source.tot_blk += 1
            uu = source.GenerateBitStr(ldpc.codelen)

            ## 对数域llr
            llr = np.zeros_like(uu, dtype = np.float64)
            llr[np.where(uu == 1)] = p
            llr[np.where(uu == 0)] = 1 - p
            llr = np.log(llr / (1 - llr))
            llr[np.isinf(llr)] = 10**(np.sign(llr[np.isinf(llr)]) * 20)

            uu_hat, iter_num, unpass_equ_rate, pass_chk = ldpc.decoder_spa(llr)
            # uu_hat2 = ldpc.encoder(uu_hat[ldpc.codechk:])
            if pass_chk:
                source.tot_iter += iter_num
                source.CntErrPass(uu, uu_hat)
            else:
                source.unpass_Equ_rate += unpass_equ_rate
                source.CntErrUnPass()

            if source.tot_blk % 2 == 0:
                source.PrintScreen(p = p)
        print("  *** *** *** *** ***");
        source.PrintScreen(p = p);
        print("  *** *** *** *** ***\n");
        source.SaveToFile(p = p, filename = logf)
    return

# main(args)

ldpc =  LDPC_Coder_llr(args)
coderargs = {'codedim':ldpc.codedim,
              'codelen':ldpc.codelen,
              'codechk':ldpc.codechk,
              'coderate':ldpc.coderate,
              'row':ldpc.num_row,
              'col':ldpc.num_col}

source = SourceSink_flip()
P = np.arange(0.1, 0.51, 0.01)
p = 0.2
# for p in P:
logf = f"BerFer_RateDistor_{p}.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs, )
source.ClrCnt()
while source.tot_blk < args.maximum_block_number: # and source.err_blk < args.maximum_error_number:
    # uu = source.GenerateBitStr(ldpc.codedim)
    # cc = ldpc.encoder(uu)
    cc = source.GenerateBitStr(ldpc.codelen)
    source.tot_blk += 1
    cc_hat, numequ, numflip, pass_chk, rest_unpass_rate = ldpc.find(copy.deepcopy(cc), p)

    ###########################
    if pass_chk:
        source.numflip_pass += numflip
        source.num_equ_pass += numequ
        source.CntErrPass(cc, cc_hat)
    else:
        source.numflip_upass += numflip
        source.num_equ_upass += numequ
        source.rest_unpass += rest_unpass_rate
        source.CntErrUnPass()

    if source.tot_blk % 1 == 0:
        source.PrintScreen(p = p)
    # break
print("  *** *** *** *** ***");
source.PrintScreen(p = p);
print("  *** *** *** *** ***\n");
# source.SaveToFile(p = p, filename = logf)


print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))





# from itertools import combinations


# def Prange_ISD(H, s, t):  # 校验矩阵H,校验子s,目标重量t
#     n = H.ncols()  # 码长n
#     k = H.ncols() - H.nrows()  # 维数k
#     S = set(range(n))  # 全集

#     e = vector(Zmod(2), [0] * n)  # 初始化错误向量
#     All_set = combinations(range(n), k)  # 所有可能的信息集
#     for ISet in All_set:
#         I = set(ISet)  # 选择一个信息集I
#         J = S - I  # I的补集J
#         HJ = H.matrix_from_columns(J)  # 抽出J中的列
#         try:
#             U = HJ.inverse()  # 检查是否可逆，是则为一个信息集
#         except:
#             continue  # 不是信息集
#         UH = U * H # 初等行变换

#         A = UH.matrix_from_columns(I)  # (UH)_I = A,(UH)_J = Id
#         s_ = s * U.T  # s'
#         if s_.hamming_weight() == t:  # 检查是否符合汉明重量
#             for j in range(n - k):
#                 e[list(J)[j]] = s_[j]  # e_I=e_0,e_J=s'
#             return e
#     return 'Not Found.'


# H = [(0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1), (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
#      (1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
#      (1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0), (0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1),
#      (1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1), (1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0),
#      (1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1), (0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0),
#      (1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)]
# H = matrix(Zmod(2), H) # 校验矩阵
# s = vector(Zmod(2), [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]) #校验子
# e = Prange_ISD(H, s, 3)
# print(e)









































































































































































































































































































































































































































































