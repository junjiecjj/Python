#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack

numpy.nonzero()
numpy.nonzero() 函数返回输入数组中非零元素的索引。

numpy.where()
numpy.where() 函数返回输入数组中满足给定条件的元素的索引。


"""
## system lib
import numpy  as np
import datetime


##  自己编写的库
from sourcesink import SourceSink
from channel import AWGN
from modulation import  BPSK
import utility
from config import args
from polar_coder import Polar

utility.set_random_seed()


polar = Polar(args)
coderargs = {'codedim':polar.codedim,
             'codelen':polar.codelen,
             'codechk':polar.codechk,
             'coderate':polar.coderate,}
source = SourceSink()
source.InitLog(promargs = args, codeargs = coderargs )


def BPSK_AWGN_Simulation(args):
    for snr in np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr):
        channel = AWGN(snr, polar.coderate)
        source.ClrCnt()

        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            uu = source.GenerateBitStr(polar.codedim)
            cc = polar.encoder_withoutCRC(uu)
            yy = BPSK(cc)
            yy = channel.forward(yy)
            yy = utility.yyToProb(yy, channel.noise_var)

            # cc_hat = np.zeros(polar.codelen, dtype = np.int8)
            # polar.decoderSC_withoutCRC(yy, cc_hat, polar.frozenbook, polar.codelen)
            # cc_hat = polar.reverse(cc_hat)
            # uu_hat = cc_hat[np.where(polar.frozenbook == 0)]


            uu_hat, cc_hat = polar.decoderSCL_withoutCRC(yy)

            source.CntErr(uu, uu_hat)
            if source.tot_blk % 500 == 0:
                source.PrintScreen(snr = snr)
                # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
        print("  *** *** *** *** ***");
        source.PrintScreen(snr = snr);
        print("  *** *** *** *** ***\n");

        source.SaveToFile(snr = snr)
    return



BPSK_AWGN_Simulation(args)

# for snr in np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr):
#     channel = AWGN(snr, polar.coderate)
#     source.ClrCnt()

#     print( f"\nsnr = {snr}(dB):\n")
#     while source.tot_blk < 1 and source.err_blk < 1:
#         uu = source.GenerateBitStr(polar.codedim)
#         cc = polar.encoder_withoutCRC(uu)
#         yy = BPSK(cc)
#         yy = channel.forward(yy)
#         yy = utility.yyToProb(yy, channel.noise_var)

#         # cc_hat = np.zeros(polar.codelen, dtype = np.int8)
#         # polar.decoderSC_withoutCRC(yy, cc_hat, polar.frozenbook, polar.codelen)
#         # cc_hat = polar.reverse(cc_hat)
#         # uu_hat = cc_hat[np.where(polar.frozenbook == 0)]


#         uu_hat, cc_hat = polar.decoderSCL_withoutCRC(yy)

#         source.CntErr(uu, uu_hat)
#         if source.tot_blk % 500 == 0:
#             source.PrintScreen(snr = snr)
#             # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
#     print("  *** *** *** *** ***");
#     source.PrintScreen(snr = snr);
#     print("  *** *** *** *** ***\n");

#     source.SaveToFile(snr = snr)


























