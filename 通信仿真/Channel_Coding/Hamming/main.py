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
from hamming_coder import Hamming


utility.set_random_seed()




hamming =  Hamming(args)
coderargs = {'codedim':hamming.codedim,
             'codelen':hamming.codelen,
             'codechk':hamming.codechk,
             'coderate':hamming.coderate,}
source = SourceSink()
source.InitLog(promargs = args, codeargs = coderargs )


def  BPSK_AWGN_Simulation(args):
    for snr in np.arange(args.minimum_snr, args.maximum_snr + args.increment_snr/2.0, args.increment_snr):
        channel = AWGN(snr)
        source.ClrCnt()

        print( f"\nsnr = {snr}(dB):\n")
        while source.tot_blk <= args.maximum_block_number and source.err_blk <= args.maximum_error_number:
            uu = source.GenerateBitStr(hamming.codedim)
            cc = hamming.encoder(uu)
            yy = BPSK(cc)
            yy = channel.forward(yy)
            ## hard decoder
            # uu_hat  = hamming.decoder_hard(yy)

            ## soft decoder
            yy = utility.yyToProb(yy, 10.0**(-snr/10.0))
            uu_hat  = hamming.decoder_soft(yy)

            source.tot_iter += 1
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































