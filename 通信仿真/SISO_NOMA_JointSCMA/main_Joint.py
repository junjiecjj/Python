

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2023

@author: jack

"""
## system lib
# import galois
import numpy  as np

##  自己编写的库
from sourcesink import SourceSink
from Config import parameters
from Channel import channelConfig
from Channel import AWGN_mac, BlockFading_mac, FastFading_mac, Large_mac
import utility
from QLDPCcoder import QLDPC_Coding, BPSK
# from SICdetector_LDPC import SeparatedDecoding_BlockFading, SeparatedDecoding_FastFading
import Modulator

utility.set_random_seed()
args = parameters()

# def QaryLDPC(args, ):
ldpc = QLDPC_Coding(args)
coderargs = {'codedim':ldpc.codedim,
             'codelen':ldpc.codelen,
             'codechk':ldpc.codechk,
             'coderate':ldpc.coderate,
             'row':ldpc.num_row,
             'col':ldpc.num_col, }

source = SourceSink()
# rho = 1
logf = f"./resultsTXT/{args.channel_type}/BER_Joint_{args.channel_type}_{args.K}U_wo_pa.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs,)

## modulator
modem, Es, bps = Modulator.modulator( args.type, args.M)
framelen = int(ldpc.codelen/bps)

## 遍历SNR
sigma2dB = np.arange(14.4, 20, 0.2)  # dB
sigma2W = args.K * 10**(-sigma2dB/10.0)  # 噪声功率 w

for sigma2db, sigma2w in zip(sigma2dB, sigma2W):
    source.ClrCnt()
    print( f"\n sigma2 = {sigma2db}(dB), {sigma2w}(w):")
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        if args.channel_type == 'AWGN':
            H = AWGN_mac(args.K, framelen)
            # H = H * P.reshape(-1,1)
        elif args.channel_type == 'block_fading':
            H = BlockFading_mac(args.K, framelen)
            # H = H * P.reshape(-1,1)
        elif args.channel_type == 'fast_fading':
            H = FastFading_mac(args.K, framelen)
            # H = H * P.reshape(-1,1)
        elif args.channel_type == 'large':
            BS_locate, users_locate, beta_Au, PL_Au = channelConfig(args.K, r = 100)
            H = Large_mac(args.K, ldpc.codelen, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigma2w)
        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)

        cc = np.zeros((args.K, ldpc.codelen), dtype = np.int8)
        for k in range(args.K):
            cc[k] =  ldpc.encoder(uu[k,:])

        ## Modulate
        symbs = np.zeros((args.K, int(ldpc.codelen/bps)), dtype = complex)
        for k in range(args.K):
            symbs[k] = BPSK(cc[k])

        ## 符号能量归一化
        symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        yy = ldpc.MACchannel(symbs, H, sigma2w)

        ##>>>>> Joint detecting & decoding
        ## llr
        pp = ldpc.post_probability(yy, H, sigma2w)
        ## Decoding
        uu_hat, iter_num = ldpc.decoder_FFTQSPA(pp, maxiter = 50)
        source.tot_iter += (iter_num * args.K)

        # break
        source.CntBerFer(uu, uu_hat)
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = sigma2db)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    # break
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = sigma2db)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = sigma2db)
    # return

# QaryLDPC(args)

























































