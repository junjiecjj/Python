

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:08:38 2024

@author: jack

numpy.nonzero()
numpy.nonzero() 函数返回输入数组中非零元素的索引。

numpy.where()
numpy.where() 函数返回输入数组中满足给定条件的元素的索引。

"""
## system lib
# import galois
import numpy  as np
import time
import copy


##  自己编写的库
from sourcesink import SourceSink
from Config import parameters
from Channel import channelConfig
from Channel import AWGN_mac, BlockFading_mac, FastFading_mac, Large_rayleigh_fast
import utility
from LDPCcoder import LDPC_Coder, BPSK
# from SICdetector_LDPC import SeparatedDecoding_BlockFading, SeparatedDecoding_FastFading
import Modulator
from CapacityOptimizer import OFDMWaterFull
utility.set_random_seed(42)

args = parameters()

# def QaryLDPC(args, ):
ldpc = LDPC_Coder(args)
coderargs = {'codedim':ldpc.codedim,
             'codelen':ldpc.codelen,
             'codechk':ldpc.codechk,
             'coderate':ldpc.coderate,
             'row':ldpc.num_row,
             'col':ldpc.num_col, }

source = SourceSink()
# rho = 1
# logf = f"./resultsTXT/{args.channel_type.split('-')[0]}/BER_Joint_Block_{args.K}u_w_powerdiv_{rho}.txt"
logf = f"./resultsTXT/{args.channel_type.split('_')[1]}/BER_OFDM_{args.channel_type}_{args.K}U_wo_pa.txt"
source.InitLog(logfile = logf, promargs = args, codeargs = coderargs,)

## modulator
modem, Es, bps = Modulator.modulator( args.type, args.M)
framelen = int(ldpc.codelen/bps)
# inteleaverM = inteleaver(args.K, int(ldpc.codelen/bps))

BS_locate, users_locate, beta_Au, PL_Au, d_Au = channelConfig(args.K, r = 100, rmin = args.rmin)

# 遍历SNR
n0     = np.arange(-120, -130, -1)        # 噪声功率谱密度, dBm/Hz
n00    = 10**(n0/10.0)/1000               # 噪声功率谱密度, Watts/Hz
N0     = n00 * args.B                     # 噪声功率, Watts

for noisePsd, noisepower in zip(n0, N0):
    source.ClrCnt()
    print( f"\n noisePsd = {noisePsd}(dBm/Hz), {noisepower}(w):")
    P = OFDMWaterFull(PL_Au, args.P_total,)
    P = np.ones(args.K) * args.P_total/args.K
    t0 = time.time()
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        # if args.channel_type == 'AWGN':
        #     H = AWGN_mac(args.K, framelen)
        # elif args.channel_type == 'block-fading':
        #     H = BlockFading_mac(args.K, framelen)
        # elif args.channel_type == 'fast-fading':
        #     H = FastFading_mac(args.K, framelen)
        if args.channel_type == 'large_fast':
            H = Large_rayleigh_fast(args.K, framelen, PL_Au, noisevar = noisepower)
        H = H * np.sqrt(P[:,None])

        ## 编码
        uu = source.SourceBits(args.K, ldpc.codedim)

        cc = np.zeros((args.K, ldpc.codelen), dtype = np.int8)
        for k in range(args.K):
            cc[k,:] =  ldpc.encoder(uu[k,:])

        ## Modulate
        symbs = np.zeros((args.K, int(ldpc.codelen/bps)), dtype = complex)
        for k in range(args.K):
            symbs[k] = BPSK(cc[k])

        ## 符号能量归一化
        symbs  = symbs / np.sqrt(Es)

        ## Pass Channel
        # yy = ldpc.MACchannel(symbs, H, 1)
        noise = np.sqrt(1/2) * (np.random.randn(*symbs.shape) + 1j*np.random.randn(*symbs.shape))
        yy = H * symbs + noise

        ##>>>>> seperated decoding
        uu_hat = np.zeros((args.K, ldpc.codedim), dtype = np.int8)
        for k in range(args.K):
            Noise = np.ones(yy.shape[1]) * 1
            llrk = Modulator.demod_fastfading(copy.deepcopy(modem.constellation), yy[k,:], 'soft', H = H[k,:],  Es = Es,  noise_var = Noise)
            uu_hat[k,:], iterk = ldpc.decoder_msa(llrk)
            source.tot_iter += iterk
        ## llr
        # pp = ldpc.post_probability(yy, H, 1)
        ## Decoding
        # uu_hat, iter_num = ldpc.decoder_FFTQSPA(pp, maxiter = 50)

        # source.tot_iter += iter_num * args.K
        source.tot_time = (time.time() - t0)*args.K/60.0
        # source.CntSumErr(uu_sum, uu_hat_sum)
        # break
        source.CntBerFer(uu, uu_hat)
        if source.tot_blk % 1 == 0:
            source.PrintScreen(snr = noisePsd)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    # break
    print("  *** *** *** *** ***")
    source.PrintScreen(snr = noisePsd)
    print("  *** *** *** *** ***\n")
    source.SaveToFile(filename = logf, snr = noisePsd)
    # return

# QaryLDPC(args)

























































