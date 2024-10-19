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
import multiprocessing

##  自己编写的库
from sourcesink import SourceSink
from channel import AWGN
from modulation import BPSK
# from modulation import demodu_BPSK
import utility
from argsLDPC import arg as topargs
from ldpc_coder import LDPC_Coder_llr




# source.InitLog(promargs = ldpcarg, codeargs = coderargs)


def BPSK_AWGN_Simulation(i, name, args, snr = 2.0, dic_berfer = '',  lock = None):
    np.random.seed(i)
    source = SourceSink()
    source.ClrCnt()

    channel = AWGN(snr)

    print( f"\nsnr = {snr}(dB): ")
    frame = 0
    while source.tot_blk < args.maximum_block_number and source.err_blk < args.maximum_error_number:
        frame += 1
        uu = source.GenerateBitStr(ldpcCoder.codedim)
        cc = ldpcCoder.encoder(uu)
        yy = BPSK(cc)
        yy = channel.forward(yy)
        yy = utility.yyToLLR(yy, channel.noise_var)
        uu_hat, iter_num = ldpcCoder.decoder_msa(yy)
        source.tot_iter += iter_num
        source.CntErr(uu, uu_hat)
        # if source.tot_blk % 2 == 0:
            # source.PrintScreen(snr = snr)
            # source.PrintResult(log = f"{snr:.2f}  {source.m_ber:.8f}  {source.m_fer:.8f}")
    dic_berfer[name] = {"ber":source.ber, "fer":source.fer, "ave_iter":source.ave_iter }
    if lock != None:
        lock.acquire()
        source.PrintScreen(snr = snr);
        source.SaveToFile(snr = snr)
        lock.release()
    return



# BPSK_AWGN_Simulation(topargs)

if __name__ == '__main__':
    utility.set_random_seed(1)
    # print(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    ldpcCoder =  LDPC_Coder_llr(topargs)
    coderargs = {'codedim':ldpcCoder.codedim,
                 'codelen':ldpcCoder.codelen,
                 'codechk':ldpcCoder.codechk,
                 'coderate':ldpcCoder.coderate,
                 'row':ldpcCoder.num_row,
                 'col':ldpcCoder.num_col}
    utility.WrLogHead(promargs = topargs, codeargs = coderargs)

    m = multiprocessing.Manager()
    # dict_param = m.dict()
    dict_berfer = m.dict()
    lock = multiprocessing.Lock()  # 这个一定要定义为全局
    jobs = []

    for i, snr in enumerate(np.arange(topargs.minimum_snr, topargs.maximum_snr + topargs.increment_snr/2.0, topargs.increment_snr)):
        ps = multiprocessing.Process(target = BPSK_AWGN_Simulation, args=(i, f"snr={snr:.2f}(dB)", topargs, snr, dict_berfer, lock ))
        jobs.append(ps)
        ps.start()

    for p in jobs:
        p.join()

    for snr in  np.arange(topargs.minimum_snr, topargs.maximum_snr + topargs.increment_snr/2.0, topargs.increment_snr):
        name = f"snr={snr:.2f}(dB)"
        print(f"{name} {dict_berfer[name]['ber']:.8f} {dict_berfer[name]['fer']:.8f} {dict_berfer[name]['ave_iter']:.3f}")


























































































































































































































































































































































































































































































