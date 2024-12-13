


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:20:38 2023
@author: JunJie Chen
"""

import numpy as np
import os, sys
# import math
import datetime


class SourceSink_flip(object):
    def __init__(self):
        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.tot_blk = 0  # 总的帧数
        self.tot_blk_pass = 0
        self.pass_rate = 0
        self.tot_bit = 0  # 总的比特数 = 总的帧数 x 信源序列长度
        self.err_blk = 0  # 总的错误帧数
        self.err_bit = 0  # 总的错误比特数 = 每帧的错误比特之和
        self.numflip_pass  = 0
        self.avg_numflip_pass = 0
        self.num_equ_pass = 0
        self.avg_num_equ_pass = 0
        # self.tot_iter = 0  #
        # self.ave_iter = 0.0

        self.ber = 0.0   # 误比特率
        self.fer  = 0    # 误帧率

        self.tot_blk_unpass = 0
        self.unpass_rate = 0
        self.numflip_upass  = 0
        self.avg_numflip_upass = 0
        self.equ_num_upass = 0
        self.avg_num_equ_upass = 0
        self.rest_unpass = 0
        self.avg_rest_unpass = 0
        return

    def InitLog(self, logfile = "SNR_BerFer.txt", promargs = '', codeargs = ''):
        current_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        logfile = current_dir + logfile
        with open(logfile, 'a+') as f:
            print("#=====================================================================================",  file = f)
            print("                      " + self.now,  file = f)
            print("#=====================================================================================\n",  file = f)
            f.write("######### [program config] #########\n")
            for k, v in promargs.__dict__.items():
                f.write(f"{k: <25}: {v: <40}\n")
            f.write("######### [code config] ##########\n")
            for k, v in codeargs.items():
                f.write(f"{k: <25}: {v: <40}\n")
            f.write("\n#=============================== args end  ===============================\n")
        return

    def GenerateBitStr(self, Len):
        uu = np.random.randint(low = 0, high = 2, size = (Len, ), dtype = np.int8 )
        return uu

    def ClrCnt(self):
        # 计数器清零
        self.tot_blk = 0  # 总的帧数
        self.tot_blk_pass = 0
        self.pass_rate = 0
        self.tot_bit = 0  # 总的比特数 = 总的帧数 x 信源序列长度
        self.err_blk = 0  # 总的错误帧数
        self.err_bit = 0  # 总的错误比特数 = 每帧的错误比特之和
        self.numflip_pass  = 0
        self.avg_numflip_pass = 0
        self.num_equ_pass = 0
        self.avg_num_equ_pass = 0
        self.ber = 0.0   # 误比特率
        self.fer  = 0    # 误帧率

        self.tot_blk_unpass = 0
        self.unpass_rate = 0
        self.numflip_upass  = 0
        self.avg_numflip_upass = 0
        self.num_equ_upass = 0
        self.avg_num_equ_upass = 0
        self.rest_unpass = 0
        self.avg_rest_unpass = 0
        return

    def CntErrPass(self, uu, uu_hat, accumulator = 1):
        assert uu.shape == uu_hat.shape
        Len = uu.shape[-1]
        temp_err = np.sum(uu != uu_hat)
        self.tot_blk_pass += 1.0
        # if accumulator == 1:
        if temp_err > 0:
            self.err_bit += temp_err
            self.err_blk += 1

        self.tot_bit += Len
        self.ber = self.err_bit / self.tot_bit
        self.fer = self.err_blk / self.tot_blk_pass

        # self.ave_iter =  self.tot_iter / self.tot_blk_pass
        self.pass_rate = self.tot_blk_pass / self.tot_blk
        self.avg_numflip_pass = self.numflip_pass / self.tot_blk_pass
        self.avg_num_equ_pass = self.num_equ_pass / self.tot_blk_pass

        return

    def CntErrUnPass(self, accumulator = 1):
        self.tot_blk_unpass += 1
        self.unpass_rate = self.tot_blk_unpass / self.tot_blk
        self.avg_numflip_upass = self.numflip_upass / self.tot_blk_unpass
        self.avg_num_equ_upass = self.num_equ_upass / self.tot_blk_unpass
        self.avg_rest_unpass = self.rest_unpass / self.tot_blk_unpass
        return


    def SaveToFile(self, filename = "SNR_BerFer.txt", p = '', open_type = 'a+'):
        log = f"[{p:.3f}, {self.fer:.8f}, {self.ber:.8f}, {self.ave_unpass:.8f}],"
        with open(filename, open_type) as f:
            f.write(log + "\n")
        return

    def PrintScreen(self, p = '',  ):
        print(f"  p = {p:.3f},[pass]  -> passrate = {self.pass_rate:.8f}, bits = {self.err_bit}/{self.tot_bit}, blk = {self.err_blk}/{self.tot_blk_pass}, avg_numflip = {self.avg_numflip_pass:.8f}, avg_num_equ = {self.avg_num_equ_pass:.8f}, fer = {self.fer:.8f}, ber = {self.ber:.8f}")
        print(f"  p = {p:.3f},[unpass]-> unpassrate = {self.unpass_rate:.8f}, avg_rest_unpass = {self.avg_rest_unpass:.8f}, tot_blk = {self.tot_blk:.3f}, avg_numflip = {self.avg_numflip_upass:.8f}, avg_num_equ = {self.avg_num_equ_upass:.8f}")
        return
































































































































































































































































