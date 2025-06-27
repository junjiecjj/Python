


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 15:20:38 2023
@author: JunJie Chen
"""

import numpy as np
import os
# import sys
import datetime


class SourceSink(object):
    def __init__(self):
        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.tot_blk = 0  # 总的帧数
        self.tot_bit = 0  # 总的比特数 = 总的帧数 x 信源序列长度
        self.err_blk = 0  # 总的错误帧数
        self.err_bit = 0  # 总的错误比特数 = 每帧的错误比特之和
        self.tot_symb = 0 # 总的错误符号数
        self.err_symb = 0 # 总的符号数
        self.tot_iter = 0    #
        self.ave_iter = 0.0  #
        self.err_sum = 0.0   # 多用户实数和错误数
        self.tot_sum = 0.0   # 多用户实数和总数
        self.ber  = 0.0   # 误比特率
        self.fer  = 0     # 误帧率
        self.ser = 0      # 误符号率
        self.sumerr = 0   #
        # self.InitLog()
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

    def SourceBits(self, J, Len):
        uu = np.random.randint(low = 0, high = 2, size = (J, Len), dtype = np.int8 )
        return uu

    def ClrCnt(self):
        # 计数器清零
        self.tot_blk = 0.0
        self.tot_bit = 0.0
        self.err_blk = 0.0
        self.err_bit = 0.0
        self.tot_symb = 0.0
        self.err_symb = 0.0
        self.tot_iter = 0.0
        self.ave_iter = 0.0
        self.err_sum = 0.0
        self.tot_sum = 0.0
        return

    # def CntErr(self, uu, uu_hat, accumulator = 1):
    #     assert uu.shape == uu_hat.shape
    #     Len = uu.shape[-1]
    #     temp_err = np.sum(uu != uu_hat)

    #     if accumulator == 1:
    #         if temp_err > 0:
    #             self.err_bit += temp_err
    #             self.err_blk += 1
    #         self.tot_blk += 1.0
    #         self.tot_bit += Len
    #         self.ber = self.err_bit / self.tot_bit
    #         self.fer = self.err_blk / self.tot_blk
    #     self.ave_iter =  self.tot_iter / self.tot_blk
    #     return

    def CntBerFer(self, uu, uu_hat, ):
        assert uu.shape == uu_hat.shape
        J, Len = uu.shape
        temp_err = np.sum(uu != uu_hat)

        if temp_err > 0:
            self.err_bit += temp_err
            self.err_blk += np.sum(np.sum(uu != uu_hat, axis = 1) > 0)
        self.tot_blk += J
        self.tot_bit += J * Len
        self.ber = self.err_bit / self.tot_bit
        self.fer = self.err_blk / self.tot_blk
        self.ave_iter =  self.tot_iter / self.tot_blk
        return

    # def CntSer(self, symbol, symbol_hat, ):
    #     assert symbol.shape == symbol_hat.shape
    #     J, Len = symbol.shape
    #     temp_err = np.sum(symbol != symbol_hat)
    #     if temp_err > 0:
    #         self.err_symb += temp_err
    #     self.tot_symb += J * Len
    #     self.ser = self.err_symb / self.tot_symb
    #     return

    # def CntSumErr(self, uu_sum, uu_hat_sum, ):
    #     assert uu_sum.shape == uu_hat_sum.shape
    #     Len = uu_sum.shape[-1]
    #     temp_err = np.sum(uu_sum != uu_hat_sum)

    #     if temp_err > 0:
    #         self.err_sum += temp_err
    #     self.tot_sum += Len
    #     self.sumerr = self.err_sum / self.tot_sum
    #     return

    def SaveToFile(self, filename = "SNR_BerFer.txt", snr = '', open_type = 'a+'):
        log = f"[{snr:.2f}, {self.fer:.8f}, {self.ber:.8f}, {self.sumerr:.8f}, {self.ser:.8f}, {self.ave_iter:.3f}],"
        with open(filename, open_type) as f:
            f.write(log + "\n")
        return

    def PrintScreen(self, snr = '',  ):
        print(f"  snr = {snr:.2f}(dB):iter = {self.ave_iter:.3}, bits = {self.err_bit}/{self.tot_bit}, blk = {self.err_blk}/{self.tot_blk}, fer = {self.fer:.10f}, ber = {self.ber:.10f}")
        return



















































































































































































































































