#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 22:25:47 2024

@author: jack
"""

import os, sys
import math
import datetime
import imageio
# import cv2
import skimage
import glob
import random
import numpy as np
import torch

import matplotlib
#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# 功能：
class checkpoint(object):
    def __init__(self, args, now = 'None'):
        # print("#=================== checkpoint 开始准备 ======================\n")
        self.args = args
        self.n_processes = 8
        if now == 'None':
            self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        else:
            self.now =  now

        tempp = '_' + args.diff_case + str([args.local_up if args.diff_case == 'batchs' else args.local_epoch][0])
        quantway = '_1bits_' + args.quantway + '_'
        tmp = self.now + f"_{args.model}_{"IID" if args.IID else "noIID"}_{args.case}{tempp if args.case == 'diff' else ''}{quantway if args.quantize else '_'}{args.optimizer}_{args.lr}_U{args.num_of_clients}_bs{args.local_bs}"
        self.savedir = os.path.join(args.save_path, tmp)
        os.makedirs(self.savedir, exist_ok = True)

        self.writeArgsLog(self.getSavePath('argsConfig.txt'))
        # print("#================== checkpoint 准备完毕 =======================\n")
        return

    def writeArgsLog(self, filename, open_type = 'w'):
        with open(filename, open_type) as f:
            f.write('====================================================================================\n')
            f.write(self.now + '\n')
            f.write('====================================================================================\n\n')

            f.write("###############################################################################\n")
            f.write("################################  args  #######################################\n")
            f.write("###############################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}\n")
            f.write("\n################################ args end  ##################################\n")
        return

    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)

    # 写日志
    def write_log(self, log, train = True ):
        if train == True:
            logfile = self.getSavePath('trainLog.txt')
        else:
            logfile = self.get_testSavepath('testLog.txt')
        with open(logfile, 'a+') as f:
            f.write(log + '\n')
        return

# >>> 测试相关函数
    # 初始化测试结果目录
    def get_testSavepath(self, *subdir):
        return os.path.join(self.testResdir, *subdir)
# <<< 测试相关函数
















