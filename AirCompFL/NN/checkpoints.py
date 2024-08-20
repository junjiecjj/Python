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
class checkpoint():
    def __init__(self, args ):
        print(color.fuchsia("\n#================================ checkpoint 开始准备 =======================================\n"))
        self.args = args
        self.n_processes = 8

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.savedir = os.path.join(args.save_path, f"{self.now}_{args.modelUse}")
        os.makedirs(self.savedir, exist_ok = True)

        print(f"训练结果保存目录 = {self.savedir} ")

        # 模型参数保存的目录
        # self.modelSaveDir = os.path.join(args.ModelSave, f"{self.now}_Model_{args.modelUse}")
        self.modelSaveDir = self.args.ModelSave
        os.makedirs(self.modelSaveDir, exist_ok = True)

        # open_type = 'a' if os.path.exists(self.getSavePath('trainLog.txt')) else 'w'
        # self.log_file = open(self.getSavePath('trainLog.txt'), open_type)
        self.writeArgsLog(self.getSavePath('argsConfig.txt'))

        # Prepare test dir and so on:
        self.testResdir = os.path.join(self.savedir, "test_results")
        os.makedirs(self.testResdir, exist_ok=True)
        print(f"测试结果保存目录 = {self.testResdir} ")

        print(color.fuchsia("\n#================================ checkpoint 准备完毕 =======================================\n"))
        return


    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)

    # 写日志
    def write_log(self, log, train = False ):
        if train == True:
            logfile = self.getSavePath('trainLog.txt')
        else:
            logfile = self.get_testSavepath('testLog.txt')
        with open(logfile, 'a+') as f:
            f.write(log + '\n')
        return

    # 写日志
    def write_attacklog(self, log ):
        logfile = self.getSavePath('AttackLog.txt')
        with open(logfile, 'a+') as f:
            f.write(log + '\n')
        return

# >>> 测试相关函数
    # 初始化测试结果目录
    def get_testSavepath(self, *subdir):
        return os.path.join(self.testResdir, *subdir)
# <<< 测试相关函数
















