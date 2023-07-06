
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

import os, sys
import math
import datetime
import imageio
import cv2
import skimage
import glob
import random
import numpy as np
import torch

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#内存分析工具
from memory_profiler import profile
import objgraph
import gc

#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



# 初始化随机数种子
def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
    return

def set_printoption(precision = 3):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(
        precision = precision,    # 精度，保留小数点后几位，默认4
        threshold = 1000,
        edgeitems = 3,
        linewidth = 150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile = None,
        sci_mode = False  # 用科学技术法显示数据，默认True
        )


# 功能：
class checkpoint():
    def __init__(self, args ):
        print(color.fuchsia("\n#================================ checkpoint 开始准备 =======================================\n"))
        self.args = args
        self.ok = True
        self.n_processes = 8

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.savedir = os.path.join(args.save, f"{self.now}_{args.modelUse}")
        os.makedirs(self.savedir, exist_ok = True)

        print(f"训练结果保存目录 = {self.savedir} \n")

        # 模型参数保存的目录
        # self.modelSaveDir = os.path.join(args.ModelSave, f"{self.now}_Model_{args.modelUse}")
        self.modelSaveDir = self.args.ModelSave
        os.makedirs(self.modelSaveDir, exist_ok = True)

        # open_type = 'a' if os.path.exists(self.getSavePath('trainLog.txt')) else 'w'
        # self.log_file = open(self.getSavePath('trainLog.txt'), open_type)
        self.writeArgsLog(self.getSavePath('argsConfig.txt'))

        # Prepare test dir and so on:
        self.TeMetricLog = {}
        self.testResdir = os.path.join(self.savedir, "test_results")
        os.makedirs(self.testResdir, exist_ok=True)
        print(f"测试结果保存目录 = {self.testResdir} \n")

        print(color.fuchsia("\n#================================ checkpoint 准备完毕 =======================================\n"))
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

    def print_parameters(self, logfile, netG = None, netD = None, net = None, name = ""):
        logfile = self.getSavePath(logfile)  # logfile = 'trainLog.txt'
        with open(logfile, 'a+') as f:
            print("#=====================================================================================",  file = f)
            print("                      " + self.now,  file = f)
            if net != None:
                print(f"#================================== {name} ====================================",  file = f)
                print(net, file = f)
                print(f"#============================= {name} Parameters ==============================",  file = f)
                for name, param in  net.named_parameters():
                    if param.requires_grad:
                        #print(f"{name}: {param.size()}, {param.requires_grad} ")
                        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ", file = f)
            print("#=====================================================================================\n",  file = f)
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

    def SaveTestLog(self):
        # self.plot_AllTestMetric()
        torch.save(self.TeMetricLog, self.get_testSavepath('TestMetric_log.pt'))
        return

    def SaveTestFig(self, DaSetName, CompRatio, SnrTest, snrTrain, figname, data):
        filename = self.get_testSavepath('results-{}'.format(DaSetName),'{}_R={}_SnrTrain={}_SnrTest={}.png'.format(figname, CompRatio,snrTrain,SnrTest))
        #print(f"filename = {filename}\n")
        normalized = data[0].mul(255 / self.args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        #print(f"tensor_cpu.shape = {tensor_cpu.shape}\n")
        imageio.imwrite(filename, tensor_cpu.numpy())
        return
# <<< 测试相关函数





































































































































































































