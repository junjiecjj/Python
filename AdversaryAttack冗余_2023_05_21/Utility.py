
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

import os, sys
import math
import time
import datetime

from torch.autograd import Variable

import random
import numpy as np
import imageio
import torch.nn as nn
import torch
# import torch.optim as optim
# import torch.optim.lr_scheduler as lrs
import collections
from torch.utils.tensorboard import SummaryWriter
from transformers import optimization
from scipy import stats

import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
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

# 功能：
class checkpoint():
    def __init__(self, args ):
        print(color.fuchsia(f"\n#================================ checkpoint 开始准备 =======================================\n"))
        self.args = args
        self.ok = True
        self.n_processes = 8

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.savedir = os.path.join(args.save, f"{self.now}_TrainLog_{args.modelUse}")
        os.makedirs(self.savedir, exist_ok = True)

        print(f"训练过程MSR、PSNR、Loss等保存目录 = {self.savedir} \n")

        # 模型参数保存的目录
        self.modelSaveDir = os.path.join(args.saveModel, f"{self.now}_Model_{args.modelUse}")
        os.makedirs(self.modelSaveDir, exist_ok = True)

        open_type = 'a' if os.path.exists(self.getSavePath('trainLog.txt')) else 'w'
        self.log_file = open(self.getSavePath('trainLog.txt'), open_type)

        self.writeArgsLog('argsConfig.txt')
        # self.writeArgsLog('trainLog.txt')
        # self.log_file = open(self.getSavePath('trainLog.txt'), open_type)

        print(color.fuchsia(f"\n#================================ checkpoint 准备完毕 =======================================\n"))
        return


    def writeArgsLog(self, filename, open_type = 'a'):
        with open(self.getSavePath(filename), open_type) as f:
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

    def print_parameters(self, netG = None, netD = None, net = None, name = "GAN"):
        print(f"#=====================================================================================",  file=self.log_file)
        print(self.now,  file = self.log_file)
        if netG != None:
            print(f"#================================== Generator ========================================",  file=self.log_file)
            print(netG, file = self.log_file)
        if netD != None:
            print(f"#================================== Discriminator ====================================",  file=self.log_file)
            print(netD, file = self.log_file)
        if net != None:
            print(f"#================================== {name} ====================================",  file=self.log_file)
            print(net, file = self.log_file)
        if netG != None:
            print(f"#================================= Generator Parameters ==============================",  file=self.log_file)
            for name, param in  netG.named_parameters():
                if param.requires_grad:
                    #print(f"{name}: {param.size()}, {param.requires_grad} ")
                    print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ", file=self.log_file)
        if netD != None:
            print(f"#============================= Discriminator Parameters ==============================",  file=self.log_file)
            for name, param in  netD.named_parameters():
                if param.requires_grad:
                    #print(f"{name}: {param.size()}, {param.requires_grad} ")
                    print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ", file=self.log_file)
        if net != None:
            print(f"#============================= {name} Parameters ==============================",  file=self.log_file)
            for name, param in  net.named_parameters():
                if param.requires_grad:
                    #print(f"{name}: {param.size()}, {param.requires_grad} ")
                    print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ", file=self.log_file)
        print(f"#=====================================================================================\n",  file=self.log_file)
        return


    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)

    # 保存模型参数
    def saveModel(self, trainer,  compratio, snr, epoch, is_best=False):
        trainer.model.save(self.modelSaveDir, compratio, snr, epoch, is_best=is_best)
        return

    # 画图和保存Loss日志
    def saveLoss(self, trainer):
        if hasattr(trainer, 'Loss_D'):
            trainer.Loss_D.save(self.savedir)
            # trainer.Loss_D.plot_loss(self.savedir)
            trainer.Loss_D.plot_AllLoss(self.savedir)
        if hasattr(trainer, 'Loss_G'):
            trainer.Loss_G.save(self.savedir)
            # trainer.Loss_D.plot_loss(self.savedir)
            trainer.Loss_G.plot_AllLoss(self.savedir)
        if hasattr(trainer, 'Loss'):
            trainer.Loss.save(self.savedir)
            # trainer.Loss_D.plot_loss(self.savedir)
            trainer.Loss.plot_AllLoss(self.savedir)
        return

    # 画图和保存Loss日志
    def savelearnRate(self, trainer):
        if hasattr(trainer, 'optim_G'):
            trainer.optim_G.save_lr(self.savedir, )
        if hasattr(trainer, 'optim_D'):
            trainer.optim_D.save_lr(self.savedir, )
        if hasattr(trainer, 'optim'):
            trainer.optim.save_lr(self.savedir, )
        return

    # 写日志
    def write_log(self, log, train=False ,refresh=True):
        # print(log)
        self.log_file.write(log + '\n')  # write() argument must be str, not dict
        if refresh:
            self.log_file.close()
            if train== True:
                self.log_file = open(self.getSavePath('trainLog.txt'), 'a')
            else:
                self.log_file = open(self.get_testpath('testLog.txt'), 'a')
        return

    # 关闭日志
    def done(self):
        self.log_file.close()
        return

# >>> 测试相关函数
    # 初始化测试结果目录
    def InittestDir(self, now = 'TestResult'):
        self.TeMetricLog = {}
        # now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.testRudir = os.path.join(self.savedir, now)
        os.makedirs(self.testRudir, exist_ok=True)
        for d in self.args.data_test:
            os.makedirs(os.path.join(self.testRudir,'results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_testpath('testLog.txt')) else 'w'
        self.log_file = open(self.get_testpath('testLog.txt'), open_type)
        print(f"====================== 打开测试日志 {self.log_file.name} ===================================")

        with open(self.get_testpath('argsTest.txt'), open_type) as f:
            f.write('#==========================================================\n')
            f.write(self.now + '\n')
            f.write('#==========================================================\n\n')

            f.write("############################################################################################\n")
            f.write("####################################  Test args  ###########################################\n")
            f.write("############################################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}\n")
            f.write('\n')
            f.write("################################ args end  #################################################\n")
        return

    def get_testpath(self, *subdir):
        return os.path.join(self.testRudir, *subdir)

    def SaveTestLog(self):
        # self.plot_AllTestMetric()
        torch.save(self.TeMetricLog, self.get_testpath('TestMetric_log.pt'))
        return

    def SaveTestFig(self, DaSetName, CompRatio, SnrTest, snrTrain, figname, data):
        filename = self.get_testpath('results-{}'.format(DaSetName),'{}_R={}_SnrTrain={}_SnrTest={}.png'.format(figname, CompRatio,snrTrain,SnrTest))
        #print(f"filename = {filename}\n")
        normalized = data[0].mul(255 / self.args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        #print(f"tensor_cpu.shape = {tensor_cpu.shape}\n")
        imageio.imwrite(filename, tensor_cpu.numpy())
        return

# <<< 测试相关函数













































































































































































