
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

import os, sys

import datetime

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['axes.spines.left'] = 1
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.labelspacing'] = 0.2




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
        print( "\n#================================ checkpoint 开始准备 =======================================\n")
        self.args = args
        self.ok = True
        self.n_processes = 8

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.savedir = os.path.join(args.save, f"{self.now}_MNIST")
        os.makedirs(self.savedir, exist_ok = True)
        print(f"训练结果保存目录 = {self.savedir} \n")

        self.writeArgsLog(self.getSavePath('argsConfig.txt'))

        # Prepare test dir and so on:

        self.testResdir = os.path.join(self.savedir, "test_results")
        os.makedirs(self.testResdir, exist_ok=True)
        print(f"测试结果保存目录 = {self.testResdir} \n")

        print( "\n#================================ checkpoint 准备完毕 =======================================\n")
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

# >>> 测试相关函数
    # 初始化测试结果目录
    def get_testSavepath(self, *subdir):
        return os.path.join(self.testResdir, *subdir)

# <<< 测试相关函数





































































































































































































