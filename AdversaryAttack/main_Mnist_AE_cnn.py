# -*- coding: utf-8 -*-
"""
Created on 2023/04/25
@author: Junjie Chen

"""
#  系统库
import torch

import warnings

warnings.filterwarnings('ignore')
import os


#内存分析工具
from memory_profiler import profile
import objgraph

# 以下是本项目自己编写的库
# checkpoint
import Utility

# 参数
from Option import args
# 数据集
import data
from data import data_generator

# 损失函数
from loss.Loss import myLoss

# 训练器

from trainers.AE_cnn_Mnist_R_SNR            import   AE_cnn_mnist_R_SNR_Trainer
from trainers.AE_cnn_Mnist_R_noiseless      import   AE_cnn_mnist_R_noiseless_Trainer


# 模型
# from model import DCGAN
from model import LeNets
from model import AutoEncoder


# Loss、PSNR等结果可视化等
from visual.summwriter import SummWriter


#==============================================================================================================
# 设置随机数种子
Utility.set_random_seed(args.seed,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
args.device = torch.device(args.device if torch.cuda.is_available() and not args.cpu else "cpu")




def main_AE_cnn_mnist_R_SNR():
    data = "MNIST"
    args.loss = "1*MSE"
    args.lr = 0.001
    args.batch_size = 256
    args.modelUse = f"AE_{data.lower()}_R_SNR"
    args.epochs = 50
    args.decay = '10-20-40-80'

    # args.epochs = 100
    # args.decay = '20-40-80-120'

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, data)

        trainer = AE_cnn_mnist_R_SNR_Trainer( args, loader,  ckp, )

        # 训练
        if  args.wanttrain:
            print(f"Starting train \n")
            trainer.train()

    return

def main_AE_cnn_mnist_R_noiseless():
    data = "MNIST"
    args.loss = "1*MSE"
    args.lr = 0.001
    args.batch_size = 256
    args.modelUse = f"AE_{data.lower()}_R"
    args.epochs = 2
    args.decay = '10-20-40-80'

    # args.epochs = 100
    # args.decay = '20-40-80-120'

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, data)

        trainer = AE_cnn_mnist_R_noiseless_Trainer( args, loader,  ckp, )

        # 训练
        if  args.wanttrain:
            trainer.train()
    return



if __name__ == '__main__':

    # main_AE_cnn_mnist_R_SNR()
    main_AE_cnn_mnist_R_noiseless()









































































































































