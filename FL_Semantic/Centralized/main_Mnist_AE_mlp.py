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

from trainers.AE_mlp_Trainer import AE_mlp_Trainer



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





def main_mlp_AutoEncoder():
    args.loss = "1*MSE"
    args.lr = 0.005
    args.batch_size = 128
    args.modelUse = "AutoEncoder"
    args.epochs = 1


    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)


    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        # 损失函数类
        loss = myLoss(args, ckp, 'minst')

        # tensorboard可视化
        wr = SummWriter(args, ckp)

        model = AutoEncoder.AED_mlp_MNIST().to(args.device)


        trainer = AE_mlp_Trainer( args, loader, model, loss, ckp, wr)

        # 训练
        if  args.wanttrain:
            print(f"Starting train \n")
            trainer.train()
        # trainer.viewMiddleFeature3D()
        # trainer.raw_recovered()
        trainer.raw_recovered_rows()
        wr.WrClose()

        #print(f"====================== 关闭日志 ===================================")
        ckp.closelog()

        return





if __name__ == '__main__':

    main_mlp_AutoEncoder()










































































































































