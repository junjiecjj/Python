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
# os.system('pip install einops')
# os.system('pip install objgraph')
# os.system('pip install memory_profiler')
# os.system('pip install psutil')
# os.system('pip install transformers')

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
# from GAN_minst_trainer import GANforMinstTrainer
# from GAN_cifer10_trainer import GANforCifar10Trainer
# from LeNet_minst_FGSM import LeNetMinst_FGSM_Trainer
# from GAN_Gauss_trainer import GANforGaussTrainer

from Trainer.GAN_minst_trainer import  GANforMinstTrainer
from Trainer.GAN_cifer10_trainer import  GANforCifar10Trainer
from Trainer.LeNet_minst_FGSM import  LeNetMinst_FGSM_Trainer
from Trainer.GAN_Gauss_trainer import  GANforGaussTrainer
from Trainer.DeepSem_trainer import  SemComTrainer



# 模型
from model import DCGAN
from model import LeNets
from model import AutoEnDecoder


# Loss、PSNR等结果可视化等
from visual.summwriter import SummWriter



#==============================================================================================================
# 设置随机数种子
torch.manual_seed(args.seed)
Utility.set_random_seed(args.seed)


# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
args.device = torch.device(args.device if torch.cuda.is_available() and not args.cpu else "cpu")


# #@profile
def main_Minst_DCGAN():
    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        # 损失函数类
        loss_G = myLoss(args, ckp, 'minstG')
        loss_D = myLoss(args, ckp, 'minstD')

        # tensorboard可视化
        wr = SummWriter(args, ckp)

        Generator = DCGAN.Minst_Generator(args).to(args.device)
        Discriminator = DCGAN.Minst_Discriminator(args).to(args.device)

        trainer =  GANforMinstTrainer( args, loader, Generator, Discriminator, loss_G, loss_D, ckp, wr)

        # 训练
        if  args.wanttrain:
            print(f"Starting train \n")
            trainer.train()

        wr.WrClose()

        #print(f"====================== 关闭日志 ===================================")
        ckp.done()

        return



# #@profile
def main_CIFAR10_DCGAN():
    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'CIFAR10')

        # 损失函数类
        loss_G = myLoss(args, ckp, 'Cifar10G')
        loss_D = myLoss(args, ckp, 'Cifar10D')

        # tensorboard可视化
        wr = SummWriter(args, ckp)

        Generator = DCGAN.CIFER_Generator().apply(DCGAN.weights_init).to(args.device)
        Discriminator = DCGAN.CIFER_Discriminator().apply(DCGAN.weights_init).to(args.device)

        trainer = GANforCifar10Trainer( args, loader, Generator, Discriminator, loss_G, loss_D, ckp, wr)

        # 训练
        if  args.wanttrain:
            print(f"Starting train \n")
            trainer.train()


        wr.WrClose()

        #print(f"====================== 关闭日志 ===================================")
        ckp.done()

        return



def main_Gauss_DCGAN():
    args.loss = "1*MSE"

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        # 损失函数类
        loss_G = myLoss(args, ckp, 'minstG')
        loss_D = myLoss(args, ckp, 'minstD')

        # tensorboard可视化
        wr = SummWriter(args, ckp)

        Generator = DCGAN.Minst_Generator(args).to(args.device)
        Discriminator = DCGAN.Minst_Discriminator(args).to(args.device)

        trainer = GANforGaussTrainer( args, loader, Generator, Discriminator, loss_G, loss_D, ckp, wr)

        # 训练
        if  args.wanttrain:
            print(f"Starting train \n")
            trainer.train()

        wr.WrClose()

        #print(f"====================== 关闭日志 ===================================")
        ckp.done()

        return



# #@profile
def main_Minst_preLeNet():
    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        args.loss = "1*CrossEntropy"
        args.epochs = 20
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        # 损失函数类
        loss = myLoss(args, ckp, 'MinstLeNetFGSM')

        # tensorboard可视化
        wr = SummWriter(args, ckp)

        model = LeNets.LeNet_csdn(args).to(args.device)

        trainer = LeNetMinst_FGSM_Trainer(args, loader, model, loss, ckp, wr)

        # 训练
        if  args.wanttrain:
            # print(f"Starting train \n")
            trainer.train()
            pass
        # 训练
        if  args.wanttest:
            print(f"Starting testing \n")
            trainer.testFGSM()

        wr.WrClose()

        #print(f"====================== 关闭日志 ===================================")
        ckp.done()

        return

# def main_SemCom_Auto():
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

    model = AutoEnDecoder.AutoEncoderMnist().to(args.device)


    trainer = SemComTrainer( args, loader, model, loss, ckp, wr)

    # 训练
    if  args.wanttrain:
        print(f"Starting train \n")
        trainer.train()
    # trainer.viewMiddleFeature3D()
    # trainer.raw_recovered()
    trainer.raw_recovered_tmpview()
    wr.WrClose()

    #print(f"====================== 关闭日志 ===================================")
    ckp.done()

        # return


# if __name__ == '__main__':
#     # main_Minst_DCGAN()
#     # main_CIFAR10_DCGAN()
#     # main_Gauss_DCGAN()


#     # main_Minst_preLeNet()
#     main_SemCom_Auto()







