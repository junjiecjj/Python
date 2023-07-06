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


# 以下是本项目自己编写的库
# checkpoint
import Utility

# 参数
from Option import args
# 数据集
from data import data_generator

# 损失函数
from loss.Loss import myLoss

# 训练器
from trainers.LeNet_minst_FGSM import  LeNetMinst_FGSM_Trainer
from trainers.LeNet_minst_new  import  LeNetMinst_Sem_Trainer


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


# #@profile
def main_Minst_preLeNet_FGSM():
    data            = "MNIST"
    args.modelUse   = f"AE_noqua_cnn_classify_joinloss_{data.lower()}_R_SNR_attack"
    # args.CompRate = [0.2, 0.9]
    # args.SNRtrain = [3, 20]
    # args.SNRtest = [3, 10, 20]
    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        args.pretrain =  True

        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        classifier = LeNets.LeNet_3().to(args.device)
        pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-06-10:17:07.pt"
        # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
        classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device ))

        trainer = LeNetMinst_FGSM_Trainer(args, loader, classifier, ckp )

        trainer.FGSM_NoCommunication()

        trainer.FGSM_R_SNR(args.SNRtest)

    return

# #@profile
def main_Minst_preLeNet_new():
    data            = "MNIST"
    args.modelUse   = f"AE_noqua_cnn_classify_joinloss_{data.lower()}_R_SNR_attack"

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        args.pretrain =  True
        args.quantize =  False

        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, 'MNIST')

        classifier = LeNets.LeNet_3().to(args.device)
        pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-06-10:17:07.pt"
        # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
        classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device ))
        trainer = LeNetMinst_Sem_Trainer(args, loader, classifier, ckp )

        trainer.Sem_R_SNR(args.SNRtest)
    return


if __name__ == '__main__':
    main_Minst_preLeNet_FGSM()
    # main_Minst_preLeNet_new()










































































































































