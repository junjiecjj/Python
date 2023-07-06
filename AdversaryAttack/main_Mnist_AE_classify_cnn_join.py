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


# #内存分析工具
# from memory_profiler import profile
# import objgraph

# 以下是本项目自己编写的库
# checkpoint
import Utility

# 参数
from Option import args
# 数据集
from data import data_generator

# 损失函数
# from loss.Loss import myLoss

# 训练器
from trainers.AE_cnn_classify_Mnist_R_noiseless_join   import   AE_cnn_classify_mnist_R_noiseless_join_Trainer
from trainers.AE_cnn_classify_Mnist_R_SNR_join         import   AE_cnn_classify_mnist_R_SNR_join_Trainer

# 模型
# from model import DCGAN
from model import LeNets
# from model import AutoEncoder




#==============================================================================================================
# 设置随机数种子
Utility.set_random_seed(args.seed,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")


def main_AE_cnn_classify_R_SNR_join():
    data            = "MNIST"
    args.loss       = "1*MSE"
    args.optimizer  = "ADAM"
    args.lr         = 0.002
    args.batch_size      = 128
    args.test_batch_size = 128
    args.modelUse   = f"AE_noqua_cnn_classify_joinloss_{data.lower()}_R_SNR"

    args.pretrain   = False
    args.quantize   = False

    args.epochs     = 100
    if args.epochs > 20:
        n = 5
        args.decay  = '-'.join([str(int(args.epochs*(i+1)/n)) for i in range(n-1)])
        print(f"args.decay = {args.decay}")
    # args.CompRate = [0.8, 0.9]
    # args.SNRtrain = [3, 10]

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, data)

        classifier = LeNets.LeNet_3().to(args.device)
        # pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
        pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-06-10:17:07.pt"
        # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
        classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device ))

        trainer = AE_cnn_classify_mnist_R_SNR_join_Trainer(classifier, args, loader,  ckp, )

        # 训练
        trainer.train()
    return

def main_AE_cnn_classify_R_noiseless_join():
    data            = "MNIST"
    args.loss       = "1*MSE"
    args.optimizer  = "ADAM"
    args.lr         = 0.002
    args.batch_size      = 128
    args.test_batch_size = 128
    args.modelUse   = f"AE_noqua_cnn_classify_joinloss_{data.lower()}_R_noiseless"

    args.pretrain   = False
    args.quantize   = False

    args.epochs     = 150
    if args.epochs  > 20:
        n = 5
        args.decay  = '-'.join([str(int(args.epochs*(i+1)/n)) for i in range(n-1)])
        print(f"args.decay = {args.decay}")

    # args.CompRate = [0.2, 0.8]

    #  加载 checkpoint, 如日志文件, 时间戳, 指标等
    ckp = Utility.checkpoint(args)

    if ckp.ok:
        ## 数据迭代器，DataLoader
        loader = data_generator.DataGenerator(args, data)

        classifier = LeNets.LeNet_3().to(args.device)
        # pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
        pretrained_model = f"/home/{args.user_name}/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-06-10:17:07.pt"
        # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
        classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device ))

        trainer = AE_cnn_classify_mnist_R_noiseless_join_Trainer(classifier, args, loader,  ckp, )

        # 训练
        trainer.train()
    return

if __name__ == '__main__':
    main_AE_cnn_classify_R_SNR_join()
    # main_AE_cnn_classify_R_noiseless_join()















