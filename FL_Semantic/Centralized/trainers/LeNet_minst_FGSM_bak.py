# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import sys, os
import datetime
import numpy as np
import imageio

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import shutil


#内存分析工具
from memory_profiler import profile
import objgraph

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

sys.path.append("..")
# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()
# print(color.fuchsia("Color Print Test Pass"))
import Optimizer
import Utility
from . import common, MetricsLog
# 数据集
# import data
from data import data_generator


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



class LeNetMinst_FGSM_Trainer():
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test[0]
        self.classify = model
        self.device = args.device

        print(f"len(self.loader_train) = {len(self.loader_train)}, len(self.loader_train.dataset) = {len(self.loader_train.dataset)}")
        print(f"len(self.loader_test ) = {len(self.loader_test )}, len(self.loader_test.dataset) = {len(self.loader_test.dataset)}")

        self.ckp.print_parameters(net = self.classify, name = "LeNet for Minst")
        return

    def test(self, model, test_loader, Loss, epsilon):
        # 精度计数器
        correct = 0
        adv_examples = []

        # 循环遍历测试集中的所有示例
        for data, target in test_loader:
            print(f"0  {data.min()}, {data.max()}")
            data, target = common.prepare(self.device, self.args.precision, data, target)        # 选择精度且to(device)

            # 设置张量的requires_grad属性，这对于攻击很关键
            data.requires_grad = True
            # 通过模型前向传递数据
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # 如果初始预测是错误的，不打断攻击，继续
            if init_pred.item() != target.item():
                continue

            # 计算损失
            loss = Loss(output, target)

            # 将所有现有的渐变归零
            model.zero_grad()


            # ###  1
            # # 计算后向传递模型的梯度
            # loss.backward()
            # # 收集datagrad
            # data_grad = data.grad.data
            # #print(f"data.shape = {data.shape}, data_grad.shape = {data_grad.shape}")
            # # 唤醒FGSM进行攻击
            # perturbed_data =  common.fgsm_attack(data, epsilon, data_grad)
            # # print(f"1  {perturbed_data.min()}, {perturbed_data.max()}")

            # ###  2
            # loss.backward()
            # grad_sign = data.grad.data.sign()
            # perturbed_data = data +  epsilon * grad_sign
            # perturbed_data = torch.clamp(perturbed_data, min = -1, max = 1)


            ## 3
            gradients_sign = torch.autograd.grad(loss, data)[0].sign()
            # print(f"gradients_sign.shape = {gradients_sign}")
            perturbed_data = data +  epsilon * gradients_sign
            perturbed_data = torch.clamp(perturbed_data, min = -1, max = 1)



            # 重新分类受扰乱的图像
            output = model(perturbed_data)

            # 检查是否成功
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if final_pred.item() == target.item():
                correct += 1
                # 保存0 epsilon示例的特例
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # 稍后保存一些用于可视化的示例
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # 计算这个epsilon的最终准确度
        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

        # 返回准确性和对抗性示例
        return final_acc,  adv_examples


    def testFGSM(self,):
        self.classify.eval()
        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.higred(f"\n#================================ 开始测试,时刻{now1} =======================================\n"))

        loss = torch.nn.NLLLoss()
        loss = torch.nn.CrossEntropyLoss()

        # 设置不同扰动大小
        epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,]

        accuracies = []
        examples = []

        print(f"device = {next(self.classify.parameters()).device}")
        # 对每个epsilon运行测试
        for eps in epsilons:
            acc, ex = self.test(self.classify, self.loader_test, loss, eps)
            accuracies.append(acc)
            examples.append(ex)

        common.plotXY(epsilons, accuracies, xlabel = r"$\mathrm{\epsilon}$", ylabel = "Accuracy", title = "Accuracy vs Epsilon", legend = "Y vs. X", figsize = (5, 5), savepath = "/home/jack/snap/", savename = "hh")
        common.FGSM_draw_image(len(epsilons), len(examples[0]), epsilons, examples,  savepath = "/home/jack/snap/", savename = "FGSM_samples")


        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        print(color.higred(f"\n#================================ 完成测试, 开始时刻:{now1}/结束时刻:{now2}  =======================================\n"))
        return











































