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
from Trainer import common, MetricsLog
# 数据集
# import data
from data import data_generator


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




class LeNetMinst_FGSM_Trainer():
    def __init__(self, args, loader, model, loss, ckp, writer):
        self.args = args
        # self.scale = args.scale
        # #print(f"trainer  self.scale = {self.scale} \n")
        self.wr = writer
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.net = model

        self.Loss = loss

        self.device = args.device
        print(f"len(self.loader_train) = {len(self.loader_train)}, len(self.loader_train.dataset) = {len(self.loader_train.dataset)}")
        print(f"len(self.loader_test[0]) = {len(self.loader_test[0])}, len(self.loader_test[0].dataset) = {len(self.loader_test[0].dataset)}")


        self.optim = Optimizer.make_optimizer(args, self.net, 'MinstLeNetPretrain')

        self.ckp.print_parameters(net = self.net, name = "LeNet for Minst")
        return


    #@profile
    def train(self):
        tm = common.myTimer()
        myAccRecord = MetricsLog.AccuracyRecorder(2)

        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.higred(f"\n#================================ 开始训练, 时刻:{now1} =======================================\n"))
        self.ckp.write_log(f"#======================================== 开始训练, 开始时刻: {now1} =============================================\n", train=True)

        for epoch in range(self.args.epochs):
            print(f"\nEpoch : {epoch+1}/{self.args.epochs}({100.0*(epoch+1)/self.args.epochs:0>5.2f}%)")
            self.ckp.write_log(f"\nEpoch : {epoch}", train=True)

            self.optim.updatelr()
            self.Loss.add_log()
            metric = MetricsLog.Accumulator(3)
            myAccRecord.addlog()

            for batch, (X, y) in enumerate(self.loader_train):
                self.net.zero_grad()
                ## (1) 训练真实图片
                X, y = common.prepare(self.device, self.args.precision, X, y)        # 选择精度且to(device)

                y_hat = self.net(X)           
                loss = self.Loss(y_hat, y)    

                self.optim.zero_grad()       # 必须在反向传播前先清零。
                loss.backward()
                self.optim.step()
                # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")
                with torch.no_grad():
                    acc = common.accuracy(y_hat, y)
                    metric.add(loss, acc, X.shape[0])
                    myAccRecord.add([loss.item(), acc], X.shape[0])
                # 输出训练状态
                if batch % 100 == 0:
                    frac1 = (epoch + 1) / self.args.epochs
                    frac2 = (batch + 1)/len(self.loader_train)
                    print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t  Train acc:{:4.2f} ".format(epoch+1, self.args.epochs, frac1, batch+1, len(self.loader_train), frac2, loss.item()/X.shape[0], acc, ))
                    self.ckp.write_log('    [epoch: %d/%d, batch: %3d/%d]\tLoss_D: %.4f ' % (epoch+1, self.args.epochs, batch, len(self.loader_train), loss.item(),  ), train=True)

            # 学习率递减
            self.optim.schedule()

            # 计算并更新epoch的loss
            epochLos = self.Loss.mean_log()[-1]
            myAccRecord.avg()

            # test data accuracy
            test_acc = common.evaluate_accuracy_gpu(self.net, self.loader_test[0], device = self.device)
            # epoch 的平均 loss
            train_l = metric[0]/metric[2]
            #train_al = myAccRecord[0]/myAccRecord[2]
            # epoch 的平均 正确率
            train_acc = metric[1]/metric[2]
            #train_aacc = myAccRecord[1]/myAccRecord[2]

            tmp = tm.toc()
            print(f"  Epoch: {epoch+1}/{self.args.epochs}({(epoch+1)*100.0/self.args.epochs:5.2f}%) | loss = {train_l:.4f}/{myAccRecord[0]:.4f}/{epochLos.item():.4f} | train acc: {train_acc:.3f}/{myAccRecord[1]:.3f}, test acc: {test_acc:.3f}| Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")
            self.ckp.write_log(f"  Epoch {epoch+1}/{self.args.epochs} | loss = {epochLos } | train acc: {train_acc:.3f}, test acc: {test_acc:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)

        # 保存模型的学习率并画图
        self.ckp.savelearnRate(self)
        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)
        
        # print(f"myAccRecord.data = {myAccRecord.data}")
        #self.ckp.save()
        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.ckp.write_log(f"#========================= 本次训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ================================",train=True)
        # 关闭日志
        self.ckp.done()
        print(color.higred(f"======== 关闭训练日志 {self.ckp.log_file.name} =============="))
        print(color.higred(f"\n#====================== 训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ==============================\n"))
        return


    def test(self, model, test_loader, Loss, epsilon):
        # 精度计数器
        correct = 0
        adv_examples = []

        # 循环遍历测试集中的所有示例
        for data, target in test_loader:
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

            # 计算后向传递模型的梯度
            loss.backward()

            # 收集datagrad
            data_grad = data.grad.data
            #print(f"data.shape = {data.shape}, data_grad.shape = {data_grad.shape}")

            # 唤醒FGSM进行攻击
            perturbed_data =  common.fgsm_attack(data, epsilon, data_grad)

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
        return final_acc, adv_examples


    def testFGSM(self,):
        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.higred(f"\n#================================ 开始测试,时刻{now1} =======================================\n"))
        self.args.test_batch_size = 1
        dataloader = data_generator.DataGenerator(self.args, 'MNIST')
        test_loader = dataloader.loader_test[0]
        loss = torch.nn.NLLLoss()
        loss = torch.nn.CrossEntropyLoss()

        # 设置不同扰动大小
        epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        accuracies = []
        examples = []

        # 预训练模型
        pretrained_model = "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/mnist_LeNet.pt"
        self.net.load_state_dict(torch.load(pretrained_model, map_location= self.device))
        self.net.eval()
        print(f"device = {next(self.net.parameters()).device}")
        # 对每个epsilon运行测试
        for eps in epsilons:
            acc, ex = self.test(self.net, test_loader, loss, eps)
            accuracies.append(acc)
            examples.append(ex)

        common.plotXY(epsilons, accuracies, xlabel = r"$\mathrm{\epsilon}$", ylabel = "Accuracy", title = "Accuracy vs Epsilon", legend = "Y vs. X", figsize = (5, 5), savepath = "/home/jack/snap/", savename = "hh")
        common.FGSM_draw_image(len(epsilons), len(examples[0]), epsilons, examples,  savepath = "/home/jack/snap/", savename = "FGSM_samples")


        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        print(color.higred(f"\n#================================ 完成测试, 开始时刻:{now1}/结束时刻:{now2}  =======================================\n"))
        return











































