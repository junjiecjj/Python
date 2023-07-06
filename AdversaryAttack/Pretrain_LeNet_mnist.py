#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:43:54 2023

@author: jack
"""

##  系统库
import os, sys
import torch
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

import socket, getpass
# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')


##  本项目自己编写的库
#  工具
from trainers import common as tcommon

# 模型
from model import LeNets

# 数据器
from data import data_generator

# checkpoint
import Utility

# 指标记录器
from trainers import MetricsLog

# 参数
from Option import args

# 损失函数
from loss.Loss import myLoss

# 优化器
import Optimizer

import ColorPrint
color = ColorPrint.ColoPrint()
#===============================================================================================

# 设置随机数种子
Utility.set_random_seed(args.seed,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

model = LeNets.LeNet_3().to(args.device)
loader = data_generator.DataGenerator(args, 'MNIST')

## 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
args.device = torch.device(args.device if torch.cuda.is_available() and not args.cpu else "cpu")


data = "MNIST"
args.loss = "1*CrossEntropy"
args.optimizer = 'ADAM'
args.lr = 0.002
args.batch_size = 256
args.modelUse = f"LeNet_{data.lower()}_classify_prerain"
args.epochs = 60
args.decay = '10-20-40-80'

ckp = Utility.checkpoint(args)
loss =  myLoss(args)

class LeNetMinst_PreTrain(object):
    def __init__(self, args, loader, model, loss, ckp):
        self.args = args
        self.device = args.device
        # self.scale = args.scale
        # #print(f"trainer  self.scale = {self.scale} \n")
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.net = model.to(self.device)
        self.Loss = loss
        self.optim = Optimizer.make_optimizer(args, self.net,  )
        self.ckp.print_parameters(logfile = 'trainLog.txt', net = self.net, name = "LeNet for Minst classify")
        return

    #@profile
    def train(self):
        tm = tcommon.myTimer()
        TraRecord = MetricsLog.TraRecorder(Len = 5)

        print(color.higred(f"\n=== 开始训练:{tm.start_str} ===\n"))
        self.ckp.write_log(f"#=== 开始训练: {tm.start_str} ===\n", train=True)

        for epoch in range(self.args.epochs):
            self.net.train()
            print(f"\nEpoch : {epoch+1}/{self.args.epochs}({100.0*(epoch+1)/self.args.epochs:0>5.2f}%)")
            self.ckp.write_log(f"\nEpoch {epoch+1}/{self.args.epochs}({100.0*(1+epoch)/self.args.epochs:0>5.2f}%)", train = True)
            lr = self.optim.updatelr()
            self.optim.updatelr()
            self.Loss.addlog()
            metric = MetricsLog.Accumulator(3)
            TraRecord.addlog(epoch)

            for batch, (X, y) in enumerate(self.loader_train):
                self.net.zero_grad()
                # print(f"1 {X.min()}, {X.max()}, {y.shape}, ")
                X, y = tcommon.prepare(self.device, self.args.precision, X, y)        # 选择精度且to(device)
                # print(f"2  {X.shape}, {y.shape}, ")
                y_hat = self.net(X)
                # print(f"3 {X.shape}, {y.shape}, {y_hat.shape}, {y_hat.min()}, {y_hat.max()}")
                loss = self.Loss(y_hat, y)

                self.optim.zero_grad()       # 必须在反向传播前先清零。
                loss.backward()
                self.optim.step()
                # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")
                with torch.no_grad():
                    acc = tcommon.accuracy(y_hat, y)
                    metric.add(loss.item(), acc, X.shape[0])
                # 输出训练状态
                if batch % 100 == 0:
                    frac1 = (epoch + 1) / self.args.epochs
                    frac2 = (batch + 1) / len(self.loader_train)
                    print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t  Train acc:{:4.2f} ".format(epoch+1, self.args.epochs, frac1, batch+1, len(self.loader_train), frac2, loss.item()/X.shape[0], acc, ))
                    self.ckp.write_log('    [epoch: %d/%d, batch: %3d/%d]\tLoss: %.4f ' % (epoch+1, self.args.epochs, batch, len(self.loader_train), loss.item(),  ), train=True)

            # 学习率递减
            self.optim.schedule()
            # 计算并更新epoch的loss
            epochLos = self.Loss.mean_log()[-1]

            # epoch 的平均 loss
            epoch_avg_loss = metric[0]/metric[2]
            # epoch 的 train data 正确率
            epoch_train_acc = metric[1]/metric[2]
            # test data accuracy
            test_acc = tcommon.evaluate_accuracy_gpu(self.net, self.loader_test[0], device = self.device)
            TraRecord.assign([lr, epoch_avg_loss, epoch_train_acc, test_acc])

            if epoch % 2 == 0:
                self.val_plot(self.net, self.loader_test[0], epoch)

            tmp = tm.toc()
            print(f"  Epoch: {epoch+1}/{self.args.epochs}({(epoch+1)*100.0/self.args.epochs:5.2f}%) | loss = {epochLos:.4f}/{TraRecord[1]:.4f} | train acc: {epoch_train_acc:.3f}/{TraRecord[2]:.3f}, test acc: {test_acc:.3f}/{TraRecord[3]:.3f}| Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")
            self.ckp.write_log(f"  Epoch {epoch+1}/{self.args.epochs} | loss = {epochLos.item():.4f} | train acc: {epoch_train_acc:.3f}/{TraRecord[1]:.3f}, test acc: {test_acc:.3f}/{TraRecord[2]:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)


        TraRecord.save(self.ckp.savedir)
        TraRecord.plot_inonefig(self.ckp.savedir, metric_str = ["lr", "Train Loss", "train acc", "val acc"])

        ### 保存网络中的参数, 速度快，占空间少
        torch.save(model.state_dict(), f"/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_{tm.start_str}.pt")   # 训练和测试都归一化

        self.ckp.write_log(f"#=== 本次训练完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ===",train=True)
        print(color.higred(f"\n#============ 训练完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 =================\n"))

        acc1 = self.validata(self.net, self.loader_train,)
        print(f"train data acc = {acc1}")

        acc2 = self.validata(self.net, self.loader_test[0],)
        print(f"test data acc = {acc2}")
        return acc1, acc2

    def val_plot(self, model, dataloader, epo):
        model.eval()
        savedir = os.path.join(self.ckp.savedir, "valiateImage" )
        os.makedirs(savedir, exist_ok = True)
        rows =  4
        cols = 5
        idx = np.arange(0, rows*cols, 1)
        labels = dataloader.dataset.targets[idx]
        # 真实图片
        real_image = dataloader.dataset.data[idx] #.numpy()
        with torch.no_grad():
            test_data = tcommon.data_tf_cnn_mnist_batch(real_image)
            test_data,  = tcommon.prepare(self.device, self.args.precision, test_data)
            predlabs = model(test_data).cpu().argmax(axis = 1)

        tcommon.grid_imgsave(savedir, real_image, labels, predlabs, dim = (rows, cols),   suptitle = f"Classify results, epoch = {epo}", basename = f"epoch={epo}")

        return


    def validata(self, model, dataloader, device = None):
        model.eval()
        if not device:
            device = next(model.parameters()).device

        metric = MetricsLog.Accumulator(2)
        with torch.no_grad():
            for X, y in dataloader:
                # print(f"X.shape = {X.shape}, y.shape = {y.shape}, size(y) = {size(y)}/{y.size(0)}") # X.shape = torch.Size([128, 1, 28, 28]), y.shape = torch.Size([128]), size(y) = 128
                X, y = X.to(device), y.to(device)
                metric.add(tcommon.accuracy(model(X), y), y.size(0))  # size(y)
        acc = metric[0] / metric[1]
        return acc



p = LeNetMinst_PreTrain(args, loader, model, loss, ckp)

ac1, ac2 = p.train()






# classifier = LeNets.LeNet_3().to(args.device)
# pretrained_model = "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/LeNet_Minst_classifier_2023-06-01-22:20:58.pt"
# # 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
# classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device))































































































































































































































































































































































































































































































































































































