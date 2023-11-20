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
from model import LeNets, AutoEncoder

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

# 在指定压缩率和信噪比下训练间隔epoch时, 在验证集上的测试结果
def validate( model, classifier, dataloader, ):
    model.eval()
    classifier.eval()

    metric  = MetricsLog.Accumulator(5)
    with torch.no_grad():
        for batch, (X, label) in enumerate(dataloader):
            X  = X.to(device)
            # 传输
            X_hat          = model(X)
            # 传输后分类
            predlabs       = classifier(X_hat).cpu()
            # 计算准确率
            acc            = tcommon.accuracy(predlabs, label )
            # 不同方法计算 PSNR
            X, X_hat = X.detach().cpu(), X_hat.detach().cpu()
            batch_01_psnr  = tcommon.PSNR_torch(X , X_hat , )
            X              = tcommon.data_inv_tf_cnn_mnist_batch_3D(X)
            X_hat          = tcommon.data_inv_tf_cnn_mnist_batch_3D(X_hat)
            batch_avg_psnr = tcommon.PSNR_torch_Batch(X, X_hat, )
            # image_avg_psnr, image_sum_psnr, batchsize = common.PSNR_torch_Image(X, X_hat,)
            metric.add(batch_01_psnr, batch_avg_psnr,  acc, 1, X.size(0))

        val_batch_01 = metric[0]/metric[3]
        val_batch    = metric[1]/metric[3]
        val_acc      = metric[2]/metric[4]
    return val_batch_01, val_batch,  val_acc




#===============================================================================================

# 设置随机数种子
Utility.set_random_seed(args.seed,  deterministic = True, benchmark = True)
Utility.set_printoption(3)


raw_size = 28*28
loader = data_generator.DataGenerator(args, 'MNIST')
loader_train = loader.loader_train
loader_test  = loader.loader_test[0]


## 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

tm = tcommon.myTimer()
print(color.higred(f"\n  开始训练:{tm.start_str}  =\n"))
for idx_c,  compr in enumerate(np.arange(0.1, 1, 0.1)):
    print(f"压缩率:{compr:.2f} ({idx_c+1}/{len(np.arange(0.1, 1, 0.1))})")
    for idx_s, Snr in enumerate([3, 10]):
        print(f"  信噪比:{Snr} dB ({idx_s+1}/2)")
        AE       = AutoEncoder.AED_cnn_mnist(encoded_space_dim = int(raw_size * compr), snr  = Snr).to(args.device)
        classify = LeNets.LeNet_3().to(args.device)

        optimAE       = torch.optim.Adam(AE.parameters(), 1e-3)
        optimClassify = torch.optim.Adam(classify.parameters(), 1e-3)

        LossMse      = torch.nn.MSELoss()
        LossCroseny  = torch.nn.CrossEntropyLoss()
        trainrecord  = MetricsLog.TraRecorder(Len = 8)

        epochs = 100
        for epoch in range(epochs):
            AE.train()
            classify.train()

            metric = MetricsLog.Accumulator(6)
            trainrecord.addlog(epoch)

            for batch, (X, y) in enumerate(loader_train):
                X, y = X.to(device), y.to(device)

                X_hat = AE(X)
                y_hat = classify(X_hat)

                loss = compr * compr * LossMse(X_hat, X) + (1 - compr) * LossCroseny(y_hat, y)

                optimAE.zero_grad()
                optimClassify.zero_grad()
                loss.backward()
                optimAE.step()
                optimClassify.step()

                with torch.no_grad():
                    X, X_hat, y, y_hat = X.detach().cpu(), X_hat.detach().cpu(), y.detach().cpu(), y_hat.detach().cpu()
                    acc           = tcommon.accuracy(y_hat, y)
                    batch_01_psnr = tcommon.PSNR_torch(X, X_hat)
                    X     =  tcommon.data_inv_tf_cnn_mnist_batch_3D(X)
                    X_hat =  tcommon.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                    batch_avg_psnr = tcommon.PSNR_torch_Batch(X, X_hat )
                    metric.add(loss.item(), batch_01_psnr, batch_avg_psnr, acc, 1, X.size(0))
                # 输出训练状态
                if batch % 100 == 0:
                    print(f"    [epoch: {epoch+1:*>5d}/{epochs}, batch: {batch+1:*>5d}/{len(loader_train)}]\tLoss: {loss.item()/X.size(0):.4f} \t acc:{acc:.3f} \t batch_avg_psnr: {batch_01_psnr:.3f}/{batch_avg_psnr:.3f} (dB)")

            # average train metrics
            avg_loss     = metric[0]/metric[4]
            avg_batch_01 = metric[1]/metric[4]
            avg_batch    = metric[2]/metric[4]
            accuracy     = metric[3]/metric[5]

            val_batch_01, val_batch,  val_acc = validate( AE, classify, loader_test )

            trainrecord.assign([avg_loss, avg_batch_01, avg_batch,  accuracy, val_batch_01, val_batch, val_acc])

            tmp = tm.toc()
            print(f"  压缩率:{compr:.2f} ({idx_c+1}/{len(np.arange(0.1, 1, 0.1))}), 信噪比:{Snr} dB ({idx_s+1}/2) | Epoch: {epoch+1}/{epochs}({(epoch+1)*100.0/epochs:5.2f}%) | train loss = {avg_loss:.4f}, psnr:{avg_batch_01:.3f}/{avg_batch:.3f}(dB), acc: {acc:.3f} | test acc: {val_acc:.3f}, psnr:{val_batch_01:.3f}/{val_batch}(dB)| Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")


























