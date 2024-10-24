#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:59:52 2023

@author: jack
"""


import sys, os
import time, datetime
import numpy as np
import imageio

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
import shutil


import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm


#内存分析工具
from memory_profiler import profile

sys.path.append("../")
# 本项目自己编写的库
from ColorPrint  import ColoPrint
# import ColorPrint.ColoPrint as ColoPrint
color = ColoPrint()
# print(color.fuchsia("Color Print Test Pass"))
import Optimizer
import Utility

from Trainer import common, MetricsLog


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



class SemComTrainer():
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

        self.optim = Optimizer.make_optimizer(args, self.net, 'minstAutoEn')

        self.ckp.print_parameters(net = self.net, name = "mnistAutoEncoder")
        return


    #@profile
    def train(self):
        tm = common.myTimer()
        self.similarMetrics = MetricsLog.MetricsRecorder(self.args, 3)
        metrics = MetricsLog.AccuracyRecorder(1, metricsname = "MSE loss")

        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.higred(f"\n#================================ 开始训练, 时刻:{now1} =======================================\n"))
        self.ckp.write_log(f"#======================================== 开始训练, 开始时刻: {now1} =============================================\n", train=True)

        accumEpoch = 0
        for epoch in range(self.args.epochs):
            print(f"\nEpoch : {epoch+1}/{self.args.epochs}({100.0*(1+epoch)/self.args.epochs:0>5.2f}%)")
            self.ckp.write_log(f"\nEpoch : {epoch}", train=True)

            self.optim.updatelr()
            self.Loss.add_log()
            self.similarMetrics.addlog()
            metrics.addlog()

            metric = common.Accumulator(2)

            for batch, (X, label) in enumerate(self.loader_train):
                # print(f"X.shape = {X.shape}, label.shape = {label.shape}")
                channel, H, W = X.size(1), X.size(2), X.size(3)

                self.net.zero_grad()

                X = X.view(-1, 28*28)
                y = X.view(-1, 28*28)
                X, y = common.prepare(self.device, self.args.precision, X, y)        # 选择精度且to(device)

                encoded, X_hat = self.net(X)
                loss = self.Loss(X_hat, y)
                # print(f"X.shape = {X.shape}, y.shape = {y.shape}, X_hat.shape = {X_hat.shape}, loss = {loss}")
                # X.shape = torch.Size([128, 784]), y.shape = torch.Size([128, 784]), X_hat.shape = torch.Size([128, 784]), loss = 6977.56298828125
                self.optim.zero_grad()       # 必须在反向传播前先清零。
                loss.backward()
                self.optim.step()
                # print(f"y.shape = {y.shape}, y_hat.shape = {y_hat.shape}")

                with torch.no_grad():
                    metric.add(loss.item(), X.shape[0])
                    metrics.add([loss.item() ], X.size(0))

                    X = X.view(-1, channel, H, W).detach().cpu()
                    X_hat = X_hat.view(-1, channel, H, W).detach().cpu()

                    batch_avg_psnr = common.PSNR_torch_Batch( X*255, X_hat*255, cal_type = '1' )
                    image_avg_psnr, image_sum_psnr, batchsize = common.PSNR_torch_Image( X*255, X_hat*255 )
                    self.similarMetrics.add([batch_avg_psnr, image_avg_psnr, image_sum_psnr], batchsize)

                # 输出训练状态
                if batch % 100 == 0:
                    frac1 = (epoch + 1) / self.args.epochs
                    frac2 = (batch + 1)/len(self.loader_train)
                    print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss: {:.4f} \t batch_avg_psnr: {:.4f}/{:.4f}/{:.4f} ".format(epoch+1, self.args.epochs, frac1, batch+1, len(self.loader_train), frac2, loss.item()/X.size(0), batch_avg_psnr,image_avg_psnr,image_sum_psnr/batchsize ))
                    self.ckp.write_log('    [epoch: %d/%d, batch: %3d/%d]\tLoss_D: %.4f ' % (epoch+1, self.args.epochs, batch, len(self.loader_train), loss.item(),  ), train=True)

            # 学习率递减
            self.optim.schedule()

            # 计算并更新epoch的loss
            epochLos = self.Loss.mean_log()[-1]  # len(self.loader_train.dataset)

            # epoch 的平均 psnr
            self.similarMetrics.avg()
            metrics.avg()
            # epoch 的平均 loss
            train_l = metric[0]/metric[1]

            tmp = tm.toc()
            # tmp = tm.stop()
            print(f"  Epoch: {epoch+1}/{self.args.epochs}({(epoch+1)*100.0/self.args.epochs:5.2f}%) | loss = {train_l:.4f}/{epochLos.item():.4f}/{metrics[0]:.4f} | avg PSNR:{self.similarMetrics[0]:.3f}/{self.similarMetrics[1]:.3f}/{self.similarMetrics[2]:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")
            self.ckp.write_log(f"  Epoch {epoch+1}/{self.args.epochs} | loss = {epochLos } | avg PSNR:{self.similarMetrics[0]:.3f}/{self.similarMetrics[1]:.3f}/{self.similarMetrics[2]:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)

        # 保存模型的学习率并画图
        self.ckp.savelearnRate(self)
        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)
        self.similarMetrics.save(self.ckp.savedir, "Train_PSNR.pt")
        self.similarMetrics.plot(self.ckp.savedir)
        metrics.save(self.ckp.savedir, "Train_MSEloss.pt")
        metrics.plot(self.ckp.savedir)

        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.ckp.write_log(f"#========================= 本次训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ================================",train=True)
        # 关闭日志
        self.ckp.done()
        print(color.higred(f"======== 关闭训练日志 {self.ckp.log_file.name} =============="))
        print(color.higred(f"\n#====================== 训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ==============================\n"))
        return


    def viewMiddleFeature3D(self):
        view_data = self.loader_train.dataset.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.0
        view_data = common.prepare(self.device, self.args.precision, view_data)[0]
        encoded_data,  _ = self.net(view_data)    # 提取压缩的特征值
        encoded_data = encoded_data.detach().cpu().numpy()
        fig = plt.figure(figsize = (8, 8))
        ax = plt.axes(projection='3d')  # 3D 图
        # x, y, z 的数据值
        print(f"encoded_data.shape = {encoded_data[:3]}")
        X = encoded_data[:, 0]#.numpy()
        Y = encoded_data[:, 1]#.numpy()
        Z = encoded_data[:, 2]#.numpy()
        # print(X[0],Y[0],Z[0])
        values = self.loader_train.dataset.train_labels[:200].numpy()  # 标签值
        for x, y, z, s in zip(X, Y, Z, values):
            c = cm.rainbow(int(255*s/9))    # 上色
            ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        out_fig = plt.gcf()
        out_fig.savefig( self.ckp.savedir +'/Feature_3D.eps', format='eps',dpi=1000, bbox_inches='tight')
        # out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()
        return



    def raw_recovered(self):
        #原数据和生成数据的比较
        plt.ion()

        for i in range(30):
            test_data = self.loader_train.dataset.train_data[i].view(-1, 28*28).type(torch.FloatTensor)/255.
            test_data = common.prepare(self.device, self.args.precision, test_data)[0]
            # print(f"test_data.shape = {test_data }")
            _, result = self.net(test_data)
            # print('输入的数据的维度', train_data.train_data[i].size())
            # print('输出的结果的维度',result.size())

            im_result = result.view(28,28).detach().cpu().numpy()
            # print(im_result.size())
            plt.figure(1, figsize=(10, 3))
            plt.subplot(121)
            plt.title('test_data')
            plt.imshow(self.loader_train.dataset.train_data[i].numpy(), cmap='Greys')
            # plt.figure(1, figsize=(10, 3))
            plt.subplot(122)
            plt.title('result_data')
            plt.imshow(im_result, cmap='Greys')
            # plt.ioff()
            # plt.show()
            plt.pause(0.5)
        # plt.show()
        plt.ioff()
        plt.show()


    def raw_recovered_tmpview(self):
        self.net.eval()
        print(f"创建 {self.args.tmpout} 文件夹！")
        comparedir = self.ckp.savedir + "/tmpout"
        os.makedirs(comparedir, exist_ok = True)
        testloaderlen = self.loader_test[0].dataset.data.size(0)
        randombatch = 10
        rows =  5
        cols = 2
        figsize = (cols*2, rows*2 )
        for batch in range(randombatch):
            idx = np.random.randint(low = 0, high = testloaderlen, size = (rows, ))
            label = self.loader_test[0].dataset.targets[idx]
            
            test_data = self.loader_test[0].dataset.data[idx].view(-1, 28*28).type(torch.FloatTensor)/255.0
            test_data,  = common.prepare(self.device, self.args.precision, test_data) 
            _, im_result = self.net(test_data)
            # 自编码器恢复的图片
            im_result = im_result.view(-1, 28, 28).detach().cpu().numpy()

            # 真实图片
            real_image = self.loader_test[0].dataset.data[idx].numpy()
            fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True) #  constrained_layout=True

            for i  in range(rows):
                axs[i, 0].imshow(real_image[i], cmap='Greys')
                font = {'family': 'Times New Roman', 'style': 'normal', 'size': 12, }
                axs[i, 0].set_title('real img', fontdict = font, )

                axs[i, 0].set_xticks([])  # #不显示x轴刻度值
                axs[i, 0].set_yticks([] ) # #不显示y轴刻度值

                font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18, 'color':'blue', }
                axs[i, 0].set_ylabel(f"ground truth: {label[i]}",  fontdict = font1, labelpad = 8)

                axs[i, 1].imshow(im_result[i], cmap='Greys')
                font = {'family': 'Times New Roman', 'style': 'normal', 'size': 12,  }
                axs[i, 1].set_title('recovered img', fontdict = font, )
                axs[i, 1].set_xticks([])  # #不显示x轴刻度值
                axs[i, 1].set_yticks([] ) # #不显示y轴刻度值

            fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            plt.suptitle(f"batch = {batch}", fontproperties=fontt,)

            out_fig = plt.gcf()
            out_fig.savefig(comparedir + "/revovered_images_%d.eps" % (batch),  bbox_inches='tight')
            # plt.show()
            plt.close(fig)
        return



    def test1(self):


        return




































































































































































































































































































































































































































