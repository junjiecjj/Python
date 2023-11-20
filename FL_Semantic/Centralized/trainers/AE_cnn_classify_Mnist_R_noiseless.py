#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:59:52 2023

@author: jack

此代码的功能：
训练语义传输模型, 训练时是不加噪声的, 分类器是预训练的, 在此不训练分类器.

统计在指定压缩率下的训练过程的指标(分类准确率, PSNR等), 以及在各个指定压缩率下训练完后在测试集上的指标,


其中各个过程的日志都被记录, 包括:
    训练过程每个epoch 的分类正确率,PSNR等
    测试过程的在每个压缩率下时每个测试信噪比下的分类正确率, psnr等
"""


import sys, os
import time, datetime
import numpy as np
import imageio

import torch
from torch.autograd import Variable

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import PIL

sys.path.append("../")
# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()
import Optimizer
from trainers import common, MetricsLog
from model import AutoEncoder
from loss import Loss

plt.rc('font', family='Times New Roman')
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


class AE_cnn_classify_mnist_R_noiseless_Trainer():
    def __init__(self, classifier, args, loader, ckp, ):
        self.args         = args
        self.ckp          = ckp
        self.loader_train = loader.loader_train
        self.loader_test  = loader.loader_test
        self.classify     = classifier
        self.net          = None
        self.Loss         = None
        self.device       = args.device
        self.source_snr   = None
        self.testRecoder  = MetricsLog.TesRecorder(Len = 4)
        return


    #@profile
    def train(self):
        # self.classify.eval()
        tm = common.myTimer()
        raw_dim = 28 * 28
        plot_interval = 30

        print(color.higred(f"\n#==================== 开始训练:{tm.start_str} =======================\n"))
        self.ckp.write_log(f"#==================== 开始训练: {tm.start_str} =======================\n", train=True)
        self.ckp.write_log(f" 压缩率:{self.args.CompRate} \n 信噪比: {self.args.SNRtrain}", train=True)

        for idx_c, comrate in enumerate(self.args.CompRate):
            encoded_dim = int(raw_dim * comrate)
            print(f"压缩率:{comrate:.1f} ({idx_c+1}/{len(self.args.CompRate)})"); self.ckp.write_log(f"压缩率:{comrate:.1f} ({idx_c+1}/{len(self.args.CompRate)})", train=True)
            self.net = AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = 0, quantize = self.args.quantize ).to(self.device)
            random_snr = None     #   np.random.uniform(1, 11, size=(1))[0]
            tsnr = "noiseless"
            self.net.set_snr(random_snr)
            self.optim = Optimizer.make_optimizer(self.args, self.net, )
            self.Loss  = Loss.myLoss(self.args, )
            self.trainrecord = MetricsLog.TraRecorder(9, name = "Train", compr = comrate, tra_snr = tsnr)

            for epoch in range(self.args.epochs):
                metric = MetricsLog.Accumulator(6)
                self.net.train()
                lr = self.optim.updatelr()
                self.Loss.addlog()
                self.trainrecord.addlog(epoch)

                print(f"\n    Epoch : {epoch+1}/{self.args.epochs}, lr = {lr:.3e}, 压缩率:{comrate:.1f} ({idx_c+1}/{len(self.args.CompRate)}), 信噪比:{tsnr}(dB) ")
                for batch, (X, label) in enumerate(self.loader_train):
                    X, = common.prepare(self.device, self.args.precision,   X)        # 选择精度且to(device)
                    X_hat = self.net(X)
                    loss = self.Loss(X_hat, X)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    with torch.no_grad():
                        y_hat          = self.classify(X_hat).detach().cpu()
                        X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                        acc            = common.accuracy(y_hat, label )
                        batch_01_psnr  = common.PSNR_torch(X , X_hat , )
                        X              =  common.data_inv_tf_cnn_mnist_batch_3D(X)
                        X_hat          =  common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                        batch_avg_psnr = common.PSNR_torch_Batch(X, X_hat , )
                        metric.add(loss, batch_01_psnr, batch_avg_psnr, acc, 1, X.size(0))
                    # 输出训练状态
                    if batch % 100 == 0:
                        print(f"    [epoch: {epoch+1:*>5d}/{self.args.epochs}, batch: {batch+1:*>5d}/{len(self.loader_train)}]\tLoss: {loss.item()/X.size(0):.4f} \t acc:{acc:.3f} \t batch_avg_psnr: {batch_01_psnr:.3f}/{batch_avg_psnr:.3f}(dB)")
                self.optim.schedule()
                epochLos = self.Loss.mean_log()[-1].item()  # len(self.loader_train.dataset)

                # average train metrics
                avg_loss     = metric[0]/metric[4]
                avg_batch_01 = metric[1]/metric[4]
                avg_batch    = metric[2]/metric[4]
                accuracy     = metric[3]/metric[5]
                # validate on test dataset
                val_batch_01, val_batch, val_acc = self.validate(self.net, self.classify, self.loader_test[0])

                self.trainrecord.assign([lr, avg_loss, avg_batch_01, avg_batch, accuracy, val_batch_01, val_batch, val_acc])

                if epoch % plot_interval == 0 or (epoch + 1) == self.args.epochs:
                    self.R_SNR_epochImgs(self.net, self.classify, self.loader_test[0], comrate, tsnr, epoch, avg_batch_01, val_batch_01, cols = 5, )
                    # common.R_SNR_epochImgs(self.ckp.savedir,self.net, self.classify, self.loader_test[0], comrate, tsnr, epoch, avg_batch_01, val_batch_01, cols = 5, )
                tmp = tm.toc()
                print("    ******************************************************")
                print(f"    loss = {epochLos:.3f}/{self.trainrecord[1]:.3f}, PSNR: {self.trainrecord[2]:.3f}/{self.trainrecord[3]:.3f}(dB), acc:{accuracy:.3f} | val psnr: {val_batch_01:.3f}/{val_batch:.3f}(dB), acc:{val_acc:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")
                print("    ******************************************************")
                self.ckp.write_log(f"  Epoch {epoch+1}/{self.args.epochs} | loss = {epochLos:.3f}/{self.trainrecord[1]:.3f}, PSNR: {self.trainrecord[2]:.3f}/{self.trainrecord[3]:.3f}(dB), acc:{accuracy:.3f} | val psnr: {val_batch_01:.3f}/{val_batch:.3f}(dB), acc:{val_acc:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)", train=True)

            self.trainrecord.save(self.ckp.savedir)
            self.trainrecord.plot_inonefig(self.ckp.savedir, metric_str = ['lr', 'train loss', '0-1_PSNR', 'batch_PSNR', 'train acc', 'val 0-1_PSNR', 'val_batch_PSNR', 'val acc'])

            self.R_SNR_valImgs(self.net, self.classify, self.loader_test[0], trainR = comrate, tra_snr = tsnr, snrlist = self.args.SNRtest)
            self.test_R_snr(self.net, self.classify, self.loader_test[0], comrate, tsnr, SNRlist = self.args.SNRtest)

            ### 保存网络中的参数, 速度快，占空间少
            torch.save(self.net.state_dict(), f"/home/{self.args.user_name}/SemanticNoise_AdversarialAttack/ModelSave/NoQuan_MSE/AE_Minst_R={comrate:.1f}_trainSnr={tsnr}.pt")

        self.ckp.write_log(f"#=========== 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ==================", train = True)
        print(color.higred(f"\n#============ 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ==================\n"))
        return

    # 在指定压缩率和信噪比下训练完所有的epoch后, 在测试集上的指标统计
    def test_R_snr(self, model, classifier, dataloader, compr, tasnr, SNRlist = np.arange(-2, 10, 2), ):
        tm = common.myTimer()
        model.eval()
        classifier.eval()
        self.ckp.write_log(f"#=============== 开始在 压缩率:{compr:.1f}, 信噪比:{tasnr}(dB)下测试, 开始时刻: {tm.start_str} ================\n")
        self.ckp.write_log("  {:>12}  {:>12}  {:>12}  {:>12} ".format("测试信噪比", "acc", "avg_batch_01", "avg_batch" ))
        print(color.green(f"    压缩率:{compr:.1f}, 信噪比: {tasnr} (dB), 测试集:"))
        # 增加 指定压缩率和信噪比下训练的模型的测试条目
        self.testRecoder.add_item(compr, tasnr,)
        with torch.no_grad():
            for snr in SNRlist:
                model.set_snr(snr)
                # 增加条目下的 一个测试信噪比
                self.testRecoder.add_snr(compr, tasnr, snr)
                metric = MetricsLog.Accumulator(5)

                for batch, (X, label) in enumerate(dataloader):
                    X,             = common.prepare(self.device, self.args.precision, X )        # 选择精度且to(device)
                    X_hat          = model(X)
                    # 传输后分类
                    predlabs       = classifier(X_hat).detach().cpu()
                    # 计算准确率
                    X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                    acc            = common.accuracy(predlabs, label )
                    batch_01_psnr  = common.PSNR_torch(X, X_hat, )
                    X              = common.data_inv_tf_cnn_mnist_batch_3D(X)
                    X_hat          = common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                    batch_avg_psnr = common.PSNR_torch_Batch(X, X_hat, )
                    metric.add(acc, batch_01_psnr, batch_avg_psnr,  1, X.size(0))
                accuracy     = metric[0]/metric[4]
                avg_batch_01 = metric[1]/metric[3]
                avg_batch    = metric[2]/metric[3]

                met = torch.tensor([accuracy, avg_batch_01, avg_batch ])
                self.testRecoder.assign(compr, tasnr, met)
                self.ckp.write_log(f"  {snr:>10}, {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} ")
                print(color.green(f"  {snr:>10}(dB), {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} "))
        self.testRecoder.save(self.ckp.testResdir,)
        self.testRecoder.plot_inonefig1x2(self.ckp.testResdir, metric_str = ['acc', '0-1_PSNR', 'batch_PSNR', ], tra_compr = compr, tra_snr = tasnr,)
        return

    # 保存指定压缩率和信噪比下训练完后, 画出在所有测试信噪比下的图片传输、恢复、分类示例.
    def R_SNR_valImgs(self, model, classifier, dataloader, trainR = 0.1, tra_snr = 2, snrlist = np.arange(-2, 10, 2) ):
        model.eval()
        classifier.eval()
        savedir = os.path.join(self.ckp.testResdir, f"Images_compr={trainR:.1f}_trainSnr={tra_snr}(dB)" )
        os.makedirs(savedir, exist_ok = True)
        rows =  4
        cols = 5
        # 固定的选前几张图片
        idx = np.arange(0, rows*cols, 1)
        labels = dataloader.dataset.targets[idx]
        # 原图
        real_image  = dataloader.dataset.data[idx] #.numpy()
        # 原图的预处理
        test_data   = common.data_tf_cnn_mnist_batch(real_image)
        test_data,  = common.prepare(self.device, self.args.precision, test_data)
        # 原图预处理后分类
        labs_raw    = classifier(test_data).detach().cpu().argmax(axis = 1)
        raw_dir     = os.path.join(self.ckp.testResdir, "raw_image")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok = True)
            for idx, (im, label) in enumerate(zip(real_image, labels)):
                im = PIL.Image.fromarray(im.numpy())
                im.save(os.path.join(raw_dir, f"{idx}_{label}.png"))
            common.grid_imgsave(raw_dir, real_image, labels, predlabs = labs_raw, dim = (rows, cols), suptitle = "Raw images", basename = "raw_grid_images")

        # 开始遍历测试信噪比
        with torch.no_grad():
            for snr in snrlist:
                subdir = os.path.join(savedir, f"testSNR={snr}(dB)")
                os.makedirs(subdir, exist_ok = True)
                model.set_snr(snr)
                # 传输
                im_result = model(test_data)
                # 传输后分类
                labs_recover = classifier(im_result).detach().cpu().argmax(axis = 1)
                # 自编码器恢复的图片
                im_result = im_result.detach().cpu()
                im_result = common.data_inv_tf_cnn_mnist_batch_2D(im_result)
                for idx, (im, label) in enumerate(zip(im_result, labels)):
                    im = PIL.Image.fromarray(im )
                    im.save(os.path.join(subdir, f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)_{idx}_{label}.png"))
                a = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr, snr)
                bs = f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)"
                common.grid_imgsave(subdir, im_result, labels, predlabs = labs_recover, dim = (rows, cols), suptitle = a, basename = "grid_images_" + bs )
        return

    # 在指定压缩率和信噪比下训练间隔epoch时, 在验证集上的测试结果
    def validate(self, model, classifier, dataloader, ):
        model.eval()
        classifier.eval()
        metric  = MetricsLog.Accumulator(5)
        with torch.no_grad():
            for batch, (X, label) in enumerate(dataloader):
                X, = common.prepare(self.device, self.args.precision, X )        # 选择精度且to(device)
                # 传输
                X_hat          = model(X)
                # 传输后分类
                predlabs       = classifier(X_hat).detach().cpu()
                X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                # 计算准确率
                acc            = common.accuracy(predlabs, label )
                # 不同方法计算 PSNR
                batch_01_psnr  = common.PSNR_torch(X , X_hat , )
                X              = common.data_inv_tf_cnn_mnist_batch_3D(X)
                X_hat          = common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                batch_avg_psnr = common.PSNR_torch_Batch(X, X_hat, )

                metric.add( batch_01_psnr, batch_avg_psnr, acc, 1, X.size(0))
            val_batch_01 = metric[0]/metric[3]
            val_batch    = metric[1]/metric[3]
            val_acc      = metric[2]/metric[4]
        return  val_batch_01, val_batch,  val_acc

    # 画出在指定压缩率和信噪比下训练 epoch 后的图片传输、恢复、分类效果
    def R_SNR_epochImgs(self, model, classifier, dataloader, trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr, cols = 5 ):
        model.eval()
        classifier.eval()
        comparedir = self.ckp.savedir + "/valiateImage"
        os.makedirs(comparedir, exist_ok = True)
        testloaderlen = dataloader.dataset.data.size(0)
        # 随机选 cols 张图片
        idx = np.random.randint(low = 0, high = testloaderlen, size = (cols, ))
        label = dataloader.dataset.targets[idx]
        # 原图
        real_image = dataloader.dataset.data[idx] #.numpy()

        with torch.no_grad():
            # 原图预处理
            test_data   = common.data_tf_cnn_mnist_batch(real_image) # .view(-1, 1, 28, 28).type(torch.FloatTensor)/255.0
            test_data,  = common.prepare(self.device, self.args.precision, test_data)
            # 原图预处理后分类
            labs_raw    = classifier(test_data).detach().cpu().argmax(axis = 1)
            # 传输
            im_result   = model(test_data)
            # 传输后分类
            labs_recover = classifier(im_result).detach().cpu().argmax(axis = 1)
            # 传输后图片的反预处理
            im_result = im_result.detach().cpu()
            im_result = common.data_inv_tf_cnn_mnist_batch_2D(im_result)

        rows =  2
        figsize = (cols*2 , rows*2)
        fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True) #  constrained_layout=True
        for j  in range(cols):
            axs[0, j].imshow(real_image[j], cmap='Greys')
            font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
            axs[0, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(label[j], labs_raw[j]),  fontdict = font1, )
            axs[0, j].set_xticks([] )  #  不显示x轴刻度值
            axs[0, j].set_yticks([] )  #  不显示y轴刻度值

            axs[1, j].imshow( im_result[j] , cmap='Greys')
            font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
            axs[1, j].set_title(r"$\mathrm{{label}}:{} \rightarrow {}$".format(label[j], labs_recover[j]),  fontdict = font1, )
            axs[1, j].set_xticks([] )  #  不显示x轴刻度值
            axs[1, j].set_yticks([] )  #  不显示y轴刻度值

            if j == 0:
                font = {'style': 'normal', 'size': 18, }
                axs[0, j].set_ylabel('Raw img', fontdict = font, labelpad = 8) # fontdict = font,
                axs[1, j].set_ylabel('Recovered img', fontdict = font, labelpad = 8)

        fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        fontt = {'style': 'normal', 'size': 20, }
        supt = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{epoch}}:{}, \mathrm{{PSNR}}_\mathrm{{train}}:{:.2f}\mathrm{{(dB)}}, \mathrm{{PSNR}}_\mathrm{{val}}:{:.2f}\mathrm{{(dB)}}$'.format(trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr)
        plt.suptitle(supt, fontproperties=fontt )

        out_fig = plt.gcf()
        out_fig.savefig(comparedir + f"/images_R={trainR:.1f}_trainSnr={trainSnr}(dB)_epoch={epoch}.png",  bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        return
