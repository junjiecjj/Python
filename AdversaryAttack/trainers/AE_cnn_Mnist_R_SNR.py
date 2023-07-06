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
from . import common, MetricsLog
from model import AutoEncoder
from loss import Loss


plt.rc('font', family='Times New Roman')
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


class AE_cnn_mnist_R_SNR_Trainer():
    def __init__(self, args, loader, ckp, ):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.net = None
        self.Loss = None
        self.device = args.device
        self.source_snr = None
        self.testRecoder = MetricsLog.TestRecorder(Len = 5)
        return


    #@profile
    def train(self):
        tm = common.myTimer()
        raw_dim = 28 * 28
        plot_interval = 2

        print(color.higred(f"\n#=============== 开始训练, 时刻:{tm.start_str} ====================\n"))
        self.ckp.write_log(f"#============ 开始训练, 开始时刻: {tm.start_str} ================\n", train=True)
        self.ckp.write_log(f" 压缩率:{self.args.CompRate} \n 信噪比: {self.args.SNRtrain}", train=True)

        for idx_c, comrate in enumerate(self.args.CompRate):
            encoded_dim = int(raw_dim * comrate)
            print(f"压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)})"); self.ckp.write_log(f"压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)})", train=True)
            for idx_s, Snr in enumerate(self.args.SNRtrain):
                print(f"  信噪比:{Snr} dB ({idx_s+1}/{len(self.args.SNRtrain)})"); self.ckp.write_log(f"  信噪比: {Snr} dB ({idx_s+1}/{len(self.args.SNRtrain)})\n\n", train=True)
                self.net = AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = Snr).to(self.device)
                self.optim = Optimizer.make_optimizer(self.args, self.net, )
                self.Loss  = Loss.myLoss(self.args, )
                self.trainrecord = MetricsLog.TrainRecorder(4, name = "Train")
                # torch.set_grad_enabled(True)
                for epoch in range(self.args.epochs):
                    metric = MetricsLog.Accumulator(6)
                    self.net.train()
                    print(f"\n    Epoch : {epoch+1}/{self.args.epochs}({100.0*(1+epoch)/self.args.epochs:0>5.2f}%), lr = {self.optim.get_lr():.3e}, 压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)}), 信噪比:{Snr} dB ({idx_s+1}/{len(self.args.SNRtrain)})")
                    self.optim.updatelr()
                    self.Loss.addlog()
                    self.trainrecord.addlog()
                    for batch, (X, label) in enumerate(self.loader_train):
                        # use noised img to train
                        X_noised = common.Awgn(X, snr = self.source_snr)
                        X_noised, X = common.prepare(self.device, self.args.precision, X_noised, X)        # 选择精度且to(device)
                        X_hat = self.net(X_noised)
                        loss = self.Loss(X_hat, X)
                        self.optim.zero_grad()       # 必须在反向传播前先清零。
                        loss.backward()
                        self.optim.step()
                        with torch.no_grad():
                            batch_01_psnr = common.PSNR_torch(X.cpu(), X_hat.cpu(), )
                            X     =  common.data_inv_tf_cnn_mnist_batch_3D(X)
                            X_hat =  common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                            batch_avg_psnr = common.PSNR_torch_Batch(X.cpu(), X_hat.cpu(), )
                            image_avg_psnr, image_sum_psnr, batchsize = common.PSNR_torch_Image(X.cpu(), X_hat.cpu(), )
                            metric.add(batch_01_psnr, batch_avg_psnr, image_avg_psnr, image_sum_psnr, 1, batchsize)
                        # 输出训练状态
                        if batch % 100 == 0:
                            print("    [epoch: {:*>5d}/{}, batch: {:*>5d}/{}]\tLoss: {:.4f} \t batch_avg_psnr: {:.3f}/{:.3f}/{:.3f}(dB) ".format(epoch+1, self.args.epochs,  batch+1, len(self.loader_train), loss.item()/X.size(0), batch_01_psnr, batch_avg_psnr, image_avg_psnr ))
                    avg_batch_01 = metric[0]/metric[4]
                    avg_batch    = metric[1]/metric[4]
                    avg_img_psnr = metric[2]/metric[4]
                    avg_sum_psnr = metric[3]/metric[5]
                    self.trainrecord.assign([avg_batch_01, avg_batch, avg_img_psnr, avg_sum_psnr])
                    self.optim.schedule()
                    epochLos = self.Loss.mean_log()[-1]  # len(self.loader_train.dataset)
                    # validate on test dataset
                    avglos, avg_batch_01, avg_batch, avg_img_psnr, avg_sum_psnr = self.validate(self.net, self.loader_test[0])
                    # if epoch % plot_interval == 0 or (epoch + 1) == self.args.epochs:
                        # self.R_SNR_epochImgs(self.net, self.loader_test[0], comrate, snr, epoch, self.trainrecord[1], avg_sum_psnr, cols = 5, )
                    tmp = tm.toc()
                    print("      ******************************************************")
                    print(f"      Epoch: {epoch+1}/{self.args.epochs}({(epoch+1)*100.0/self.args.epochs:5.2f}%) | loss = {epochLos.item():.3f}, avg PSNR: {self.trainrecord[0]:.3f}/{self.trainrecord[1]:.3f}/{self.trainrecord[2]:.3f}/{self.trainrecord[3]:.3f}(dB) | val loss:{avglos:.3f}, val psnr: {avg_batch_01:.3f}/{avg_batch:.3f}/{avg_img_psnr:.3f}{avg_sum_psnr:.3f}(dB) | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")
                    self.ckp.write_log(f"    Epoch {epoch+1}/{self.args.epochs}({100.0*(1+epoch)/self.args.epochs:0>5.2f}%) | loss = {epochLos.item():.3f} | avg PSNR: {self.trainrecord[1]:.3f} | val loss:{avglos:.3f}, val psnr: {avg_batch_01:.3f}/{avg_batch:.3f}/{avg_img_psnr:.3f}{avg_sum_psnr:.3f}(dB) | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)", train=True)
                    print("      ******************************************************")

                self.optim.save_lr(self.ckp.savedir, compr = comrate, tra_snr = Snr)
                self.Loss.save(self.ckp.savedir, compr = comrate, tra_snr = Snr)
                self.trainrecord.save(self.ckp.savedir, compr = comrate, tra_snr = Snr)
                self.trainrecord.plot_onlypsnr(self.ckp.savedir, compr = comrate, tra_snr = Snr, metric_str = ['0-1_PSNR', 'batch_PSNR', 'bat_img_PSNR', 'img_psnr'])
                self.R_SNR_valImgs(self.net, self.loader_test[0], trainR = comrate, tra_snr = Snr, snrlist = self.args.SNRtest)
                valres =  self.test_R_snr(self.net, self.loader_test[0], comrate, Snr, SNRlist = self.args.SNRtest)
                print(color.green(f"    压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)}), 信噪比:{Snr} dB ({idx_s+1}/{len(self.args.SNRtrain)}), 测试集:"))
                common.formatPrint2DArray(valres)
                ### 保存网络中的参数, 速度快，占空间少
                torch.save(self.net.state_dict(), f"/home/jack/SemanticNoise_AdversarialAttack/ModelSave/AE_Minst_R={comrate:.1f}_trainSnr={Snr}.pt")

        self.ckp.write_log(f"#============= 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 =================", train = True)
        print(color.higred(f"\n#=========== 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ==============\n"))
        return

    ## 在指定压缩率和信噪比下训练完所有的epoch后,在测试集上测试
    def test_R_snr(self, model, dataloader, compr, tasnr, SNRlist = np.arange(-2, 10, 2), ):
        tm = common.myTimer()
        # torch.set_grad_enabled(False)
        model.eval()
        self.ckp.write_log(f"#=============== 开始在 压缩率:{compr:.2f}, 信噪比:{tasnr}(dB)下测试, 开始时刻: {tm.start_str} =================\n")
        self.ckp.write_log("  {:>12}  {:>12}  {:>12}".format("测试信噪比", "batImg_PSNR", "img_PSNR"))

        # 增加 指定压缩率和信噪比下训练的模型的测试条目
        self.testRecoder.add_item(compr, tasnr,)
        with torch.no_grad():
            for snr in SNRlist:
                model.set_snr(snr)
                # 增加条目下的 一个测试信噪比
                self.testRecoder.add_snr(compr, tasnr, snr)
                metric = MetricsLog.Accumulator(6)

                for batch, (X, label) in enumerate(dataloader):
                    X,  = common.prepare(self.device, self.args.precision, X )        # 选择精度且to(device)
                    X_hat = model(X)
                    batch_01_psnr = common.PSNR_torch(X.cpu(), X_hat.cpu(), )
                    X =  common.data_inv_tf_cnn_mnist_batch_3D(X)
                    X_hat =   common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                    batch_avg_psnr = common.PSNR_torch_Batch(X.cpu(), X_hat.cpu(), )
                    image_avg_psnr, image_sum_psnr, batchsize = common.PSNR_torch_Image(X.cpu(), X_hat.cpu(), )
                    metric.add(batch_01_psnr, batch_avg_psnr, image_avg_psnr, image_sum_psnr, 1, batchsize)
                avg_batch_01 = metric[0]/metric[4]
                avg_batch    = metric[1]/metric[4]
                avg_img_psnr = metric[2]/metric[4]
                avg_sum_psnr = metric[3]/metric[5]
                met = torch.tensor([avg_batch_01, avg_batch, avg_img_psnr, avg_sum_psnr])
                self.testRecoder.assign(compr, tasnr, met)
                self.ckp.write_log(f"  {snr:>10}, {avg_batch_01:>12.3f}, {avg_batch:>12.3f} {avg_img_psnr:>12.3f}, {avg_sum_psnr:>12.3f}")
        self.testRecoder.save(self.ckp.testResdir,)
        self.testRecoder.plot_onlypsnr(self.ckp.testResdir, metric_str = ['0-1_PSNR', 'batch_PSNR', 'bat_img_PSNR', 'img_psnr'], tra_compr = compr, tra_snr = tasnr,)

        tmpS = "TestMetrics:Compr={:.1f},SNRtrain={}(dB)".format( compr, tasnr)
        return  self.testRecoder.TeMetricLog[tmpS]

    ## 画出在指定压缩率和信噪比下训练完一定 epoch 后, 在测试集上计算误差, psnr等
    def validate(self, model, dataloader, ):
        model.eval()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        metric = MetricsLog.Accumulator(7)
        with torch.no_grad():
            for batch, (X, label) in enumerate(dataloader):

                X_noised = common.Awgn(X, snr = self.source_snr)
                X_noised, X = common.prepare(self.device, self.args.precision, X_noised, X)        # 选择精度且to(device)

                X_hat = model(X_noised)
                loss = loss_fn(X_hat, X).item()
                batch_01_psnr = common.PSNR_torch(X.cpu(), X_hat.cpu(), )
                X = common.data_inv_tf_cnn_mnist_batch_3D(X)
                X_hat =   common.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                batch_avg_psnr = common.PSNR_torch_Batch(X.cpu(), X_hat.cpu(), )
                image_avg_psnr, image_sum_psnr, batchsize = common.PSNR_torch_Image(X.cpu(), X_hat.cpu(),)
                metric.add(loss, batch_01_psnr, batch_avg_psnr, image_avg_psnr, image_sum_psnr, 1, X.size(0))
            avglos = metric[0]/metric[6]
            avg_batch_01 = metric[1]/metric[5]
            avg_batch    = metric[2]/metric[5]
            avg_img_psnr = metric[3]/metric[5]
            avg_sum_psnr = metric[4]/metric[6]
        return avglos, avg_batch_01, avg_batch, avg_img_psnr, avg_sum_psnr

    ## 画出在指定压缩率和信噪比下训练完所有的epoch后, 在不同的测试信噪比下模拟传输测试集上图片, 并保存传输前后的图片
    def R_SNR_valImgs(self, model, dataloader, trainR = 0.1, tra_snr = 2, snrlist = np.arange(-2, 10, 2) ):
        model.eval()
        savedir = os.path.join(self.ckp.testResdir, f"Images_compr={trainR:.1f}_trainSnr={tra_snr}(dB)" )
        os.makedirs(savedir, exist_ok = True)
        rows =  4
        cols = 5
        idx = np.arange(0, rows*cols, 1)
        labels = dataloader.dataset.targets[idx]
        # 真实图片
        real_image = dataloader.dataset.data[idx] #.numpy()
        test_data = common.data_tf_cnn_mnist_batch(real_image) # .view(-1, 1, 28, 28).type(torch.FloatTensor)/255.0
        test_data,  = common.prepare(self.device, self.args.precision, test_data)

        raw_dir = os.path.join(self.ckp.testResdir, "raw_image")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok = True)
            for idx, (im, label) in enumerate(zip(real_image, labels)):
                im = PIL.Image.fromarray(im.numpy())
                im.save(os.path.join(raw_dir, f"{idx}_{label}.png"))
            common.grid_imgsave(raw_dir, real_image, labels, predlabs = '', dim = (rows, cols), suptitle = "Raw images", basename = "raw_grid_images")

        with torch.no_grad():
            for snr in snrlist:
                subdir = os.path.join(savedir, f"testSNR={snr}(dB)")
                os.makedirs(subdir, exist_ok = True)
                model.set_snr(snr)
                im_result = model(test_data).detach().cpu()
                # 自编码器恢复的图片
                im_result = common.data_inv_tf_cnn_mnist_batch_2D(im_result )

                for idx, (im, label) in enumerate(zip(im_result, labels)):
                    im = PIL.Image.fromarray(im )
                    im.save(os.path.join(subdir, f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)_{idx}_{label}.png"))
                a = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr, snr)
                bs = f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)"
                common.grid_imgsave(subdir, im_result, labels, predlabs = '', dim = (rows, cols), suptitle = a, basename = "grid_images_" + bs )
        return

    ## 画出在指定压缩率和信噪比下训练完所有的epoch后, 在训练信噪比下模拟传输测试集图片, 并保存传输前后的图片
    def R_SNR_epochImgs(self, model, dataloader, trainR, trainSnr, epoch, trainavgpsnr, valavgpsnr, cols = 5 ):
        model.eval()
        comparedir = self.ckp.savedir + "/valiateImage"
        os.makedirs(comparedir, exist_ok = True)
        testloaderlen = dataloader.dataset.data.size(0)
        idx = np.random.randint(low = 0, high = testloaderlen, size = (cols, ))
        label = dataloader.dataset.targets[idx]
        # 真实图片
        real_image = dataloader.dataset.data[idx] #.numpy()

        with torch.no_grad():
            test_data = common.data_tf_cnn_mnist_batch(real_image) # .view(-1, 1, 28, 28).type(torch.FloatTensor)/255.0
            test_data,  = common.prepare(self.device, self.args.precision, test_data)
            im_result = model(test_data).detach().cpu()
            # 自编码器恢复的图片
            im_result = common.data_inv_tf_cnn_mnist_batch_2D(im_result )

        rows =  2
        figsize = (cols*2 , rows*2)
        fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout=True) #  constrained_layout=True
        for j  in range(cols):
            axs[0, j].imshow(real_image[j], cmap='Greys')
            font1 = {'style': 'normal', 'size': 18, 'color':'blue', }
            axs[0, j].set_title(f"Ground Truth: {label[j]}",  fontdict = font1, )
            axs[0, j].set_xticks([] )  #  不显示x轴刻度值
            axs[0, j].set_yticks([] )  #  不显示y轴刻度值
            axs[1, j].imshow( im_result[j] , cmap='Greys')
            axs[1, j].set_xticks([] )  #  不显示x轴刻度值
            axs[1, j].set_yticks([] )  #  不显示y轴刻度值

            if j == 0:
                font = {'style': 'normal', 'size': 18, }
                axs[0, j].set_ylabel('Raw img', fontdict = font, labelpad = 8) # fontdict = font,
                axs[1, j].set_ylabel('Recovered img', fontdict = font, labelpad = 8)

        fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        fontt = {'style': 'normal', 'size': 20, }
        plt.suptitle(f"R={trainR:.1f}, trainSnr={trainSnr}(dB), epoch:{epoch}, train psnr:{trainavgpsnr:.2f}(dB), val psnr:{valavgpsnr:.2f}(dB)", fontproperties=fontt )

        out_fig = plt.gcf()
        out_fig.savefig(comparedir + f"/images_R={trainR:.1f}_trainSnr={trainSnr}(dB)_epoch={epoch}.png",  bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        return


















































