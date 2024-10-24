# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import sys,os
import datetime
import numpy as np
import imageio

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
import shutil


#内存分析工具
from memory_profiler import profile
import objgraph

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()
# print(color.fuchsia("Color Print Test Pass"))
import Optimizer
import Utility
from model import common


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


class GANforGaussTrainer():
    def __init__(self, args, loader, generator, discriminator, loss_G, loss_D, ckp, writer):
        self.args = args
        
        
        
        self.wr = writer
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.netG = generator
        self.netD = discriminator

        self.Loss_G = loss_G
        self.Loss_D = loss_D
        self.device = args.device

        print(f"len(self.loader_train) = {len(self.loader_train)}, len(self.loader_train.dataset) = {len(self.loader_train.dataset)}")
        print(f"len(self.loader_test[0]) = {len(self.loader_test[0])}, len(self.loader_test[0].dataset) = {len(self.loader_test[0].dataset)}")

        self.optim_G = Optimizer.make_optimizer(args, self.netG, 'minstG')
        self.optim_D = Optimizer.make_optimizer(args, self.netD, 'minstD')

        self.ckp.print_parameters(netG = self.netG, netD = self.netD, )
        return

    def prepare(self, *Args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(self.args.device)

        return [_prepare(a) for a in Args]


    #@profile
    def train(self):
        iters = 0
        if os.path.exists(self.args.tmpout):
            print(f"删除 {self.args.tmpout} 文件夹！")
            if sys.platform.startswith("win"):
                shutil.rmtree(self.args.tmpout)
            else:
                os.system(f"rm -r {self.args.tmpout}")
        print(f"创建 {self.args.tmpout} 文件夹！")
        os.mkdir(self.args.tmpout)

        tm = common.myTimer()
        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.higred(f"\n#================================ 开始训练, 时刻:{now1} =======================================\n"))
        self.ckp.write_log(f"#======================================== 开始训练, 开始时刻: {now1} =============================================\n", train=True)

        examples = 25
        fixed_noise = torch.randn(examples, self.args.noise_dim).to(self.args.device)

        accumEpoch = 0
        for epoch in range(self.args.epochs):
            print(f"\nEpoch : {epoch+1}/{self.args.epochs}({100.0*(1+epoch)/self.args.epochs:0>5.2f}%)")
            self.ckp.write_log(f"\nEpoch : {epoch}", train=True)

            self.optim_G.updatelr()
            self.optim_D.updatelr()
            self.Loss_D.add_log()
            self.Loss_G.add_log()

            metricD = common.Accumulator(2)
            metricG = common.Accumulator(2)

            for batchidx, (real_imgs, real_num) in enumerate(self.loader_train):
                #####################################################################
                # (一) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                #####################################################################
                self.netD.zero_grad()
                ## (1) 训练真实图片
                real_label = torch.ones(real_imgs.size(0), 1)   # 定义真实的图片label为1
                real_imgs, real_label = self.prepare(real_imgs, real_label)        # 选择精度且to(device) real_imgs.shape = torch.Size([128, 1, 28, 28]), real_label.shape = torch.Size([128, 1])

                real_out = self.netD(real_imgs)       # 将真实图片放入判别器中,得到真实图片的判别值，输出的值越接近1越好 # real_out.shape = torch.Size([128, 1])
                D_x = real_out.mean().item()

                loss_real_D = self.Loss_D(real_out, real_label)                    # 得到真实图片的loss

                ## (2) 训练生成图片
                noise =  torch.randn(real_imgs.size(0), self.args.noise_dim).to(self.args.device)        # 随机生成一些噪声, 大小为(128, 100)

                # 使用G生成图片
                fake_img = self.netG(noise).detach()                         #   避免梯度传到G，因为G不用更新, detach分离
                fake_label = torch.zeros(fake_img.size(0), 1)                # 定义假的图片的label为0
                fake_img, fake_label = self.prepare(fake_img, fake_label)    # 选择精度且to(device)

                fake_out = self.netD(fake_img)                               # 判别器判断假的图片

                loss_fake_D = self.Loss_D(fake_out, fake_label)              # 得到假的图片的loss

                D_G_z1 = fake_out.mean().item()

                # 累加误差，参数更新
                total_loss_D = loss_real_D + loss_fake_D                    # 损失包括判真损失和判假损失
                self.optim_D.zero_grad()                                    # 必须在反向传播前先将梯度归0
                total_loss_D.backward()                                     # 将误差反向传播
                self.optim_D.step()                                         # 更新参数

                with torch.no_grad():
                    metricD.add(total_loss_D.item() ,  real_imgs.shape[0])

                #print(f" [epoch: {epoch}, {batchidx}] self.Loss_D[-1,0] = {self.Loss_D.losslog[-1,0]}")
                #################################################
                # (二) Update G network: maximize log(D(G(z)))
                #################################################
                self.netG.zero_grad()
                # 对生成图再进行一次判别
                fake_img = self.netG(noise)
                fake_out = self.netD(fake_img)
                # 计算生成图片损失，梯度反向传播
                loss_fake_G = self.Loss_G(fake_out, real_label)
                self.optim_G.zero_grad()    # 必须在反向传播前先清零。
                loss_fake_G.backward()
                self.optim_G.step()
                D_G_z2 = fake_out.mean().item()
                #print(f"  D_G_z2 = {D_G_z2} ")

                with torch.no_grad():
                    metricG.add(loss_fake_G.item() ,  fake_img.shape[0])

                fake_img = common.de_norm(fake_img)

                # 输出训练状态
                if batchidx % 100 == 0:
                    frac1 = (epoch + 1) / self.args.epochs
                    frac2 = (1 + batchidx)/len(self.loader_train)
                    print("    [epoch: {:*>5d}/{}({:0>6.2%}), batch: {:*>5d}/{}({:0>6.2%})]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}, D(G(z)): {:.4f}/{:.4f}".format(epoch+1, self.args.epochs, frac1, batchidx+1, len(self.loader_train), frac2, total_loss_D.item(),loss_fake_G.item(), D_x, D_G_z1, D_G_z2))
                    self.ckp.write_log('    [epoch: %d/%d, batch: %3d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f, D(G(z)): %.4f/%.4f' % (epoch+1, self.args.epochs, batchidx, len(self.loader_train), total_loss_D.item(), loss_fake_G.item(), D_x, D_G_z1, D_G_z2), train=True)
                ## 对固定的噪声向量，存储生成的结果
                # if (iters % 100 == 0) or ((epoch == self.args.epochs - 1) and (batchidx == len(self.loader_train) - 1)):
                #     with torch.no_grad():
                #         plot_image = self.netG(fixed_noise).detach().cpu()
                #         plot_image = common.de_norm(plot_image)
                #     self.draw_images(plot_image, epoch, batchidx, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (10, 11))
                # iters += 1
                # save_image(plot_image, self.args.tmpout+"Generated_images_epoch=%d_batch=%d.png" % (epoch,batchidx), nrow=5, normalize=True)
                # print(f"fake_img.max() = {fake_img.max()}, fake_img.min() = {fake_img.min()}, plot_image.max() = {plot_image.max()}, plot_image.min() = {plot_image.min()}, real_imgs.max() = {real_imgs.max()}, real_imgs.min() = {real_imgs.min()},")
            if epoch == 0 or (( 1 + epoch ) % 10 == 0 ):
                plot_image = self.netG(fixed_noise).detach().cpu()
                plot_image = common.de_norm(plot_image)

                common.draw_images1(self.args.tmpout, plot_image, epoch+1, batchidx, H = 28, W = 28, examples = 25,  dim = (5, 5), figsize = (10, 10))
                #save_image(plot_image, self.args.tmpout+"Generated_images_epoch=%d.png" % (epoch), nrow=5, normalize=True)

            # 学习率递减
            self.optim_G.schedule()
            self.optim_D.schedule()

            # 计算并更新epoch的loss
            epochLossG = self.Loss_G.mean_log(len(self.loader_train.dataset))[-1]
            epochLossD = self.Loss_D.mean_log(len(self.loader_train.dataset))[-1]
            #print(f"epochLossG = {epochLossG}, epochLossG[-1]= {epochLossG[-1]}")

            trainLossG = metricG[0]/metricG[1]
            trainLossD = metricD[0]/metricD[1]

            tmp = tm.toc()
            print(f"  Epoch {epoch+1}/{self.args.epochs}({(epoch+1)*100.0/self.args.epochs:4.2f}%) | loss_D = {trainLossD:.3f}/{epochLossD:.3f}, loss_G = {trainLossG:.3f}/{epochLossG:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")
            self.ckp.write_log(f"  Epoch {epoch+1}/{self.args.epochs} | loss_D = {epochLossD:.3f}, loss_G = {epochLossG:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)

        # 保存模型的学习率并画图
        self.ckp.savelearnRate(self)

        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)

        #self.ckp.save()
        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.ckp.write_log(f"#========================= 本次训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ================================",train=True)
        # 关闭日志
        self.ckp.done()
        print(color.higred(f"====================== 关闭训练日志 {self.ckp.log_file.name} ==================================="))

        print(color.higred(f"\n#====================== 训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ==============================\n"))
        return


    def test1(self):


        return

























