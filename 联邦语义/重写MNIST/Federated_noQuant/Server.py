#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 21:04:47 2025

@author: jack
"""


import os, sys
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import PIL

from Logs import Accumulator
import Tools
from Tools import prepare
from Tools import data_tf_cnn_mnist_batch
from Tools import data_inv_tf_cnn_mnist_batch_3D
from Tools import data_inv_tf_cnn_mnist_batch_2D
from Tools import PSNR_torch_Batch
from Tools import myTimer

# use this general fun, images可以是tensor可以是numpy, 可以是(batchsize, 28, 28) 可以是(batchsize, 1/3, 28, 28)
def grid_imgsave(savedir, images, labels,  predlabs = '', dim = (4, 5), suptitle = '', basename = "raw_image"):
    rows = dim[0]
    cols = dim[1]
    if images.shape[0] != rows*cols:
        print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
        raise ValueError("img num and preset is inconsistent")
    figsize = (cols*2 , rows*2 + 1)
    fig, axs = plt.subplots(dim[0], dim[1], figsize = figsize, constrained_layout=True) #  constrained_layout=True

    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            if len(images[cnt].shape) == 2:
                axs[i, j].imshow(images[cnt], cmap = 'Greys', interpolation='none') # Greys   gray
            elif len(images[cnt].shape) == 3:
                if torch.is_tensor(images):
                    axs[i, j].imshow(images[cnt].permute(1,2,0), cmap = 'Greys', interpolation='none') # Greys   gray
                else:
                    axs[i, j].imshow(np.transpose(images[cnt], (1,2,0)), cmap = 'Greys', interpolation='none') # Greys   gray
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([] ) # #不显示y轴刻度值
            font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 18, 'color':'blue', }
            if predlabs != '':
                axs[i, j].set_title( r"$\mathrm{{label}}:{} \rightarrow {}$".format(labels[cnt], predlabs[cnt]),  fontdict = font1, )
            else:
                axs[i, j].set_title("label: {}".format(labels[cnt]),  fontdict = font1, )
            cnt += 1
    if suptitle != '':
        fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 22,   }
        plt.suptitle(suptitle, fontproperties=fontt,)

    out_fig = plt.gcf()
    out_fig.savefig( os.path.join(savedir, f"{basename}.png"),  bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    return


#%% Base station
class BS(object):
    def __init__(self, args, net, global_weight, test_dataloader):
        self.args = args
        self.global_model  = net
        self.testloader   = test_dataloader
        self.global_weight = global_weight
        return

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error-free %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def aggregate_diff_erf(self, mess_lst, ):
        w_avg = copy.deepcopy(mess_lst[0])
        for key in w_avg.keys():
            for i in range(1, len(mess_lst)):
                w_avg[key] += mess_lst[i][key]
            w_avg[key] = torch.div(w_avg[key], len(mess_lst))

        for param in self.global_weight.keys():
            self.global_weight[param] += w_avg[param].type(self.global_weight[param].dtype)

        self.global_model.load_state_dict(self.global_weight, strict=True)

        return copy.deepcopy(self.global_weight)


    ###%% validate on test dataset
    def eval_SemModel(self, args, classifier):
        self.global_model.eval()
        classifier.eval()

        loss_fn = torch.nn.MSELoss(reduction = "sum")
        metric = Accumulator(5)
        with torch.no_grad():
            for X, y in self.testloader:
                # X              = X.to(args.device)
                X,             = prepare(args.device, args.precision, X)
                X_hat          = self.global_model(X)
                # 传输后分类
                y_hat          = classifier(X_hat).detach().cpu()
                mse            = loss_fn(X, X_hat).item()
                # 计算准确率
                X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                acc            = (y_hat.argmax(axis=1) == y).sum().item()

                X              = Tools.data_inv_tf_cnn_mnist_batch_3D(X)
                X_hat          = Tools.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                batch_avg_psnr = Tools.PSNR_torch_Batch(X, X_hat, )
                metric.add(acc,  batch_avg_psnr, mse, 1, X.size(0))
        accuracy     = metric[0]/metric[4]
        avg_batch    = metric[1]/metric[3]
        avg_los      = metric[2]/metric[4]
        return accuracy, avg_batch, avg_los

    ## 保存指定压缩率和信噪比下训练完后, 画出在所有测试信噪比下的 10张 图片传输、恢复、分类示例.
    def R_SNR_valImgs(self, ckp, args, classifier, trainR = 0.1, tra_snr = 2, snrlist = np.arange(-2, 10, 2) ):
        self.global_model.eval()
        classifier.eval()
        savedir = os.path.join( ckp.testResdir, f"Images_compr={trainR:.1f}_trainSnr={tra_snr}(dB)" )
        os.makedirs(savedir, exist_ok = True)
        rows =  4
        cols = 5
        # 固定的选前几张图片
        idx = np.arange( rows*cols )
        y = self.testloader.dataset.targets[idx]
        # 原图
        X  = self.testloader.dataset.data[idx]
        # 原图的预处理
        X_raw   = data_tf_cnn_mnist_batch(X)
        X_raw,  = prepare(args.device, args.precision, X_raw)
        # 原图预处理后分类
        y_raw    = classifier(X_raw).detach().cpu().argmax(axis = 1)
        raw_dir  = os.path.join(ckp.testResdir, "raw_image")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok = True)
            for idx, (im, label) in enumerate(zip(X, y)):
                im = PIL.Image.fromarray(im.numpy())
                im.save(os.path.join(raw_dir, f"{idx}_{label}.png"))
            grid_imgsave(raw_dir, X, y, predlabs = y_raw, dim = (rows, cols), suptitle = "Raw images", basename = "raw_grid_images")

        # 开始遍历测试信噪比
        with torch.no_grad():
            for snr in snrlist:
                subdir = os.path.join(savedir, f"testSNR={snr}(dB)")
                os.makedirs(subdir, exist_ok = True)
                self.global_model.set_snr(snr)
                # 传输
                X_trans = self.global_model(X_raw)
                # 传输后分类
                y_hat = classifier(X_trans).detach().cpu().argmax(axis = 1)
                # 自编码器恢复的图片
                X_trans = X_trans.detach().cpu()
                X_trans = data_inv_tf_cnn_mnist_batch_2D(X_trans)
                for idx, (im, yr, yp) in enumerate(zip(X_trans, y, y_hat)):
                    im = PIL.Image.fromarray(im )
                    im.save(os.path.join(subdir, f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)_{idx}_{yr}_{yp}.png"))
                a = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr, snr)
                bs = f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)"
                grid_imgsave(subdir, X_trans, y, predlabs = y_hat, dim = (rows, cols), suptitle = a, basename = "grid_images_" + bs )
        return

    # 在指定压缩率和信噪比下训练完所有的epoch后, 在测试集上的指标统计
    def test_R_snr(self, ckp, testRecoder, args, classifier, compr, tasnr, SNRlist = np.arange(-2, 10, 2), ):
        tm = myTimer()
        self.global_model.eval()
        classifier.eval()
        ckp.write_log(f"#=============== 开始在 压缩率:{compr:.1f}, 信噪比:{tasnr}(dB)下测试, 开始时刻: {tm.start_str} ================\n")
        ckp.write_log("  {:>12}  {:>12}  {:>12} ".format("测试信噪比", "acc", "avg_batch" ))
        print( f"    压缩率:{compr:.1f}, 信噪比: {tasnr} (dB), 测试集:")
        # 增加 指定压缩率和信噪比下训练的模型的测试条目
        testRecoder.add_item(compr, tasnr,)
        # with torch.no_grad():
        for snr in SNRlist:
            self.global_model.set_snr(snr)
            # 增加条目下的 一个测试信噪比
            testRecoder.add_snr(compr, tasnr, snr)
            metric             = Accumulator(4)
            for batch, (X, y) in enumerate(self.testloader):
                X,             = prepare(args.device, args.precision, X)
                X_hat          = self.global_model(X)
                # 传输后分类
                y_hat          = classifier(X_hat).detach().cpu()
                # 计算准确率
                X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                acc            = (y_hat.argmax(axis=1) == y).sum().item()
                # batch_01_psnr  = PSNR_torch(X, X_hat, )
                X              = data_inv_tf_cnn_mnist_batch_3D(X)
                X_hat          = data_inv_tf_cnn_mnist_batch_3D(X_hat)
                batch_avg_psnr = PSNR_torch_Batch(X, X_hat, )
                metric.add(acc,  batch_avg_psnr, 1, X.size(0))
            acc          = metric[0]/metric[3]
            avg_batch    = metric[1]/metric[2]

            met = torch.tensor([acc, avg_batch ])
            testRecoder.assign(compr, tasnr, met)
            ckp.write_log(f"  {snr:>10}, {acc:>12.3f}, {avg_batch:>12.3f} ")
            print( f"  {snr:>10}(dB), {acc:>12.3f}, {avg_batch:>12.3f} ")
        testRecoder.save(ckp.testResdir,)
        testRecoder.plot_inonefig1x2(ckp.testResdir, metric_str = ['acc', 'batch_PSNR', ], tra_compr = compr, tra_snr = tasnr,)
        return



#%%

























