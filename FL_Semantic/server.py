#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:43:42 2023

@author: jack
"""

import os
import sys
import  copy
import PIL

import  torch

## build myself
import  MetricsLog
import  tools
import numpy as np

# from model import get_model

class Server(object):
    def __init__(self, Args, test_dataloader, model = None, init_params = None):
        self.args         = Args
        self.device       = self.args.device
        self.global_model = model # get_model(self.args.model_name).to(self.device)
        # for name, var in self.global_model.state_dict().items():
            # print(f"{name}: {var.is_leaf}, {var.shape}, {var.requires_grad}, {var.type()}  ")
        self.eval_loader  = test_dataloader
        # 存储上次最新模型的字典
        self.last_pamas   = init_params
        return

    def model_aggregate(self, weight_accumulator, weight = None, cnt = None):
        ## 取平均值，得到本次通信中Server得到的更新后的模型参数
        # num_in_comm = int(max(self.args.num_of_clients * self.args.cfraction, 1))
        # print(f"cnt:   {cnt.values()}")
        for key in weight_accumulator:
            if 'float' in str(weight_accumulator[key].dtype):
                weight_accumulator[key].div_(sum(weight))
            else:
                weight_accumulator[key].copy_(torch.tensor(0))

        ## 传输的是模型差值
        if self.args.transmit_diff:
            ## 先加载上次的模型，再加上这次的更新
            self.global_model.load_state_dict(self.last_pamas, strict=True)
            for key, val in weight_accumulator.items():
                # print(f"{self.global_model.state_dict()[key].dtype}, {val.dtype}")
                self.global_model.state_dict()[key].add_(val)  # val.type(self.global_model.state_dict()[key].dtype)
        else: ## 传输的是模型参数，直接赋值
            # print("传输的是模型参数")
            for key, val in weight_accumulator.items():
                self.global_model.state_dict()[key].copy_(val.clone())

        # for key, params in  self.global_model.state_dict().items():
        #     if params.dtype != torch.float32:
        #         print(f"      Server:  {key}, {params},         param.dtype = {params.dtype} ")

        ## 得到当前最新的全局模型并赋值给存储上次最新模型的字典, 方法1
        global_parameters = {}
        for key, var in self.global_model.state_dict().items():
            global_parameters[key] = var.detach().clone()  ## 必须加.clone()，或者使用copy.deepcopy, 否则会导致global_parameters在每一个客户端训练时被改变
            self.last_pamas[key]   = var.detach().clone()   ## 必须加.clone()，或者使用copy.deepcopy, 否则会导致global_parameters在每一个客户端训练时被改变
        return global_parameters

    def semantic_model_eval(self, classifier):
        self.global_model.eval()
        classifier.eval()
        ## 训练结束之后，我们要通过测试集来验证方法的泛化性，注意:虽然训练时，Server没有得到过任何一条数据，但是联邦学习最终的目的还是要在Server端学习到一个鲁棒的模型，所以在做测试的时候，是在Server端进行的
        ##  加载Server在最后得到的模型参数
        # self.global_model.load_state_dict(global_parameters, strict=True)
        # sum_accu    = 0.0
        # sum_loss    = 0.0
        # examples    = 0
        loss_fn = torch.nn.MSELoss(reduction = self.args.reduction)
        metric = MetricsLog.Accumulator(6)
        # examples = self.eval_loader.dataset.data.shape[0]
        # 载入测试集
        for X, y in self.eval_loader:
            X              = X.to(self.device)      # 选择精度且to(device)
            X_hat          = self.global_model(X)
            # 传输后分类
            y_hat       = classifier(X_hat).detach().cpu()
            mse         = loss_fn(X, X_hat).item()
            # 计算准确率
            X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
            acc            = tools.accuracy(y_hat, y )
            batch_01_psnr  = tools.PSNR_torch(X, X_hat, )
            X              = tools.data_inv_tf_cnn_mnist_batch_3D(X)
            X_hat          = tools.data_inv_tf_cnn_mnist_batch_3D(X_hat)
            batch_avg_psnr = tools.PSNR_torch_Batch(X, X_hat, )
            metric.add(acc, batch_01_psnr, batch_avg_psnr, mse, 1, X.size(0))
        accuracy     = metric[0]/metric[5]
        avg_batch_01 = metric[1]/metric[4]
        avg_batch    = metric[2]/metric[4]
        avg_los      = metric[3]/metric[5]
        return accuracy, avg_batch_01, avg_batch, avg_los

    # 在指定压缩率和信噪比下训练完所有的epoch后, 在测试集上的指标统计
    def R_SNR_testdata(self, ckp, testRecoder, classifier, compr, tasnr, SNRlist = np.arange(-2, 10, 2), ):
        tm = tools.myTimer()
        self.global_model.eval()
        classifier.eval()
        ckp.write_log(f"#=============== 开始在 压缩率:{compr:.1f}, 信噪比:{tasnr}(dB)下测试, 开始时刻: {tm.start_str} ================\n")
        ckp.write_log("  {:>12}  {:>12}  {:>12}  {:>12} ".format("测试信噪比", "acc", "avg_batch_01", "avg_batch" ))
        print( f"    压缩率:{compr:.1f}, 信噪比: {tasnr} (dB), 测试集:")
        # 增加 指定压缩率和信噪比下训练的模型的测试条目
        testRecoder.add_item(compr, tasnr,)
        with torch.no_grad():
            for snr in SNRlist:
                self.global_model.set_snr(snr)
                # 增加条目下的 一个测试信噪比
                testRecoder.add_snr(compr, tasnr, snr)
                metric = MetricsLog.Accumulator(5)
                for batch, (X, label) in enumerate(self.eval_loader):
                    X,             = tools.prepare(self.device, self.args.precision, X )        # 选择精度且to(device)
                    X_hat          = self.global_model(X)
                    # 传输后分类
                    predlabs       = classifier(X_hat).detach().cpu()
                    # 计算准确率
                    X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                    acc            = tools.accuracy(predlabs, label )
                    batch_01_psnr  = tools.PSNR_torch(X, X_hat, )
                    X              = tools.data_inv_tf_cnn_mnist_batch_3D(X)
                    X_hat          = tools.data_inv_tf_cnn_mnist_batch_3D(X_hat)
                    batch_avg_psnr = tools.PSNR_torch_Batch(X, X_hat, )
                    metric.add(acc, batch_01_psnr, batch_avg_psnr,  1, X.size(0))
                accuracy     = metric[0]/metric[4]
                avg_batch_01 = metric[1]/metric[3]
                avg_batch    = metric[2]/metric[3]

                met = torch.tensor([accuracy, avg_batch_01, avg_batch ])
                testRecoder.assign(compr, tasnr, met)
                ckp.write_log(f"[{snr:>10}, {accuracy:>12.3f}, {avg_batch_01:>12.3f}, {avg_batch:>12.3f}],")
                print( f"  {snr:>10}(dB), {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} ")
        testRecoder.save(ckp.testResdir,)
        testRecoder.plot_inonefig1x2(ckp.testResdir, metric_str = ['acc', '0-1_PSNR', 'batch_PSNR', ], tra_compr = compr, tra_snr = tasnr,)
        return testRecoder

    ## 保存指定压缩率和信噪比下训练完后, 画出在所有测试信噪比下的图片传输、恢复、分类示例.
    def R_SNR_plotval(self, test_resultdir, classifier, trainR = 0.1, tra_snr = 2, snrlist = np.arange(-2, 10, 2) ):
        self.global_model.eval()
        classifier.eval()
        savedir = os.path.join(test_resultdir, f"Images_compr={trainR:.1f}_trainSnr={tra_snr}(dB)" )
        os.makedirs(savedir, exist_ok = True)
        rows =  4
        cols = 5
        # 固定的选前几张图片
        idx = np.arange(0, rows*cols, 1)
        labels = self.eval_loader.dataset.tensors[1][idx]
        # 原图
        real_image  = self.eval_loader.dataset.tensors[0][idx] #.numpy()

        # 原图的预处理
        # test_data   = tools.data_tf_cnn_mnist_batch(real_image)
        test_data,  = tools.prepare(self.device, self.args.precision, real_image)
        # 原图预处理后分类
        pred_labs    = classifier(test_data).detach().cpu().argmax(axis = 1)
        raw_dir     = os.path.join(test_resultdir, "raw_image")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir, exist_ok = True)
            for idx, (im, label) in enumerate(zip(real_image, labels)):
                # print(f"1 im.shape = {im.shape}, {im.max()}, {im.min()}, label.shape = {label.shape}")
                im = tools.data_inv_tf_cnn_mnist_batch_2D(im).squeeze()
                # print(f"2 im.shape = {im.shape}, {im.max()}, {im.min()}, label.shape = {label.shape}")
                im = PIL.Image.fromarray(im)
                im.save(os.path.join(raw_dir, f"{idx}_{label}.png"))
            tools.grid_imgsave(raw_dir, real_image, labels, predlabs = pred_labs, dim = (rows, cols), suptitle = "Raw images", basename = "raw_grid_images")

        # 开始遍历测试信噪比
        with torch.no_grad():
            for snr in snrlist:
                subdir = os.path.join(savedir, f"testSNR={snr}(dB)")
                os.makedirs(subdir, exist_ok = True)
                self.global_model.set_snr(snr)
                # 传输
                X_hat = self.global_model(test_data)
                # 传输后分类
                pred_labs = classifier(X_hat).detach().cpu().argmax(axis = 1)
                # 自编码器恢复的图片
                X_hat = X_hat.detach().cpu()
                X_hat = tools.data_inv_tf_cnn_mnist_batch_2D(X_hat)
                for idx, (im, label) in enumerate(zip(X_hat, labels)):
                    im = PIL.Image.fromarray(im )
                    im.save(os.path.join(subdir, f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)_{idx}_{label}.png"))
                a = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(trainR, tra_snr, snr)
                bs = f"R={trainR:.1f}_trainSnr={tra_snr}(dB)_testSnr={snr}(dB)"
                tools.grid_imgsave(subdir, X_hat, labels, predlabs = pred_labs, dim = (rows, cols), suptitle = a, basename = "grid_images_" + bs )
        return



































