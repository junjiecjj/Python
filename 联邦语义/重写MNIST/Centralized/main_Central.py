#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:31:28 2025

@author: jack


此代码的功能：
训练语义传输模型, 训练时是在指定的不同的信噪比下训练的, 分类器是预训练的, 在此不训练分类器.

统计在指定压缩率和信噪比下的训练过程的指标(分类准确率, PSNR等), 以及在各个指定压缩率和信噪比下训练完后在测试集上的指标,


其中各个过程的日志都被记录, 包括:
    训练过程每个 epoch 的分类正确率,PSNR 等
    测试过程的在每个压缩率和信噪比下时每个测试信噪比下的分类正确率, PSNR 等
"""


## 系统库
import torch

## My lib
import Utility
from Args import args
from DataLoader import DataGenerator
from Models import LeNet_3, AED_MNIST

from Logs import Accumulator, TraRecorder, TesRecorder

import Tools
###========================================================

# 设置随机数种子
Utility.set_random_seed(42,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

ckp = Utility.checkpoint(args)

testRecoder  = TesRecorder(Len = 3)
tm           = Tools.myTimer()
###========================================================

classifier = LeNet_3().to(args.device)
pretrained_model = f"{args.home}/FL_Sem2026/MNIST_classifier.pt"
classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device, weights_only=True))

loader       = DataGenerator(args, "MNIST")
loader_train = loader.loader_train
loader_test  = loader.loader_test
raw_dim      = 28 * 28

for idx, (comrate, Snr) in enumerate(zip(args.CompRate, args.SNRtrain)):
    print(f"压缩率:{comrate:.2f}, 信噪比:{Snr} dB,  ({idx+1}/{len(args.CompRate)})")
    encoded_dim = int(raw_dim * comrate)
    AutoED = AED_MNIST(encoded_space_dim = encoded_dim, snr = Snr).to(args.device)
    optimizer = torch.optim.Adam(AutoED.parameters(), lr = args.lr, betas = (0.5, 0.999), eps = 1e-08,)  # 使用SGD无法收敛

    Loss = torch.nn.MSELoss(reduction='sum')   #
    trainrecord = TraRecorder(7, name = "Train", compr = comrate, tra_snr = Snr)

    for epoch in range(args.epochs):
        metric = Accumulator(5)
        AutoED.train()
        lr = optimizer.param_groups[0]['lr']
        trainrecord.addlog(epoch)

        print(f"\n    Epoch : {epoch+1}/{args.epochs}, lr = {lr:.3e}, 压缩率:{comrate:.1f}, 信噪比:{Snr}(dB)({idx+1}/{len(args.SNRtrain)})")
        for batch, (X, y) in enumerate(loader_train):
            # X = X.to(args.device)
            X, = Tools.prepare(args.device, args.precision, X)
            X_hat = AutoED(X)
            loss =  Loss(X_hat, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_hat          = classifier(X_hat).cpu().detach()
                X, X_hat       = X.detach().cpu(), X_hat.detach().cpu()
                acc = (y_hat.argmax(axis=1) == y).sum().item()

                X     =  Tools.data_inv_tf_cnn_mnist_batch_3D(X)
                X_hat =  Tools.data_inv_tf_cnn_mnist_batch_3D(X_hat)

                batch_avg_psnr = Tools.PSNR_torch_Batch(X, X_hat , )
                metric.add(loss.item(), batch_avg_psnr, acc, 1, X.size(0))
            # 输出训练状态
            if batch % 100 == 0:
                print(f"    [epoch: {epoch+1:*>5d}/{args.epochs}, batch: {batch+1:*>5d}/{len(loader_train)}]\tLoss: {loss.item()/X.size(0):.4f} \t acc:{acc:.3f} \t batch_avg_psnr: {batch_avg_psnr:.3f}(dB)")

        avg_loss     = metric[0]/metric[4]
        avg_batch    = metric[1]/metric[3]
        accuracy     = metric[2]/metric[4]

        val_batch, val_acc = Tools.validate(args, AutoED, classifier, loader_test)
        trainrecord.assign([lr, avg_loss, avg_batch, accuracy, val_batch, val_acc])
        if epoch % 30 == 0 or (epoch + 1) == args.epochs:
            Tools.R_SNR_epochImgs(args, ckp, AutoED, classifier, loader_test, comrate, Snr, epoch, avg_batch, val_batch, cols = 5, )

        tmp = tm.toc()
        print("    ******************************************************")
        print(f"    loss =  {avg_loss:.3f}, PSNR: {avg_batch:.3f}(dB), acc:{accuracy:.3f} | val psnr: {val_batch:.3f}(dB), acc:{val_acc:.3f} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")
        print("    ******************************************************")

    trainrecord.save(ckp.savedir)
    trainrecord.plot_inonefig(ckp.savedir, metric_str = ['lr', 'train loss', 'batch_PSNR', 'train acc', 'val_batch_PSNR', 'val acc'])

    Tools.R_SNR_valImgs(ckp, args, AutoED, classifier, loader_test, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    Tools.test_R_snr(ckp, testRecoder, args, AutoED, classifier, loader_test, comrate, Snr, SNRlist = args.SNRtest)

print( f"\n#============ 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ==================\n")









































































































































