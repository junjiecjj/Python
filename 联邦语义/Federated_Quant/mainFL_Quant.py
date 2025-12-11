#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 22:15:38 2025
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
import copy
import numpy as np
# import multiprocessing as mp


## My lib
import Utility
from Args import args
from DataLoader import GetDataSet
from Models import LeNet_3, AED_MNIST
from Transceiver import B_Bit, OneBit_Grad_G
from Logs import Accumulator, TraRecorder, TesRecorder
from Clients import GenClientsGroup
from Server import BS
import Tools
###========================================================
# 设置随机数种子
Utility.set_random_seed(42,  deterministic = True, benchmark = True)
Utility.set_printoption(3)

args.save = args.home + f'/FL_Sem2026/{args.dataset}_{"IID" if args.IID else "noIID"}_{str(args.B) if args.Quantization else ""}{"Quant" if args.Quantization else "noQuant"}_'
ckp = Utility.checkpoint(args)

testRecoder  = TesRecorder(Len = 3)
tm           = Tools.myTimer()
###========================================================

classifier = LeNet_3().to(args.device)
pretrained_model = f"{args.home}/FL_Sem2026/MNIST_classifier.pt"
classifier.load_state_dict(torch.load(pretrained_model, map_location = args.device, weights_only=True))


local_dt_dict, testloader = GetDataSet(args)

raw_dim = 28 * 28
for idx, (comrate, Snr) in enumerate(zip(args.CompRate, args.SNRtrain)):
    trainrecord = TraRecorder(6, name = "Train", compr = comrate, tra_snr = Snr)

    print(f"压缩率:{comrate:.2f}, 信噪比:{Snr} dB,  ({idx+1}/{len(args.CompRate)})")
    encoded_dim   = int(raw_dim * comrate)
    AutoED        = AED_MNIST(encoded_space_dim = encoded_dim, snr = Snr).to(args.device)
    global_weight = AutoED.state_dict()
    param_tol     = np.sum([p.numel() for  p in AutoED.state_dict().values()])
    # param_tol = np.sum([p.numel() for p in AutoED.parameters() if p.requires_grad])
    ## 全部可导参数, 后续只对可导参数量化
    key_grad = []
    for name, param in AutoED.named_parameters():
        key_grad.append(name)

    Users  = GenClientsGroup(args, local_dt_dict, copy.deepcopy(AutoED) )
    server = BS(args, copy.deepcopy(AutoED), copy.deepcopy(global_weight), testloader)

    for comm_round in range(args.epochs):
        trainrecord.addlog(comm_round)
        candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
        lr = args.lr  # /(1 + 0.001 * comm_round)
        metric = Accumulator(2)

        message_lst = []
        for name in candidates:
            message, local_los = Users[name].local_update_diff1(copy.deepcopy(global_weight), lr, args.local_epoch)
            message_lst.append(message)
            metric.add(local_los, 1)

        ##>>>>>>>>>>>>>>>>> quantization >>>>>>>>>>>>>>>>>
        if args.B >= 1:
            mess_recv, err = B_Bit(message_lst, args, rounding = 'sr', ber = 0, B = args.B, key_grad = key_grad)
        elif args.B == 1:
            mess_recv, err = OneBit_Grad_G(message_lst, args, rounding = 'sr', err_rate = 0, key_grad = key_grad, G = args.G)

        ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        global_weight = server.aggregate_diff_erf(mess_recv)

        train_loss     = metric[0]/metric[1]

        run_acc, run_psnr, run_los = server.eval_SemModel(args, classifier )
        trainrecord.assign([lr, train_loss, run_acc, run_psnr, run_los])
        tmp = tm.toc()

        print(f"    round = {comm_round+1}/{args.epochs}, train loss = {train_loss:.3f} | val los: {run_los:.3f}, val acc:{run_acc:.3f}, val psnr: {run_psnr:.3f}(dB) | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")

    trainrecord.save(ckp.savedir)
    trainrecord.plot_inonefig(ckp.savedir, metric_str = ['lr', 'train loss', 'run acc', 'run psnr', 'run los'])

    server.R_SNR_valImgs(ckp, args, classifier, trainR = comrate, tra_snr = Snr, snrlist = args.SNRtest)
    server.test_R_snr(ckp, testRecoder, args, classifier, comrate, Snr, SNRlist = args.SNRtest)

print( f"\n#============ 完毕,开始时刻:{tm.start_str},结束时刻:{tm.now()},用时:{tm.hold()/60.0:.3f}分钟 ==================\n")









































































































































