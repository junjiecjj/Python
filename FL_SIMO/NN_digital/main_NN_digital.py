


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/08/25

@author: Junjie Chen


(1) IID, diff本地5轮，lr=0.01, SGD， 1bit-NR, U=100, BS =128, 接收方/2**8;
(2) IID, grad，lr=0.01, SGD， 1bit-NR, U=100, BS = 64, 接收方 / 1;


non-IID:
    (1) non-IID, diff本地5轮，lr = 0.01, SGD，no-quantize, U = 100, BS = 128;
        non-IID, diff本地5轮，lr = 0.01, SGD，no-quantize, U = 100, BS = 64;
        non-IID, diff本地5轮，lr = 0.01, SGD，1bit-quantize, U = 100, BS = 64, 接收方/2**8;
    (2) non-IID, grad，lr = 0.1, SGD，no-quantize, U = 200, 10,  BS = 128;
        non-IID, grad，lr = 0.1, SGD，no-quantize, U = 100, 10,  BS = 128;

"""

import numpy as np
import datetime
import copy
import torch

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Utility import mess_stastic
from Transmit_1bitERF import OneBitNR
from Transmit_SIMO import  OneBit_MMSE, OneBit_LDPC_MMSE
from Transmit_NOMA_SINR import OneBit_SINR, OneBit_LPDC_SINR
from Transmit_NOMA_SNRzf import OneBit_SNRzf, OneBit_LDPC_SNRzf
from Transmit_NOMA_NormZf import OneBit_NormZF, OneBit_LDPC_NormZF
from read_data import GetDataSet
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
import models
from config import args_parser
import MetricsLog

from mimo_channel import MIMO_Channel, channelConfig, Generate_hd

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()

args.IID = True         # True, False
args.dataset = "MNIST"  #  MNIST,  CIFAR10
cur_lr = args.lr = 0.01
args.num_of_clients = 10
args.active_client = 4
args.case = 'grad'        # "grad", "diff"
args.diff_case = 'epoch'  # diff:'batchs', 'epoch'
args.optimizer = 'sgd'    # 'sgd', 'adam'
args.quantize = False     # True, False
args.quantway = 'ldpc'    # 'nr',  'mimo', 'ldpc'
args.local_bs = 128
args.local_up = 1
args.local_epoch = 5
args.snr_dB = None
args.norm_fact = 2**8

## seed
args.seed = 1
set_random_seed(args.seed) ## args.seed
set_printoption(5)

##>>>>>>>>>>>>>>>>> channel
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(args)
args.sigmaK2 = -50                        # dBm
sigmaK2 = 10**(args.sigmaK2/10.0)/1000    # 噪声功率

#<<<<<<<<<<<<<<<<<<<<<<<<<<

recorder = MetricsLog.TraRecorder(3, name = "Train", )

local_dt_dict, testloader = GetDataSet(args)
if args.dataset.lower() == "MNIST":
    # global_model = models.Mnist_1MLP().to(args.device)
    global_model = models.CNNMnist(1, 10, True).to(args.device)
else:
    global_model = models.CNNCifar1(3, 10,).to(args.device)
global_weight = global_model.state_dict()
##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model) )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================
### checkpoint
ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)

    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    # print(f"candidates = {candidates}")
    message_lst = []

    ##  channel, only small fading, religh fading, y = Hx+n~CN(0, sigma^2)
    # channel = MIMO_Channel(Nr = 16, Nt = args.num_of_clients, )
    # channel.circular_gaussian()
    # h = channel.H[:, candidates]

    ## channel, large scale + small scale fading, in this case, H has been normalized by sigmaK2, so y = Hx + n~CN(0,1)
    H = Generate_hd(args.Nr, args.num_of_clients, BS_locate, users_locate, beta_Au, PL_Au, sigma2 = sigmaK2)
    h = H[:, candidates]

    ## grdient
    if args.case == 'grad':
        for name in candidates:
            message = Users[name].local_update_gradient1(copy.deepcopy(global_weight), cur_lr)
            message_lst.append(message)
        if args.quantize == True:
            if args.quantway == 'nr':
                print(f"  {args.case} -> quantize -> NR -> {args.norm_fact}")
                mess_recv, err = OneBitNR(message_lst, args, normfactor = args.norm_fact)
            elif args.quantway == 'mimo':
                print(f"  {args.case} -> quantize -> MIMO -> {args.norm_fact}")
                mess_recv, err =  OneBit_NormZF(message_lst, args, copy.deepcopy(h), snr_dB = args.snr_dB, normfactor = args.norm_fact)
            elif args.quantway == 'ldpc':
                print(f"  {args.case} -> quantize -> LDPC -> {args.norm_fact}")
                mess_recv, err =  OneBit_LDPC_NormZF(message_lst, args, copy.deepcopy(h), snr_dB = args.snr_dB, normfactor = args.norm_fact)
            server.aggregate_gradient_erf(mess_recv, cur_lr)
        else:
            print(f"  {args.case} -> without quantization")
            err = 0
            server.aggregate_gradient_erf(message_lst, cur_lr)
    ### diff
    if args.case == 'diff':
        for name in candidates:
            if args.diff_case == 'batchs':
                message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
            elif args.diff_case == 'epoch':
                message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            message_lst.append(message)
        if args.quantize == True:
            if args.quantway == 'nr':
                print(f"  {args.case} -> quantize -> NR -> {args.norm_fact}")
                mess_recv, err = OneBitNR(message_lst, args, normfactor = args.norm_fact)
            elif args.quantway == 'mimo':
                print(f"  {args.case} -> quantize -> MIMO -> {args.norm_fact}")
                mess_recv, err  =  OneBit_NormZF(message_lst, args, copy.deepcopy(h), snr_dB = args.snr_dB, normfactor = args.norm_fact)
            elif args.quantway == 'ldpc':
                print(f"  {args.case} -> quantize -> LDPC -> {args.norm_fact}")
                mess_recv, err  =  OneBit_LDPC_NormZF(message_lst, args, copy.deepcopy(h), snr_dB = args.snr_dB, normfactor = args.norm_fact)
            server.aggregate_diff_erf(mess_recv)
        else:
            print(f"  {args.case} -> without quantization")
            err = 0
            server.aggregate_diff_erf(message_lst)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    # if (comm_round + 1) % 2 == 0:
    print(f"  [  round = {comm_round+1}, lr = {cur_lr:.3f}, train los = {test_los:.3f}, test acc = {acc:.3f}, ber = {err}]")
    recorder.assign([acc, test_los, cur_lr, ])
    # recorder.plot(ckp.savedir, args)
    # if (comm_round + 1) % 20 == 0:
    recorder.save(ckp.savedir, args)
print(args)













































