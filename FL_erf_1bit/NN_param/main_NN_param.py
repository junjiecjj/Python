


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/08/25

@author: Junjie Chen

MNIST
    IID: diff, 1epoch, 量化(256)和不量化, seed = 1, lr = 0.01, bs128, 100(10)
    IID: diff, 2 local_up, 量化(256)和不量化, seed = 1, lr = 0.01, bs128, 100(10)
    IID: grad, 不量化, seed = 9999, lr = 0.01, bs128,  100(10)
    IID: grad, 1bit量化, norm_fact = 1, seed = 9999, lr = 0.01, bs128, 不是SignSGD

    non-IID:  diff, 2epoch, 不量化, seed = 1, lr = 0.01, bs128, 100(10)
    non-IID:  diff, 4batchs, 不量化, seed = 1, lr = 0.01, bs128, 100(10)
    non-IID:  grad, 不量化, seed = 1, lr = 0.01, bs128, 100(10)

"""

import numpy as np
import datetime
import copy
import torch

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Utility import mess_stastic
from Transmission import OneBitNR, OneBitSR
from read_data import GetDataSet
from clients import GenClientsGroup

from server import Server
from checkpoints import checkpoint
import models
from config import args_parser
import MetricsLog

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()

recorder = MetricsLog.TraRecorder(3, name = "Train", )

args.IID = True
args.dataset = "CIFAR10"
cur_lr = args.lr = 0.01
args.case = 'diff'              # "grad", "diff"
args.diff_case = 'epoch'       # diff:'batchs', 'epoch'
args.optimizer = 'sgd'          # 'sgd', 'adam'
args.quantize = True            # True, False
args.quantway = 'nr'            # 'nr',  'sr'
args.local_bs = 128
args.local_up = 4
args.local_epoch = 5
args.norm_fact = 256

args.num_of_clients = 50
args.cfrac = 10 / args.num_of_clients

## seed
args.seed = 1
set_random_seed(args.seed) ## args.seed
set_printoption(5)

local_dt_dict, testloader = GetDataSet(args)
# global_model = models.Mnist_1MLP().to(args.device)
# global_model = models.CNNMnist(1, 10, True).to(args.device)
global_model = models.CNNCifar1(3, 10,).to(args.device)

global_weight = global_model.state_dict()
D1 = np.sum([param.numel() for param in global_weight.values()])
D2 = np.sum([var.numel() for key, var in global_model.named_parameters()])
print(f"D1 = {D1}, D2 = {D2}")

num_in_comm = int(max(args.num_of_clients * args.cfrac, 1))
print(f"%%%%%%% Number of participated Users {num_in_comm}")
args.num_in_comm = num_in_comm

##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model))
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================
### checkpoint
ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    cur_lr = args.lr/(1 + 0.001 * comm_round)
    recorder.addlog(comm_round)
    candidates = np.random.choice(args.num_of_clients, num_in_comm, replace = False)
    # print(f"candidates = {candidates}")
    message_lst = []

    ## grdient
    if args.case == 'grad':
        for name in candidates:
            message = Users[name].local_update_gradient1(copy.deepcopy(global_weight), cur_lr)
            message_lst.append(message)
        if args.quantize == True: # 此处的传输梯度然后一比特量化是不对的，并不是SGD
            if args.quantway == 'nr':
                mess_recv = OneBitNR(message_lst, args, args.norm_fact)
            elif args.quantway == 'sr':
                mess_recv = OneBitSR(message_lst, args)
            # server.aggregate_gradient_erf_sign(mess_recv, cur_lr)
            server.aggregate_gradient_erf(mess_recv, cur_lr)
        else:
            server.aggregate_gradient_erf(message_lst, cur_lr)
    ### diff
    if args.case == 'diff':
        for name in candidates:
            if args.diff_case == 'batchs':
                message = Users[name].local_update_diff_mini_batch(copy.deepcopy(global_weight), cur_lr, args.local_up)
            elif args.diff_case == 'epoch':
                # print( "diff-epoch")
                message = Users[name].local_update_diff_epoch(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            message_lst.append(message)
        if args.quantize == True:
            if args.quantway == 'nr':
                # print( "args.quantway == 'nr'")
                mess_recv = OneBitNR(message_lst, args, args.norm_fact)
            elif args.quantway == 'sr':
                mess_recv = OneBitSR(message_lst, args)
            server.aggregate_diff_erf(mess_recv)
        else:
            server.aggregate_diff_erf(message_lst)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    if (comm_round + 1) % 2 == 0:
        print(f"   [  round = {comm_round+1}, lr = {cur_lr:.3f}, train los = {test_los:.3f}, test acc = {acc:.3f} ]")
    recorder.assign([acc, test_los, cur_lr, ])
    recorder.plot(ckp.savedir, args)
recorder.save(ckp.savedir, args)














































