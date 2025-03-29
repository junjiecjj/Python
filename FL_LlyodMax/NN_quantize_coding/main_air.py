#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:31:44 2024

@author: jack
"""


import numpy as np
import datetime
import copy
import torch

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Utility import mess_stastic
from Transmit_1bit import OneBit
from Transmit_Bbit import B_Bit
# from Transmit_1bitFlipping import  OneBitNR_flip, OneBitSR_flip

from read_data import GetDataSet
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
import models
from config import args_parser
import MetricsLog
from Channel import  FastFading_scma, FastFading_Mac

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()

args.IID = True              # True, False
args.dataset = "MNIST"     #  MNIST,  CIFAR10
cur_lr = args.lr = 0.01
args.num_of_clients = 100
args.active_client = 6
args.case = 'diff'          # "diff"
args.diff_case = 'batchs'   # diff:'batchs', 'epoch'
args.optimizer = 'sgd'      # 'sgd', 'adam'
args.quantize = False       # True, False

if args.IID == True:
    args.diff_case = 'batchs'
    if args.dataset == "MNIST":
        args.local_up = 3
        args.local_bs = 128
    elif args.dataset == "CIFAR10":
        args.local_up = 15
        args.local_bs = 32
elif args.IID == False:
    args.diff_case = 'epoch'
    if args.dataset == "MNIST":
        args.local_epoch = 1
        args.local_bs = 64
    elif args.dataset == "CIFAR10":
        args.local_epoch = 5
        args.local_bs = 64

args.P0 = 1.0 / args.active_client
args.SNR = 15
## seed
args.seed = 1
set_random_seed(args.seed) ## args.seed
set_printoption(5)

##>>>>>>>>>>>>>>>>> channel >>>>>>>>>>>>>>>>>>>
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

recorder = MetricsLog.TraRecorder(3, name = "Train", )

local_dt_dict, testloader = GetDataSet(args)
if args.dataset.lower() == "mnist":
    global_model = models.CNNMnist(1, 10, True).to(args.device)
else:
    global_model = models.CNNCifar1(3, 10,).to(args.device)
    # global_model = models.myModel().to(args.device)
global_weight = global_model.state_dict()
##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model) )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================
### checkpoint
ckp =  checkpoint(args, now)
H = np.sqrt(1/2) * (np.random.randn(args.num_comm, args.num_of_clients) + 1j * np.random.randn(args.num_comm, args.num_of_clients))
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    # cur_lr = args.lr/(1 + 0.001 * comm_round)
    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    # print(f"candidates = {candidates}")
    message_lst = []

    # channel, only small fading, religh fading, y = Hx+n~CN(0, sigma^2)
    h = H[comm_round, candidates]

    ### diff
    if args.case == 'diff':
        for name in candidates:
            if args.diff_case == 'batchs':
                message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
            elif args.diff_case == 'epoch':
                message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            message_lst.append(message)
    noise_var = 1 * 10**(-args.SNR/10.0)
    # noise_var = 0
    server.aggregate_diff_rician(message_lst, args.SNR, noise_var, 2 * args.P0, h, args.device)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    print(f"  [  round = {comm_round+1}, lr = {cur_lr:.6f}, train los = {test_los:.3f}, test acc = {acc:.3f},  ]")
    recorder.assign([acc, test_los, cur_lr, ])
    # recorder.plot(ckp.savedir, args)
    recorder.save(ckp.savedir, args)
print(args)













































