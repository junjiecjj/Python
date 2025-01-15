#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:25:48 2025

@author: jack
"""


import numpy as np
import datetime
import copy
import torch
import os

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
# from Transmit_1bit import OneBit
from Transmit_Bbit import B_Bit, mess_stastic

from read_data import GetDataSet
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
import models
from config import args_parser
import MetricsLog
# from Channel import  FastFading_scma, FastFading_Mac

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()

args.IID = False              # True, False
args.dataset = "CIFAR10"       #  MNIST,  CIFAR10

datapart = "IID" if args.IID else "nonIID"

cur_lr = args.lr = 0.1
args.num_of_clients = 100
args.active_client = 6
args.case = 'grad'          # "diff", "grad", "signSGD"
# args.diff_case = 'batchs'   # diff:'batchs', 'epoch'
args.optimizer = 'sgd'      # 'sgd', 'adam'
args.local_bs = 128
if args.IID == True:
    args.diff_case = 'batchs'
    if args.dataset == "MNIST":
        args.local_up = 3
    elif args.dataset == "CIFAR10":
        args.local_up = 10
elif args.IID == False:
    args.diff_case = 'epoch'
    if args.dataset == "MNIST":
        args.local_epoch = 1
    elif args.dataset == "CIFAR10":
        args.local_epoch = 4


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

key_grad = []
for name, param in global_model.named_parameters():
    if 'norm' not in name:
        key_grad.append(name)

##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model) )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================
statistics3 = []
# ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    # recorder.addlog(comm_round)
    # cur_lr = args.lr/(1 + 0.001 * comm_round)
    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    # print(f"candidates = {candidates}")
    message_lst = []

    ### signSGD
    if args.case == 'signSGD':
        for name in candidates:
            message = Users[name].local_update_gradient1(copy.deepcopy(global_weight), cur_lr)
            message_lst.append(message)
        else:
            server.aggregate_gradient_erf(message_lst, cur_lr)
    ### grdient
    if args.case == 'grad':
        for name in candidates:
            message = Users[name].local_update_gradient(copy.deepcopy(global_weight), cur_lr)
            message_lst.append(message)
        os.makedirs(args.home + '/FL_1bitJoint/statistics/', exist_ok = True)
        savename = args.home + f'/FL_1bitJoint/statistics/distribution_{args.case}_{args.dataset}_{datapart}.pdf'
        if comm_round in [1, 50, 100 ]:
            statistics3.append(message_lst[-1])
        err = 0
        server.aggregate_gradient_erf(message_lst, cur_lr)
    ## diff
    if args.case == 'diff':
        for name in candidates:
            if args.diff_case == 'batchs':
                message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
            elif args.diff_case == 'epoch':
                message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            message_lst.append(message)
        os.makedirs(args.home + '/FL_1bitJoint/statistics/', exist_ok = True)
        savename = args.home + f'/FL_1bitJoint/statistics/distribution_{args.case}_{args.dataset}_{datapart}.pdf'
        if comm_round in [1, 50, 100 ]:
            statistics3.append(message_lst[-1])
        err = 0
        server.aggregate_diff_erf(message_lst)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    print(f"  [  round = {comm_round+1}, lr = {cur_lr:.6f}, train los = {test_los:.3f}, test acc = {acc:.3f}, ber = {err}]")
    # recorder.assign([acc, test_los, cur_lr, ])
    if comm_round == 101:
        break
print(args)


mess_stastic(statistics3, args, savename, key_grad )













































