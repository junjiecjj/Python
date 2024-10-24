


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/08/25

@author: Junjie Chen

"""

import numpy as np
import datetime
import copy
import torch

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Utility import mess_stastic
from Transmission import OneBitTransmit
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

## seed
set_random_seed(args.seed, )
set_printoption(5)

recorder = MetricsLog.TraRecorder(3, name = "Train", )

local_dt_dict, testloader = GetDataSet(args)

# global_model = models.Mnist_2MLP().to(args.device)
global_model = models.CNNMnist(1, 10, True).to(args.device)

global_weight = global_model.state_dict()
D1 = np.sum([param.numel() for param in global_weight.values()])
# print(f"# of parameters =  {args.dimension}.")
D2 = 0
for key, var in global_model.named_parameters():
    D2 += var.numel()
print(f"D1 = {D1}, D2 = {D2}")
Users = GenClientsGroup(args, local_dt_dict, global_model )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)

##============================= 完成以上准备工作 ================================#
num_in_comm = int(max(args.num_of_clients * args.cfrac, 1))
print(f"%%%%%%% Number of participated Users {num_in_comm}")
args.num_in_comm = num_in_comm

##=======================================================================
##                              迭代
##=======================================================================
cur_lr = args.lr = 0.001

### checkpoint
ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)

    candidates = np.random.choice(args.num_of_clients, num_in_comm, replace = False)

    message_lst = []

    ## grdient
    for name in candidates:
        message = Users[name].local_update_gradient1(copy.deepcopy(global_weight), cur_lr)
        message_lst.append(message)
    # if comm_round == 0 or (comm_round + 1) % 50 == 0:
        # mess_stastic(message_lst, D, args, comm_round+1, ckp.savedir)

    # mess_recv = OneBitTransmit(message_lst, args, D1)

    server.aggregate_gradient_erf(message_lst, cur_lr)


    ## diff
    # args.local_up = 2
    # for name in candidates:
    #     message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
    #     # message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, 3)
    #     message_lst.append(message)
    # server.aggregate_diff_erf(message_lst)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    if (comm_round + 1) % 2 == 0:
        print(f"   [  round = {comm_round+1}, lr = {cur_lr:.3f}, acc = {acc:.3f}, los = {test_los:.3f}")
    recorder.assign([acc, test_los, cur_lr, ])
    recorder.plot(ckp.savedir, args)
recorder.save(ckp.savedir, args)














































