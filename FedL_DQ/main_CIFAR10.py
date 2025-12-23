#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:04:37 2025

@author: jack
"""

import numpy as np
import datetime
import copy
# import torch

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Transmit_1bit import OneBit_Grad_G
from Transmit_Bbit import B_Bit

from DataGenarator import GetDataSet
from Clients import GenClientsGroup
from Server import Server
from checkpoints import checkpoint
import Models
from Options import args_parser
import MetricsLog

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================

args = args_parser()

args.IID = True                # True, False
args.dataset = "CIFAR10"       #  CIFAR10,

datapart = "IID" if args.IID else "nonIID"
args.save_path = args.home + f'/FL_DQ/{args.dataset}_{datapart}/'

cur_lr = args.lr = 0.01

args.optimizer = 'adam'      # 'sgd', 'adam'

if args.IID == True:
    args.diff_case = 'epoch'
    args.local_epoch = 2
    args.local_bs = 64
elif args.IID == False:
    args.diff_case = 'epoch'
    args.local_epoch = 1
    args.local_bs = 64

args.quantize = True       # True, False
if args.quantize == True:
    args.rounding = 'sr'       # 'nr', 'sr',
    args.G         = 2**6

    args.quantize_way = 'fixed'
    if args.quantize_way == 'fixed':
        args.bitswidth = 4
    args.transmit_way = 'flip'     # 'erf', 'flip'
    if args.transmit_way.lower() == 'flip':
        args.flip_rate = 0.03898
    if args.transmit_way.lower() == 'erf':
        args.flip_rate = 0
else:
    args.bitswidth = 32

## seed
args.seed = 42
set_random_seed(args.seed) ## args.seed
set_printoption(5)

### checkpoint
ckp =  checkpoint(args, now)

## Log
recorder = MetricsLog.TraRecorder(4, name = "Train", )

local_dt_dict, testloader = GetDataSet(args)
global_model = Models.resnet20().to(args.device)


global_weight = global_model.state_dict()
key_grad = []
for name, param in global_model.named_parameters():
    # if "norm" not in name:
    key_grad.append(name)

##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model) )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================

for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    cur_lr = args.lr/(1 + 0.001 * comm_round)
    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    message_lst = []

    for name in candidates:
        if args.diff_case == 'batchs':
            message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
        elif args.diff_case == 'epoch':
            message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)

        message_lst.append(message)
    if args.quantize == True:
        print(f"{args.diff_case} -> {str(args.bitswidth) + "bit-quant" if args.quantize_way == 'fixed' else 'DQ'} -> {args.rounding} -> {'flip'+str(args.flip_rate) if args.transmit_way == 'flip' else 'erf'}")
        if args.bitswidth == 1:
            mess_recv, err = OneBit_Grad_G(message_lst, args, rounding = args.rounding, ber = args.flip_rate, key_grad = key_grad, G = args.G)
        elif args.bitswidth > 1:
            mess_recv, err = B_Bit(message_lst, args, rounding = args.rounding, ber = args.flip_rate, B = args.bitswidth, key_grad = key_grad)
        server.aggregate_diff_erf(mess_recv)
    else:
        print(f"  {args.diff_case} -> error-free")
        err = 0
        server.aggregate_diff_erf(message_lst)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    print(f"   [round = {comm_round+1}, lr = {cur_lr:.6f}, train los = {test_los:.3f}, test acc = {acc:.3f}, ber = {err}]")
    recorder.assign([acc, test_los, cur_lr, args.bitswidth])
    recorder.save(ckp.savedir, )
    if (comm_round + 1) % 10 == 0:
        recorder.plot(ckp.savedir, )

print(args)













































