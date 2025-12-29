#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:27:52 2025

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
from Transmit_Bbit import mess_stastic, ParamRange

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

args.num_of_clients = 40
args.active_client = 40

args.IID = True              # True, False
args.dataset = "MNIST"       #  MNIST,

datapart = "IID" if args.IID else "nonIID"
args.save_path = args.home + f'/FL_DQ/{args.dataset}_{datapart}_range/'
# args.save_path = args.home + '/FL_DQ/test/'
cur_lr = args.lr = 0.01

args.optimizer = 'adam'      # 'sgd', 'adam'

if args.IID == True:
    args.diff_case = 'epoch'
    args.local_epoch = 1
    # args.local_up = 5
    args.local_bs = 128
elif args.IID == False:
    args.diff_case = 'epoch'
    args.local_epoch = 1
    args.local_bs = 64

args.quantize = False             # True, False
if args.quantize == True:
    args.rounding = 'sr'          # 'nr', 'sr',
    args.G        = 2**8

    args.quantize_way = 'fixed'   # 'fixed', 'DQ'
    if args.quantize_way == 'fixed':
        args.bitswidth = 1
    args.transmit_way = 'flip'    # 'erf', 'flip'
    if args.transmit_way.lower() == 'flip':
        args.flip_rate = 0.2513
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
recorder = MetricsLog.TraRecorder(4, name = "TraRecorder", )
minmaxRecod = MetricsLog.TraRecorder(4, name = "minmax", )

local_dt_dict, testloader = GetDataSet(args)
global_model = Models.CNNMnist(1, 10, True).to(args.device)

global_weight = global_model.state_dict()
key_grad = []
for name, param in global_model.named_parameters():
    key_grad.append(name)
key_want = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
##============================= 完成以上准备工作 ================================#
Users = GenClientsGroup(args, local_dt_dict, copy.deepcopy(global_model) )
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)
##=======================================================================
##                              迭代
##=======================================================================
statistics3 = []
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    minmaxRecod.addlog(comm_round)
    cur_lr = args.lr/(1 + 0.001 * comm_round)

    # candidates = np.arange(args.active_client)
    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    candidates.sort()
    if 0 not in candidates:
        candidates = np.append(0, candidates)
    message_lst = []

    for name in candidates:
        if args.diff_case == 'batchs':
            message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
        elif args.diff_case == 'epoch':
            message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            # message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
        message_lst.append(message)
    if comm_round in [1, 50, 100 ]:
        statistics3.append(message_lst[-1])
    print(f"  {args.diff_case} -> error-free")

    err = 0
    server.aggregate_diff_erf(message_lst)
    Delta = ParamRange(message_lst, key_want)

    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    print(f"   [round = {comm_round+1}, lr = {cur_lr:.6f}, train los = {test_los:.3f}, test acc = {acc:.3f}, ber = {err}]")
    recorder.assign([acc, test_los, cur_lr, args.bitswidth])
    recorder.save(ckp.savedir, )
    minmaxRecod.assign(Delta[0, :])
    minmaxRecod.save(ckp.savedir, )

    if (comm_round + 1) % 10 == 0:
        recorder.plot(ckp.savedir, )
        minmaxRecod.plot(ckp.savedir, labels = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight'], name = "MinMaxLog.eps" )

savename = args.save_path + '/distribution_{args.dataset}_{datapart}.pdf'
mess_stastic(statistics3, args, savename, key_grad )
print(args)













































