


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
from Transmit_1bit import  OneBit_Grad_G
from Transmit_Bbit import B_Bit
import Transmit_SIC
# from Transmit_1bitFlipping import  OneBitNR_flip, OneBitSR_flip

from read_data import GetDataSet
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
import models
from config import args_parser
import MetricsLog
from Channel import channelConfig, Large_rayleigh_fast

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()

args.IID = True             # True, False
args.dataset = "MNIST"      #  MNIST,

datapart = "IID" if args.IID else "nonIID"
# args.save_path = args.home + f'/FL_1bitJoint/{args.dataset}_CNN_{datapart}/'
args.save_path = args.home + '/FL_1bitJoint/test/'

cur_lr = args.lr = 0.01
args.num_of_clients = 100
args.active_client = 6
args.case = 'diff'          # "diff", "grad", "signSGD"
# args.diff_case = 'batchs'   # diff:'batchs', 'epoch'
args.optimizer = 'sgd'      # 'sgd', 'adam'

args.quantize = True       # True, False
if args.quantize == True:
    args.rounding = 'sr'       # 'nr', 'sr',
    args.bitswidth = 1         #  1,  8
    args.G         = 2**8
    args.transmitWay = 'sic'    # 'perfect', 'erf', 'flip', 'joint', 'sic'

    if args.transmitWay.lower() == 'flip':
        args.flip_rate = 0.1
    if args.transmitWay.lower() == 'erf':
        args.flip_rate = 0
    if  args.transmitWay.lower() =='joint' or args.transmitWay.lower() == 'sic':
        args.noisePSD = -120 # dBm
if args.IID == True:
    args.diff_case = 'batchs'
    if args.dataset == "MNIST":
        args.local_up = 3
        args.local_bs = 128
elif args.IID == False:
    args.diff_case = 'epoch'
    if args.dataset == "MNIST":
        args.local_epoch = 1
        args.local_bs = 64

## seed
args.seed = 42
set_random_seed(args.seed) ## args.seed
set_printoption(5)
##>>>>>>>>>>>>>>>>> channel >>>>>>>>>>>>>>>>>>>
BS_locate, users_locate, beta_Au, PL_Au, d_Au = channelConfig(args.K, r = 100, rmin = 0.6)

##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
recorder = MetricsLog.TraRecorder(3, name = "Train", )

local_dt_dict, testloader = GetDataSet(args)
global_model = models.CNNMnist(1, 10, True).to(args.device)

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
### checkpoint
ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    # cur_lr = args.lr/(1 + 0.001 * comm_round)
    candidates = np.random.choice(args.num_of_clients, args.active_client, replace = False)
    # print(f"candidates = {candidates}")
    message_lst = []

    ### diff
    if args.case == 'diff':
        for name in candidates:
            if args.diff_case == 'batchs':
                message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr, args.local_up)
            elif args.diff_case == 'epoch':
                message = Users[name].local_update_diff1(copy.deepcopy(global_weight), cur_lr, args.local_epoch)
            message_lst.append(message)
        if args.transmitWay == 'perfect':
            print(f"  {args.case} -> without quantization")
            err = 0
            server.aggregate_diff_erf(message_lst)
        elif args.transmitWay == 'erf' or args.transmitWay == 'flip':
            if args.bitswidth == 1:
                print(f"  {args.case} -> {args.bitswidth}bit-quant -> {args.rounding} -> {args.transmitWay} ")
                mess_recv, err = OneBit_Grad_G(message_lst, args, rounding = args.rounding, err_rate = args.flip_rate, key_grad = key_grad, G = args.G)
            elif args.bitswidth > 1:
                print(f"  {args.case} -> {args.bitswidth}bit-quant -> {args.rounding} -> {args.transmitWay} ")
                mess_recv, err = B_Bit(message_lst, args, rounding = args.rounding, ber = args.flip_rate, B = args.bitswidth, key_grad = key_grad)
            server.aggregate_diff_erf(mess_recv)
        elif args.transmitWay == 'sic':
            if args.bitswidth == 1:
                print(f"  {args.case} -> {args.bitswidth}bit-quant -> {args.rounding} -> {args.transmitWay} ")
                mess_recv, err = Transmit_SIC.OneBit_SIC(message_lst, args, rounding = args.rounding, key_grad = key_grad, G = args.G)
            elif args.bitswidth > 1:
                print(f"  {args.case} -> {args.bitswidth}bit-quant -> {args.rounding} -> {args.transmitWay} ")
                mess_recv, err = Transmit_SIC.B_Bit_SIC(message_lst, args, rounding = args.rounding, B = args.bitswidth, key_grad = key_grad)
            server.aggregate_diff_erf(mess_recv)

        # if comm_round == 1:
        #     break
    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    print(f"  [  round = {comm_round+1}, lr = {cur_lr:.6f}, train los = {test_los:.3f}, test acc = {acc:.3f}, ber = {err}]")
    recorder.assign([acc, test_los, cur_lr, ])
    # recorder.plot(ckp.savedir, args)
    recorder.save(ckp.savedir, args)
print(args)













































