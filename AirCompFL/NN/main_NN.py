#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024/08/25

@author: Junjie Chen

空中计算的FL, MNIST NN模型：
    考虑上行，SISO。
    三种信道(无错，AWGN, 瑞丽)，三种传输方式(传输更新后的模型本身、传输模型差值、传输梯度)

"""

import numpy as np
# import matplotlib.pyplot as plt
import datetime
# import torch
import copy
import warnings
warnings.filterwarnings("ignore")


## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from read_data import GetDataSet
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
from models import Mnist_2NN
from config import args_parser
import MetricsLog



now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
#======================== main ==================================
# def run(info = 'gradient', channel = 'rician', snr = "None", local_E = 1):
args = args_parser()
## seed
set_random_seed(args.seed, )
set_printoption(5)

## Log
recorder = MetricsLog.TraRecorder(2, name = "Train", )

## Get data
local_dt_dict, testloader = GetDataSet(args)

## Model
global_model = Mnist_2NN().to(args.device)
global_weight = global_model.state_dict()

##  Clients group
Users = GenClientsGroup(args, local_dt_dict, global_model )

## Server
server = Server(args, copy.deepcopy(global_model), testloader)

##============================= 完成以上准备工作 ================================#
##  选取的 Clients 数量
num_in_comm = int(max(args.num_of_clients * args.cfrac, 1))

H = np.sqrt(1/2) * (np.random.randn(args.num_comm, args.num_of_clients) + 1j * np.random.randn(args.num_comm, args.num_of_clients))
H = np.abs(H)

##=======================================================================
##          迭代
##=======================================================================
print(f"Info = {args.case}, E = {args.local_up}, channel = {args.channel}, snr = {args.SNR}")

lr_init = args.lr

# checkpoint
# ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)

    if args.lr_decrease:
        # lr = args.lr / (0.004*comm_round + 1)
        cur_lr = server.set_lr(comm_round, lr_init)

    if (comm_round + 1) % 100 == 0:
        print(f"   [{args.case}:{args.local_up if args.case != 'gradient' else ''}, {args.channel}:{args.SNR if args.channel != 'erf' else ''}(dB), ] ---> round = {comm_round+1},  ")

    ####################### random choice client ##################
    candidates = np.random.choice(args.num_of_clients, num_in_comm, replace = False)
    h = H[comm_round, candidates]

    ######################## Distribution & Local Update ####################
    message_lst = []
    for name in candidates:
        if args.case == "gradient":
            message = Users[name].local_update_gradient(copy.deepcopy(global_weight), cur_lr)
        elif args.case == "diff":
            message = Users[name].local_update_diff(copy.deepcopy(global_weight), cur_lr)
        elif args.case == "model":
            message = Users[name].local_update_model(copy.deepcopy(global_weight), cur_lr)
        message_lst.append(message)

    ######################## Upload & Aggregation ##########################
    ####>>> error-free

    ###>>> Rice channel

    ########################### 更新回归系数 ###############################
    # theta = copy.deepcopy(server.theta)

    # recorder.assign([abs(gap_t - optim_Fw), lr, ])
# recorder.save(ckp.savedir, args)


    # return












# def main():
#     cases = ["gradient", "diff", "model"]
#     channels = ['erf', 'rician']
#     local_E = [1, 5, 10]
#     SNR = np.arange(-20, 21, 5)
#     # cases = ["model", "gradient", "diff", ]
#     # channels = ['rician', 'awgn', 'erf', ]
#     # local_E = [1, 5, 10]
#     # SNR = np.arange(-20, 20, 5)

#     error_lst = []

#     for info in cases:
#         for channel in channels:
#             if info != "gradient"  and channel != 'erf':
#                 for E in local_E:
#                     for snr in SNR:
#                         try:
#                             run(info = info, channel = channel, snr = snr, local_E = E)
#                         except Exception as e:
#                             error_lst.append([info, E, channel, snr])
#                             print(f"{e}: {info}, {E}, {channel}, {snr} error !!!!!!!!")
#             elif info == 'gradient' and channel != 'erf':
#                 for snr in SNR:
#                     try:
#                         run(info = info, channel = channel, snr = snr)
#                     except Exception as e:
#                         error_lst.append([info, E, channel, snr])
#                         print(f"{e}: {info}, {E}, {channel}, {snr} error !!!!!!!!")
#             elif info != 'gradient' and channel == 'erf':
#                 for E in local_E:
#                     run(info = info, channel = channel, local_E = E)
#             elif info == 'gradient' and channel == 'erf':
#                 run(info = info, channel = channel)
#     return error_lst

# if __name__=="__main__":
# error_lst = main()
# print(error_lst)

























































