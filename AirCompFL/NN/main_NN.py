


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
# import warnings
# warnings.filterwarnings("ignore")


## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
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

args.case = "model"        # "gradient", "diff", "model"
args.channel = 'rician'       # 'erf', rician'
args.model = "mnist_2mlp"   # "mnist_2nn", "mnist_1mlp", "mnist_2mlp", "mnist_cnn"
args.SNR = 20
args.local_up = 5

print(f">>> Info: {args.case}({args.local_up if args.case != 'gradient' else '1'}), channel: {args.channel}(snr: {args.SNR if args.channel != 'erf' else 'none'})")

## seed
set_random_seed(args.seed, )
set_printoption(5)

## Log
recorder = MetricsLog.TraRecorder(3, name = "Train", )

## Get data
local_dt_dict, testloader = GetDataSet(args)

## Model
if args.model.lower() == "mnist_2nn":
    global_model = models.Mnist_2NN().to(args.device)
elif args.model.lower() == "mnist_1mlp":
    global_model = models.Mnist_1MLP().to(args.device)
elif args.model.lower() == "mnist_2mlp":
    global_model = models.Mnist_2MLP().to(args.device)
elif args.model.lower() == "mnist_cnn":
    global_model = models.Mnist_CNN().to(args.device)

global_weight = global_model.state_dict()
args.dimension = np.sum([param.numel() for param in global_weight.values()]); print(f"# of parameters =  {args.dimension}.")

##  Clients group
Users = GenClientsGroup(args, local_dt_dict, global_model )

## Server
server = Server(args, copy.deepcopy(global_model), copy.deepcopy(global_weight), testloader)

##============================= 完成以上准备工作 ================================#
##  选取的 Clients 数量
num_in_comm = int(max(args.num_of_clients * args.cfrac, 1)); print(f"%%%%%%% Number of participated Users {num_in_comm}")
args.num_in_comm = num_in_comm
H = np.sqrt(1/2) * (np.random.randn(args.num_comm, args.num_of_clients) + 1j * np.random.randn(args.num_comm, args.num_of_clients))
# H = np.ones(args.num_comm, args.num_of_clients)
H = np.abs(H)

##=======================================================================
##                               迭代
##=======================================================================
cur_lr = args.lr

### checkpoint
ckp =  checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    if args.lr_decrease: cur_lr = server.set_lr(comm_round, args.lr)
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
    if args.case == "gradient" and args.channel.lower() == 'erf':
        server.aggregate_gradient_erf(message_lst, cur_lr)
    elif args.case == "diff" and args.channel.lower() == 'erf':
        server.aggregate_diff_erf(message_lst)
    elif args.case == "model" and args.channel.lower() == 'erf':
        server.aggregate_model_erf(message_lst)

    ####>>> Rician channel
    noise_var = args.P0 * 10**(-args.SNR/10.0)
    if args.case == "gradient" and args.channel.lower() == 'rician':
        server.aggregate_gradient_rician(message_lst, cur_lr, noise_var, args.P0, h, args.device)
    elif args.case == "diff" and args.channel.lower() == 'rician':
        server.aggregate_diff_rician(message_lst, args.SNR, noise_var, args.P0, h, args.device)
    elif args.case == "model" and args.channel.lower() == 'rician':
        server.aggregate_model_rician(message_lst, args.SNR, noise_var, args.P0, h, args.device)

    ########################### Update & Test ###############################
    global_weight = copy.deepcopy(server.global_weight)
    acc, test_los = server.model_eval(args.device)
    if (comm_round + 1) % 20 == 0:
        print(f"   [{args.case}:{args.local_up if args.case != 'gradient' else ''}, {args.channel}:{args.SNR if args.channel != 'erf' else ''}(dB), ] ---> round = {comm_round+1}, lr = {cur_lr:.3f}, acc = {acc:.3f}, los = {test_los:.3f}")

    recorder.assign([acc, test_los, cur_lr, ])
recorder.save(ckp.savedir, args)


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

























































