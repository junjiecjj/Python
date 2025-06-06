#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:11:55 2024

@author: Junjie Chen

空中计算的FL：
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
from Utility import Initial
from Utility import optimal_linear_reg
from Utility import mess_stastic
from clients import GenClientsGroup
from server import Server
from checkpoints import checkpoint
from config import args_parser
import MetricsLog


now = "Stastic_erf"
#======================== main ==================================
# def run():
args = args_parser()
#======================== seed ==================================
# 设置随机数种子
set_random_seed(args.seed, )
set_printoption(5)

recorder = MetricsLog.TraRecorder(2, name = "Train", )

## Initial
theta_true, theta0, X, Y, frac = Initial(args)
theta_optim, optim_Fw = optimal_linear_reg(X, Y, frac)

## 创建 Clients 群
Users = GenClientsGroup(args, X, Y, frac)

## 创建 server
server = Server(args, theta0)

##============================= 完成以上准备工作 ================================#
##  选取的 Clients 数量
num_in_comm = int(max(args.num_of_clients * args.cfrac, 1))

H = np.sqrt(1/2) * (np.random.randn(args.num_comm, args.num_of_clients) + 1j * np.random.randn(args.num_comm, args.num_of_clients))
H = np.abs(H)

##=======================================================================
##          迭代
##=======================================================================
theta = theta0
lr0 = args.lr
args.case = "model"   # "gradient", "diff", "model"
args.channel = 'erf'       # 'erf', 'awgn', rician'
args.SNR = 1
args.local_up = 1
print(f"Info = {args.case}, E = {args.local_up}, channel = {args.channel}, snr = {args.SNR}")

### checkpoint
ckp = checkpoint(args, now)
for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    lr = server.set_learning_rate(comm_round, lr0, args.lr_decrease)

    ################# 根据当前的回归系数获取 F(w^(t)) - F(w^*),即 optimal gap
    gap_t = np.sum([frac[name] * user.local_loss(theta) for name, user in Users.items()])
    if (comm_round + 1) % 100 == 0:
        print(f"   [{args.case}:{args.local_up if args.case != 'gradient' else ''}, {args.channel}:{args.SNR if args.channel != 'erf' else ''}(dB), ] -----> round = {comm_round+1}, gap = {gap_t:.5f}, lr = {lr}")

    ####################### random choice client ##################
    clients_idx = np.random.choice(args.num_of_clients, num_in_comm, replace = False)
    candidates = ['client{}'.format(int(i)) for i in clients_idx]

    h = H[comm_round, clients_idx]
    ####################### Distribution & Local Update #################
    message_lst = []
    for name in candidates:
        if args.case == "gradient":
            message = Users[name].local_gradient(theta, args.local_bs)
        elif args.case == "diff":
            message = Users[name].model_diff(theta, args.local_up, args.local_bs, lr)
        elif args.case == "model":
            message = Users[name].updated_model(theta, args.local_up, args.local_bs, lr)
        message_lst.append(message)

    if (comm_round + 1) % 100 == 0:
        mess_stastic(args, comm_round+1, ckp.savedir, message_lst)

    ######################## Upload & Aggregation ##########################
    ####>>> error-free
    if args.case == "gradient" and args.channel.lower() == 'erf':
        server.aggregate_erf_gradient(message_lst, lr)
    elif args.case == "diff" and args.channel.lower() == 'erf':
        server.aggregate_erf_diff(message_lst)
    elif args.case == "model" and args.channel.lower() == 'erf':
        server.aggregate_erf_model(message_lst)

    ###>>> AWGN channel
    # print(f"{args.SNR}")
    if args.channel != "erf":
        noise_var = args.P0 * 10**(-(args.SNR/10.0))
    if args.case == "gradient" and args.channel.lower() == 'awgn':
        server.aggregate_awgn_gradient(message_lst, lr, noise_var, args.P0)
    elif args.case == "diff" and args.channel.lower() == 'awgn':
        server.aggregate_awgn_diff(message_lst, noise_var, args.P0)
    elif args.case == "model" and args.channel.lower() == 'awgn':
        server.aggregate_awgn_model(message_lst, noise_var, args.P0)

    ###>>> Rice channel
    if args.case == "gradient" and args.channel.lower() == 'rician':
        server.aggregate_rician_gradient(message_lst, lr, noise_var, args.P0, h)
    elif args.case == "diff" and args.channel.lower() == 'rician':
        server.aggregate_rician_diff(message_lst, noise_var, args.P0, h)
    elif args.case == "model" and args.channel.lower() == 'rician':
        server.aggregate_rician_model(message_lst, noise_var, args.P0, h)

    ########################### 更新回归系数 ###############################
    theta = copy.deepcopy(server.theta)

    recorder.assign([abs(gap_t - optim_Fw), lr, ])
# recorder.save(ckp.savedir, args)


# return














































