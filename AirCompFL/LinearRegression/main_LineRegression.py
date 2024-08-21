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

# import torch
import copy

## 以下是本项目自己编写的库
from Utility import set_random_seed, set_printoption
from Utility import Initial
from Utility import optimal_linear_reg

from clients import GenClientsGroup

from server import Server

from checkpoints import checkpoint

from config import args_parser

import MetricsLog


args = args_parser()
#======================== seed ==================================
# 设置随机数种子
set_random_seed(args.seed, )
set_printoption(5)

# checkpoint
# ckp =  checkpoint(args)

#======================== main ==================================
# def main():
recorder = MetricsLog.TraRecorder(1, name = "Train", )

## Initial
theta_true, theta0, X, Y, frac = Initial(args)
theta_optim, optim_Fw = optimal_linear_reg(X, Y, frac)

## 创建 Clients 群
Users = GenClientsGroup(args, X, Y, theta0, frac)

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
lr = args.lr
args.case = "updated model"   # "gradient", "model diff", "updated model"
args.channel = 'rician'       # 'error free', 'awgn', rician'
print(f"args.case = {args.case}, channel = {args.channel}")

for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)

    if args.lr_decrease:
        lr = lr/(0.004*comm_round + 1)
    ## 根据当前的回归系数获取 F(w^(t)) - F(w^*),即 optimal gap
    gap_t = np.sum([frac[name] * user.local_loss(theta) for name, user in Users.items()])
    print(f"  round = {comm_round}, gap = {gap_t}")
    ## 从 K 个客户端随机选取 k 个
    clients_idx = np.random.choice(args.num_of_clients, num_in_comm, replace = False)
    candidates = ['client{}'.format(int(i)) for i in clients_idx]

    h = H[comm_round, clients_idx]

    message_lst = []
    for name in candidates:
        if args.case == "gradient":
            message = Users[name].local_gradient(theta, args.local_bs)
        elif args.case == "model diff":
            message = Users[name].model_diff(theta, args.local_up, args.local_bs, lr)
        elif args.case == "updated model":
            message = Users[name].updated_model(theta, args.local_up, args.local_bs, lr)
        message_lst.append(message)
    ####>>> error-free
    if args.case == "gradient" and args.channel.lower() == 'error free':
        server.erf_aggregate_local_gradient(message_lst, lr)
    elif args.case == "model diff" and args.channel.lower() == 'error free':
        server.erf_aggregate_model_diff(message_lst)
    elif args.case == "updated model" and args.channel.lower() == 'error free':
        server.erf_aggregate_updated_model(message_lst)

    ###>>> AWGN channel
    if args.case == "gradient" and args.channel.lower() == 'awgn':
        server.awgn_aggregate_local_gradient(message_lst, lr, args.SNR)
    elif args.case == "model diff" and args.channel.lower() == 'awgn':
        server.awgn_aggregate_model_diff(message_lst, args.SNR)
    elif args.case == "updated model" and args.channel.lower() == 'awgn':
        server.awgn_aggregate_updated_model(message_lst, args.SNR)

    ###>>> Rice channel
    if args.case == "gradient" and args.channel.lower() == 'rician':
        server.rician_aggregate_local_gradient(message_lst, lr, args.SNR, h)
    elif args.case == "model diff" and args.channel.lower() == 'rician':
        server.rician_aggregate_model_diff(message_lst, args.SNR, h)
    elif args.case == "updated model" and args.channel.lower() == 'rician':
        server.rician_aggregate_updated_model(message_lst, args.SNR, h)

    ## 更新回归系数
    theta = copy.deepcopy(server.theta)



# print(f"Data volume = {data_valum} (floating point number) ")
# return


# if __name__=="__main__":
#     main()




























































