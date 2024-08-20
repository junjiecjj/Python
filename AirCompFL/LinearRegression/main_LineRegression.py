#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:11:55 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import copy

## 以下是本项目自己编写的库

import Utility
from Utility import Initial
from Utility import optimal_linear_reg

from clients import GenClientsGroup

from server import Server

from checkpoints import checkpoint
# 参数
from config import args_parser

import MetricsLog


args = args_parser()
#======================== seed ==================================
# 设置随机数种子
Utility.set_random_seed(args.seed, )
Utility.set_printoption(5)

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

##=======================================================================
##          迭代
##=======================================================================
theta = theta0
lr = args.lr
args.case = "updated model"
print(f"args.case = {args.case}")

for comm_round in range(args.num_comm):
    recorder.addlog(comm_round)
    if args.lr_decrease:
        lr = lr/(0.004*comm_round + 1)

    gap_t = np.sum([frac[name] * user.local_loss(theta) for name, user in Users.items()])
    print(f"  round = {comm_round}, gap = {gap_t}")
    ## 从 K 个客户端随机选取 k 个
    candidates = ['client{}'.format(int(i)) for i in np.random.choice(args.num_of_clients, num_in_comm, replace = False)]


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

    ####>>> AGWN channel
    # if args.case == "gradient" and args.channel.lower() == 'agwn':
    #     theta = server.awgn_aggregate_local_gradient(message_lst)
    # elif args.case == "model diff" and args.channel.lower() == 'agwn':
    #     theta = server.awgn_aggregate_model_diff(message_lst)
    # elif args.case == "updated model" and args.channel.lower() == 'agwn':
    #     theta = server.awgn_aggregate_updated_model(message_lst)

    ####>>> Rice channel
    # if args.case == "gradient" and args.channel.lower() == 'rician':
    #     theta = server.rician_aggregate_local_gradient(message_lst)
    # elif args.case == "model diff" and args.channel.lower() == 'rician':
    #     theta = server.rician_aggregate_model_diff(message_lst)
    # elif args.case == "updated model" and args.channel.lower() == 'rician':
    #     theta = server.rician_aggregate_updated_model(message_lst)

    theta = copy.deepcopy(server.theta)



# print(f"Data volume = {data_valum} (floating point number) ")
# return


# if __name__=="__main__":
#     main()




























































