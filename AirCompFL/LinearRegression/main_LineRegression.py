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
##          核心代码
##=======================================================================

## num_comm 表示通信次数
for round_idx in range(args.num_comm):
    pass

# print(f"Data volume = {data_valum} (floating point number) ")
# return


# if __name__=="__main__":
#     main()




























































