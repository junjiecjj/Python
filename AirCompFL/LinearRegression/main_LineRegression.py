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

from clients import ClientsGroup

from server import Server

# 参数
from config import args_parser

import MetricsLog


args = args_parser()
#======================== seed ==================================
# 设置随机数种子
Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(5)

# checkpoint
ckp = Utility.checkpoint(args)

#======================== main ==================================
def main():
    recorder = MetricsLog.TraRecorder(1, name = "Train", )

    ## Initial
    theta_true, dataset = Initial(args)



    ## 创建 Clients 群
    myClients = ClientsGroup(args )

    ## 创建 server
    server = Server(args, )

    ##============================= 完成以上准备工作 ================================#
    ##  选取的 Clients 数量
    num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

    ##=======================================================================
    ##          核心代码
    ##=======================================================================

    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):
        pass

    # print(f"Data volume = {data_valum} (floating point number) ")
    return


if __name__=="__main__":
    main()




























































