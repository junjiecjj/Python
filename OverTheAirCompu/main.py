# -*- coding: utf-8 -*-
"""
Created on 2023/06/30
@author: Junjie Chen
"""

# import os
# import sys
# from tqdm import tqdm
import numpy as np
import torch

# 以下是本项目自己编写的库
# checkpoint

import Utility

from clients import ClientsGroup

from server import Server


# 参数
from config import args

import MetricsLog


#==================================================  seed =====================================================
# 设置随机数种子
Utility.set_random_seed(args.seed, deterministic = True, benchmark = True)
Utility.set_printoption(5)

#  加载 checkpoint, 如日志文件, 时间戳, 指标等
ckp = Utility.checkpoint(args)

#=================================================== main =====================================================
def main():
    recorder = MetricsLog.TraRecorder(5, name = "Train", )
    ## 创建 Clients 群
    myClients = ClientsGroup(  args.dir_minst, args )

    ## 创建 server
    server = Server(args,  )

    ##==================================================================================================
    ##                                核心代码
    ##==================================================================================================
    ## num_comm 表示通信次数
    for round_idx in range(args.num_comm):

        recorder.addlog(round_idx)
        print(f"Communicate round : {round_idx + 1} / {args.num_comm} : ")
        ckp.write_log(f"Communicate round : {round_idx + 1} / {args.num_comm} : ", train = True)


        sum_parameters = {}
        cnt = {}
        for name, params in server.global_model.state_dict().items():
            sum_parameters[name] = torch.zeros_like(params)
            cnt[name]            = 0.0

        global_parameters = server.model_aggregate(sum_parameters, cnt)
        ## 训练结束之后，Server端进行测试集来验证方法的泛化性，
        acc, evl_loss = server.model_eval()

        recorder.assign([ ])

        if (round_idx + 1) % 100 == 0:
            recorder.save(ckp.savedir)
    return


if __name__=="__main__":
    main()














































































































































































