




# -*- coding: utf-8 -*-
"""
Created on 2023/06/30
@author: Junjie Chen
"""


import argparse
import os
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    # 获取当前系统用户目录
    home = os.path.expanduser('~')
    parser.add_argument('--home', type=str, default = home, help='user home')

    ## 设备相关
    parser.add_argument('--gpu',    type = int, default = True,      help = 'use gpu')
    parser.add_argument('--gpu_idx', type = str, default = 'cuda:0',   help = 'cuda device')
    parser.add_argument('--seed',   type = int, default = 9999,       help = 'random seed')

    # 模型和数据
    parser.add_argument('--D', type=int, default=100, help="dimension of linear regession")   ## 线性回归的维度
    parser.add_argument('--local_ds', type=int, default=512, help="local data size")   ## 本地数据大小
    parser.add_argument('--sigma', type=float, default=0.2, help="noise std")   ##
    parser.add_argument('--iid', type = int, default = 0, help='Default set to IID. Set to 0 for non-IID.') ## 数据是否 IID
    # parser.add_argument('--isBalance', type=int, default = 1,  help = 'numer of the clients') ## 每个客户端是否一样多的数据

    ## 联邦学习相关参数
    parser.add_argument('--local_up', type=int, default = 10, help = "the number of local epochs: E") ## 训练次数(客户端更新次数)
    parser.add_argument('--local_bs', type=int, default = 128, help="local batch size: B") ## local_batchsize 大小
    parser.add_argument('--num_of_clients', type=int,   default = 100, help = 'numer of the clients') ## 客户端的数量
    parser.add_argument('--cfrac', type=float, default = 0.1, help = 'the fraction of clients: C') ## 随机挑选的客户端的数量

    parser.add_argument('--num_comm', type=int, default = 1000, help = 'number of communications') ## num_comm 表示通信次数，此处设置为1k
    parser.add_argument('--case', type=str, default = 'gradient', choices = ('gradient', 'diff', 'model'), help = 'the join comm-learning case') ## 传输的是什么，本地训练多少轮等配置

    ## 通信相关参数
    parser.add_argument('--channel', type=str, default = 'error free', choices = ('erf', 'awgn', 'rician'),) ## 信道类型
    parser.add_argument('--P0', type=float, default = 1, help = "average transmit power"  ) ## 单个信号平均发送功率
    parser.add_argument('--SNR', type=float, default = 10, help = "dB"  ) ## 信噪比

    ## 数据根目录/日志保存目录
    parser.add_argument('--save_path', type = str, default = home + '/AirFL/LinearRegression/',    help = 'file name to save')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('--lr_decrease', type = bool , default = True, help = 'learning rate')

    # args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            print(f"arg = {arg}")
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            print(f"arg = {arg}")
            vars(args)[arg] = False
    # 如果想用GPU且存在GPU, 则用GPU; 否则用CPU;
    args.device = torch.device(args.gpu_idx if torch.cuda.is_available() and args.gpu else "cpu")

    return args

args = args_parser()






















