

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
    parser.add_argument('--gpu_idx', type = str, default = 'cuda:0', help = 'cuda device')
    parser.add_argument('--seed',   type = int, default = 9999,      help = 'random seed')

    # 模型和数据
    parser.add_argument('--dataset', type=str, default = 'MNIST', help = "name of dataset")      ## 所用的数据集
    parser.add_argument('--dir_data', type = str, default = home+'/AirFL/Dataset', help = 'dataset directory')
    parser.add_argument('--IID', type = bool, default = False, help = 'Default set to IID. Set to 0 for non-IID.') ## 数据是否 IID

    ## 联邦学习相关参数
    parser.add_argument('--local_up', type=int, default = 5, help = "the number of local epochs: E") ## 训练次数(客户端更新次数)
    parser.add_argument('--local_bs', type=int, default = 128, help = "local batch size: B")         ## local_batchsize 大小
    parser.add_argument('--test_bs', type=int, default = 128, help = 'test batch size')              ## test_batchsize 大小
    parser.add_argument('--num_of_clients', type=int, default = 100, help = 'numer of the clients')  ## 客户端的数量
    parser.add_argument('--cfrac', type=float, default = 0.1, help = 'the fraction of clients: C')     ## 随机挑选的客户端的数量
    parser.add_argument('--num_comm', type=int, default = 1000, help = 'number of communications')   ## num_comm 表示通信次数，此处设置为1k
    parser.add_argument('--case', type=str, default = 'gradient', choices = ('gradient', 'diff', 'model'), help = 'the join comm-learning case')

    ## 数据根目录/日志保存目录
    parser.add_argument('--save_path', type = str, default = home + '/AirFL/NN/', help = 'file name to save')

    ### 优化器
    parser.add_argument('--optimizer', type = str, default = 'SGD', choices = ('SGD', 'ADAM', 'RMSprop'), help = 'optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--lr', type = float, default = 0.04, help = 'learning rate')
    parser.add_argument('--lr_decrease', type = bool , default = False, help = 'learning rate diminishing')

    ## 通信相关参数

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






















