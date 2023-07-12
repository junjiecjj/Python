

# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""

import argparse
import socket, getpass , os
import numpy as np

# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')


parser = argparse.ArgumentParser( description="FedAvg with Communication")
parser.add_argument('--host_name', type=str, default = host_name, help='host name')
parser.add_argument('--user_name', type=str, default = user_name, help='user name')
parser.add_argument('--user_home', type=str, default = user_home, help='user home')

# model  specifications
parser.add_argument('--modelUse',        type = str, default = 'FedAvg',   help='model name')

# 设备相关
parser.add_argument('--cpu',             type = int, default = False,      help = 'use cpu only')
parser.add_argument('--device',          type = str, default = 'cuda:0',   help = 'cuda device')
parser.add_argument('--seed',            type = int, default = 1,          help = 'random seed')

##=============================================================================================================================================
##==============================================  联邦学习相关参数 ======================================================================
##=============================================================================================================================================

## 客户端的数量
parser.add_argument('--num_of_clients',  type=int,   default = 100,           help = 'numer of the clients')
## 随机挑选的客户端的数量
parser.add_argument('--cfraction',       type=float, default = 0.1,           help = 'C fraction, 0 means 1 client, 1 means total clients')
## 训练次数(客户端更新次数)
parser.add_argument('--loc_epochs',      type=int,   default = 10,            help = 'local train epoch')
## local_batchsize 大小
parser.add_argument('--local_batchsize', type=int,   default = 128,           help = 'local train batch size')
## test_batchsize 大小
parser.add_argument('--test_batchsize',  type=int,   default = 128,           help = 'test batch size')
## 模型名称
parser.add_argument('--model_name',      type=str,   default = "mnist_cnn",   help = 'the model to train')
## 所用的数据集
parser.add_argument("--dataset",         type=str,   default = "mnist",       help = "需要训练的数据集")
## 模型验证频率（通信频率）
parser.add_argument("--val_freq",        type=int,   default = 5,             help = "model validation frequency(of communications)")
parser.add_argument('--save_freq',       type=int,   default = 20,            help = "global model save frequency(of communication)")
## num_comm 表示通信次数，此处设置为1k
parser.add_argument('--num_comm',        type=int,   default = 300,           help = 'number of communications')
## 数据是否 IID
parser.add_argument('--isIID',             type=int,   default = 0 ,          help = 'the way to allocate data to clients')
## 传输的是模型参数还是模型更新
parser.add_argument('--transmitted_diff',  type=int,   default = 1,           help = 'the way to allocate data to clients')
##==============================================  差分隐私 ======================================================================
## 是否使用 Local DP
parser.add_argument('--LDP',                type=int,     default = 1 ,            help = '是否使用local 差分隐私')
## 是否使用 Client DP
parser.add_argument('--CDP',                type=int,     default = 0 ,            help = '是否使用 client 差分隐私')

parser.add_argument('--eps',               type=float,   default = 8,             help = '隐私预算')
parser.add_argument('--clip',              type=float,   default = 10 ,          help = '梯度剪切')
parser.add_argument('--delta',             type=float,   default = 0.0001 ,       help = '超出隐私预算的概率')
parser.add_argument('--q',                 type=float,   default = 0.1,         help = '每epoch数据百分比')
parser.add_argument('--sigma',             type=float,   default = 1,             help = '高斯噪声的标准差的一部分,总的为 clip*sigma')



##==============================================  模型稀疏：随机掩码 ==============================================================
## 是否使用模型稀疏
parser.add_argument('--Random_Mask',       type=float,   default = 0,             help = '是否使用随机掩码')
parser.add_argument('--prop',              type=float,   default = 0.7,           help = '掩码是1的概率')

##==============================================  模型压缩 ==============================================================
## 是否使用模型稀疏
parser.add_argument('--Compression',       type=float,   default = 0,             help = '是否使用压缩')
parser.add_argument('--crate',             type=float,   default = 0.9,           help = '压缩率:选取前crate的模型参数进行传输')


##=============================================================================================================================================
##==============================================  机器学习本身相关参数 ======================================================================
##=============================================================================================================================================

##====================================================================================================================
parser.add_argument('--precision',       type=str,   default='single', choices=('single', 'half'),   help='FP precision for test (single | half)')

# Data specifications
# 数据根目录
parser.add_argument('--dir_minst',    type = str, default = home+'/FedAvg_DataResults/Data/', help = 'dataset directory')
parser.add_argument('--dir_cifar10',  type = str, default = home+'/FedAvg_DataResults/Data/CIFAR10', help = 'dataset directory')

#  地址
parser.add_argument('--pretrain',     type = int, default = False,                                    help = 'whether use pretrain model')
parser.add_argument('--save_path',    type = str, default = home+'/FedAvg_DataResults/results/',      help = 'file name to save')
parser.add_argument('--ModelSave',    type = str, default = home + '/FedAvg_DataResults/ModelSave/',  help = 'file name to save')

# warm up参数, polynomial动态学习率调整先是在最初的 warm_up_ratio*total_setp 个steps中以线性的方式进行增长，之后便是多项式的方式进行递减，直到衰减到lr_end后保持不变。
parser.add_argument('--warm_up_ratio', type = float,   default = 0.1,   help='warm up的步数占比')
parser.add_argument('--lr_end',        type = float,   default = 1e-4,  help='学习率终止值')
parser.add_argument('--power',         type = int,     default = 2,     help='warm up多项式的次数，当power=1时（默认）等价于 get_linear_schedule_with_warmup函数')

# Optimization specifications
parser.add_argument('--learning_rate', type = float, default = 0.01,            help = 'learning rate')
parser.add_argument('--optimizer',     type = str,   default = 'ADAM',           choices = ('SGD', 'ADAM', 'RMSprop'), help = 'optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--decay',         type = str,   default = '20-40-80-120',   help = 'learning rate decay type')
parser.add_argument('--gamma',         type = float, default = 0.6,              help = 'learning rate decay factor for step decay')
parser.add_argument('--momentum',      type = float, default = 0.9,              help = 'SGD momentum')
parser.add_argument('--betas',         type = tuple, default = (0.5, 0.999),     help = 'ADAM beta')
parser.add_argument('--epsilon',       type = float, default = 1e-8,             help = 'ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',  type = float, default = 0,                help = 'weight decay')
parser.add_argument('--gclip',         type = float, default = 0,                help = 'gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss',         type = str, default = '1*CrossEntropy',   help = 'loss function configuration')
parser.add_argument('--reduction',    type = str, default = 'sum',              help = 'loss function configuration')

args, unparsed = parser.parse_known_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        print(f"arg = {arg}")
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        print(f"arg = {arg}")
        vars(args)[arg] = False

# https://github.com/Yangfan-Jiang/Federated-Learning-with-Differential-Privacy/tree/master


# https://github.com/AdamWei-boop/Federated-Learning-with-Local-Differential-Privacy

# https://github.com/TheWitcher05/Federated_learning_with_differential_privacy



