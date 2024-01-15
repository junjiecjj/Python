

# -*- coding: utf-8 -*-
"""
Created on 2023/06/30

@author: Junjie Chen

"""

import argparse
import socket, getpass , os
import numpy as np

import torch


# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')


parser = argparse.ArgumentParser(description="Semantic Communication Based on Federated Learning")
parser.add_argument('--host_name',       type=str, default = host_name, help='host name')
parser.add_argument('--user_name',       type=str, default = user_name, help='user name')
parser.add_argument('--user_home',       type=str, default = user_home, help='user home')

# model  specifications
parser.add_argument('--modelUse',        type = str, default = 'FLSemantic',   help='model name')

# 设备相关
parser.add_argument('--cpu',             type = int, default = 0,      help = 'use cpu only')
parser.add_argument('--device',          type = str, default = 'cuda:0',   help = 'cuda device')
parser.add_argument('--seed',            type = int, default = 1,          help = 'random seed')

##=============================================================================================================================================
##==============================================  联邦学习相关参数 ======================================================================
##=============================================================================================================================================
## 数据是否 IID
parser.add_argument('--isIID',           type=int,   default = 0,              help = '每个客户端的数据是否IID, 0代表non-IID，1代表IID')

## 训练次数(客户端更新次数)
parser.add_argument('--loc_epochs',      type=int,   default = 10,             help = 'local train epoch')
## local_batchsize 大小
parser.add_argument('--local_batchsize', type=int,   default = 50,             help = 'local train batch size')
## test_batchsize 大小
parser.add_argument('--test_batchsize',  type=int,   default = 128,            help = 'test batch size')
## 模型名称
parser.add_argument('--model_name',      type=str,   default = "AutoEncoder",    help = 'the model to train')
## 所用的数据集
parser.add_argument("--dataset",         type=str,   default = "mnist",        help = "训练数据集")
## 数据较多的客户端的数据量

## 客户端的数量
parser.add_argument('--num_of_clients',  type=int,   default = 100,            help = ' ')
## 随机挑选的客户端的数量
parser.add_argument('--cfraction',       type=float, default = 0.1,            help = 'C fraction, 0 means 1 client, 1 means total clients')
## num_comm 表示通信次数，此处设置为1k
parser.add_argument('--num_comm',        type=int,   default = 500,           help = 'number of communications')
## 传输的是模型参数还是模型更新
parser.add_argument('--transmit_diff',   type=int,   default = 1,              help = 'the way to allocate data to clients')


##================================================= 量化 ========================================================================
## 是否使用模型稀疏
parser.add_argument('--Quantz',            type = int,     default = 0,            help = '是否使用量化')
parser.add_argument('--B',                 type = int,     default = 8,            help = '量化比特数')

##================================================= 1 Bit 量化 ========================================================================
## 是否使用模型稀疏
# parser.add_argument('--Quantz1Bit',             type = int,     default = 0,            help = '是否使用1 Bit 量化')
parser.add_argument('--G',             type = int,     default = 4,            help = 'G')

##=============================================================================================================================================
##==============================================  机器学习本身相关参数 ======================================================================
##=============================================================================================================================================

##====================================================================================================================
parser.add_argument('--precision',     type=str,   default='single', choices=('single', 'half'),   help='FP precision for test (single | half)')

# Data specifications
# 数据根目录
parser.add_argument('--dir_minst',     type = str, default = home+'/FL_semantic/Data/', help = 'dataset directory')


#  地址
parser.add_argument('--pretrain',      type = int, default = False,                             help = 'whether use pretrain model')
parser.add_argument('--save_path',     type = str, default = home + '/FL_semantic/results/',    help = 'file name to save')
parser.add_argument('--ModelSave',     type = str, default = home + '/FL_semantic/ModelSave/',  help = 'file name to save')  #cjj

# Optimization specifications
parser.add_argument('--learning_rate', type = float, default = 0.001,            help = 'learning rate')
parser.add_argument('--optimizer',     type = str,   default = 'ADAM',           choices = ('SGD', 'ADAM', 'RMSprop'), help = 'optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--decay',         type = str,   default = '40-80-120-200',  help = 'learning rate decay type')
parser.add_argument('--gamma',         type = float, default = 0.996,           help = 'learning rate decay factor for step decay')
parser.add_argument('--momentum',      type = float, default = 0.9,              help = 'SGD momentum')
parser.add_argument('--betas',         type = tuple, default = (0.5, 0.999),     help = 'ADAM beta')
parser.add_argument('--epsilon',       type = float, default = 1e-8,             help = 'ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay',  type = float, default = 0,                help = 'weight decay')
parser.add_argument('--gclip',         type = float, default = 0,                help = 'gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss',          type = str, default = '1*MSE',            help = 'loss function configuration')
parser.add_argument('--reduction',     type = str, default = 'sum',              help = 'loss function configuration')

# warm up参数, polynomial 动态学习率调整先是在最初的 warm_up_ratio*total_setp 个 steps 中以线性的方式进行增长，之后便是多项式的方式进行递减，直到衰减到lr_end后保持不变。
parser.add_argument('--warm_up_ratio', type = float,   default = 0.1,   help='warm up的步数占比')
parser.add_argument('--lr_end',        type = float,   default = 1e-4,  help='学习率终止值')
parser.add_argument('--power',         type = int,     default = 2,     help='warm up多项式的次数，当power=1时（默认）等价于 get_linear_schedule_with_warmup函数')

parser.add_argument('--CompRate',      type = np.ndarray,     default = [0.2, 0.5, 0.9],  help = 'Compress rate for test')
parser.add_argument('--SNRtrain',      type = np.ndarray,     default = [2, 10, 20],  help = 'SNR for train ')
parser.add_argument('--SNRtest',       type = np.ndarray,     default = np.arange(-5, 36, 1),  help = 'SNR for  test')
##====================================================
args, unparsed = parser.parse_known_args()
##====================================================



args.decay = '-'.join([str((i+1)) for i in range(args.num_comm - 1)])
# args.decay = '-'.join([str((i+1)) for i in range(args.num_comm  - 1)])

#==================================================== device ===================================================
# 如果不想用CPU且存在GPU, 则用GPU; 否则用CPU;

args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
print(f"PyTorch is running on {args.device}, {torch.cuda.get_device_name(0)}")




