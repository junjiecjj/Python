

# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

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


parser = argparse.ArgumentParser(description=' Deep learning 模型的参数')

parser.add_argument('--host_name', default = host_name, help='You can set various templates in option.py')
parser.add_argument('--user_name', default = user_name, help='You can set various templates in option.py')
parser.add_argument('--user_home', default = user_home, help='You can set various templates in option.py')

# model  specifications
parser.add_argument('--modelUse', default = 'LeNet', help='You can set various templates in option.py')
# parser.add_argument('--model', default='ipt', help='model name')


parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')

# CPU GPU相关等参数
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
# cjj add
parser.add_argument('--cpu', action = 'store_true', default = False, help = 'use cpu only')
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'cuda device')
parser.add_argument('--seed', type = int, default = 1, help = 'random seed')

# Data specifications
# 数据根目录
parser.add_argument('--dir_fashionminst', type = str, default = home+'/SemanticNoise_AdversarialAttack/Data/', help = 'dataset directory')  # cjj
parser.add_argument('--dir_minst', type = str, default = home+'/SemanticNoise_AdversarialAttack/Data/', help = 'dataset directory')  # cjj
parser.add_argument('--dir_cifar10', type = str, default = home+'/SemanticNoise_AdversarialAttack/Data/CIFAR10', help = 'dataset directory')  # cjj

parser.add_argument('--dir_demo', type = str, default = '../test', help = 'demo image directory')
parser.add_argument('--tmpout', type = str, default = home+'/SemanticNoise_AdversarialAttack/tmpout/', help = 'tmpout directory')  # cjj

# 预训练模型地址
parser.add_argument('--pretrain', action = 'store_true', default = False,  help = 'whether use pretrain model')  # cjj
parser.add_argument('--save', type = str, default = home+'/SemanticNoise_AdversarialAttack/results/',  help = 'file name to save')  #cjj

parser.add_argument('--ModelSave', type = str, default = home + '/SemanticNoise_AdversarialAttack/ModelSave/',  help = 'file name to save')  #cjj
parser.add_argument('--SummaryWriteDir', type = str, default = home+'/SemanticNoise_AdversarialAttack/results/TensorBoard', help = 'demo image directory')
parser.add_argument('--TrainImageSave', type = str, default = home + '/SemanticNoise_AdversarialAttack/results/',  help='file name to save image during train process')  #cjj


# Minst 数据集相关参数
parser.add_argument('--Minst_channel', type = int, default = 1, help = 'Channel of Minst dataset')
parser.add_argument('--Minst_heigh', type = int, default = 28, help = 'Height of Minst dataset')
parser.add_argument('--Minst_width', type = int, default = 28, help = 'Weight of Minst dataset')
parser.add_argument('--noise_dim', type = int, default = 100, help = 'Weight of Minst dataset')
parser.add_argument('--real_label', type = int, default = 1, help = 'Weight of Minst dataset')
parser.add_argument('--fake_label', type = int, default = 0, help = 'Weight of Minst dataset')



# Training and test  specifications
parser.add_argument('--epochs', type=int, default = 10,  help='number of epochs to train')
parser.add_argument('--batch_size', type = int, default = 128, help = 'input batch size for training')
parser.add_argument('--test_batch_size', type=int,  default = 128, help='input batch size for test')

# warm up参数, polynomial动态学习率调整先是在最初的 warm_up_ratio*total_setp 个steps中以线性的方式进行增长，之后便是多项式的方式进行递减，直到衰减到lr_end后保持不变。
parser.add_argument('--warm_up_ratio', type = float, default = 0.1, help='warm up的步数占比')
parser.add_argument('--lr_end', type = float,  default = 1e-4, help='学习率终止值')
parser.add_argument('--power', type = int,  default = 2, help='warm up多项式的次数，当power=1时（默认）等价于 get_linear_schedule_with_warmup函数')


# Optimization specifications
parser.add_argument('--lr', type = float, default = 0.002, help = 'learning rate')
parser.add_argument('--decay', type = str, default = '20-40-80-120',  help = 'learning rate decay type')
parser.add_argument('--gamma',  type = float, default = 0.6, help = 'learning rate decay factor for step decay')
parser.add_argument('--optimizer', default = 'ADAM', choices = ('SGD', 'ADAM', 'RMSprop'), help = 'optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'SGD momentum')
parser.add_argument('--betas',  type = tuple, default = (0.5, 0.999), help = 'ADAM beta')
parser.add_argument('--epsilon', type = float, default = 1e-8, help = 'ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay')
parser.add_argument('--gclip', type = float, default = 0, help = 'gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type = str, default = '1*MSE', help = 'loss function configuration')

parser.add_argument('--CompRate',  type = np.ndarray,     default = np.arange(0.1, 1, 0.1),  help = 'Compress rate for test')
parser.add_argument('--SNRtrain',  type = str,            default = '1, 3, 10, 20',  help = 'SNR for train ')
parser.add_argument('--SNRtest',   type = np.ndarray,     default = np.arange(-2, 21, 1),  help = 'SNR for  test')
parser.add_argument('--metrics',   type = str,            default = 'PSNR', help = 'loss function configuration')

parser.add_argument('--quantize', action = 'store_true', default = False,  help = 'whether use quantize')  # cjj

args, unparsed = parser.parse_known_args()

args.SNRtest = np.append(args.SNRtest, 25)
args.SNRtest = np.append(args.SNRtest, 30)
args.SNRtest = np.append(args.SNRtest, 35)
args.SNRtest = np.append(args.SNRtest, 40)
args.SNRtest.sort()


if args.epochs == 0:
    args.epochs = 1e8

args.metrics = list(map(lambda x: x.strip(" "), args.metrics.split(',')))

# args.CompRate = list(map(lambda x: float(x), args.CompRate.split(',')))
args.SNRtrain      = list(map(lambda x: int(x), args.SNRtrain.split(',')))


for arg in vars(args):
    if vars(args)[arg] == 'True':
        print(f"arg = {arg}")
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        print(f"arg = {arg}")
        vars(args)[arg] = False











