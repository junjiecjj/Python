

# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

import argparse

parser = argparse.ArgumentParser(description='IPT模型的参数')

parser.add_argument('--debug', action='store_true', help='Enables debug mode')
parser.add_argument('--template', default='.', help='You can set various templates in option.py')

# model  specifications
parser.add_argument('--modelUse', default='ipt', help='You can set various templates in option.py')
# parser.add_argument('--model', default='ipt', help='model name')

parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single | half)')

# 预训练模型地址
parser.add_argument('--pretrain', type=str, default='/home/jack/IPT-Pretrain/IPT_pretrain.pt')  # cjj

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
# parser.add_argument('--cpu', action='store_true', help='use cpu only')
# cjj add
parser.add_argument('--cpu', action='store_false', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
# 数据根目录
parser.add_argument('--dir_data', type=str, default='/home/jack/IPT-Pretrain/Data/', help='dataset directory')  # cjj
parser.add_argument('--dir_demo', type=str, default='../test', help='demo image directory')
parser.add_argument('--SummaryWriteDir', type=str, default='/home/jack/IPT-Pretrain/results/summary', help='demo image directory')

# 训练数据名称
parser.add_argument('--data_train', type=str, default='DIV2K_cut', help='train dataset name')

# 测试集数据名称
parser.add_argument('--data_test',type=str,  default='Set2+Set3', help='test dataset name')
# parser.add_argument('--data_test', type=str, default='Set5+Set14+B100+Urban100+DIV2K', help='test dataset name')  # cjj

parser.add_argument('--useBIN',  action='store_false', help='是否使用bin图像')

#parser.add_argument('--data_range', type=str, default='1-800/801-810', help='train/test data range')
parser.add_argument('--data_range', type=str, default='1-64', help='train/test data range')
parser.add_argument('--ext', type=str, default='sep', help='dataset file extension')
# parser.add_argument('--scale', type=str, default='2+3+4+5+6+1', help='super resolution scale')
parser.add_argument('--scale', type=str, default='1', help='super resolution scale') # cjj

parser.add_argument('--patch_size', type=int, default=48, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',  help='do not use data augmentation')

parser.add_argument('--CompressRateTrain', type=str, default='0.17, 0.33 ',  help='Compress rate for test')
parser.add_argument('--SNRtrain',  type=str, default='2, 10',  help='SNR for train')

#parser.add_argument('--CompressRateTest', type=str, default='0.17, 0.33, 0.4',  help='Compress rate for test')
parser.add_argument('--SNRtest',  type=str, default='-6,-4,-2, 0, 2, 6, 10, 14, 18',  help='SNR for train')


# Training and test  specifications
# cjj
parser.add_argument('--wanttest',  action='store_false', help='set this option to test the model')
parser.add_argument('--wanttrain', action='store_false', help='set this option to train the model')
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=1000, help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=100,  help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--test_batch_size', type=int,  default=1,help='input batch size for training')
parser.add_argument('--crop_batch_size', type=int, default=64, help='input batch size for training')
parser.add_argument('--split_batch', type=int,default=1, help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble',  action='store_true', help='use self-ensemble method for test')
# parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1, help='k value for adversarial loss')


#denoise
parser.add_argument('--denoise', action='store_false')
parser.add_argument('--sigma', type=float, default=30)

#derain
parser.add_argument('--derain', action='store_false')
parser.add_argument('--derain_test', type=int, default=1)


# 压缩层和解压缩层参数设置
parser.add_argument('--cpKerSize', type=int,  default=12, help='压缩层的卷积核大小')
parser.add_argument('--cpStride', type=int,  default=4, help='压缩层的步长')
parser.add_argument('--cpPad', type=int,  default=2, help='压缩层的padding')

parser.add_argument('--dcpKerSize', type=int,  default=10, help='压缩层的卷积核大小')
parser.add_argument('--dcpStride', type=int,  default=4, help='压缩层的步长')
parser.add_argument('--dcpPad', type=int,  default=1, help='压缩层的padding')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=str, default='20-40-60-80-100-120',  help='learning rate decay type')
parser.add_argument('--gamma',  type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas',  type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*MSE+0.7*L1', help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8', help='skipping batch that has large error')

parser.add_argument('--metrics', type=str, default='Psnr, MSE', help='loss function configuration')

# Log specifications
parser.add_argument('--save', type=str, default='/home/jack/IPT-Pretrain/results/',  help='file name to save')  #cjj
parser.add_argument('--load', type=str, default='/home/jack/IPT-Pretrain/results/', help='file name to load')
parser.add_argument('--resume', type=int,  default=0, help='resume from specific checkpoint')
parser.add_argument('--saveModelEveryEpoch', action='store_false', help='save all intermediate models')
parser.add_argument('--print_every',type=int, default=100,help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_false', help='save output results')
parser.add_argument('--save_gt',action='store_false',help='save low-resolution and high-resolution images together')


#transformer
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--patch_dim', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--freeze_norm', action='store_true')
parser.add_argument('--post_norm', action='store_true')
parser.add_argument('--no_mlp', action='store_true')
parser.add_argument('--pos_every', action='store_true')
parser.add_argument('--no_pos', action='store_true')
#  parser.add_argument('--num_queries', type=int, default=1)
parser.add_argument('--num_queries', type=int, default=6)



args, unparsed = parser.parse_known_args()

# [2]
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')  #  ['DIV2K']

args.metrics = list(map(lambda x: x.strip(" "), args.metrics.split(',')))


args.CompressRateTrain = list(map(lambda x: float(x), args.CompressRateTrain.split(',')))
args.SNRtrain = list(map(lambda x: int(x), args.SNRtrain.split(',')))
args.SNRtest = list(map(lambda x: int(x), args.SNRtest.split(',')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        print(f"arg = {arg}")
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        print(f"arg = {arg}")
        vars(args)[arg] = False





