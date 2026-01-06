#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=500,
                        help="number of rounds of training")
    # parser.add_argument('--num_users', type=int, default=100,
    #                     help="number of users: K")
    # parser.add_argument('--frac', type=float, default=0.5,
    #                     help='the fraction of clients: C')
    parser.add_argument('--num_users', type=int, default=40,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.3,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20,
                        help="local batch size: B")
    parser.add_argument('--Vb', type=int, default=20,
                        help="Vb-dimensional quantization")
    parser.add_argument('--M', type=int, default=2**6,
                        help="quantization level")
    parser.add_argument('--L', type=int, default=64,       # int(args.num_users * args.frac)   int(6 * 2**args.Qbit/8)
                        help="Length of each codeword in the codebook")
    args = parser.parse_args()
    Rl = np.random.rand(args.L, args.M).astype(np.float32)
    Im = np.random.rand(args.L, args.M).astype(np.float32)
    Rl[Rl >= 0.5] = 1
    Rl[Rl < 0.5] = -1
    Im[Im >= 0.5] = 1
    Im[Im < 0.5] = -1
    UMmat = (1 / np.sqrt(2)) * (Rl + 1j * Im)
    parser.add_argument('--UM', type=np.float32, default=UMmat,  help="Length of each codeword in the codebook") #                     help="Length of each codeword in the codebook")
    # parser.add_argument('--UM', type=np.float32, default=(1 / np.sqrt(2)) * (np.random.randn(args.L, args.M).astype(np.float32) + 1j * np.random.randn(args.L, args.M).astype(np.float32)),
    #                     help="Length of each codeword in the codebook")
    parser.add_argument('--SNR', type=int, default=20, help="SNR")
    parser.add_argument('--local_lr', type=float, default=0.01,
                        help='learning rate for local update')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='learning rate for global update')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam')
    parser.add_argument('--eps', type=float, default=0, help='eps for adam')
    parser.add_argument('--max_init', type=float, default=1e-3, help='initialize max_v for adam')



    # model arguments
    parser.add_argument('--model', type=str, default='convmixer', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='fedavg', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save', type=int, default=1, help='whether to save results')
    parser.add_argument('--outfolder', type=str, default='./results')

    parser.add_argument('--compressor', type=str, default='Kmeans', help='compressor strategy')
    args = parser.parse_args()
    return args
