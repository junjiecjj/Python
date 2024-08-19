
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""

import os, sys
import math
import datetime
import imageio
# import cv2
import skimage
import glob
import random
import numpy as np
import torch

import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')


#### 本项目自己编写的库
# sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



# 初始化随机数种子
def set_random_seed(seed = 999999,):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

def set_printoption(precision = 3):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    torch.set_printoptions(
        precision = precision,    # 精度，保留小数点后几位，默认4
        threshold = 1000,
        edgeitems = 3,
        linewidth = 150,  # 每行最多显示的字符数，默认80，超过则换行显示
        profile = None,
        sci_mode = False  # 用科学技术法显示数据，默认True
    )


def Initial(args):
    theta_true = np.random.randn(args.D, 1)
    theta0 = np.zeros((args.D, 1))
    X = {}
    Y = {}
    for client in range(args.num_of_clients):
        X[f"client{client:d}"] = np.random.randn(args.local_ds, args.D)
        noise = np.random.normal(loc = 0.0, scale = args.sigma, size = (args.local_ds, 1))
        Y[f"client{client:d}"] = X[f"client{client:d}"] @ theta_true + noise
    frac = {}
    tot_data_size = np.sum([len(val) for key, val in X.items()])
    for key, val in X.items():
        frac[key] = len(val)/tot_data_size
    return  theta_true, theta0, X, Y, frac


def optimal_linear_reg(X, Y, frac):
    XTX = np.sum([data.T @ data for client, data in X.items()], axis = 0)
    XTY = np.sum([data.T @ Y[client] for client, data in X.items()], axis = 0)
    theta_optim = np.linalg.inv(XTX) @ XTY
    optim_Fw = np.sum([0.5 * frac[client] * np.linalg.norm(data @ theta_optim - Y[client], ord = 2 )**2/len(data) for client, data in X.items()])
    return theta_optim, optim_Fw













































































































































































