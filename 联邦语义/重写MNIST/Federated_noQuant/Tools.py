#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 03:54:22 2025

@author: jack
"""


# 系统库
import math
import os, sys
import time
import datetime
import torch
import PIL
import numpy as np
# from scipy import stats
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#### 本项目自己编写的库
from Logs import Accumulator

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype_np = lambda x, *args, **kwargs: x.astype(*args, **kwargs)
astype_tensor = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def try_gpu(i=0):
    """Return gpu device if exists, otherwise return cpu device."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


#============================================================================================================================
#                                                   data preporcess and recover
#============================================================================================================================

def data_inv_tf_mlp_mnist(x):
    """
    :param x:
    :return:
    """
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((28, 28))
    recover_data = np.around(recover_data.detach().cpu().numpy() ).astype(np.uint8)
    return recover_data


def data_tf_cnn_mnist_batch(x):
    # ## 1
    # x = torchvision.transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1, 1, 28, 28))

    ### 2
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))  # (-1, 28*28)
    x = x.reshape((-1, 1, 28, 28))  # ( 1, 28, 28)
    x = torch.from_numpy(x)
    return x

# x.shape = (128, 1, 28, 28)
# recover_data = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_3D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape(( -1, 1, 28, 28))  # (-1, 28, 28)
    # recover_data = np.around(recover_data.detach().numpy() ).astype(np.uint8)
    # recover_data =  recover_data.detach().type(torch.uint8)
    return recover_data  #   (128, 1, 28, 28)

# x.shape = (128, 1, 28, 28)
def data_inv_tf_cnn_mnist_batch_2D(x):
    """
    :param x:
    :return:
    """
    # recover_data = x * 0.5 + 0.5
    # recover_data = x
    recover_data = x * 0.5 + 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((-1, 28, 28))  # (-1, 28, 28)
    recover_data = np.around(recover_data.numpy()).astype(np.uint8)
    # recover_data =  recover_data.round().type(torch.uint8)
    return recover_data  # (128, 28, 28)


#============================================================================================================================
#                                                   计算正确率
#============================================================================================================================

# 去归一化
def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def formatPrint2DArray(array):
    for linedata in array:
        for idx, coldata in enumerate(linedata):
            if idx == 0:
                print(f"{coldata:>8.3f}(dB)  ", end = ' ')
            else:
                print(f"{coldata:>8.3f}  ", end = ' ')
        print("\n", end = ' ')

    return
#============================================================================================================================
#                                                   定时器
#============================================================================================================================

class myTimer(object):
    def __init__(self, name = 'epoch'):
        self.acc = 0
        self.name = name
        self.timer = 0
        self.tic()
        self.start_str = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    def tic(self):  # time.time()函数返回自纪元以来经过的秒数。
        self.t0 = time.time()
        self.ts = self.t0

    # 返回从ts开始历经的秒数。
    def toc(self):
        diff = time.time() - self.ts
        self.ts = time.time()
        self.timer  += diff
        return diff

    def reset(self):
        self.ts = time.time()
        tmp = self.timer
        self.timer = 0
        return tmp

    def now(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # 从计时开始到现在的时间.
    def hold(self):
        return time.time() - self.t0

#============================================================================================================================
#                                                   指标计算
#============================================================================================================================

def MSE_np_Batch(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    im, jm = np.float64(im), np.float64(jm)
    # X, Y = np.array(X).astype(np.float64), np.array(Y).astype(np.float64)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    return mse

def MSE_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)
    D = len(im.shape)
    if D < 2:
        raise ValueError('Input images must have >= 2D dimensions.')
    MSE = 0
    for i in range(im.shape[0]):
        MSE += MSE_np_Batch(im[i], jm[i], )

    avgmse = MSE/im.shape[0]
    return  avgmse, MSE, im.shape[0]


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr

def PSNR_np_Batch(im, jm, rgb_range = 255.0, cal_type = 'y'):
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)
    G = np.array( [65.481,   128.553,    24.966] )
    G = G.reshape(3, 1, 1)

    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        diff = diff * G
        diff = np.sum(diff, axis = -3) / rgb_range
    #print(f"diff.shape = {diff.shape}")
    mse = np.mean(diff**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = -10.0 * math.log10(mse)

    return psnr

def PSNR_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    D = len(im.shape)
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    im, jm = np.array(im, dtype = np.float64), np.array(jm, dtype = np.float64)

    PSNR = 0
    for i in range(im.shape[0]):
        if im[i].shape[0] == 1:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].shape[0] == 3:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = 'y')

    avgsnr = PSNR/im.shape[0]
    return  avgsnr, PSNR, im.shape[0]

#========================================================================================================================

def PSNR_torch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    maxp = max(im.max(), jm.max() )
    cr = torch.nn.MSELoss()
    mse = cr(jm, im)
    # out_np = X_hat.detach().numpy()
    psnr = 10 * np.log10(maxp ** 2 / mse.detach().numpy() ** 2)
    if psnr > 200:
        psnr = 200.0
    return  psnr

def PSNR_torch_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    # im, jm = torch.Tensor(im), torch.Tensor(jm)
    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        # gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs = [65.481, 128.553, 24.966]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1)
        diff = diff.mul(convert).sum(dim = -3) / rgb_range
    mse = diff.pow(2).mean().item()
    # print(f"mse = {mse}")
    if mse <= 1e-20:
        mse = 1e-20
    psnr =  -10.0 * math.log10( mse)

    return  psnr


def PSNR_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    # im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must 4-dimensions.')
    PSNR = 0
    for i in range(im.size(0)):
        if im[i].size(0) == 1:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].size(0) == 3:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = 'y')

    avgpsnr = PSNR/im.size(0)
    return  avgpsnr, PSNR, im.size(0)


def MSE_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    MSE = 0
    for i in range(im.size(0)):
        MSE += MSE_torch_Batch(im[i], jm[i] )

    avgmse = MSE/im.size(0)
    return  avgmse, MSE, im.size(0)



def MSE_torch_Batch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    diff = (im - jm)

    mse = diff.pow(2).mean().item()
    return mse

#============================================================================================================================
#
#============================================================================================================================

#  功能：将img每个像素点的至夹在[0,255]之间
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def prepare(device, precision, *Args):
    def _prepare(tensor):
        if  precision == 'half': tensor = tensor.half()
        return tensor.to( device)

    return [_prepare(a) for a in Args]


#============================================================================================================================
#                              AWGN: add noise
#============================================================================================================================

# 以实际信号功率计算噪声功率，再将信号加上噪声。
def Awgn(x, snr = 3):
    if snr == None:
        return x
    SNR = 10.0**(snr/10.0)
    # signal_power = ((x**2)*1.0).mean()
    signal_power = (x*1.0).pow(2).mean()
    noise_power = signal_power/SNR
    noise_std = torch.sqrt(noise_power)
    #print(f"x.shape = {x.shape}, signal_power = {signal_power}, noise_power={noise_power}, noise_std={noise_std}")

    noise = torch.normal(mean = 0, std = float(noise_std), size = x.shape)
    return x + noise.to(x.device)

#============================================================================================================================
#                                                画图代码
#============================================================================================================================


















