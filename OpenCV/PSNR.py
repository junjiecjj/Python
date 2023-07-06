#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 23:06:12 2023

@author: jack
"""

import PIL
import math
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import sys

"""
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import mean_squared_error as ski_mse

im和jm的形状一致:
如: size = (2, 28, 28, 3, )
    size = (2, 3, 28, 28,)
    size = ( 3, 28, 28,)
    size = ( 28, 28, 3)
"""
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import mean_squared_error as ski_mse
from skimage.metrics import structural_similarity as ski_ssim
from skimage.color import rgb2ycbcr


"""
im和jm的形状一致: 输入的im, jm 可以是单通道也可以是多通道, 通道在前在后无所谓, 因为是 全部元素求mse,
如: size = (2, 28, 28, 3, )
    size = (2, 3, 28, 28,)
    size = ( 3, 28, 28,)
    size = ( 28, 28, 3)
    size = ( 28, 28 )
"""
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

    D = len(im.shape)
    if D < 3:
        raise ValueError('Input images must have >= 3D dimensions.')
    MSE = 0
    for i in range(im.shape[0]):
        MSE += MSE_np_Batch(im[i], jm[i], )

    avgmse = MSE/im.shape[0]
    return  avgmse, MSE, im.shape[0]



"""
im和jm的形状一致: 输入的im, jm 可以是单通道也可以是多通道, 通道在前在后无所谓, 因为是 全部元素求mse,
如: size = (2, 28, 28, 3, )
    size = (2, 3, 28, 28,)
    size = ( 3, 28, 28,)
    size = ( 28, 28, 3)
    size = ( 28, 28 )
"""
def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr

"""
im和jm的形状一致: 输入的im, jm 只能是三通道, 且为RGB顺序, 通道必须在最后,  因为rgb2ycbcr默认是通道在后,

如: size = (2, 28, 28, 3, )
    size = ( 28, 28, 3)
    通道必须在最后
"""
def PSNR_2y_np_RGBchannelLast(im, jm):
    im, jm = im/255.0, jm/255.0
    im = rgb2ycbcr(im)[..., 0]
    jm = rgb2ycbcr(jm)[..., 0]
    # print(f"im.shape = {im.shape}, jm.shape = {jm.shape}")

    #im, jm = im/255.0, jm/255.0
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)

    #psnr = ski_psnr(im, jm)
    return psnr


"""
逐像素的计算 PSNR

im和jm的形状一致:

(1) 输入的im, jm 可以是三通道, 如果是三通道, 且 cal_type = 'y', 则这时的通道顺序只能是RGB顺序,  且通道的维数必须在长宽维数前, 这时计算的是将RGB转化为YCbCr通道后在Y通道上计算PSNR.
如:
    size = (batchsize, 3, 28, 28,)
            (1, 3, 28, 28,)
            ( 3, 28, 28,)
            (28, 28) # 也可以, 这时是单通道, 此时若cal_type = 'y', 则会将单通道拓展为三通道, 所以如果是(28, 28), 则建议使用 cal_type != 'y'
(2) 输入的im, jm 可以是三通道, 如果是三通道, 且 cal_type != 'y', 这时是直接求全部像素点的mse然后计算PSNR.
如:
    size = (batchsize, 3, 28, 28,)

(3) 输入的im, jm 可以是单通道, 如果是单通道,  cal_type 必须不等于 'y', 这时是直接求全部像素点的mse然后计算PSNR.
如:
    size = (batchsize, 1, 28, 28,)
           (1, 28, 28,)
           (28, 28,)
"""
def PSNR_np_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
    im, jm = np.array(im, dtype = np.float32), np.array(jm, dtype = np.float32)
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


"""
    size = (batchsize, 3, 28, 28,)
           (batchsize, 1, 28, 28,)
"""
def PSNR_np_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    D = len(im.shape)
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    PSNR = 0
    for i in range(im.shape[0]):
        if im[i].shape[0] == 1:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].shape[0] == 3:
            PSNR += PSNR_np_Batch(im[i], jm[i], cal_type = 'y')

    avgsnr = PSNR/im.shape[0]
    return  avgsnr, PSNR, im.shape[0]




#  功能：将img每个像素点的至夹在[0,255]之间
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)




"""
逐像素的计算 PSNR

im和jm的形状一致:

(1) 输入的im, jm 可以是三通道, 如果是三通道, 且 cal_type = 'y', 则这时的通道顺序只能是RGB顺序,  且通道的维数必须在长宽维数前, 这时计算的是将RGB转化为YCbCr通道后在Y通道上计算PSNR.
如:
    size = (batchsize, 3, 28, 28,)
           (1, 3, 28, 28,)
            (3, 28, 28,)
            (28, 28) # 也可以, 这时是单通道, 此时若cal_type = 'y', 则会将单通道拓展为三通道, 所以如果是(28, 28), 则建议使用 cal_type != 'y'
(2) 输入的im, jm 可以是三通道, 如果是三通道, 且 cal_type != 'y', 这时等价于直接求全部像素点的mse然后计算PSNR, 这时只要 im, jm 的shape一样即可, shape是任何都行.
如:
    size = (batchsize, 3, 28, 28, )
            (1, 3, 28, 28, )
            ( 3, 28, 28, )
            ( 28, 28, 3 )
(3) 输入的im, jm 可以是单通道, 如果是单通道,  cal_type 必须不等于 'y', 这时是直接求全部像素点的mse然后计算PSNR. cal_type  等于 'y'也是可以运行的, 但是结果没意义, 因为本身就是单通道.
如:
    size = (batchsize, 1, 28, 28,)
    size = ( 1, 28, 28,)
"""
def PSNR_torch_Batch(im, jm, rgb_range = 255.0, cal_type='y'):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)

    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        # gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs = [65.481, 128.553, 24.966]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1)
        diff = diff.mul(convert).sum(dim = -3) / rgb_range
    mse = diff.pow(2).mean().item()
    # print(f"mse = {mse}")
    if mse <= 1e-10:
        mse = 1e-10
    psnr =  -10.0 * math.log10( mse)

    return  psnr

"""
计算 Batch的 PSNR, 之前的解法虽然RGB转YCbCr后计算Y通道的PSNR, 但都是以batch为单位, 而不是以batch内的每张图片为单位, 此实现方法将batch内的图片分开统计,然后求和取平均.

im和jm的形状一致: 必须为以下两种,
    size = (batchsize, 3, 28, 28,)
    size = (batchsize, 1, 28, 28,)

经过测试发现, 对于 PSNR
            (一) 如果先转换到Y通道后再计算PSNR, 不管是 (1) 先计算每张图片的 psnr, 然后求和取均值还是 (2)直接以 batch 为单位计算, 当batchsize 足够大的时候, 两种方法基本一致.
            (二) 如果不经过通道转换, 直接求全局的mse, 不管是 (1) 先计算每张图片的 psnr, 然后求和取均值还是 (2)直接以 batch 为单位计算, 当batchsize 足够大的时候, 两种方法基本一致.
            但经过通道转换, 转换到y通道后计算 psnr 明显比不转换的大.

            对于MSE, 则不存在是否转通道的问题, 经过测试发现:
            不管是 直接计算 batch 的 MSE 还是计算每张图的 mse, 然后求和取均值, 计算的 MSE 是一样的.
"""
def PSNR_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D != 4:
        raise ValueError('Input images must have 4D dimensions.')
    PSNR = 0
    for i in range(im.size(0)):
        if im[i].size(0) == 1:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = '1')
        elif im[i].size(0) == 3:
            PSNR += PSNR_torch_Batch(im[i], jm[i], cal_type = 'y')

    avgsnr = PSNR/im.size(0)
    return  avgsnr, PSNR, im.size(0)



"""
im和jm的形状一致: 输入的im, jm 可以是三通道也可以说单通道, 通道在前在后无所谓, 因为是 全部元素求mse,
如: size = (2, 28, 28, 3, )
    size = (2, 3, 28, 28,)
    size = ( 3, 28, 28,)
    size = ( 28, 28, 3)
    size = ( 28, 28 )
"""
def MSE_torch_Batch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, hr = torch.Tensor(im), torch.Tensor(jm)
    diff = (im - jm)

    mse = diff.pow(2).mean().item()
    return mse

def MSE_torch_Image(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    D = len(im.size())
    if D < 3 :
        raise ValueError('Input images must have >= 3D dimensions.')
    MSE = 0
    for i in range(im.size(0)):
        MSE += MSE_torch_Batch(im[i], jm[i] )

    avgmse = MSE/im.size(0)
    return  avgmse, MSE, im.size(0)


"""
逐像素的计算均方误差
"""
def calc_psnr(sr, hr,  rgb_range = 255.0, cal_type='y'):
    sr, hr = torch.Tensor(sr), torch.Tensor(hr)
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range

    if cal_type=='y':
        #gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs = [65.481, 128.553, 24.966]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / rgb_range
        diff = diff.mul(convert).sum(dim=1)

    mse = diff.pow(2).mean().item()
    if mse <= 1e-10:
        mse = 1e-10
    return   -10 * math.log10(mse)


# source = "/home/jack/公共的/Python/OpenCV/"

# im = imageio.imread(source + "baby.png").transpose(2, 0 ,1)
# jm = imageio.imread(source + "baby_poll.jpg").transpose(2, 0 ,1)


# # jm = im + np.random.normal(0, 1 , size = im.shape)
# a = PSNR_np_simple(im, jm)
# b = PSNR_np_Batch(im, jm)

# np.random.seed(1)
# # im = np.random.randint(low = 0, high = 256, size = (2, 28, 28, 3, )).astype(np.uint8)
# # jm = np.random.randint(low = 0, high = 256, size = (2, 28, 28, 3, )).astype(np.uint8)

# #===============================================================================================
# ycbcr_from_rgb = np.array([[    65.481,   128.553,    24.966],
#                             [   -37.797,   -74.203,   112.0  ],
#                             [   112.0  ,   -93.786,   -18.214]])

# G = np.array( [65.481,   128.553,    24.966] )



# # im = np.random.randint(low = 0, high = 10, size = (2, 2, 2, 3, ))#.astype(np.uint8)
# # jm = np.random.randint(low = 0, high = 10, size = (2, 2, 2, 3, ))#.astype(np.uint8)

# im = np.random.randint(low = 0, high = 256, size = ( 28, 28)).astype(np.uint8)
# jm = np.random.randint(low = 0, high = 256, size = ( 28, 28)).astype(np.uint8)

# # im = imageio.imread(source + "baby.png")
# # jm = imageio.imread(source + "baby_poll.jpg")
# # im = np.expand_dims(im.transpose(2,0,1), axis = 0)
# # jm = np.expand_dims(jm.transpose(2,0,1), axis = 0)

# print(f"im.shape = {im.shape}, jm.shape = {jm.shape}\n\n")


# mse_np = MSE_np_Batch(im, jm)
# print(f"mse_np = {mse_np} ")

# mse_tr = MSE_torch_Batch(im, jm)
# print(f"mse_tr = {mse_tr},  ")

# # avgmse_np, Mse_np, size_np = MSE_np_Image(im, jm)
# # print(f"avgmse_np = {avgmse_np}, Mse_np = {Mse_np}, size_np = {size_np}")

# # avgmse_torch, Mse_torch, size = MSE_torch_Image(im, jm)
# # print(f"avgmse_torch = {avgmse_torch}, Mse_torch = {Mse_torch}, size = {size}\n\n")



# pnsr_np_simple = PSNR_np_simple(im, jm)
# print(f"pnsr_np_simple = {pnsr_np_simple} ")

# color = '1'
# psnr_2y_np = PSNR_np_Batch(im , jm, cal_type = color)
# print(f"psnr_2y_np = {psnr_2y_np}")

# PSNR_2y_ts  = PSNR_torch_Batch(im, jm,  cal_type = color)
# print(f"PSNR_2y_ts = {PSNR_2y_ts}")

# psnr_ipt = calc_psnr(im, jm, cal_type=color)
# print(f"psnr_ipt = {psnr_ipt}")


# # psnr2y_np = PSNR_2y_np_RGBchannelLast(im.transpose( 0,2,3,1), jm.transpose( 0,2,3,1))  # ( 1,2,0 )  0,2,3,1
# # print(f"psnr2y_np = {psnr2y_np}")

# # avgpsnr_np, Psnr_np, size_np = PSNR_np_Image(im, jm)
# # print(f"avgpsnr_np = {avgpsnr_np}, Psnr_np = {Psnr_np}, size_np = {size_np}")


# # avgpsnr_ts, Psnr_ts, size_ts = PSNR_torch_Image(im, jm)
# # print(f"avgpsnr_ts = {avgpsnr_ts}, Psnr_ts = {Psnr_ts}, size_ts = {size_ts}\n\n")


# def data_transform(x):
#     x = np.array(x, dtype='float32') / 255
#     x = (x - 0.5) / 0.5
#     # x = x / 0.5
#     x = x.reshape((-1,))
#     x = torch.from_numpy(x)
#     return x
# import torch

# im1 = data_transform(im)
# jm1 = data_transform(jm)
# cr1 = torch.nn.MSELoss()
# mse = cr1(jm1, im1)
# out_np = jm1.detach().numpy()
# psnr = 10 * np.log10(np.max(out_np) ** 2 / mse.detach().numpy() ** 2)
# print(f"psnr = {psnr}")


# # #===================================================================
import  torchvision
# root='/home/jack/公共的/MLData/CIFAR10'
root='/home/jack/公共的/MLData/'
# root='/home/jack/公共的/MLData/FashionMNIST'

def PSNR_torch(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    maxp = max(im.max(), jm.max() )
    cr1 = torch.nn.MSELoss()
    mse = cr1(jm, im).cpu()
    out_np = jm.detach().numpy()
    psnr = 10 * np.log10(np.max(out_np) ** 2  / mse.detach().numpy() ** 2)
    return  psnr

def PSNR_torch1(im, jm, ):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = torch.Tensor(im), torch.Tensor(jm)
    maxp = max(im.max(), jm.max() )
    cr1 = torch.nn.MSELoss()
    mse = cr1(jm, im).cpu()
    # out_np = X_hat.detach().numpy()
    psnr = 10 * np.log10(maxp ** 2 / mse.detach().numpy() ** 2)
    return  psnr

def data_transform(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    # x = x / 0.5
    x = x.reshape((1, 28, 28))
    x = torch.from_numpy(x)
    return x


def data_inv_transform(x):
    """
    :param x:
    :return:
    """
    recover_data = x * 0.5 + 0.5
    # recover_data = x * 0.5
    recover_data = recover_data * 255
    recover_data = recover_data.reshape((-1, 1, 28, 28))
    recover_data = recover_data.detach().numpy()
    return recover_data

batch_size = 128


def data_tf(x):
    x = np.array(x, dtype='float32')
    x = x * 1.0
    return x


# FashionMNIST  MNIST  CIFAR10
trainset =  datasets.MNIST(root = root, train = True,  download = True,   transform = data_transform ) # 表示是否需要对数据进行预处理，none为不进行预处理
testset  =  datasets.MNIST(root = root, train = False,  download = True, transform = data_transform )

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 0

train_iter = DataLoader(trainset, batch_size=batch_size, shuffle = False,  )
test_iter = DataLoader(testset, batch_size=batch_size, shuffle  = False, )

print(f"len(trainset) = {len(trainset)}, len(testset) = {len(testset)}, len(train_iter) = {len(train_iter)}, len(test_iter) = {len(test_iter)}")
# batch_size = 25, len(trainset) = 60000, len(testset) = 10000, len(train_iter) = 2400, len(test_iter) = 400

PSNR_torch_batch = 0
Psnr_np_simple = 0
PSNR_np_batch = 0


avgpsnr_torch_image = 0
PSNR_torch_image = 0
size_torch = 0

avgpsnr_np_image = 0
PSNR_np_image = 0
size_np = 0

psnr_sh = 0



color = '1'
for epoch, (X, y) in enumerate(train_iter):
      # print(f"X.shape = {X.shape}, y.shape = {y.shape}")
      # draw_images1(tmpout, X,  epoch, 1, H = 32, W = 32, examples = 25,  dim = (5, 5), figsize = (10, 10))
      # plt.imshow(X[0].numpy().transpose(1,2,0))
      X_hat = X + torch.randn(size = X.size() ) * 0.1
      X_hat = torch.clamp(X_hat, -1, 1,  )

      psnr_sh +=  PSNR_torch1(X, X_hat)

      X1 = data_inv_transform(X)
      X_hat1 = data_inv_transform(X_hat)

      # PSNR
      PSNR_torch_batch += PSNR_torch_Batch(X1 , X_hat1 ,  cal_type = color)
      PSNR_np_batch    += PSNR_np_Batch(X1 , X_hat1 ,  cal_type = color)
      Psnr_np_simple   += PSNR_np_simple(X1, X_hat1)

      a, b, c  = PSNR_torch_Image(X1 , X_hat1 )
      avgpsnr_torch_image += a
      PSNR_torch_image += b
      size_torch += c

      d, e, f  = PSNR_np_Image(X1 , X_hat1 )
      avgpsnr_np_image += d
      PSNR_np_image += e
      size_np += f

      # # MSE
      # MSE_torch_batch += MSE_torch_Batch( X*255, X_hat*255,)
      # MSE_np_batch += MSE_np_Batch( X*255, X_hat*255,)


      # a, b, c  = MSE_torch_Image(X*255, X_hat*255)
      # avgMSE_torch_image += a
      # MSE_torch_image += b
      # size_torch2 += c

      # d, e, f  = MSE_np_Image(X*255, X_hat*255)
      # avgMSE_np_image += d
      # MSE_np_image += e
      # size_np2 += f

avgpsnr_sh = psnr_sh / len(train_iter)
print(f"00  avgpsnr_sh = {avgpsnr_sh}, psnr_sh = {psnr_sh},  len(train_iter) = { len(train_iter)} \n")



# PSNR
avgpsnr_np_simple = Psnr_np_simple / len(train_iter)
print(f"1 avgpsnr_np_simple = {avgpsnr_np_simple}, Psnr_np_simple = {Psnr_np_simple},  len(train_iter) = { len(train_iter)} \n")


avgpsnr_torch_batch = PSNR_torch_batch / len(train_iter)
print(f"2 avgpsnr_torch_batch = {avgpsnr_torch_batch}, PSNR_torch_batch = {PSNR_torch_batch},  len(train_iter) = { len(train_iter)} \n")

avgpsnr_np_batch = PSNR_np_batch / len(train_iter)
print(f"3 avgpsnr_np_batch = {avgpsnr_np_batch}, PSNR_np_batch = {PSNR_np_batch},  len(train_iter) = { len(train_iter)} \n")

avgpsnr_torch_image_1 = avgpsnr_torch_image / len(train_iter)
avgpsnr_torch_image_2 = PSNR_torch_image / size_torch
print(f"4 avgpsnr_torch_image_2 = {avgpsnr_torch_image_2}, avgpsnr_torch_image_1 = {avgpsnr_torch_image_1},  PSNR_torch_image = {PSNR_torch_image}, size_torch = {size_torch}, len(train_iter) = { len(train_iter)} \n")

avgpsnr_np_image_1 = avgpsnr_np_image / len(train_iter)
avgpsnr_np_image_2 = PSNR_np_image / size_np
print(f"5 avgpsnr_np_image_2 = {avgpsnr_np_image_2}, avgpsnr_np_image_1 = {avgpsnr_np_image_1},  PSNR_np_image = {PSNR_np_image}, size_np = {size_np},  len(train_iter) = { len(train_iter)} \n")

# # MSE
# avgmse_torch_batch = MSE_torch_batch / len(train_iter)
# print(f"1 avgmse_torch_batch = {avgmse_torch_batch}, MSE_torch_batch = {MSE_torch_batch},  len(train_iter) = { len(train_iter)} \n")

# avgmse_np_batch = MSE_np_batch / len(train_iter)
# print(f"2 avgmse_np_batch = {avgmse_np_batch}, MSE_np_batch = {MSE_np_batch},  len(train_iter) = { len(train_iter)} \n")

# avgmse_torch_image_1 = avgMSE_torch_image / len(train_iter)
# avgmse_torch_image_2 = MSE_torch_image / size_torch2
# print(f"3 avgmse_torch_image_2 = {avgmse_torch_image_2}, avgmse_torch_image_1 = {avgmse_torch_image_1},  MSE_torch_image = {MSE_torch_image}, size_torch2 = {size_torch2}, len(train_iter) = { len(train_iter)} \n")

# avgmse_np_image_1 = avgMSE_np_image / len(train_iter)
# avgmse_np_image_2 = MSE_np_image / size_np2
# print(f"4 avgmse_np_image_2 = {avgmse_np_image_2}, avgmse_np_image_1 = {avgmse_np_image_1},  MSE_np_image = {MSE_np_image}, size_np2 = {size_np2},  len(train_iter) = { len(train_iter)} \n")








