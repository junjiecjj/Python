#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:43:33 2023

@author: jack
"""


import numpy as np
from torchvision import datasets
import os, sys
import math
import datetime
import imageio

import glob


import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


# from ..Trainer import common
# import Utility

import os, sys
import shutil

def deletemkdir(path):
    if os.path.exists(path):
        # print(f"删除 {path} 文件夹！")
        if sys.platform.startswith("win"):
            shutil.rmtree(path)
        else:
            os.system(f"rm -r {path}")
    # print(f"创建 {path} 文件夹！")
    os.makedirs(path, exist_ok = True)
    return


#============================================================================================================================
#                                                    Channel Capacity Achieve code
#============================================================================================================================


def getSaveQuality_imageio(image, filesize, filename):
    l = 0
    r = 100
    cnt = 0
    while l < r:
        tmp_img = "/home/jack/tmp/cifar10/%s_%s.jpg"%(filename, cnt)
        m = math.ceil((l + r) * 1.0/2)
        imageio.imwrite(tmp_img, image, quality = m)
        # time.sleep(0.100)
        fsize = os.path.getsize(tmp_img)
        if fsize <= filesize:
            l = m
        else:
            r = m - 1
        cnt += 1
    m = l

    return m


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')
    rgb_range = 255.0   # max(im.max(), jm.max()) * 1.0
    im, jm = np.float64(im), np.float64(jm)
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(rgb_range**2 / mse)
    return psnr


def JPEG_withoutCapacity_imageio(rootDir, DataSetFolder, savedir, compress, caltype = 'y', channel = '3', ):
    print(f"Compress Rate =  {compress}")
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logfile = f"./Capacity-achieve_cifar10_{now}.txt"
    f = open(logfile,  mode = 'a+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    f.write(f" Without Capacity  \n")
    f.write(f" filesize =  comp * im.size \n")
    f.write(f" no R_min \n")
    f.write(f" Raw data: bmp \n")

    # f.close()
    PSNR_res = {}

    ext = [".png", ".jpg", ".bmp"]
    for d in  DataSetFolder:
        f.write(f"\n{'Dataset':<15}: {d} \n")
        f.write(f"{'Compress Rate':<15}: {compress} \n")
        # f.write(f"{'SNR':<15}: {SNR} \n\n")
        f.flush()

        print(f"Dataset: {d}")
        PSNR_res[f"{d}"] = np.zeros((1, len(compress) ))
        files = sorted(glob.glob(os.path.join(rootDir, d, "*" + ext[2])))
        # print(f"files  = {files }")
        f.write(f"[  ")
        for i, comp in enumerate(compress):
            print(f"  compress rate = {comp:.2f}")
            PSNR = 0.0
            folder = savedir + d + "comp={:.2f}/".format(comp)
            os.makedirs(folder, exist_ok = True)

            Filesize = 0.0
            for file in files:
                filename = os.path.splitext(os.path.basename(file))[0]
                savename = folder + f"comp={comp:.2f}_{filename}.jpg"
                im = imageio.v2.imread(file, )
                # print(f"im = {im.shape}")
                assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "
                fsize = os.path.getsize(file)
                filesize =  comp * im.size  #  im.size  fsize
                correct_quality = getSaveQuality_imageio(im, filesize, filename)
                imageio.imwrite(savename, im, quality = correct_quality )
                fsize_save = os.path.getsize(savename)
                Filesize += fsize_save
                jm = imageio.v2.imread(savename, )
                # 考虑当 压缩质量 quality = 0 时, 以原图每个通道的均值来代替恢复的图片, 参考文献： "Deep Joint Source-Channel Coding for Wireless Image Transmission"
                # if correct_quality <= 0:
                #     mean = im.mean(axis = (0,1))
                #     jm = np.ones_like(im)*mean
                # else:
                #     imageio.imwrite(savename, im, quality = correct_quality )
                #     jm = imageio.v2.imread(savename, )

                # print(f"{im.shape}, im.max() = {im.max()}, jm.max() = {jm.max()}")
                PSNR +=  PSNR_np_simple(im, jm )  # 10.0 * math.log10(255.0**2 / MSE(im, jm))
            PSNR /= len(files)
            avgfsize = Filesize/len(files)
            f.write(f"{PSNR:.3f}, ")
            f.flush()
            print(f"    ave psnr = {PSNR:.2f}, avgfsize = {avgfsize}")
            PSNR_res[f"{d}"][0, i] = PSNR
        f.write(" ] \n")
        f.flush()
    f.close()
    print(f"logfile = {logfile}")
    return PSNR_res


# root = '/home/jack/公共的/MLData/CIFAR10'
# trainset =  datasets.CIFAR10(root = root, train = True,  download = True, transform = None)
# testset =  datasets.CIFAR10(root = root, train = False, download = True,  transform = None)


# pngdir_test = '/home/jack/SemanticNoise_AdversarialAttack/Data/Cifar10_testset_png/'
# deletemkdir(pngdir_test)

# for idx, (img, y) in  enumerate(testset):
#     print(f"{idx},  img.shape = {img.size}, y  = {y }, {np.array(img).max()}")   # X的每个元素都是 0 - 1的.
#     name = f"{idx}_{y}.png" # png  bmp  jpg
#     img.save(pngdir_test + name, quality = 100)  # , quality = 100
#     ##X.show()
#     ##draw_images1(tmpout, X,  epoch, 1, H = 28, W = 28, examples = 28,  dim = (5, 5), figsize = (10, 10))


# pngdir_train = '/home/jack/SemanticNoise_AdversarialAttack/Data/Cifar10_train_png/'
# deletemkdir(pngdir_train)


# for idx, (img, y) in  enumerate(trainset):
#     print(f"{idx},  img.shape = {img.size}, y  = {y }")   # X的每个元素都是 0 - 1的.
#     name = f"{idx}_{y}.png"
#     img.save(pngdir_train + name, quality = 100)
#     ## X.show()
#     ## draw_images1(tmpout, X,  epoch, 1, H = 28, W = 28, examples = 28,  dim = (5, 5), figsize = (10, 10))



compress = np.arange(0.1, 1.0, 0.1)
# SNR = np.arange(0, 21, 1)
# SNR = np.append(SNR, 20)
# SNR = np.append(SNR, 30)
# SNR = [0, 1, 3, 8, 10, 12, 20]


rootDir = "/home/jack/SemanticNoise_AdversarialAttack/Data/"
DataSetFolder = [ 'Cifar10_testset_bmp/', ] #  Cifar10_testset_png/  Cifar10_testset_jpg/   Cifar10_testset_bmp/

savetmp = "/home/jack/SemanticNoise_AdversarialAttack/Data/tmpdata_cifar10/"
deletemkdir(savetmp)

PSNR_cifar10 = JPEG_withoutCapacity_imageio(rootDir, DataSetFolder, savetmp, compress,  caltype = '1', channel = '3',)











































































































































































































































































































































































































































































































