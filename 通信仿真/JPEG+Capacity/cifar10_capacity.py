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

import os, sys
import shutil

import glob
import socket, getpass , os
import numpy as np


# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')


def deletemkdir(path):
    if os.path.exists(path):
        print(f"删除 {path} 文件夹！")
        if sys.platform.startswith("win"):
            shutil.rmtree(path)
        else:
            os.system(f"rm -r {path}")
    print(f"创建 {path} 文件夹！")
    os.makedirs(path, exist_ok = True)
    return


#============================================================================================================================
#                                                    Channel Capacity Achieve code
#============================================================================================================================


def getSaveQuality_imageio(image, filesize, filename):
    l = 1
    r = 100
    cnt = 0
    while l < r:
        tmp_img = f"{user_home}/tmp/cifar10/%s_%s.jpg"%(filename, cnt)
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


def JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, savedir, compress, SNR, caltype = 'y', channel = '3', ):
    print(f"Compress Rate =  {compress}")
    print(f"SNR           =  {SNR}\n")
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logfile = f"./Capacity-achieve_cifar10_{now}.txt"
    f = open(logfile,  mode = 'a+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    f.write(f" cap = 0.5 * math.log2(1 + 10**(snr/10.0) )  \n")
    f.write(f" filesize = cap * comp * im.size / 4 \n")
    f.write(f" no R_min \n")
    f.write(f" Raw data: png \n")
    f.flush()

    PSNR_res = {}

    ext = [".png", ".bmp"]
    for d in  DataSetFolder:
        f.write(f"\n{'Dataset':<15}: {d} \n")
        f.write(f"{'Compress Rate':<15}: {compress} \n")
        f.write(f"{'SNR':<15}: {SNR} \n\n")
        f.flush()

        print(f"Dataset: {d}")
        PSNR_res[f"{d}"] = np.zeros((len(compress), len(SNR) ))
        files = sorted(glob.glob(os.path.join(rootDir, d, "*" + ext[1])))
        # print(f"files  = {files }")
        for i, comp in enumerate(compress):
            print(f"  compress rate = {comp:.2f}")
            f.write(f"[ ")
            for j, snr in enumerate(SNR):
                PSNR = 0.0
                print( f"    snr = {snr:.2f}(dB)", end=',  ')
                folder = savedir + d + "comp={:.2f}".format(comp) + "/snr={:.2f}/".format(snr)
                os.makedirs(folder, exist_ok = True)
                cap = 0.5 * math.log2(1 + 10**(snr/10.0) )  # 10**(snr/10.0)
                # if snr >= 9.0:
                    # cap = 1
                Filesize = 0.0
                for file in files:
                    filename = os.path.splitext(os.path.basename(file))[0]
                    savename = folder + f"comp={comp:.2f}_snr={snr:.2f}_{filename}.jpg"
                    im = imageio.imread(file, )
                    # print(f"im = {im.shape}")
                    assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "
                    fsize = os.path.getsize(file)
                    filesize = cap * comp * im.size / 4.0 #  im.size  fsize
                    correct_quality = getSaveQuality_imageio(im, filesize, filename)
                    imageio.imwrite(savename, im, quality = correct_quality )
                    Filesize +=  os.path.getsize(savename)
                    jm = imageio.imread(savename, )
                    ### 考虑当 压缩质量 quality = 0 时, 以原图每个通道的均值来代替恢复的图片, 参考文献： "Deep Joint Source-Channel Coding for Wireless Image Transmission"
                    if correct_quality <= 1:
                        mean = im.mean(axis = (0, 1))
                        jm = np.ones_like(im)*mean

                    # print(f"{im.shape}, im.max() = {im.max()}, jm.max() = {jm.max()}")
                    PSNR +=  PSNR_np_simple(im, jm )  # 10.0 * math.log10(255.0**2 / MSE(im, jm))
                PSNR /= len(files)
                avgfsize = Filesize/len(files)
                f.write(f"{PSNR:.3f}, ")
                f.flush()
                print(f"ave psnr = {PSNR:.2f}(dB), avgfsize = {avgfsize}")
                PSNR_res[f"{d}"][i, j] = PSNR
            f.write(" ],\n")
            f.flush()
    f.close()
    print(f"logfile = {logfile}")
    return PSNR_res


# root = f"{user_home}/公共的/MLData/CIFAR10"
# trainset =  datasets.CIFAR10(root = root, train = True,  download = True, transform = None)
# testset =  datasets.CIFAR10(root = root, train = False, download = True,  transform = None)


# pngdir_test = f"{user_home}/SemanticNoise_AdversarialAttack/Data/Cifar10_testset_bmp/"
# deletemkdir(pngdir_test)

# for idx, (img, y) in  enumerate(testset):
#     print(f"{idx},  img.shape = {img.size}, y  = {y }, {np.array(img).max()}")   # X的每个元素都是 0 - 1的.
#     name = f"{idx}_{y}.bmp" # png  bmp  jpg
#     img.save(pngdir_test + name, quality = 100)  # , quality = 100
#     ##X.show()
#     ##draw_images1(tmpout, X,  epoch, 1, H = 28, W = 28, examples = 28,  dim = (5, 5), figsize = (10, 10))



##  JPEG + Capacity
compress = np.arange(0.1, 1.0, 0.1)
SNR = np.arange(-2, 21, 1)
SNR = np.append(SNR, 25)
SNR = np.append(SNR, 30)
SNR = np.append(SNR, 35)
SNR = np.append(SNR, 40)
SNR.sort()


rootDir = f"{user_home}/SemanticNoise_AdversarialAttack/Data/"
DataSetFolder = [ 'Cifar10_testset_bmp/', ] #  Cifar10_testset_png/  Cifar10_testset_jpg/   Cifar10_testset_bmp/

savetmp = f"{user_home}/SemanticNoise_AdversarialAttack/Data/tmpdata_cifar10/"
deletemkdir(savetmp)

PSNR_cifar10 = JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, savetmp, compress, SNR, caltype = '1', channel = '3',)




































































































































































































































































































































































































































































































