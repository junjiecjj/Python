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



import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
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
    l = 1
    r = 100
    cnt = 0
    while l < r:
        tmp_img = f"{user_home}/tmp/Mnist/%s_%s.jpg"%(filename, cnt)
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

    # im, jm = np.float64(im), np.float64(jm)
    rgb_range = 255.0  # max(im.max(), jm.max()) * 1.0
    mse = np.mean((im * 1.0 - jm * 1.0)**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(rgb_range**2 / mse)
    return psnr


def JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, savedir, compress, SNR, caltype = 'y', channel = '3', ):
    print(f"Compress Rate =  {compress}")
    print(f"SNR           =  {SNR}\n")
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logfile = f"./Capacity-achieve_mnist_{now}.txt"
    f = open(logfile,  mode = 'a+')

    # logfileq = "./Capacity-achieve_q.txt"
    #fq = open(logfileq,  mode = 'w+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    f.write(f" cap = 0.5 * math.log2(1 + 10**(snr/10.0) )  \n")
    f.write(f" filesize = cap * comp * im.size / 4 \n")
    f.write(f" no R_min \n")
    f.write(f" Raw data: bmp \n")
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
            print(f"  compress rate = {comp:.1f}")
            f.write(f"[ ")
            for j, snr in enumerate(SNR):
                PSNR = 0.0
                print( f"    snr = {snr:d}(dB)", end=',  ')
                folder = savedir + d + "comp={:.1f}".format(comp) + "/snr={}(dB)/".format(snr)
                os.makedirs(folder, exist_ok = True)
                cap = 0.5 * math.log2(1 + 10**(snr/10.0) )
                # cap = 0.5 * math.log2(1 + snr )

                Filesize = 0.0
                for file in files:
                    filename = os.path.splitext(os.path.basename(file))[0]
                    savename = folder + f"comp={comp:.1f}_snr={snr}(dB)_{filename}.jpg"
                    im = imageio.imread(file, )
                    # print(f"im.shape = {im.shape}")
                    assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "
                    fsize = os.path.getsize(file)
                    filesize = cap * comp * im.size / 4.0
                    correct_quality = getSaveQuality_imageio(im, filesize, filename)

                    imageio.imwrite(savename, im, quality = correct_quality )
                    jm = imageio.imread(savename, )
                    if correct_quality <= 1:
                        mean = im.mean()
                        jm = np.ones_like(im)*mean

                    Filesize += os.path.getsize(savename)
                    PSNR += PSNR_np_simple(im, jm)
                PSNR /= len(files)
                avgfsize = Filesize/len(files)
                f.write(f"{PSNR:.3f}, ")
                f.flush()
                print(f"ave psnr = {PSNR:.2f}(dB), avgfsize = {avgfsize}")
                PSNR_res[f"{d}"][i, j] = PSNR
            f.write(" ],\n")
            f.flush()
    f.close()
    #fq.close()

    print(f"logfile = {logfile}")
    return PSNR_res


root = f"{user_home}/公共的/MLData/"
trainset =  datasets.MNIST(root = root, train = True,  download = True, transform = None)
testset =  datasets.MNIST(root = root, train = False, download = True,  transform = None)


## test
# pngdir_test = '/home/jack/SemanticNoise_AdversarialAttack/Data/Mnist_test_bmp/'
# deletemkdir(pngdir_test)

# for idx, (img, y) in  enumerate(testset):
#     print(f"{idx},  img.size = {img.size}, y  = {y }")
#     name = f"{idx}_{y}.bmp"  # .png   bmp
#     img.save(pngdir_test + name,)
#     ### imageio.v2.imwrite(pngdir_test + name, img,   )



##  JPEG + Capacity
compress = np.arange(0.1, 1.0, 0.1)
SNR = np.arange(-2, 21, 1)
SNR = np.append(SNR, 25)
SNR = np.append(SNR, 30)
SNR = np.append(SNR, 35)
SNR = np.append(SNR, 40)
SNR.sort()

rootDir = f"{user_home}/SemanticNoise_AdversarialAttack/Data/"
DataSetFolder = [ 'Mnist_test_bmp/', ] # ' Mnist_train_png  Mnist_train_bmp

savetmp = f"{user_home}/SemanticNoise_AdversarialAttack/Data/tmpdata_mnist/"
deletemkdir(savetmp)

PSNR_mnist = JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, savetmp, compress, SNR, caltype = '1', channel = '1',)











































































































































































































































































































































































































































































































