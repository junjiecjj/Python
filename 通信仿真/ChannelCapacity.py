#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/12a8207149b0


import numpy as np
import sys, os
import imageio
import cv2
import skimage
import math
import time, datetime
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def getSaveQuality_imageio(image, filesize, filename):
    l = 0
    r = 100
    cnt = 0
    tmp_img = "/tmp/%s.jpg"%filename
    while l < r:
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

    return l

def getSaveQuality_cv2(image, filesize, filename):
    l = 0
    r = 100
    cnt = 0
    tmp_img = "/tmp/%s.jpg"%filename
    while l < r:
        m = math.ceil((l + r)/2)
        cv2.imwrite(tmp_img, image, [int(cv2.IMWRITE_JPEG_QUALITY), m]) # imageio.imwrite(tmp_img, image, quality = q)
        # time.sleep(0.100)
        fsize = os.path.getsize(tmp_img)
        if fsize <= filesize:
            l = m
        else:
            r = m - 1
        cnt += 1
    m = l

    return l

def getSaveQuality_skimage(image, filesize, filename):
    l = 0
    r = 100
    cnt = 0
    tmp_img = "/tmp/%s.jpg"%filename
    while l < r:
        m = math.ceil((l + r)/2)
        skimage.io.imsave(tmp_img, image, quality = m)
        # time.sleep(0.100)
        fsize = os.path.getsize(tmp_img)
        if fsize <= filesize:
            l = m
        else:
            r = m - 1
        cnt += 1
    m = l

    return l

def MSE(X, Y):
    mse = np.mean((X - Y)**2)
    return mse

def PSNR_np_2y(im, jm, rgb_range = 255.0, cal_type='y'):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    G = np.array( [65.481,   128.553,    24.966] )
    G = G.reshape(3, 1, 1)

    ## 方法1
    diff = (im / 1.0 - jm / 1.0) / rgb_range

    if cal_type == 'y':
        diff = diff * G
        diff = np.sum(diff, axis = -3) / rgb_range

    mse = np.mean(diff**2)
    if mse <= 1e-20:
        mse = 1e-20
    psnr = -10.0 * math.log10(mse)
    return psnr




def JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, compress, SNR):
    logfile = "./Capacity-achieve.txt"
    f = open(logfile,  mode = 'w+')

    logfileq = "./Capacity-achieve_q.txt"
    #fq = open(logfileq,  mode = 'w+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    # f.close()
    PSNR_res = {}

    ext = [".png", ".jpg"]
    for d in  DataSetFolder:
        f.write(f"\n{'Dataset':<15}: {d} \n")
        f.write(f"{'Compress Rate':<15}: {compress} \n")
        f.write(f"{'SNR':<15}: {SNR} \n\n")
        f.flush()

        print(f"Dataset: {d}")
        PSNR_res[f"{d}"] = np.zeros((len(compress), len(SNR) ))
        files = sorted(glob.glob(os.path.join(rootDir, d, "*" + ext[0])))
        for i, comp in enumerate(compress):
            print(f"  compress rate = {comp:.2f}")
            for j, snr in enumerate(SNR):
                PSNR = 0.0
                print( f"    snr = {snr:.2f}", end=',  ')
                folder = "/home/jack/IPT-Pretrain-DataResults/CapacityAchievePicture_py/" + d + "comp={:.2f}".format(comp) + "/snr={:.2f}/".format(snr)
                os.makedirs(folder, exist_ok = True)
                cap = 0.5 * math.log2(1 + 10**(snr/10.0) )
                for file in files:
                    filename = os.path.splitext(os.path.basename(file))[0]
                    savename = folder + f"comp={comp:.2f}_snr={snr:.2f}_{filename}.jpg"
                    im = imageio.v2.imread(file, )
                    assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "

                    filesize = cap * comp * im.size / 8
                    correct_quality = getSaveQuality_imageio(im, filesize, filename)
                    #fq.write(f"compress rate = {comp:6.2f}, snr = {snr:6.2f}, filename = {filename:15s}, quality = {correct_quality:3d}\n")
                    imageio.imwrite(savename, im, quality = correct_quality )
                    # time.sleep(0.100)
                    jm = imageio.v2.imread(savename, )
                    PSNR +=  PSNR_np_2y(im.transpose(2,0,1), jm.transpose(2,0,1), cal_type='1')  # 10.0 * math.log10(255.0**2 / MSE(im, jm))
                PSNR /= len(files)
                f.write(f"{PSNR:.3f}, ")
                f.flush()
                print(f"ave psnr = {PSNR:.2f}")
                PSNR_res[f"{d}"][i, j] = PSNR
            f.write("\n")
            f.flush()

    f.close()
    #fq.close()
    return PSNR_res

def JPEG_CapacityAchieve_cv2(rootDir, DataSetFolder, compress, SNR):
    logfile = "./Capacity-achieve.txt"
    f = open(logfile,  mode = 'w+')

    logfileq = "./Capacity-achieve_q.txt"
    #fq = open(logfileq,  mode = 'w+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    # f.close()
    PSNR_res = {}

    ext = [".png", ".jpg"]
    for d in  DataSetFolder:
        f.write(f"\n{'Dataset':<15}: {d} \n")
        f.write(f"{'Compress Rate':<15}: {compress} \n")
        f.write(f"{'SNR':<15}: {SNR} \n\n")
        f.flush()

        print(f"Dataset: {d}")
        PSNR_res[f"{d}"] = np.zeros((len(compress), len(SNR) ))
        files = sorted(glob.glob(os.path.join(rootDir, d, "*" + ext[0])))
        for i, comp in enumerate(compress):
            print(f"  compress rate = {comp:.2f}")
            for j, snr in enumerate(SNR):
                PSNR = 0.0
                print( f"    snr = {snr:.2f}", end=',  ')
                folder = "/home/jack/IPT-Pretrain-DataResults/CapacityAchievePicture_py/" + d + "comp={:.2f}".format(comp) + "/snr={:.2f}/".format(snr)
                os.makedirs(folder, exist_ok = True)
                cap = 0.5 * math.log2(1 + 10**(snr/10.0) )
                for file in files:
                    filename = os.path.splitext(os.path.basename(file))[0]
                    savename = folder + f"comp={comp:.2f}_snr={snr:.2f}_{filename}.jpg"
                    im = cv2.imread(file, )  # im = imageio.v2.imread(file, )
                    assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "

                    filesize = cap * comp * im.size/8.0
                    correct_quality = getSaveQuality_cv2(im, filesize, filename)
                    #fq.write(f"compress rate = {comp:6.2f}, snr = {snr:6.2f}, filename = {filename:15s}, quality = {correct_quality:3d}\n")
                    cv2.imwrite(savename, im, [int(cv2.IMWRITE_JPEG_QUALITY), correct_quality]) #  cv2.imwrite(savename, im, quality = correct_quality )
                    # time.sleep(0.100)
                    jm = cv2.imread(savename, ) # jm = imageio.v2.imread(savename, )
                    PSNR += PSNR_np_2y(im.transpose(2,0,1), jm.transpose(2,0,1), cal_type='1')  # 10.0 * math.log10(255.0**2 / MSE(im, jm))
                PSNR /= len(files)
                f.write(f"{PSNR:.3f}, ")
                f.flush()
                print(f"ave psnr = {PSNR:.2f}")
                PSNR_res[f"{d}"][i, j] = PSNR
            f.write("\n")
            f.flush()
    f.flush()
    f.close()
    #fq.close()
    return PSNR_res


def JPEG_CapacityAchieve_skimage(rootDir, DataSetFolder, compress, SNR):
    logfile = "./Capacity-achieve.txt"
    f = open(logfile,  mode = 'w+')

    logfileq = "./Capacity-achieve_q.txt"
    #fq = open(logfileq,  mode = 'w+')

    now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    f.write(f"\n\n=====================================================================================\n")
    f.write(f"                               {now1}\n")
    f.write("======================================================================================\n\n")
    # f.close()
    PSNR_res = {}

    ext = [".png", ".jpg"]
    for d in  DataSetFolder:
        f.write(f"\n{'Dataset':<15}: {d} \n")
        f.write(f"{'Compress Rate':<15}: {compress} \n")
        f.write(f"{'SNR':<15}: {SNR} \n\n")
        f.flush()

        print(f"Dataset: {d}")
        PSNR_res[f"{d}"] = np.zeros((len(compress), len(SNR) ))
        files = sorted(glob.glob(os.path.join(rootDir, d, "*" + ext[0])))
        for i, comp in enumerate(compress):
            print(f"  compress rate = {comp:.2f}")
            for j, snr in enumerate(SNR):
                PSNR = 0.0
                print( f"    snr = {snr:.2f}", end=',  ')
                folder = "/home/jack/IPT-Pretrain-DataResults/CapacityAchievePicture_py/" + d + "comp={:.2f}".format(comp) + "/snr={:.2f}/".format(snr)
                os.makedirs(folder, exist_ok = True)
                cap = 0.5 * math.log2(1 + 10**(snr/10.0) )
                for file in files:
                    filename = os.path.splitext(os.path.basename(file))[0]
                    savename = folder + f"comp={comp:.2f}_snr={snr:.2f}_{filename}.jpg"
                    im = skimage.io.imread(file, )
                    assert (im.dtype == 'uint8'), "图片的dtype不为 'uint8' "

                    filesize = cap * comp * im.size/8.0
                    correct_quality = getSaveQuality_cv2(im, filesize, filename)
                    # fq.write(f"compress rate = {comp:6.2f}, snr = {snr:6.2f}, filename = {filename:15s}, quality = {correct_quality:3d}\n")
                    skimage.io.imsave(savename, im, quality = correct_quality )
                    # time.sleep(0.100)
                    jm = skimage.io.imread(savename, )
                    PSNR += 10.0 * math.log10(255.0**2 / MSE(im, jm))
                PSNR /= len(files)
                f.write(f"{PSNR:.3f}, ")
                f.flush()
                print(f"ave psnr = {PSNR:.2f}")
                PSNR_res[f"{d}"][i, j] = PSNR
            f.write("\n")
            f.flush()
    f.flush()
    f.close()
    #fq.close()
    return PSNR_res



compress = np.arange(1, 1.1, 0.1)
SNR = np.arange(10, 11, 1)

rootDir = "/home/jack/IPT-Pretrain-DataResults/Data/benchmark/"
DataSetFolder = [ 'B100/HR/', 'Set5/HR/'] # 'B100/HR/',




PSNR_res = JPEG_CapacityAchieve_imageio(rootDir, DataSetFolder, compress, SNR)
# PSNR_res = JPEG_CapacityAchieve_cv2(rootDir, DataSetFolder, compress, SNR)
# PSNR_res = JPEG_CapacityAchieve_skimage(rootDir, DataSetFolder, compress, SNR)




















































