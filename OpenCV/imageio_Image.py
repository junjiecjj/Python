#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:23:16 2023

@author: jack
https://blog.csdn.net/hfuter2016212862/article/details/104763385


Install:
pip install imageio
conda install -c conda-forge imageio

(二) imageio:

import imageio


(1) 读取
    image = imageio.imread(imagepath)
    type(image) = <class 'imageio.core.util.Array'>
(2) 展示
    无显示方法, 但其读取后的类型为numpy数据, 故可用之后的plt.imshow()显示.
(3) 保存
    imageio.imsave('image.jpg', image, quality=75)
    imageio.imsave('./Figures/imageio1.png',im2, quality=75)
    imageio.imwrite('./Figures/imageio2.png',im2, quality=75)  # 与imsave完全一样

imageio 可以读取/写入 Bmp 格式的图片

"""



from scipy import misc, ndimage
import imageio

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

#=================================================================================================
# imageio读取，plt画图(正常),cv2画图(正常),
im2 = imageio.imread('./Figures/lena.png')  # 读取为RGB
plt.imshow(im2)
cv2.imshow('src',im2[:,:,::-1]) # 将RGB转换为BGR
cv2.waitKey()
cv2.destroyAllWindows()

#=================================================================================================
# imageio读取，imageio保存,imageio再读取，plt画图(正常),cv2画图(正常),
im2 = imageio.imread('./Figures/lena.png')  # 读取为RGB
imageio.imwrite('./Figures/imageio1.png',im2)
im2 = imageio.imread('./Figures/imageio1.png')  # 读取为RGB
plt.imshow(im2)
cv2.imshow('src',im2[:,:,::-1]) # 将RGB转换为BGR
cv2.waitKey()
cv2.destroyAllWindows()



#=================================================================================================
# 实验20
# imageio读取,plt画图(正常)，cv2画图(正常), imageio保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2imageio.png',img)  # 保存时为BGR


img1 = imageio.imread('./Figures/lenaPltCv2imageio.png')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8

plt.imshow(img1)  # 将BGR转成RGB



# 实验 17
#==================================================
# imageio 读，乘以1.0，再 imageio 保存，再 imageio 读
#==================================================
img = imageio.imread('./Figures/baby.png', )
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img,)


img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)



#==================================================
# imageio 但彩图读进的是RGB，与opencv有区别
#==================================================
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('./Figures/lena.png')  # 读取为BGR，
im1 = im[:,:,::-1]

# imageio读取，

im2 = imageio.imread('./Figures/lena.png')  # 读取为RGB
cv2.imshow('src',im2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
plt.imshow(im2)
print(f"im2.dtype = {im2.dtype}\n")
print(f"im2.size = {im2.size}\n")
print(f"im2.shape = {im2.shape}\n")



diff = im2 - im1

print(f"diff.max() = {diff.max()}, diff.min() = {diff.min()}")
# 可以看出, imageio 和 CV2 除了通道不一样，其他基本一样，像素点可能差 1;


imageio.imsave('./Figures/imageio1.png',im2)
imageio.imwrite('./Figures/imageio2.png',im2)  # 与imsave完全一样


im2_1 = imageio.imread('./Figures/imageio1.png')  # 读取为RGB
print(f"im2_1.dtype = {im2_1.dtype}\n")
print(f"im2_1.size = {im2_1.size}\n")
print(f"im2_1.shape = {im2_1.shape}\n")

im2_2 = imageio.imread('./Figures/imageio2.png')  # 读取为RGB
print(f"im2_2.dtype = {im2_2.dtype}\n")
print(f"im2_2.size = {im2_2.size}\n")
print(f"im2_2.shape = {im2_2.shape}\n")

diff = im2_1 - im2_2
print(f"diff.max() = {diff.max()}, diff.min() = {diff.min()}")
# diff.max() = 0, diff.min() = 0




#========================================================================
# 反色
#========================================================================
import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = imageio.imread('./Figures/lena.png')
img1 = 255 - img
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img)


cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)

plt.imshow(img1)
#==============================================================================================

import math
import imageio

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


def PSNR_np_simple(im, jm):
    if not im.shape == jm.shape:
        raise ValueError('Input images must have the same dimensions.')

    im, jm = np.float64(im), np.float64(jm)
    # print(f"3 (im-jm).max() = {(im-jm).max()}, (im-jm).min() = {(im-jm).min()}")
    # mse = np.mean((im * 1.0 - jm * 1.0)**2)
    mse = np.mean((im  - jm )**2)

    if mse <= 1e-20:
        mse = 1e-20
    psnr = 10.0 * math.log10(255.0**2 / mse)
    return psnr


import os  # baby    baby_1 /   0_3  0_3_1  /  0_7   0_7_1  / lena  lena_1
figdir = '/home/jack/公共的/Python/OpenCV/Figures/0_7.png'   # bmp  jpg  png
savedir = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jpg'
imgio = imageio.v2.imread(figdir )

imageio.v2.imwrite(savedir, imgio, quality = 100 ) #  quality = 100
imgio1 = imageio.v2.imread(savedir, )

size1 = os.path.getsize(figdir)
size2 = os.path.getsize(savedir)

print(f"size1 = {size1}, size2 = {size2}")
print(f"1 (imgio-imgio1).max() = {(imgio-imgio1).max()}, (imgio-imgio1).min() = {(imgio-imgio1).min()}")
print(f"2 (imgio*1.0-imgio1*1.0).max() = {(imgio*1.0-imgio1*1.0).max()}, (imgio*1.0-imgio1*1.0).min() = {(imgio*1.0-imgio1*1.0).min()}")


# psnr = PSNR_np_2y(imgio.transpose(2,0,1), imgio1.transpose(2,0,1), cal_type='y')
# print(f"psnr = {psnr}")

psnr = PSNR_np_simple(imgio, imgio1)
print(f"psnr = {psnr}")

# 0_3:   1: 16.2643933  ->  41.6374930   y: 20.2056  -> 56.43126653411094
# baby:  1: 22.64763    ->  48.0870440   y: 26.1289  -> 57.150011
# lena:  1: 21.42240    ->  38.3282353   y: 25.7071  -> 56.092733
# 0010:  1: 21.77987    ->  44.4113176   y: 25.331   ->  56.1121696



savedir1 = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jpg'
imageio.v2.imwrite(savedir1, imgio , quality = 100) #  quality = 100
imgio2 = imageio.v2.imread(savedir1, )
size3 = os.path.getsize(savedir1)
print(f" size3 = {size3}")




# print(f"imgio1-imgio = \n{imgio1*1.0 - imgio*1.0}\n")
mean = imgio1.mean(axis = (0,1))
imgio2 = np.ones_like(imgio)*mean
psnr = PSNR_np_simple(imgio, imgio2)
print(f"psnr = {psnr}")



#===========================================================================


















































