#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:34:20 2023

@author: jack

(一) plt
(1) 读取
    img = plt.imread(imagepath): 如果 imagepath为 PNG 图像, 以浮点数组 (0-1) 的形式返回，所有其他格式都作为 int 型数组返回，位深由具体图像决定。
    type(img) = <class 'numpy.ndarray'>
(2) 展示
    plt.imshow(img)
(3) 保存
    plt.imsave('./Figures/lenaPltCv2Imageio.png',img)

需要注意的是, 不论是哪种方法函数读入图片, 显示图片皆可用plt.imshow()来操作。


"""


#========================================================================
# cv2读取,plt画图(正常)
#========================================================================
import matplotlib.pyplot as plt
import numpy as np
#与opencv结合使用
import cv2
im3 = cv2.imread('./Figures/lena.png')
plt.imshow(im3)
plt.axis('off')
plt.show()
#发现图像颜色怪怪的，原因当然是我们前面提到的RGB顺序不同的原因啦,转一下就好
im3 = cv2.cvtColor(im3,cv2.COLOR_BGR2RGB)
plt.imshow(im3)
plt.axis('off')
plt.show()
#所以无论用什么库读进图片，只要把图片改为矩阵，那么matplotlib就可以处理了

# 或者手动调整
im3 = cv2.imread('./Figures/lena.png')
plt.imshow(im3[:,:,::-1])
plt.axis('off')
plt.show()


#========================================================================
# plt读取 png ,plt画图(正常) , plt保存，再plt读取, plt画图(正常)
#========================================================================

img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
plt.imsave('./lenaplt.png', img)



imgplt = plt.imread('./Figures/lenaplt.png')
plt.imshow(imgplt)
plt.show()
print(f"imgplt.shape = {imgplt.shape}\n") # (h,w,c)
print(f"imgplt.size = {imgplt.size}\n") # 像素总数目
print(f"imgplt.dtype = {imgplt.dtype}\n")


imgcv = cv2.imread('./Figures/lenaplt.png')
print(f"imgcv.shape = {imgcv.shape}\n") # (h,w,c)
print(f"imgcv.size = {imgcv.size}\n") # 像素总数目
print(f"imgcv.dtype = {imgcv.dtype}\n")
plt.imshow(imgcv[:,:,::-1])
plt.show()
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32

# imgplt.shape = (512, 512, 4)
# imgplt.size = 1048576
# imgplt.dtype = float32

# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint32

# plt.imsave后再用plt.imread读取会比原图多出一维，都是1，但是用cv读取不会多出一维。

#========================================================================
# plt读取 jpg  , plt保存，再plt读取,
#========================================================================

img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
plt.imsave('./Figures/flowerplt.jpg', img)



imgplt = plt.imread('./Figures/flowerplt.jpg')
plt.imshow(imgplt)
plt.show()
print(f"imgplt.shape = {imgplt.shape}\n") # (h,w,c)
print(f"imgplt.size = {imgplt.size}\n") # 像素总数目
print(f"imgplt.dtype = {imgplt.dtype}\n")


imgcv = cv2.imread('./Figures/flowerplt.jpg')
print(f"imgcv.shape = {imgcv.shape}\n") # (h,w,c)
print(f"imgcv.size = {imgcv.size}\n") # 像素总数目
print(f"imgcv.dtype = {imgcv.dtype}\n")
plt.imshow(imgcv[:,:,::-1])
plt.show()
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32

# imgplt.shape = (512, 512, 4)
# imgplt.size = 1048576
# imgplt.dtype = float32

# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint32

# plt.imsave后再用plt.imread读取会比原图多出一维，都是1，但是用cv读取不会多出一维。



#==================================================
# matplotlib是一个科学绘图神器，用的人非常多。但彩图读进的是RGB，与opencv有区别
#==================================================
import matplotlib.pyplot as plt
import numpy as np


img = plt.imread('./Figures/lena.png')
print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
plt.imsave('./Figures/lenaplt.png', img)
plt.imshow(img)



img1 = plt.imread('./Figures/lenaplt.png')
plt.imshow(img1)
plt.axis('off')
plt.show()

#也可以关闭显示x，y轴上的数字
img2 = plt.imread('./Figures/lena.png')
plt.imshow(img2)
plt.axis('off')
plt.show()


#plt.imread读入的就是一个矩阵，跟opencv一样，但彩图读进的是RGB，与opencv有区别
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
print(img)


im_r = img[:,:,0] # 红色通道
plt.imshow(im_r)
plt.show()
#此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数解决
plt.imshow(im_r, cmap='Greys_r')
plt.show()


im_g = img[:,:,1] # 绿色通道
plt.imshow(im_g)
plt.show()
#此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数解决
plt.imshow(im_g, cmap='Greys_r')
plt.show()

im_b = img[:,:,2] # 蓝色通道
plt.imshow(im_b)
plt.show()
#此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数解决
plt.imshow(im_b,cmap='Greys_r')
plt.show()


#========================================================================
# 反色
#========================================================================
import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = plt.imread('./lena.png')
img1 = 1 - img
cv2.imshow('src',img[:,:,::-1])
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()
plt.imshow(img)


cv2.imshow('src',img1[:,:,::-1])
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()
plt.imshow(img1)


#================================================================================================================================
##  使用plt 显示多个image的使用案例
#================================================================================================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#=================================================================================================
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

im1 = cv2.imread('./Figures/baby.png')
print(f"im1.shape = {im1.shape}")
im1 = ('baby', "Baby", im1)

im2 = cv2.imread('./Figures/flower.jpg')
print(f"im2.shape = {im2.shape}")
im2 = ('flower', "Flower", im2)

im3 = cv2.imread('./Figures/lena.png')
print(f"im3.shape = {im3.shape}")
im3 = ('lena', "Lena", im3)

im4 = cv2.imread('./Figures/bird.png')
print(f"im4.shape = {im4.shape}")
im4 = ('bird', "Bird", im4)

Image = []
Image.append(im1)
Image.append(im2)
Image.append(im3)
Image.append(im4)


cnt = 0
M = 2
N = 2
fig, axs = plt.subplots(M, N, figsize=(10, 10))
for i in range(M):
    for j in range(N):
        orig, adv, image = Image[cnt]
        axs[i, j].set_title("{} -> {}".format(orig, adv))
        axs[i, j].imshow(image[:,:,::-1] )
        axs[i, j].set_xticks([])  # #不显示x轴刻度值
        axs[i, j].set_yticks([] ) # #不显示y轴刻度值
        if j == 0:
            axs[i, j].set_ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        cnt += 1
plt.tight_layout()

plt.subplots_adjust(left=0, bottom=0, right=1, top=0.92 , wspace=0.0, hspace=0.1)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5, y=0.99, )

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'hh.eps',format='eps', bbox_inches = 'tight')

plt.show()

#=======================================================================================
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

im1 = cv2.imread('./Figures/baby.png')
print(f"im1.shape = {im1.shape}")
im1 = ('baby', "Baby", im1)

im2 = cv2.imread('./Figures/flower.jpg')
print(f"im2.shape = {im2.shape}")
im2 = ('flower', "Flower", im2)

im3 = cv2.imread('./Figures/lena.png')
print(f"im3.shape = {im3.shape}")
im3 = ('lena', "Lena", im3)

im4 = cv2.imread('./Figures/bird.png')
print(f"im4.shape = {im4.shape}")
im4 = ('bird', "Bird", im4)

Image = []
Image.append(im1)
Image.append(im2)
Image.append(im3)
Image.append(im4)



cnt = 0
M = 2
N = 2
plt.figure(figsize=(10, 10))
for i in range(M):
    for j in range(N):
        
        plt.subplot(M, N, cnt+1)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, image = Image[cnt]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(image[:,:,::-1])
        cnt += 1
plt.tight_layout()
plt.show()





#================================================================================================================================

 































