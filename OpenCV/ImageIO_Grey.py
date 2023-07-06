#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:11:27 2023

@author: jack

总结读取和显示单通道图片的方法;




"""


import PIL

from PIL import Image
from scipy import misc, ndimage
import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import sys
from skimage import io


#====================================================================================================
#==============  使用 plt ==============================
#====================================================================================================
import matplotlib.pyplot as plt
import matplotlib.image as mp


figdir = '/home/jack/公共的/Python/OpenCV/Figures/man.png'
# 读取图片
img = matplotlib.image.imread(figdir)[:,:,0]
# 输出图片维度
print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")




#====================================================================================================
#==============  使用 PIL ==============================
#====================================================================================================

figdir = '/home/jack/公共的/Python/OpenCV/Figures/man.png'
# 读取图片
img = PIL.Image.open(figdir)
# 输出图片维度
print(f"img.size = {img.size}\n") # (h,w,c)
print(f"type(img) = {type(img)}\n")



# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img,  cmap='Greys_r')
plt.title(" 2")
plt.show()

# 展示方法二
cv2.imshow('src',  np.array(img), )
cv2.waitKey()
cv2.destroyAllWindows()


# 展示方法三
skimage.io.imshow(np.array(img))
skimage.io.show()

# 展示方法三
img.show()


savefile = '/home/jack/公共的/Python/OpenCV/Figures/man_PILsave.jpg'
img.save(savefile, quality=1 )
img1 = PIL.Image.open(savefile) # 在正常的cv.imread后加上-1即可

import torchvision.transforms as transforms
image_transforms = transforms.Compose([ transforms.Grayscale(1) ])
image = image_transforms(img)
print("灰度图像维度： ", np.array(image).shape)

#====================================================================================================
#============== 使用 cv2  ==============================
#====================================================================================================

figdir = '/home/jack/公共的/Python/OpenCV/Figures/man.png'
# 读取图片
img = cv2.imread(figdir, cv2.IMREAD_GRAYSCALE) # 在正常的cv.imread后加上-1即可
# 输出图片维度
print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")

# type(img) = <class 'numpy.ndarray'>
# img.shape = (512, 512)
# img.size = 262144
# img.dtype = uint8

# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img,  cmap='Greys_r')
plt.title(" 2")
plt.axis('off')
plt.show()


# 展示方法二
cv2.imshow('src', img, )
cv2.waitKey()
cv2.destroyAllWindows()


# 展示方法三
skimage.io.imshow(np.array(img))
skimage.io.show()

# 展示方法三
PIL.Image.fromarray(img).show()

savefile = '/home/jack/公共的/Python/OpenCV/Figures/man_cvsave.png'
cv2.imwrite(savefile, img, )
img1 = cv2.imread(savefile, cv2.IMREAD_GRAYSCALE) # 在正常的cv.imread后加上-1即可


#============== 使用 cv2  ==============================
#也可以这么写，先读入彩色图，再转灰度图


figdir = '/home/jack/公共的/Python/OpenCV/Figures/man.png'
# 读取图片
img = cv2.imread(figdir,)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 输出图片维度
print(f"type(img1) = {type(img1)}\n")
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")

# type(img) = <class 'numpy.ndarray'>
# img.shape = (512, 512)
# img.size = 262144
# img.dtype = uint8

# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img1,  cmap='Greys_r')
plt.title(" 2")
plt.axis('off')
plt.show()


# 展示方法二
cv2.imshow('src', img1, )
cv2.waitKey()
cv2.destroyAllWindows()


# 展示方法三
skimage.io.imshow(np.array(img1))
skimage.io.show()

# 展示方法三
PIL.Image.fromarray(img1).show()





#====================================================================================================
#============== 使用 imageio ==============================
#====================================================================================================


import imageio

figdir = '/home/jack/公共的/Python/OpenCV/Figures/man.png'

# figdir = '/home/jack/SemanticNoise_AdversarialAttack/Data/Mnist_test_png/0_7.png'
# figdir = '/home/jack/SemanticNoise_AdversarialAttack/Data/Cifar10_test_png/0_3.png'

imgio = imageio.v2.imread(figdir , mode = 'L')


# 输出图片维度
print(f"type(imgio) = {type(imgio)}\n")
print(f"imgio.shape = {imgio.shape}\n") # (h,w,c)
print(f"imgio.size = {imgio.size}\n") # 像素总数目
print(f"imgio.dtype = {imgio.dtype}\n")

# type(img) = <class 'numpy.ndarray'>
# img.shape = (512, 512)
# img.size = 262144
# img.dtype = uint8

# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img,  cmap='Greys_r')
plt.title(" 1")
plt.show()

# 展示方法二
cv2.imshow('src', img) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法三
skimage.io.imshow(img)
skimage.io.show()



# 展示方法三
PIL.Image.fromarray(img).show()



savedir = '/home/jack/公共的/Python/OpenCV/Figures/man_imageio_save.jpg'
imageio.v2.imwrite(savedir, imgio, mode = 'L',  quality = 100)
imgio1 = imageio.v2.imread(savedir, mode = 'L')

print(f"imgio1-imgio = \n{imgio1*1.0 - imgio*1.0}\n")



# figdir = '/home/jack/SemanticNoise_AdversarialAttack/Data/Cifar10_test_png/0_3.png'
# savedir = '/home/jack/SemanticNoise_AdversarialAttack/Data/Cifar10_test_png/0_3_1.jpg'
# imgio = imageio.v2.imread(figdir  )

# imageio.v2.imwrite(savedir, imgio,  quality = 100)
# imgio1 = imageio.v2.imread(savedir, )


# print(f"imgio1-imgio = \n{imgio1*1.0 - imgio*1.0}\n")
























































































