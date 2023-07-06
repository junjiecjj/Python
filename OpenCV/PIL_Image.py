#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:28:13 2023

@author: jack

(一)  from PIL import Image
    (1) 读取
        img = PIL.Image.open(imagepath): 如果 imagepath为 PNG 图像, 以浮点数组 (0-1) 的形式返回，所有其他格式都作为 int 型数组返回，位深由具体图像决定。
        返回值: Image对象
        type(img) = PIL.JpegImagePlugin.JpegImageFile
    (2) 展示
        img.show()
    (3) 保存   PIL保存图片的方式也是非常简单，调用方法 Image.save() 即可，保存的是RGB格式的图片。
    首先PIL保存图片的时候,图片类型一定要是ndarray类型,不能是tensor类型,否则报错
        img.save('./Figures/lenaPltCv2Imageio.png', quality=95)
        quality参数： 保存图像的质量，值的范围从1（最差）到95（最佳）。 默认值为75，使用中应尽量避免高于95的值; 100会禁用部分JPEG压缩算法，并导致大文件图像质量几乎没有任何增益。

PIL读取图片不直接返回numpy对象, 可以用numpy提供的函数np.array()进行转换, 亦可用Image.fromarray()再从numpy对象转换为原来的Image对象, 读取, 显示, 保存以及数据格式转换方法见如下代码：

Image.fromarray(这里面的数据只能是 uint8 型的):
    Image.fromarray(np.uint8(img))
    Image.fromarray( img.astype(np.uint8))

如果图像为RGBA格式(这里的A表示透明度)则使用Image方法读入的是4通道的数据, cv2.imread不具备这种能力.


PIL 可以读取/写入 Bmp 格式的图片

"""

import PIL
from PIL import Image
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import sys
from skimage import io


#==========================================================================
# 由tensor转换为图片
# 经过tensor的处理过程之后，要将tensor形式转换为图片。
# 用到numpy()函数，fromarray()函数。uint8函数

## 这里将tensor形式用numpy()函数转为数组形式，
## 并且用transpose将数组转置为PIL能够处理的WxHxC形式。
nimg = img.numpy().transpose(1,2,0)
img = nimg * 255 # 将原来tensor里0-1的数值乘以255，以便转为uint8数据形式，uint8是图片的数值形式。
#那么此时img就是原料，通过两种方式将img化为图片
# 第一种
Image.fromarray(np.uint8(img)) # eg1
# 第二种
Image.fromarray(img.astype(np.uint8)) #eg2






#===================================================================================================================

img = np.random.randn(128, 128, 3)
img = PIL.Image.fromarray(img.astype('uint8'), mode='RGB')


#===================================================================================================================
arr = (np.eye(300)*255) # 二维数组
im = PIL.Image.fromarray(arr.astype('uint8'))   # 转换为图像
im.show()
#===================================================================================================================









#==========================================================================













# 原图是 png 还是 jpg
# source = './Figures/baby.png'

source = '/home/jack/公共的/Python/OpenCV/Figures/flower.jpg'


## 保存文件的拓展名
ext = '.jpg'

# ext = '.png'

savefile = '/home/jack/公共的/Python/OpenCV/Figures/PILSave' + ext

#==========================================================================

img = PIL.Image.open(source)


img.save(savefile, quality = 100 )

# 展示方法一
img.show()


### 查看图像实例的属性
print(f"img = {img}")
print(f"type(img) = {type(img)}, img.format = {img.format}, img.width = {img.width}, img.height = {img.height}, img.size = {img.size}, img.mode = {img.mode}\n")
# img = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=481x321 at 0x7F24D0B69130>
# type(img) = <class 'PIL.JpegImagePlugin.JpegImageFile'>, img.format = JPEG, img.width = 481, img.height = 321, img.size = (481, 321), img.mode = RGB

# image to array
img1 = np.array(img)  #获得numpy对象,RGB
print(type(img1))   #结果为<class 'numpy.ndarray'>
print(img1.shape)   #结果为(694, 822, 3)

# 展示方法二, plt 可以显示<class 'PIL.JpegImagePlugin.JpegImageFile'> 也可以显示 np.array
fig = plt.figure()
plt.tight_layout()
plt.imshow(img)
plt.title(" 2")
plt.show()

# 展示方法二
fig = plt.figure()
plt.tight_layout()
plt.imshow(img1)
plt.title(" 2")
plt.show()

# 展示方法三, cv2 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()


# 展示方法四, io 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
io.imshow(img1)
io.show()


# array to image,
img2 = PIL.Image.fromarray((img1*2.123).astype(np.uint8))
print(type(img2))   #结果为<class 'PIL.Image.Image'>
img2.show()


#img2.save(savefile, quality = 1)

img3 = PIL.Image.open(savefile)
# np.array(img3) == img1

#========================================================================
#   Resize
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

# 语法: Image.resize(size, resample=0, resample=image.BICUBIC,)
# 参数 :
# size - 请求的尺寸，以像素为单位，是一个2元组：（宽度，高度）。
# resample - 一个可选的重采样过滤器。这可以是PIL.Image.NEAREST（使用最近的邻居）、PIL.Image.BILINEAR（线性插值）、PIL.Image.BICUBIC（三次样条插值）或PIL.Image.LANCZOS（一个高质量的下采样过滤器）中的一个。如果省略，或者图像有模式 "1 "或 "P"，则设置为PIL.Image.NEAREST。
# 返回类型 :一个图像对象。
img1 = img.resize((256, 256), resample = PIL.Image.BICUBIC,)

# 2.水平翻转
img2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
# transpose有这么几种模式:
    # FLIP_LEFT_RIGHT：左右镜像
    # FLIP_TOP_BOTTOM：上下镜像
    # ROTATE_90：逆时针转90度
    # ROTATE_180：逆时针转180度
    # ROTATE_270：逆时针转270度
    # TRANSPOSE：像素矩阵转置
    # TRANSVERSE
# 4.垂直翻转
img2 = img.rotate(180)

# 6.水平+垂直翻转
img3 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT).rotate(180)


# 旋转
img3 = img.rotate(45)

# 图像裁剪操作
(left, upper, right, lower) = (20, 20, 100, 100)
img4 = img.crop((left, upper, right, lower))


#========================================================================
# 水平翻转
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

img1 = transforms.RandomHorizontalFlip(p=0.9)(img)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])
plt.show()

cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img1[:,:,::-1])
plt.show()


#========================================================================
# 垂直翻转
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

img1 = transforms.RandomVerticalFlip(p=0.9)(img)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])
plt.show()

cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img1[:,:,::-1])
plt.show()



#========================================================================
# Resize
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

img1 = transforms.Resize((224, 900))(img)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])
plt.show()

cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img1[:,:,::-1])
plt.show()




#========================================================================
#    中心裁剪 transforms.CenterCrop(size)
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

img1 = transforms.CenterCrop(256)(img)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])
plt.show()

cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img1[:,:,::-1])
plt.show()


#========================================================================
#   随机旋转  transforms.RandomRotation((30,60))(img)
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
source =  '/home/jack/公共的/Python/OpenCV/Figures/lena.png'
img = PIL.Image.open(source)

img1 = transforms.RandomRotation((30,60))(img)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])
plt.show()

cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img1[:,:,::-1])
plt.show()


#===================================================================================================================



import PIL
import os
import math

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



figdir = '/home/jack/公共的/Python/OpenCV/Figures/0_3.bmp'
savedir = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jpg'
imgio = PIL.Image.open(figdir  )

imgio.save(savedir,  quality = 1 )
imgio1 = PIL.Image.open(savedir, )

size1 = os.path.getsize(figdir)
size2 = os.path.getsize(savedir)

print(f"size1 = {size1}, size2 = {size2}")

imgio = np.array(imgio)
imgio1 = np.array(imgio1)
# psnr = PSNR_np_2y(imgio.transpose(2,0,1), imgio1.transpose(2,0,1), cal_type='y')
# print(f"psnr = {psnr}")
print(f"1 (imgio-imgio1).max() = {(imgio-imgio1).max()}, (imgio-imgio1).min() = {(imgio-imgio1).min()}")
print(f"2 (imgio*1.0-imgio1*1.0).max() = {(imgio*1.0-imgio1*1.0).max()}, (imgio*1.0-imgio1*1.0).min() = {(imgio*1.0-imgio1*1.0).min()}")

psnr = PSNR_np_simple(imgio, imgio1)
print(f"psnr = {psnr}")





savedir1 = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jpg'
imgio.save(savedir1,  quality = 100) #  quality = 100
imgio2 = PIL.Image.open(savedir1, )
size3 = os.path.getsize(savedir1)
print(f" size3 = {size3}")









