#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:57:23 2023

@author: jack


cv2:
(1) 读取
    img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
    type(img) = <class 'numpy.ndarray'>
(2) 展示
    cv2.imshow('src', img1)
    plt.close()
    cv2.waitKey()
    cv2.destroyAllWindows()
(3) 保存
    cv2.imwrite('./Figures/lenaPltCv2Imageio.png', img1)



需要注意的是, 不论是哪种方法函数读入图片, 显示图片皆可用 plt.imshow() 来操作。
(一) opencv对于读进来的图片的通道排列是BGR, 而不是主流的RGB! 由于cv2读取的是BGR图像, 因此使用 plt, imageio 等 保存图像的时候会将BGR图像转换成RGB, 即将第一个通道和第三个通道的顺序改变后再保存, 这样会保证读取的保存的图像一致, 不出错.
    1. cv2.imread 它默认认为图片存储的方式是RGB, 而不管实际意义上的顺序, 但是读取的时候反序了, 它读出的图片的通道是BGR。
    2. cv2.imshow 它把被显示的图片当成BGR的顺序, 而不管实际意义上的顺序,然后显示的时候变成RGB显示。
    3. cv2.imwrite 它认为被保存的图片是BGR顺序, 而不管实际意义上的顺序, 保存的时候反序成RGB再保存。
    4. plt, imageio 默认认为被打开, 保存, 显示的图片的通道顺序为RGB。
    所以将cv2和plt混用时需要注意顺序;
"""



from scipy import misc, ndimage
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import sys


img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB

cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")


#=================================================================================================
# 实验11
# cv2读取 png, plt画图(正常), cv2画图(正常), cv2保存png, 再cv2读取 png, plt画图(正常), cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8

cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8

#=================================================================================================
# 实验11
# cv2读取 png, plt画图(正常), cv2画图(正常), cv2保存 jpg, 再cv2读取 png, plt画图(正常), cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8

cv2.imwrite('./Figures/lenaPltCv2Imageio.jpg',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.jpg')
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8

#=================================================================================================
# 实验11
# cv2读取 jpg, plt画图(正常)，cv2画图(正常), cv2保存 png, 再cv2读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/flower.jpg')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8

cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8

#=================================================================================================
# 实验11
# cv2读取 jpg, plt画图(正常)，cv2画图(正常), cv2保存 jpg, 再cv2读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/flower.jpg')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8

cv2.imwrite('./Figures/lenaPltCv2Imageio.jpg',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.jpg')
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8


# 实验1
#==================================================
# cv读 png, 乘以1.0，再cv保存 png, 再cv读 png
#==================================================
img = cv2.imread('./Figures/baby.png', )
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()



img1 = img*1.123
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()

# 实验1
#==================================================
# cv读 png, 乘以1.0，再cv保存 jpg, 再cv读 jpg
#==================================================
img = cv2.imread('./Figures/baby.png', )
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()



img1 = img*1.123
cv2.imwrite('./Figures/babyFloat.jpg', img1)

img2 = cv2.imread('./Figures/babyFloat.jpg', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()

# 实验1
#==================================================
# cv读 jpg, 乘以1.0，再cv保存 png, 再cv读 png
#==================================================
img = cv2.imread('./Figures/flower.jpg', )
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()



img1 = img*1.123
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()

# 实验1
#==================================================
# cv读 jpg, 乘以1.0，再cv保存 jpg, 再cv读 jpg
#==================================================
img = cv2.imread('./Figures/flower.jpg', )
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()



img1 = img*1.123
cv2.imwrite('./Figures/babyFloat.jpg', img1)

img2 = cv2.imread('./Figures/babyFloat.jpg', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img[:,:,::-1], )
plt.show()


#=================================================================================================
# 实验12
# cv2读取, cv2画图(正常), cv2保存，再cv2读取， cv2画图(正常),
img5 = cv2.imread('./Figures/lena.png')
cv2.imshow('src', img5)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('./Figures/lenaCV2write.png',img5)


img6 = cv2.imread('./Figures/lenaCV2write.png')
cv2.imshow('src',img6)
cv2.waitKey()
cv2.destroyAllWindows()


#==============================================================================================


#==============================================================================================
#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
#==============================================================================================


#==============================================================================================
# 展示四分之一的图片
#==============================================================================================
import cv2
import numpy as np
img = cv2.imread('./Figures/lena.png')
cv2.imshow('src',img[256:,256:,:])
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()

#==============================================================================================
# 读取灰度图
#==============================================================================================
gray = cv2.imread('./Figures/lena.png', cv2.IMREAD_GRAYSCALE) #灰度图
cv2.imshow('gray', gray)
print(f"gray.shape = {gray.shape}\n") # (h,w,c)
print(f"gray.size = {gray.size}\n") # 像素总数目
print(f"gray.dtype = {gray.dtype}\n")
#print(gray)
cv2.waitKey()
cv2.destroyAllWindows()

#也可以这么写，先读入彩色图，再转灰度图
src = cv2.imread('./Figures/lena.png')

gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray1)
cv2.waitKey()
cv2.destroyAllWindows()

print(f"gray.shape = {gray.shape}\n") # (h,w,c)
print(f"gray.size = {gray.size}\n") # 像素总数目
print(f"gray.dtype = {gray.dtype}\n")
#print(gray)


#也可以这么写，先读入彩色图，再手动转灰度图
src = cv2.imread('./Figures/lena.png')
gray2 = src[:,:,0] * 0.114 + src[:,:,1] * 0.387 + src[:,:,2] * 0.299
gray2 = gray2.astype(np.uint8)
cv2.imshow('gray',gray2)
cv2.waitKey()
cv2.destroyAllWindows()

print(f"gray2.shape = {gray2.shape}\n") # (h,w,c)
print(f"gray2.size = {gray2.size}\n") # 像素总数目
print(f"gray2.dtype = {gray2.dtype}\n")
#print(gray)


import cv2
#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED 包含alpha通道
img = cv2.imread('./Figures/lena.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# (512, 512, 3)
# 786432
# print(img)



#注意到，opencv读入的图片的彩色图是一个channel last的三维矩阵（h,w,c），即（高度，宽度，通道）
#有时候在深度学习中用到的的图片矩阵形式可能是channel first，那我们可以这样转一下
print(f"img.shape = {img.shape}\n") # (h,w,c)
img = img.transpose(2,0,1)
print(f"img.shape = {img.shape}\n") # (h,w,c)
# (3, 512, 512)

#有时候还要扩展维度，比如有时候我们需要预测单张图片，要在要加一列做图片的个数，可以这么做
img = np.expand_dims(img, axis=0)
print(f"img.shape = {img.shape}\n")

# (1, 3, 512, 512)


#因为opencv读入的图片矩阵数值是0到255，有时我们需要对其进行归一化为0~1
img3 = cv2.imread('./Figures/lena.png')
img3 = img3.astype("float") / 255.0  #注意需要先转化数据类型为float
print(f"img3.shape = {img3.shape}\n") # (h,w,c)
print(f"img3.size = {img3.size}\n") # 像素总数目
print(f"img3.dtype = {img3.dtype}\n")
print(f"img3 = \n{img3}\n")

#存储图片
cv2.imwrite('test1.jpg',img3) #得到的是全黑的图片，因为我们把它归一化了
#所以要得到可视化的图，需要先*255还原
img3 = img3 * 255
cv2.imwrite('test2.jpg',img3)  #这样就可以看到彩色原图了


img3 = img3 * 355
cv2.imwrite('test3.jpg',img3)  #这样就可以看到彩色原图了

img3 = img3 * 365
cv2.imwrite('test4.jpg',img3)  #这样就可以看到彩色原图了


# opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！
#opencv读入的矩阵是BGR，如果想转为RGB，可以这么转
img4 = cv2.imread('./Figures/lena.png')
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)

# cv2保存时不需要将BGR转换为RGB, 因为它会转换，它会把被保存的图像的通道顺序当做BGR，然后转成RGB再保存，所以下面这行保存后再查看会发现图像的颜色不对了
cv2.imwrite('test3.jpg',img4)
# 同样的，cv2.imshow()会把需要显示的图像的通道顺序看成BGR，而不管实际意义上的顺序，然后转换为RGB再显示，所以当将cv2与其他imageIO库混用时经常出错。

#=========================================================
#分离通道
#=========================================================
img5 = cv2.imread('./Figures/lena.png')
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

b,g,r = cv2.split(img5)


#合并通道
img5 = cv2.merge((b,g,r))
cv2.imshow('src',img5)
cv2.imwrite('bgr.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#合并通道
img5 = cv2.merge((b,r,g))
cv2.imshow('src',img5)
cv2.imwrite('brg.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#合并通道
img5 = cv2.merge((r,g,b))
cv2.imshow('src',img5)
cv2.imwrite('rgb.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#合并通道
img5 = cv2.merge((r,b,g))
cv2.imshow('src',img5)
cv2.imwrite('rbg.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#合并通道
img5 = cv2.merge((g,r,b))
cv2.imshow('src',img5)
cv2.imwrite('grb.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#合并通道
img5 = cv2.merge((g,b,r))
cv2.imshow('src',img5)
cv2.imwrite('gbr.jpg',img5)
cv2.waitKey()
cv2.destroyAllWindows()



#将红色通道值全部设0
img5 = cv2.imread('./Figures/lena.png')
img5[:,:,2] = 0
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#将绿色通道值全部设0
img5 = cv2.imread('./Figures/lena.png')
img5[:,:,1] = 0
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

#将蓝色通道值全部设0
img5 = cv2.imread('./Figures/lena.png')
img5[:,:,0] = 0
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

# 只显示蓝色通道
img5 = cv2.imread('./Figures/lena.png')
img5  = img5[:,:,0]
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

# 显示绿色通道
img5 = cv2.imread('./Figures/lena.png')
img5  = img5[:,:,1]
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

# 显示红色通道
img5 = cv2.imread('./Figures/lena.png')
img5  = img5[:,:,2]
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

#========================================================================
# 反色, imageio, io 通用, PIL需要转成np.array或者使用 torchvision.transforms
#========================================================================
import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./Figures/lena.png')
img1 = 255 - img
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
# 水平翻转
#========================================================================
import cv2
import numpy as np
from torchvision import transforms

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./Figures/lena.png')
img1 = np.fliplr(img)
img1 = transforms.RandomHorizontalFlip()(img)
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

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./Figures/lena.png')
img1 = np.flipud(img)
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
# 随机噪声
#========================================================================
import cv2
import numpy as np

n = 4   # n max  is 4 RGBA
img = np.random.randint(0, 255, (512, 512, n), dtype=np.uint8)

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img[:,:,::-1])





#========================================================================
# https://www.jianshu.com/p/5134e90955e6
# 对图像的alpha通道进行处理。
#========================================================================
# 首先从当前目录下读取文件lenacolor.png，然后将其进行色彩空间变换，将其由BGR色彩空间转换到BGRA色彩空间，得到bgra，即为原始图像lena添加alpha通道。
# 接下来，分别将提取得到的alpha通道的值设置为125、0，并将新的alpha通道与原有的BGR通道进行组合，得到新的BGRA图像bgra125、bgra0。
# 接着，分别显示原始图像、原始BGRA图像bgra、重构的BGRA图像bgra125和bgra0。
# 最后，将3个不同的BGRA图像保存在当前目录下。

import cv2
img = cv2.imread('./Figures/lena.png')
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()


bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
cv2.imshow("bgra",bgra)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("./Figures/bgra.png", bgra)

b,g,r,a = cv2.split(bgra)

a[:,:] = 125
bgra125 = cv2.merge([b,g,r,a])
cv2.imshow("bgra125",bgra125)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("./Figures/bgra125.png", bgra125)

a[:,:] = 0
bgra0 = cv2.merge([b,g,r,a])
cv2.imshow("bgra0",bgra0)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("./Figures/bgra0.png", bgra0)





#===================================================================================
# https://www.ryanxin.cn/archives/340
#===================================================================================
r = np.array([[243 for i in range(50)] for j in range(50)]).astype(int)
g = np.array([[67 for i in range(50)] for j in range(50)]).astype(int)
b = np.array([[24 for i in range(50)] for j in range(50)]).astype(int)
img6 = cv2.merge([r,g,b])  #合并一张50x50的纯色RGB图像
cv2.imwrite('./Figures/save.png', img6)  #图像的保存




#==============================================================================================
#  修改图片的大小和通道的像素值
#==============================================================================================

def image2label(path, size_):
    w = size_[0]
    h = size_[1]
    label_im = cv2.imread(path)
    #label_im=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    #修改图像的尺寸大小
    new_array = cv2.resize(label_im, (w, h), interpolation=cv2.INTER_CUBIC)
    data = np.array(new_array,dtype='int32')
    #修改B通道的像素值
    for i in range(data[:,:,0].shape[0]):
        for j in range(data[:,:,0].shape[1]):
            if data[:,:,0][i][j] > 155:
                data[:,:,0][i][j] = 255
            else:
                data[:,:,0][i][j] = 0
    #修改G通道的像素值
    for i in range(data[:,:,1].shape[0]):
        for j in range(data[:,:,1].shape[1]):
            if data[:,:,1][i][j] > 155:
                data[:,:,1][i][j] = 255
            else:
                data[:,:,1][i][j] = 0
    #修改R通道的像素值
    for i in range(data[:,:,2].shape[0]):
        for j in range(data[:,:,2].shape[1]):
            if data[:,:,2][i][j] > 155:
                data[:,:,2][i][j] = 255
            else:
                data[:,:,2][i][j] = 0

    return data

# if __name__ =='__main__':

#修改的尺寸大小
size_ = [320,480]
img_path = '/home/jack/图片/Wallpapers/3DBed.jpg'
label = image2label(img_path, size_)
#修改后的尺寸和修改后的像素值保存下来
save_img ='./Figures/0002_c1s1_000451_03.png'
cv2.imwrite(save_img, label)
print(label[100])



#===================================================================================
# https://www.cnblogs.com/hanxiaosheng/p/9559996.html
#===================================================================================
# 读取一张四川大录古藏寨的照片
img3 = cv2.imread('./Figures/origin.png')

# 缩放成200x200的方形图像
img_200x200 = cv2.resize(img3, (200, 200))

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_200x300 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

# 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
img_300x300 = cv2.copyMakeBorder(img3, 50, 50, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# 对照片中树的部分进行剪裁
patch_tree = img3[20:150, -180:-50]

cv2.imwrite('./Figures/cropped_tree.png', patch_tree)
cv2.imwrite('./Figures/resized_200x200.png', img_200x200)
cv2.imwrite('./Figures/resized_200x300.png', img_200x300)
cv2.imwrite('./Figures/bordered_300x300.png', img_300x300)









