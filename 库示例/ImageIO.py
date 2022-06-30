#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:09:10 2022

@author: jack
"""


#==============================================================================================
# https://zodiac911.github.io/blog/imread_differences.html
#==============================================================================================

#encoding=utf8
from PIL import Image
from scipy import misc, ndimage
import imageio
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt 
import skimage
import sys
from skimage import io



def display(imagepath):
    # PIL
    im_PIL = Image.open(imagepath)
    print("PIL type1", type(im_PIL))
    im_PIL = np.array(im_PIL)# PIL 方法读取后需要通过转换获得numpy对象,RGB
    print("PIL type2", type(im_PIL))
    print("PIL shape", im_PIL.shape)
    print("PIL min {} max {}".format(im_PIL.min(), im_PIL.max()))


    # imageio
    im_imageio = imageio.imread(imagepath)
    print("imageio type", type(im_imageio))
    print("imageio shape", im_imageio.shape)
    print("imageio min {} max {}".format(im_imageio.min(), im_imageio.max()))


    # # scipy.ndimage
    # im_scipy_ndimage = imageio.imread(imagepath)
    # print(type(im_scipy_ndimage))
    # print(im_scipy_ndimage.shape)


    # matplotlib
    im_matplotlib = plt.imread(imagepath)
    print("matplotlib type", type(im_matplotlib))
    print("matplotlib shape", im_matplotlib.shape)
    print("matplotlib min {} max {}".format(im_matplotlib.min(), im_matplotlib.max()))


    # cv2
    im_cv2=cv2.imread(imagepath)
    print("cv2 type", type(im_cv2))
    print("cv2 shape", im_cv2.shape)
    print("cv2 min {} max {}".format(im_cv2.min(), im_cv2.max()))


    # skimge
    im_skimge = io.imread(imagepath)
    print("skimge type", type(im_skimge))
    print("skimge shape", im_skimge.shape)
    print("skimge min {} max {}".format(im_skimge.min(), im_skimge.max()))


    # cv2.imshow('test',im4)
    # cv2.waitKey()
    #统一使用plt进行显示，不管是plt还是cv2.imshow,在python中只认numpy.array，但是由于cv2.imread 的图片是BGR，cv2.imshow 时相应的换通道显示

    plt.figure(figsize=(6,9))
    plt.subplot(321)
    plt.title('PIL read')
    plt.imshow(im_PIL)

    plt.subplot(322)
    plt.title('imageio read')
    plt.imshow(im_imageio)

    
    plt.subplot(324)
    plt.title('matplotlib read')
    plt.imshow(im_matplotlib)

    plt.subplot(325)
    plt.title('cv2 read')
    plt.imshow(im_cv2)


    plt.subplot(326)
    plt.title('skimge read')
    plt.imshow(im_skimge)


    plt.tight_layout()
    plt.show()
    
    print(np.allclose(im_imageio, im_PIL))
    try:
        print(np.allclose(im_imageio, im_cv2))
        print(np.allclose(im_imageio, im_cv2[:,:,::-1]))
    except ValueError as e:
        print(e)
    print(np.allclose(im_imageio, im_matplotlib))
    print(np.allclose(im_imageio, im_skimge))
    
    return im_PIL, im_imageio, im_cv2, im_matplotlib, im_skimge
    
#     try:
#         print(np.array_equal(im_PIL, im_imageio, im_cv2, im_matplotlib, im_skimge))
#     except ValueError as e:
#         print(e)
#         print( ) 




imagepath='/home/jack/图片/Wallpapers/3DBed.jpg'
im_PIL, im_imageio, im_cv2, im_matplotlib, im_skimge = display(imagepath)
plt.imshow(im_cv2[:,:,::-1]) #交换 BGR 中的 BR


print(f"(im_PIL-im_imageio).min() = {(im_PIL-im_imageio).min()},  (im_PIL-im_imageio).max() = {(im_PIL-im_imageio).max()},\
      (im_PIL-im_matplotlib).min() = {(im_PIL-im_matplotlib).min()}, (im_PIL-im_matplotlib).max() = {(im_PIL-im_matplotlib).max()},\
           (im_PIL-im_skimge).min() = {(im_PIL-im_skimge).min()}, (im_PIL-im_skimge).max() = {(im_PIL-im_skimge).max()}")

"""
图片为三通道时，五种读取方式读取的结果几乎都一样，只是 cv2 读取的结果通道顺序为 BGR, 所以显示上出现了一点不同，通过转换将 BR 通道交换之后显示效果正常，但是 np allclose 函数显示两张图片还是有一些不同

看一看为什么
"""

index = np.argwhere(im_imageio-im_cv2[:,:,::-1]!=0) # 不相等的 index
print(index.shape)

print(im_imageio[index[:,0], index[:,1],index[:,2]], im_cv2[:,:,::-1][index[:,0], index[:,1],index[:,2]]) # 对比差异

# 作差比较
s = im_imageio - im_cv2[:,:,::-1]
s[index[:,0], index[:,1],index[:,2]]



"""
原来imageio 和 cv2 读取到的图片数值会有略微的差异，比如 236 和 237 这样的差异

值得注意的是， im_imageio 和 im_cv2image 作差之后仍为 image 对象，由于 image 对象不允许存在负值，所以有些负数会被加上 256，比如 -1 会自动转换为 255
"""
imagepath='/home/jack/图片/Wallpapers/3DBed.jpg'
im_PIL, im_imageio, im_cv2, im_matplotlib, im_skimge = display(imagepath)
plt.imshow(im_cv2[:,:,::-1]) #交换 BGR 中的 BR
plt.show()



"""
图片为灰度图片时，除了 cv2 其他方式读取的图片都是单通道的，cv2 读取的图片依然是三通道的。

可以看出，cv读取的三通道图片无论交换通道顺序与否，显示效果都相同。

matplotlib 读取的结果也与其他两种不同，matplotlib 读取的结果为 0~1 之间的浮点数
"""
cmp1 = im_cv2[:,:,0]==im_cv2[:,:,1]
cmp2 = im_cv2[:,:,0]==im_cv2[:,:,2]
print(np.all(cmp1)) # all True
print(np.all(cmp2)) # all True


np.allclose(im_imageio, im_cv2[:,:,0])

"""
cv2 对灰度图片的处理，是三通道完全一样，且任一通道都和其他读取方式（除了 matplotlib）读取的结果一样

和 imageio 读取结果略有不同的现象在但通道情况下似乎没有出现
"""
print(f"im_imageio = {im_imageio}")



print(f"im_matplotlib = {im_matplotlib}")
























