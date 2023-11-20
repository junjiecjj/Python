#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:58:29 2022

@author: jack



(一) plt
    (1) 读取
        img = plt.imread(imagepath): 如果 imagepath为 PNG 图像, 以浮点数组 (0-1) 的形式返回，所有其他格式都作为 int 型数组返回，位深由具体图像决定。
        返回值: numpy.array
        type(img) = <class 'numpy.ndarray'>
    (2) 展示
        plt.imshow(img)
    (3) 保存
        plt.imsave('./Figures/lenaPltCv2Imageio.png',img, format = 'jpg')
        matplotlib.pyplot.imsave(fname, arr, **kwargs)
        plt.savefig(savefile, dpi=300, quality = 95)  # 注意, plt.savefig 用来保存当前画布, 而不能用来保存数组.
    参数说明：

    fname: 保存图像的文件名, 可以是相对路径或绝对路径。
    arr: 表示图像的NumPy数组。
    format 指明图片格式,可能的格式有png, pdf, svg, etc.
    dpi 分辨率
    cmap: 颜色映射，对于彩色图片这个参数被忽略，只对灰度图片有效。
    origin: {'upper', 'lower'}选其一,指明图片原点在左上还是左下,默认左上'upper'

(二) imageio, import imageio
    (1) 读取
        image = imageio.imread(imagepath, mode = '')
        mode类型:
        'L' (8-bit pixels, grayscale)
        'P' (8-bit pixels, mapped to any other mode using a color palette)
        'RGB'  (3x8-bit pixels, true color)
        'RGBA' (4x8-bit pixels, true color with transparency mask)
        'CMYK' (4x8-bit pixels, color separation)
        'YCbCr' (3x8-bit pixels, color video format)
        'I' (32-bit signed integer pixels)
        'F' (32-bit floating point pixels)
        ————————————————

        返回值: numpy.array
        type(image) = <class 'imageio.core.util.Array'>
    (2) 展示
        无显示方法, 但其读取后的类型为numpy数据, 故可用之后的plt.imshow()显示.
    (3) 保存
        imageio.imsave('image.jpg', image, quality = 25)  # 压缩至25%



(三) cv2
    (1) 读取
        img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')

        imread函数有两个参数,第一个参数是图片路径，第二个参数表示读取图片的形式，有三种：
        cv2.IMREAD_COLOR: 加载彩色图片,这个是默认参数,可以直接写1。
        cv2.IMREAD_GRAYSCALE: 以灰度模式加载图片,可以直接写0。
        cv2.IMREAD_UNCHANGED: 包括alpha,可以直接写-1

        返回值: numpy.array
        type(img) = <class 'numpy.ndarray'>

    (2) 展示
        cv2.imshow('src', img1)
        plt.close()
        cv2.waitKey()
        cv2.destroyAllWindows()
    (3) 保存
        cv2.imwrite('./Figures/lenaPltCv2Imageio.png', img1)
        cv2.imwrite(存储路径，图像变量[,存盘标识])

        存盘标识：
        cv2.CV_IMWRITE_JPEG_QUALITY  设置图片格式为.jpeg或者.jpg的图片质量, 其值为0---100 (数值越大质量越高), 默认95
        cv2.CV_IMWRITE_WEBP_QUALITY  设置图片的格式为.webp格式的图片质量, 值为0--100
        cv2.IMWRITE_PNG_COMPRESSION  设置.png格式的压缩比, 其值为0--9(数值越大, 压缩比越大), 默认为3, cv2.IMWRITE_PNG_COMPRESSION的值为0, 即不压缩.

        cv2.imwrite(savefile, img, [int(cv2.IMWRITE_JPEG_QUALITY), 1])
        cv2.imwrite(savefile, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        cv2.imwrite(savefile, img, [int(cv2.CV_IMWRITE_WEBP_QUALITY), 3])

(四) from skimage import io
    (1) 读取
        img1 = io.imread('./Figures/lenaPltCv2Imageio.png')
        图像通道, 这里的图像通道顺序为RGB, 如果图像是RGBA格式的话, 读入的也是一个4通道的数据
        type(img) = <class 'numpy.ndarray'>
        返回值: numpy.array
    (2) 展示
        io.imshow(img)
        io.show()
    (3) 保存
        io.imsave('./Figures/lenaPltCv2Imageio.png', img1, quality = 95) # 0 - 100

经过实验发现, 以上方法都可以保存为 pdf, 只是不能读取 pdf.
            (1) 针对 jpg 图像的压缩, imageio.imsave, cv2.imwrit, skimage.io.imsave  调节压缩比参数的时候存储的文件大小有很明显的区别, 且视觉上存在很明显的差别; 但是针对png格式的图片时压缩时文件大小存在明显的差别, 但是视觉上前后看不出差别.
            (2) 针对 png 图像的压缩, imageio.imsave, cv2.imwrit, skimage.io.imsave  原图和压缩后的图的像素点的差异随着压缩比的增大而增大, 也就是压缩得越厉害像素变换的越多; 但是针对png格式的图片压缩前后的像素点完全一致, 与压缩比无关.
            说明 jpg 是有损压缩, 但是 png 是无损压缩.
========================================================================================================
(1) 需要注意的是, 不论是哪种方法函数读入图片, 显示图片皆可用 plt.imshow(img), cv2.imshow(img), skimage.io.imshow(img) 来操作. 如果 img 为 0-1 之间的浮点数, 则 plt.imshow(img) 和 cv2.imshow(img), skimage.io.imshow(img) 会将浮点数转换为 0-255 再显示; 如果 img 为 0-255 之间的浮点数, 则 plt.imshow(img) 和 cv2.imshow(img), skimage.io.imshow() 会直接显示; 使用 cv2.imshow(img) 时需要注意通道顺序即可; 一般画多子图会用 plt.imshow(img).

(2) 使用 plt 读取 PNG 图像, 读取的图片矩阵 img 是 0-1 之间的浮点数矩阵, 而 imageio, io 和 cv2 读取 PNG 图像的返回值 img1 都是 0-255 的像素点矩阵, 基本上 img1 = img * 255, 也就是 plt 读取的是归一化的像素点.

(3) 读取 jpg 图像,  plt, imageio 和io, cv2 读取的 img  都是 0-255 的像素点矩阵, 返回值都是 np.array.

(4) plt 读取 PNG 图像, 读取的图片矩阵是 0-1 之间的浮点数, 使用 cv2 保存时, 不管保存为 png 还是 jpg, 都会失去所有像素信息, 因为 cv 只能保存大于1的整数和浮点数, 而 plt 读取的图片矩阵否是0-1之间的浮点数, 所以用 cv2 保存和读取 0-1 之间的浮点数组成的矩阵会得到黑色。
    plt 读取 PNG 图像, 读取的图片矩阵是 0-1 之间的浮点数, 使用 io 和 imageio 保存时, 不管保存为 png 还是 jpg, 都会报错, 因为 imageio 只能保存大于1的整数和浮点数。

(6) plt读取 非PNG 图像, 如 jpg 时, 使用 cv2.imwrite, imageio.imsave 保存都正常, 只不过注意 cv2 需要调换通道顺序, 而 imageio 不需要.

(7) opencv对于读进来的图片的通道排列是BGR, 而不是主流的RGB! 由于cv2读取的是BGR图像, 因此使用 plt, imageio 等 保存图像的时候会将BGR图像转换成RGB, 即将第一个通道和第三个通道的顺序改变后再保存, 这样会保证读取的保存的图像一致, 不出错.
    1. cv2.imread 它默认认为图片存储的方式是RGB, 而不管实际意义上的顺序, 但是读取的时候反序了, 它读出的图片的通道是BGR。
    2. cv2.imshow 它把被显示的图片当成BGR的顺序, 而不管实际意义上的顺序,然后显示的时候变成RGB显示。
    3. cv2.imwrite 它认为被保存的图片是BGR顺序, 而不管实际意义上的顺序, 保存的时候反序成RGB再保存。
    4. plt, imageio 默认认为被打开, 保存, 显示的图片的通道顺序为RGB。
    所以将cv2和plt混用时需要注意顺序;

(一) 当使用 plt 读取 png 图片时, 只能使用 plt 保存, 使用 cv2, io, imageio 保存时要么 error 要么 像素出错;
    在使用 plt 读取 png 图片时, 读取后, 使用 plt 保存为 png 时, 再使用 plt, imageio 打开时, 图片变为 4 通道; 使用 cv2, io 打开时, 图片还是 3 通道;
    在使用 plt 读取 png 图片时, 读取后, 使用 plt 保存为 jpg 时,  再使用 plt, imageio, cv2, io 打开时, 图片还是 3 通道;

(二) 当使用 其他方法 读取 png 图片时, 一切正常, 除了注意通道的顺序.

(三) 当使用 plt imageio cv2 PIL  读取 jpg 图片时, 一切正常, 除了注意通道的顺序.




本程序是测试:
当原图是 png 还是 jpg 格式时

读取方式是 plt   imageio   cv2等三种方式时

保存方式是 plt   imageio   cv2等三种方式时

再次读取方式是 plt   imageio   cv2等三种方式时

的效果.




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






# 原图是 png 还是 jpg
source = './Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

# source = './Figures/man.png'

## 保存文件的拓展名
ext = '.jpg'

ext = '.png'


### ext = '.pdf'

savefile = './Figures/PltCv2Imageio' + ext



#==========================================================================

#====================================================
# 读取的三种方式
#====================================================
# img = plt.imread(source)

img = cv2.imread(source, cv2.IMREAD_COLOR)

# img = imageio.imread(source, mode = 'RGB')

# img = skimage.io.imread(source)
ep = [cv2.IMWRITE_JPEG2000_COMPRESSION_X, 10]

cv2.imwrit("hhhh.jp2", img, ep)

#====================================================
#  第一次读取的展示


# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img, )
plt.title(" 1")
plt.show()

# 展示方法二
cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法三
skimage.io.imshow(img)
skimage.io.show()


### 展示方法四
## img.show()

print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32


# # 查看图像实例的属性
# print(f"img.format = {img.format}, img.width = {img.width}, img.height = {img.height}, img.size = {img.size}, img.mode = {img.mode}\n")





#====================================================
# 保存的三种方式
#====================================================
# plt.imsave(savefile, img, dpi = 100 )


# if ext == '.jpg':
#     cv2.imwrite(savefile, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# elif ext == '.png':
#     cv2.imwrite(savefile, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
# elif ext == '.webp':
#     cv2.imwrite(savefile, img, [int(cv2.CV_IMWRITE_WEBP_QUALITY), 3])



# imageio.imwrite(savefile, img, quality = 1 )



# skimage.io.imsave(savefile, img, quality = 100)



#====================================================
# 再次读取的三种方式
#====================================================
img1 = plt.imread(savefile)

# img1 = cv2.imread(savefile)

# img1 = imageio.imread(savefile)

# img1 = skimage.io.imread(savefile)

## img1 = PIL.Image.open(savefile)
## img1 = np.array(img1)
#==========================================================================





#====================================================
#  第二次读取的展示
#====================================================

# 展示方法一
fig = plt.figure()
plt.tight_layout()
plt.imshow(img1)
plt.title(" 2")
plt.show()

# 展示方法二
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()


# 展示方法三
skimage.io.imshow(img1)
skimage.io.show()


### 展示方法四
### img1.show()

#====================================================
print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}\n") # (h,w,c)
print(f"img.size = {img.size}\n") # 像素总数目
print(f"img.dtype = {img.dtype}\n")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32

print(f"type(img1) = {type(img1)}\n")
print(f"img1.shape = {img1.shape}\n") # (h,w,c)
print(f"img1.size = {img1.size}\n") # 像素总数目
print(f"img1.dtype = {img1.dtype}\n")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8



# # 查看图像实例的属性
# print(f"img.format = {img.format}, img.width = {img.width}, img.height = {img.height}, img.size = {img.size}, img.mode = {img.mode}\n")











































