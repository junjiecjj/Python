#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:09:10 2022

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
(1) 需要注意的是, 不论是哪种方法函数读入图片, 显示图片皆可用 plt.imshow(img), cv2.imshow(img) 来操作. 如果 img 为 0-1 之间的浮点数, 则 plt.imshow(img) 和 cv2.imshow(img) 会将浮点数转换为 0-255 再显示; 如果 img 为 0-255 之间的浮点数, 则 plt.imshow(img) 和 cv2.imshow(img) 会直接显示; 使用 cv2.imshow(img) 时需要注意通道顺序即可; 一般画多子图会用 plt.imshow(img).

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


"""

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



#==============================================================================================
# 常用的图片IO的方法总结
#==============================================================================================
# 第一, opencv,即cv2

figdir = '/home/jack/公共的/Python/OpenCV/Figures/baby.png'

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
    print("PIL min {} max {}\n".format(im_PIL.min(), im_PIL.max()))


    # imageio
    im_imageio = imageio.imread(imagepath)
    print("imageio type", type(im_imageio))
    print("imageio shape", im_imageio.shape)
    print("imageio min {} max {}\n".format(im_imageio.min(), im_imageio.max()))


    # # scipy.ndimage
    # im_scipy_ndimage = imageio.imread(imagepath)
    # print(type(im_scipy_ndimage))
    # print(im_scipy_ndimage.shape)


    # matplotlib
    im_matplotlib = plt.imread(imagepath)
    print("matplotlib type", type(im_matplotlib))
    print("matplotlib shape", im_matplotlib.shape)
    print("matplotlib min {} max {}\n".format(im_matplotlib.min(), im_matplotlib.max()))


    # cv2
    im_cv2 = cv2.imread(imagepath)
    print("cv2 type", type(im_cv2))
    print("cv2 shape", im_cv2.shape)
    print("cv2 min {} max {}\n".format(im_cv2.min(), im_cv2.max()))


    # skimge
    im_io = io.imread(imagepath)
    print("skimge type", type(im_io))
    print("skimge shape", im_io.shape)
    print("skimge min {} max {}\n".format(im_io.min(), im_io.max()))


    # cv2.imshow('test',im4)
    # cv2.waitKey()
    #统一使用plt进行显示，不管是plt还是cv2.imshow,在python中只认numpy.array，但是由于cv2.imread 的图片是BGR，cv2.imshow 时相应的换通道显示

    plt.figure(figsize=(6,9))
    plt.subplot(321)
    plt.title('PIL read')
    plt.imshow(im_PIL, interpolation='none', cmap='Greys')

    plt.subplot(322)
    plt.title('imageio read')
    plt.imshow(im_imageio, interpolation='none', cmap='Greys')


    plt.subplot(324)
    plt.title('matplotlib read')
    plt.imshow(im_matplotlib, interpolation='none', cmap='Greys')

    plt.subplot(325)
    plt.title('cv2 read')
    plt.imshow(im_cv2, interpolation='none', cmap='Greys') # 此时会反色, 除非 im_cv2[:,:,::-1]


    plt.subplot(326)
    plt.title('IO read')
    plt.imshow(im_io, interpolation='none', cmap='Greys')


    plt.tight_layout()
    plt.show()

    print(np.allclose(im_imageio, im_PIL))
    try:
        print(np.allclose(im_imageio, im_cv2))
        print(np.allclose(im_imageio, im_cv2[:,:,::-1]))
    except ValueError as e:
        print(e)
    print(np.allclose(im_imageio, im_matplotlib))
    print(np.allclose(im_imageio, im_io))
    print(np.allclose(im_imageio, im_PIL))


    print(f"\n (im_PIL-im_imageio).min() = {(im_PIL-im_imageio).min()}\n (im_PIL-im_imageio).max() = {(im_PIL-im_imageio).max()}\n (im_PIL-im_matplotlib).min() = {(im_PIL-im_matplotlib).min()}\n (im_PIL-im_matplotlib).max() = {(im_PIL-im_matplotlib).max()}\n (im_PIL-im_io).min() = {(im_PIL-im_io).min()}\n (im_PIL-im_io).max() = {(im_PIL-im_io).max()}")

    return im_PIL, im_imageio, im_cv2, im_matplotlib, im_io



#===================================================================================================================================
#  jpg 格式彩色图片
#===================================================================================================================================

imagepath='/home/jack/图片/Wallpapers/3DBed.jpg'
im_PIL, im_imageio, im_cv2, im_matplotlib, im_io = display(imagepath)
plt.imshow(im_cv2[:,:,::-1])  #交换 BGR 中的 BR



"""
因为图片为 jpg 格式的，所以plt也是返回 0-255 而不是0-1;
图片为三通道时，五种读取方式读取的结果几乎都一样，只是 cv2 读取的结果通道顺序为 BGR, 其他方式的读出顺序都是 RGB, 所以显示上出现了一点不同，通过转换将 BR 通道交换之后显示效果正常，
因此, 在pytorch中如果需要读取文件, 则需要通过 im_cv2.permute(2, 0, 1) 将图像由HWC->CHW
对于非 png 格式的图片, plt, Image, imageio, io读取出来的数据是完全一样的,
但是 以上读取方式和 cv2 还是有一些不同, 看一看为什么
"""

index = np.argwhere(im_imageio-im_cv2[:,:,::-1] != 0) # 不相等的 index
print(f"index.shape = {index.shape}")

print(im_imageio[index[:,0], index[:,1],index[:,2]], im_cv2[:,:,::-1][index[:,0], index[:,1],index[:,2]]) # 对比差异

# 作差比较
s = im_imageio - im_cv2[:,:,::-1]
print(s[index[:,0], index[:,1],index[:,2]])



"""
原来imageio 和 cv2 读取到的图片数值会有略微的差异，比如 236 和 237 这样的差异;

值得注意的是, im_imageio 和 im_cv2image 作差之后仍为 image 对象，由于 image 对象不允许存在负值，所以有些负数会被加上 256，比如 -1 会自动转换为 255;
"""

#===================================================================================================================================
#  png 格式 单通道 图片
#===================================================================================================================================

imagepath='./Figures/bridge.png'
brige = imageio.imread(imagepath)


im_PIL, im_imageio, im_cv2, im_matplotlib, im_io = display(imagepath)
plt.imshow(im_cv2[:,:,::-1]) #交换 BGR 中的 BR
plt.show()



"""
图片为灰度图片时，除了 cv2 其他方式读取的图片都是单通道的，cv2 读取的图片依然是三通道的。

可以看出, cv读取的三通道图片无论交换通道顺序与否，显示效果都相同。

matplotlib 读取的结果也与其他两种不同，matplotlib 读取的结果为 0~1 之间的浮点数
"""
cmp1 = im_cv2[:,:,0]==im_cv2[:,:,1]
cmp2 = im_cv2[:,:,0]==im_cv2[:,:,2]
print(np.all(cmp1)) # all True
print(np.all(cmp2)) # all True
print(np.allclose(im_imageio, im_cv2[:,:,0]))

"""
cv2 对灰度图片的处理，是三通道完全一样，且任一通道都和其他读取方式（除了 matplotlib）读取的结果一样

和 imageio 读取结果略有不同的现象在但通道情况下似乎没有出现
"""
print(f"im_imageio = {im_imageio}")
print(f"im_matplotlib = {im_matplotlib}")




#=========================================================================================================
# 以下是总结 xx 读取图片, yy 保存图片， zz再读取图片的功能，xx,yy,zz可以取值为plt，cv2, imageio,共有27种可能
#=========================================================================================================
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio

#=================================================================================================
# 实验1
# plt读取 png ,plt画图(正常)，cv2画图(正常), cv2保存，再cv2读取，plt画图(一片黑)，cv2画图(一片黑),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()

cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32

cv2.imwrite('./Figures/lenaPltCv2Imageio.png', img[:,:,::-1])

img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8



# 实验1
# plt读取 jpg, plt画图(正常)，cv2画图(正常), cv2保存，再cv2读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32

cv2.imwrite('./Figures/flowerPltCv2Imageio.png', img[:,:,::-1])

img1 = cv2.imread('./Figures/flowerPltCv2Imageio.png')
plt.imshow(img1[:,:,::-1])
plt.show()
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8

#=================================================================================================
# 实验2
# plt读取 png ,plt画图(正常)，cv2画图(正常), cv2保存 png，再plt读取，plt画图(一片黑0)，cv2画图(一片黑),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

# 实验2
# plt读取 png ,plt画图(正常)，cv2画图(正常), cv2保存 png，再plt读取，plt画图(一片黑0)，cv2画图(一片黑),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
cv2.imwrite('./Figures/lenaPltCv2Imageio.jpg',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.jpg')
plt.imshow(img1)
plt.show()
cv2.imshow('src', img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

# 实验2
# plt读取 jpg, plt画图(正常)，cv2画图(正常), cv2保存，再plt读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()

cv2.imshow('src', img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
cv2.imwrite('./Figures/flowerPltCv2Imageio.png', img[:,:,::-1])


img1 = plt.imread('./Figures/flowerPltCv2Imageio.png')
plt.imshow(img1)
plt.show()

cv2.imshow('src', img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32


#=================================================================================================
# 实验3
# plt读取,plt画图(正常)，cv2画图(正常), cv2 保存，再 imageio 读取，plt画图(一片黑0)，cv2画图(一片黑),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)

img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32


# 实验3
# plt读取 jpg, plt画图(正常)，cv2画图(正常), cv2 保存，再 imageio 读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
cv2.imwrite('./Figures/flowerPltCv2Imageio.png',img[:,:,::-1])


img1 = imageio.imread('./Figures/flowerPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

#=================================================================================================
# 实验4
# plt读取png, plt画图(正常)，cv2画图(正常), plt保存 png，再cv2读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()

# 实验4
# plt读取 png, plt画图(正常)，cv2画图(正常), plt保存 jpg，再cv2读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.jpg',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.jpg')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()

# 实验4
# plt读取 jpg, plt画图(正常)，cv2画图(正常), plt保存 png，再cv2读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()

# 实验4
# plt读取 jpg, plt画图(正常)，cv2画图(正常), plt保存 jpg ，再cv2读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.jpg',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.jpg')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1]) # 需要转换通道顺序
plt.show()

#=================================================================================================
# 实验5
# plt读取 png, plt画图(正常), cv2画图(正常), plt保存 png, 再plt读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,-2::-1]) # 这时img1多出了最后一维，需要去掉然后翻转
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1)
plt.show()

# 实验5
# plt读取 png, plt画图(正常), cv2画图(正常), plt保存 jpg, 再plt读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.jpg',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.jpg')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1)
plt.show()

# 实验5
# plt读取 jpg, plt画图(正常)，cv2画图(正常), plt保存 png，再plt读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imsave('./Figures/flowerPltCv2Imageio.png',img)


img1 = plt.imread('./Figures/flowerPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,-2::-1]) # 这时img1多出了最后一维，需要去掉然后翻转
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1)
plt.show()

# 实验5
# plt读取 jpg, plt画图(正常)，cv2画图(正常), plt保存，再plt读取 jpg, plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imsave('./Figures/flowerPltCv2Imageio.jpg',img)


img1 = plt.imread('./Figures/flowerPltCv2Imageio.jpg')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1)
plt.show()

"""

2. plt保存，再plt读取的时候，发现是4通道的
"""
#=================================================================================================
# 实验6
# plt读取,plt画图(正常)，cv2画图(正常), plt保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imsave('./Figures/lenaPltCv2Imageio.png', img)


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,-2::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = uint8
plt.imshow(img1) # 需要转换通道顺序
plt.show()

#=================================================================================================
# 实验7
# plt读取,plt画图(正常)，cv2画图(正常), imageio保存(** TypeError: Cannot handle this data type: (1, 1, 3), <f4)，再imageio读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
imageio.imwrite('./Figures/lenaPltCv2Imageio.png', img)


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8

# 实验7
# plt读取jpg, plt画图(正常)，cv2画图(正常), imageio保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/flower.jpg')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1]) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
imageio.imwrite('./Figures/flowerPltCv2Imageio.png', img)


img1 = imageio.imread('./Figures/flowerPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8

"""
从这个实验可以看出, plt 读取的图片是 0-1之间的, 但是使用 imageio保存时会自动将像素点 x255 转换为[0, 255] 的像素点.
"""

#=================================================================================================
# 实验8
# plt读取,plt画图(正常)，cv2画图(正常), imageio保存，再plt读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32


imageio.imsave('./Figures/lenaPltCv2Imageio.png',img)  # imageio.imsave 和 imageio.imwrite一样


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

#=================================================================================================
# 实验9
# plt读取,plt画图(正常)，cv2画图(正常), imageio保存，再  cv2 读取，plt画图(正常)，cv2画图(正常),
img = plt.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"type(img) = {type(img)}\n")
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# type(img) = <class 'numpy.ndarray'>
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = float32
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"type(img1) = {type(img1)}\n")
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# type(img1) = <class 'numpy.ndarray'>
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1[:,:,::-1])
plt.show()


"""
从以上九个实验可以看出:
    1 .对于 png 图片, plt 读取进来的是 0-1 之间的值, 此时如果用 plt 再保存, 则会自动转为 0-255 保存, 不会有错; 如果是用 imageio 则会自动转为 0-255 保存, 此时相当于又恢复了像素点; 但是如果使用 cv 保存, 则不会自动转换为 0-255, 此时相当于保存后像素点全为 [0,1], 再读取的时候就是全黑图像;
    2. 对于 jpg 图片, 就不会有上述问题, 只需要注意通道顺序;

"""
#=================================================================================================
# 实验10
# cv2读取 png, plt画图(正常)，cv2画图(正常), cv2保存 png, 再plt读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

# 实验10
# cv2读取 png,plt画图(正常)，cv2画图(正常), cv2保存 jpg, 再plt读取, plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
cv2.imwrite('./Figures/lenaPltCv2Imageio.jpg',img)  # 保存时为BGR


img1 = plt.imread('./Figures/lenaPltCv2Imageio.jpg')
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32

# 实验10
# cv2读取 png, plt画图(正常)，cv2画图(正常), cv2保存 png, 再plt读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/flower.jpg')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
cv2.imwrite('./Figures/flowerPltCv2Imageio.png',img)  # 保存时为BGR


img1 = plt.imread('./Figures/flowerPltCv2Imageio.png')
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32


# 实验10
# cv2读取 png, plt画图(正常)，cv2画图(正常), cv2保存 png, 再plt读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/flower.jpg')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
cv2.imwrite('./Figures/flowerPltCv2Imageio.jpg',img)  # 保存时为BGR


img1 = plt.imread('./Figures/flowerPltCv2Imageio.jpg')
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
#=================================================================================================
# 实验11
# cv2读取,plt画图(正常)，cv2画图(正常), cv2保存，再cv2读取，plt画图(正常)，cv2画图(正常),
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
# 实验12
# cv读取,plt画图(正常)，cv2画图(正常), cv2保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])
plt.show()
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8

cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')

plt.imshow(img1)
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"type(img1) = {type(img1)}\n")
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# type(img1) = <class 'imageio.core.util.Array'>
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = uint8


#=================================================================================================
# 实验13
# cv2读取,plt画图(正常)，cv2画图(正常), plt保存，再plt读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8

plt.imsave('./Figures/lenaCvReadPltWrite.png',img)  # 保存时为BGR


img1 = plt.imread('./Figures/lenaCvReadPltWrite.png')
plt.imshow(img1[:,:,-2::-1])
plt.show()
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = float32

#=================================================================================================
# 实验14
# cv2读取,plt画图(正常)，cv2画图(正常), plt保存，再cv2读取，plt画图(正常)，cv2画图(正常),
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

plt.imsave('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8



"""
从以上八个实验可知：
1. cv2.imread 它默认认为图片存储的方式是RGB，而不管实际意义上的顺序，但是读取的时候反序了, 它读出的图片的通道是BGR。
2. cv2.imshow 它把被显示的图片当成BGR的顺序，而不管实际意义上的顺序，然后显示的时候变成RGB显示。
3. cv2.imwrite 它认为被保存的图片是BGR顺序，而不管实际意义上的顺序，保存的时候反序成RGB再保存。
4. plt默认认为被打开，保存，显示的图片的通道顺序为RGB。

所以将cv2和plt混用时需要注意顺序
"""

#=================================================================================================
# 实验15
# cv2读取,plt画图(正常)，cv2画图(正常), plt保存，再 imageio 读取，plt画图(正常)，cv2画图(正常),
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
plt.imsave('./Figures/lenaCvReadPltWrite.png',img)  # 保存时为BGR


img1 = imageio.imread('./Figures/lenaCvReadPltWrite.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 786432
# img1.dtype = uint8

plt.imshow(img1[:,:,-2::-1])  # 将BGR转成RGB
plt.show()
#=================================================================================================
# 实验16
# cv2读取,plt画图(正常)，cv2画图(正常), imageio保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])
plt.show()
cv2.imshow('src',img) # 需要转换通道顺序
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img1.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)

img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1])
plt.show()

#=================================================================================================
# 实验17
# cv2读取,plt画图(正常)，cv2画图(正常), imageio保存，再cv2读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])
plt.show()
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1)
plt.show()
#=================================================================================================
# 实验18
# cv2读取,plt画图(正常)，cv2画图(正常), imageio保存，再 plt 读取，plt画图(正常)，cv2画图(正常),
img = cv2.imread('./Figures/lena.png')
plt.imshow(img[:,:,::-1])
plt.show()
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1[:,:,::-1])
plt.show()

#=================================================================================================


# 实验19
# imageio读取,plt画图(正常)，cv2画图(正常), imageio保存，再plt读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)
plt.show()
cv2.imshow('src',img[:,:,::-1] ) # 将 RGB 转成 BGR
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = float32
plt.imshow(img1)  # 将BGR转成RGB
plt.show()
#=================================================================================================
# 实验20
# imageio读取,plt画图(正常)，cv2画图(正常), imageio保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
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
plt.show()
#=================================================================================================
# 实验21
# imageio读取,plt画图(正常)，cv2画图(正常), imageio保存，再cv读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
imageio.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imshow(img1[:,:,::-1])  # 将BGR转成RGB
plt.show()
#=================================================================================================
# 实验22
# imageio读取,plt画图(正常)，cv2画图(正常), plt保存，再plt读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()
cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,-2::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = float32
plt.imshow(img1[:,:,:3])
plt.show()
#=================================================================================================

# 实验23
# imageio读取,plt画图(正常)，cv2画图(正常), plt保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()

cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,-2::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = uint8

plt.imshow(img1[:,:,:3])  # 将BGR转成RGB
plt.show()


#=================================================================================================
# 实验24
# imageio读取,plt画图(正常)，cv2画图(正常), plt保存，再 cv2 读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()

cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imsave('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 4)
# img1.size = 1048576
# img1.dtype = uint8

plt.imshow(img1[:,:,::-1] )  # 将BGR转成RGB
plt.show()



#=================================================================================================
# 实验25
# imageio读取,plt画图(正常)，cv2画图(正常), cv2保存，再cv2读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()

cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 786432
# img1.dtype = uint8
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = cv2.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8
plt.imshow(img1)
plt.show()


#=================================================================================================

# 实验26
# imageio读取,plt画图(正常)，cv2画图(正常), cv2保存，再imageio读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()

cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = imageio.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1 )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = uint8

plt.imshow(img1[:,:,::-1])  # 将BGR转成RGB
plt.show()


#=================================================================================================
# 实验27
# imageio读取,plt画图(正常)，cv2画图(正常), cv2保存，再 plt 读取，plt画图(正常)，cv2画图(正常),
img = imageio.imread('./Figures/lena.png')
plt.imshow(img)  # 将BGR转成RGB
plt.show()

cv2.imshow('src',img[:,:,::-1] )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
cv2.imwrite('./Figures/lenaPltCv2Imageio.png',img)  # 保存时为BGR


img1 = plt.imread('./Figures/lenaPltCv2Imageio.png')
cv2.imshow('src',img1 )
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# img1.shape = (512, 512, 3)
# img1.size = 1048576
# img1.dtype = float32

plt.imshow(img1[:,:,::-1])  # 将BGR转成RGB
plt.show()




#========================================================================================
# 验证cv2和imageio能都保存和读取浮点数的图片
#========================================================================================

# 实验1
#==================================================
# cv读，乘以1.0，再cv保存, 再cv读
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



img1 = img*1.0001
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


# 实验 2
#==================================================
# cv读，乘以1.0，再cv保存, 再image读
#==================================================
img = cv2.imread('./Figures/baby.png', )
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1],)
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

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
plt.show()


# 实验 3
#==================================================
# cv读，乘以1.0，再cv保存, 再 plt 读
#==================================================
img = cv2.imread('./Figures/baby.png', )
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1],)
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = float32
plt.imshow(img2,)
plt.show()


# 实验 4
#==================================================
# cv读，乘以1.0，再plt保存(出错，plt如果保存浮点数，则只能是0-1的小数，不能是大于1的浮点数)，再cv读
#==================================================
img = cv2.imread('./Figures/baby.png', )
img1 = img*1.0001
plt.imsave('./Figures/babyFloat.png', img1) # ValueError: Floating point image RGB values must be in the 0..1 range.

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
print(f"img.shape = {img2.shape}") # (h,w,c)
print(f"img.size = {img2.size}") # 像素总数目
print(f"img.dtype = {img2.dtype}")
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()


# 实验 5
#==================================================
# cv读，乘以1.0，再image保存，再cv读
#==================================================
img = cv2.imread('./Figures/baby.png', )

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1],)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img2,)
plt.show()


# 实验 6
#==================================================
# cv读，乘以1.0，再image保存，再plt读
#==================================================
img = cv2.imread('./Figures/baby.png', )

cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1],)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# (512, 512, 3)
# 786432
# print(img)
plt.imshow(img2[:,:,::-1],)
plt.show()


# 实验 7
#==================================================
# cv读，乘以1.0，再 image 保存，再 image 读
#==================================================
img = cv2.imread('./Figures/baby.png', )
cv2.imshow('src',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = uint8
plt.imshow(img[:,:,::-1],)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2[:,:,::-1],)
plt.show()



# 实验 8
#==================================================
# plt读，乘以1.0，再cv保存, 再cv读，一片黑白
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2) # 一片黑白
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)# 一片黑白
plt.show()


# 实验 9
#==================================================
# plt读，乘以1.0，再 cv 保存, 再 imageio 读，一片黑白
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2) # 一片黑白
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)# 一片黑白
plt.show()



# 实验 10
#==================================================
# plt读，乘以1.0，再 cv 保存, 再 plt 读，一片黑白
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2) # 一片黑白
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)# 一片黑白
plt.show()


"""
以上三个实验可以看出，plt读取,cv2保存时，会失去所有像素信息，因为cv只能保存大于1的整数和浮点数，而plt读取的图片矩阵否是0-1之间的浮点数，所以用cv2保存和读取0-1之间的浮点数组成的矩阵会得到黑色。

"""

# 实验 11
#==================================================
# plt读，乘以1.0，再 imageio 保存, 再 plt 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)
plt.show()


# 实验 12
#==================================================
# plt读，乘以1.0，再 imageio 保存, 再 cv 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2[:,:,::-1],)
plt.show()


# 实验 13
#==================================================
# plt读，乘以1.0，再 imageio 保存, 再 imageio 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)
plt.show()


# 实验 14
#==================================================
# plt读，乘以1.0，再 plt 保存, 再 imageio 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.000
plt.imsave('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,-2::-1])
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 4)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2,)
plt.show()


# 实验 15
#==================================================
# plt读，乘以1.0，再 plt 保存, 再 cv2 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.0
plt.imsave('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2[:,:,::-1],)
plt.show()


# 实验 16
#==================================================
# plt读，乘以1.0，再 plt 保存, 再 plt 读，
#==================================================
img = plt.imread('./Figures/baby.png', )

cv2.imshow('src',img[:,:,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# img.shape = (512, 512, 3)
# img.size = 786432
# img.dtype = float32
plt.imshow(img,)
plt.show()



img1 = img*1.000
plt.imsave('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2[:,:,-2::-1])
cv2.waitKey()
cv2.destroyAllWindows() #
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 4)
# img2.size = 786432
# img2.dtype = float32
plt.imshow(img2,)
plt.show()


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
plt.show()



img1 = img*1.0001
img1 = img.astype(np.uint8)
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
plt.show()


# 实验 18
#==================================================
# imageio 读，乘以1.0，再 imageio 保存，再 cv2 读
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
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

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
plt.imshow(img2[:,:,::-1],)
plt.show()


# 实验 19
#==================================================
# imageio 读，乘以1.0，再 imageio 保存，再 plt 读
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
plt.show()



img1 = img*1.0001
imageio.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
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
plt.show()



# 实验 20
#==================================================
# imageio 读，乘以1.0，再 plt 保存(出错，plt如果保存浮点数，则只能是0-1的小数，不能是大于1的浮点数)，再 plt 读
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
plt.show()



img1 = img*1.0001
plt.imsave('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
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
plt.show()


# 实验 21
#==================================================
# imageio 读，乘以1.0，再 cv2 保存 ，再 plt 读
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
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = plt.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2[:,:,::-1],)
plt.show()


# 实验 22
#==================================================
# imageio 读，乘以1.0，再 cv2 保存 ，再 cv2 读
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
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = cv2.imread('./Figures/babyFloat.png', )
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
plt.show()


# 实验 23
#==================================================
# imageio 读，乘以1.0，再 cv2 保存 ，再 imageio 读
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
plt.show()



img1 = img*1.0001
cv2.imwrite('./Figures/babyFloat.png', img1)

img2 = imageio.imread('./Figures/babyFloat.png', )
cv2.imshow('src',img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8
plt.imshow(img2[:,:,::-1],)
plt.show()





