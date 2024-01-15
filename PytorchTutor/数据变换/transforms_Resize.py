#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:20:47 2023

@author: jack

transforms.Resize(size)
对载入的图片数据按照我们的需要进行缩放，传递给这个类的size可以是一个整型数据，也可以是一个类似于 (h ,w) 的序列。如果输入是个(h,w)的序列，h代表高度，w代表宽度，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸，相当于force。如果使用的是一个整型数据，则将图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。

transforms.Resize( 122) 一般的作用对象是 PIL image 或 tensor, 否则 TypeError: img should be PIL Image. Got <class 'numpy.ndarray'>, 所以读取图片的时候一般顺序是
transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.


transforms.Scale(size)
对载入的图片数据我们的需要进行缩放，用法和torchvision.transforms.Resize类似。。传入的size只能是一个整型数据，size是指缩放后图片最小边的边长。举个例子，如果原图的height>width,那么改变大小后的图片大小是(size*height/width, size)。

"""



import numpy as np
from torchvision import transforms
import PIL
from PIL import Image
from scipy import misc, ndimage
import imageio.v2 as imageio

import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import sys
from skimage import io


#======================================= PIL  transforms.Resize(224) =================================================
import PIL


# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

img = PIL.Image.open(source)
### 查看图像实例的属性
print(f"img1 = {img}")
print(f"type(img) = {type(img)}, img.format = {img.format}, img.width = {img.width}, img.height = {img.height}, img.size = {img.size}, img.mode = {img.mode}\n")
# img1 = <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7F248E116040>
# type(img) = <class 'PIL.PngImagePlugin.PngImageFile'>, img.format = PNG, img.width = 512, img.height = 512, img.size = (512, 512), img.mode = RGB


img2 = np.array(img)
print(f"\n\ntype(img2) = {type(img2)}")
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8


CompTrans = transforms.Compose([ transforms.Resize((224, 900)) ])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

#  TypeError: img should be PIL Image. Got <class 'numpy.ndarray'>
img1 =  CompTrans(img)
### 查看图像实例的属性
print(f"img1 = {img1}")
print(f"type(img1) = {type(img1)}, img1.format = {img1.format}, img1.width = {img1.width}, img1.height = {img1.height}, img1.size = {img1.size}, img1.mode = {img1.mode}\n")
# img1 = <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7F248E116040>
# type(img) = <class 'PIL.PngImagePlugin.PngImageFile'>, img.format = PNG, img.width = 512, img.height = 512, img.size = (512, 512), img.mode = RGB



# 展示方法一
img1.show()

# 展示方法二, plt 可以显示<class 'PIL.JpegImagePlugin.JpegImageFile'> 也可以显示 np.array
fig = plt.figure()
plt.tight_layout()
plt.imshow(img1)
plt.title(" 1")
plt.show()

# 展示方法三, cv2 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
cv2.imshow('src', np.array(img1)[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法四, io 只能显示 np.array,  
skimage.io.imshow(np.array(img1))
skimage.io.show()




# 在自己建立 dataset 迭代器时，一般操作是检索数据集图像的路径，然后使用 PIL 库或 opencv库读取图片路径。
#======================================= PIL  transforms.Resize(224) =================================================
import PIL


# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

img = PIL.Image.open(source)
### 查看图像实例的属性
print(f"img1 = {img}")
print(f"type(img) = {type(img)}, img.format = {img.format}, img.width = {img.width}, img.height = {img.height}, img.size = {img.size}, img.mode = {img.mode}\n")
# img1 = <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7F248E116040>
# type(img) = <class 'PIL.PngImagePlugin.PngImageFile'>, img.format = PNG, img.width = 512, img.height = 512, img.size = (512, 512), img.mode = RGB


img2 = np.array(img)
print(f"\n\ntype(img2) = {type(img2)}")
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8


CompTrans = transforms.Compose([ transforms.ToTensor(), transforms.Resize(224) ])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

#  TypeError: img should be PIL Image. Got <class 'numpy.ndarray'>
img1 =  CompTrans(img)
print(f"\n\ntype(img1) = {type(img1)}")
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# type(img1) = <class 'torch.Tensor'>
# img1.shape = torch.Size([3, 224, 224])
# img1.size = <built-in method size of Tensor object at 0x7fe5817e3e50>
# img1.dtype = torch.float32

img2 = np.array(img1)
print(f"\n\ntype(img2) = {type(img2)}")
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (3, 224, 224)
# img2.size = 150528
# img2.dtype = float32


# 展示方法二, plt 可以显示<class 'PIL.JpegImagePlugin.JpegImageFile'> 也可以显示 np.array
fig = plt.figure()
plt.tight_layout()
plt.imshow(img2.transpose((1, 2, 0)))
plt.title(" 1")
plt.show()

# 展示方法三, cv2 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
cv2.imshow('src', np.transpose(img2, (1, 2, 0))[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法四, io 只能显示 np.array,  
skimage.io.imshow(np.transpose(img2, (1, 2, 0)))
skimage.io.show()




#======================================= cv2 =================================================
import PIL


# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

img = cv2.imread(source )[:,:,::-1]
print(f"\n\ntype(img) = {type(img)}")
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8


CompTrans = transforms.Compose([ transforms.Resize(224) ])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

img1 = PIL.Image.fromarray(img.astype(np.uint8))
img1 = CompTrans(img1)
### 查看图像实例的属性
print(f"img1 = {img1}")
print(f"type(img1) = {type(img1)}, img1.format = {img1.format}, img1.width = {img1.width}, img1.height = {img1.height}, img1.size = {img1.size}, img1.mode = {img1.mode}\n")
# img1 = <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x7F248E116040>
# type(img) = <class 'PIL.PngImagePlugin.PngImageFile'>, img.format = PNG, img.width = 512, img.height = 512, img.size = (512, 512), img.mode = RGB




# 展示方法一
img1.show()

# 展示方法二, plt 可以显示<class 'PIL.JpegImagePlugin.JpegImageFile'> 也可以显示 np.array
fig = plt.figure()
plt.tight_layout()
plt.imshow(img1 )
plt.title(" 1")
plt.show()

# 展示方法三, cv2 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
cv2.imshow('src', np.array(img1)[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法四, io 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
skimage.io.imshow(np.array(img1) )
skimage.io.show()



#======================================= cv2 =================================================
import PIL


# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

img = cv2.imread(source )[:,:,::-1].copy()
print(f"\n\ntype(img) = {type(img)}")
print(f"img.shape = {img.shape}") # (h,w,c)
print(f"img.size = {img.size}") # 像素总数目
print(f"img.dtype = {img.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (512, 512, 3)
# img2.size = 786432
# img2.dtype = uint8


CompTrans = transforms.Compose([ transforms.ToTensor(), transforms.Resize(224) ])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean


img1 = CompTrans(img)
### 查看图像实例的属性
print(f"\n\ntype(img1) = {type(img1)}")
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# type(img1) = <class 'torch.Tensor'>
# img1.shape = torch.Size([3, 224, 224])
# img1.size = <built-in method size of Tensor object at 0x7fe57df8a720>
# img1.dtype = torch.float32


img2 = np.array(img1)
print(f"\n\ntype(img2) = {type(img2)}")
print(f"img2.shape = {img2.shape}") # (h,w,c)
print(f"img2.size = {img2.size}") # 像素总数目
print(f"img2.dtype = {img2.dtype}")
# type(img2) = <class 'numpy.ndarray'>
# img2.shape = (3, 224, 224)
# img2.size = 150528
# img2.dtype = float32


# 展示方法二, plt 可以显示<class 'PIL.JpegImagePlugin.JpegImageFile'> 也可以显示 np.array
fig = plt.figure()
plt.tight_layout()
plt.imshow(img2.transpose((1, 2, 0)))
plt.title(" 1")
plt.show()

# 展示方法三, cv2 只能显示 np.array, 所以得先把 PIL 的读取结果转为  np.array
cv2.imshow('src', np.transpose(img2, (1, 2, 0))[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法四, io 只能显示 np.array,  
skimage.io.imshow(np.transpose(img2, (1, 2, 0)))
skimage.io.show()






























































































































