#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:26:57 2023

@author: jack

使用opencv读取的图像是[H,W,C]大小的，数据格式是 np.uint8 ，经过 ToTensor() 会进行归一化。而其他的数据类型（如 np.int8）经过 ToTensor() 数值不变，不进行归一化，后面会详细讲述。并且经过ToTensor()后图像格式变为 [C,H,W]。

transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))]) 使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
这一组值是怎么来的？ 这一组值是从imagenet训练集中抽样算出来的。


经过transforms.Normalize数据不一定服从正态分布！

这里减去均值，除以标准差只是将数据进行标准化处理，并不会改变原始数据的分布！

另外是每一个channels上所有batch的数据服从均值为0，标准差为1。

transforms.Normalize 作用的对象必须是 tensor, 否则报错：
TypeError: img should be Tensor Image. Got <class 'PIL.PngImagePlugin.PngImageFile'>
因此一般在 transforms.Normalize 前加 transforms.ToTensor(),

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

# （1） np.uint8 类型
data = np.array([
    [0, 5, 10, 20, 0],
    [255, 125, 180, 255, 196]
], dtype=np.uint8)

CompTrans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(std=(0.5 ),mean=(0.5 ))])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean


tensor = CompTrans(data)
print(tensor)
# tensor([[[0.0000, 0.0196, 0.0392, 0.0784, 0.0000],
#          [1.0000, 0.4902, 0.7059, 1.0000, 0.7686]]])


# （2）非 np.uint8 类型
import numpy as np
from torchvision import transforms

data = np.array([
    [0, 5, 10, 20, 0],
    [255, 125, 180, 255, 196]
])*1.0      # data.dtype = int32

CompTrans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(std=(0.5 ),mean=(0.5 ))])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean


tensor = CompTrans(data)
print(tensor)
"""
tensor([[[  0,   5,  10,  20,   0],
         [255, 125, 180, 255, 196]]], dtype=torch.int32)
"""


"""
（1）np.array 整型的默认数据类型为 np.int32，经过 ToTensor() 后数值不变，不进行归一化。
（2）np.array 整型的默认数据类型为 np.float64，经过 ToTensor() 后数值不变，不进行归一化。
（3）opencv 读取的图像格式为 np.array，其数据类型为 np.uint8。
    经过 ToTensor() 后数值由 [0,255] 变为 [0,1]，通过将每个数据除以255进行归一化。
（4）经过 ToTensor() 后，HWC 的图像格式变为 CHW 的 tensor 格式。
（5）np.uint8 和 np.int8 不一样，uint8是无符号整型，数值都是正数。
（6）ToTensor() 可以处理任意 shape 的 np.array，并不只是三通道的图像数据。

"""

# 在自己建立 dataset 迭代器时，一般操作是检索数据集图像的路径，然后使用 PIL 库或 opencv库读取图片路径。
#======================================= PIL =================================================
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


# 展示方法一
img.show()


# 展示方法2
fig = plt.figure()
plt.tight_layout()
plt.imshow(img,  cmap='Greys_r')
plt.title(" 1")
plt.show()

# 展示方法3
cv2.imshow('src', img2[:,:,::-1]) # 需要转换通道顺序 . #  [:,:,::-1]  img   [:,:,-2::-1]
cv2.waitKey()
cv2.destroyAllWindows()



# 展示方法4
skimage.io.imshow(img2)
skimage.io.show()






CompTrans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

img1 = CompTrans(img)
print(f"\n\ntype(img1) = {type(img1)}")
print(f"img1.shape = {img1.shape}") # (h,w,c)
print(f"img1.size = {img1.size}") # 像素总数目
print(f"img1.dtype = {img1.dtype}")
# type(img1) = <class 'torch.Tensor'>
# img1.shape = torch.Size([3, 512, 512])
# img1.size = <built-in method size of Tensor object at 0x7f24887580e0>
# img1.dtype = torch.float32


#======================================= opencv =================================================
import numpy as np
import cv2


# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

imgcv = cv2.imread(source, )

### 查看图像实例的属性
print(f"\n\ntype(imgcv) = {type(imgcv)}")
print(f"imgcv.shape = {imgcv.shape}") # (h,w,c)
print(f"imgcv.size = {imgcv.size}") # 像素总数目
print(f"imgcv.dtype = {imgcv.dtype}")
# type(imgcv) = <class 'numpy.ndarray'>
# imgcv.shape = (512, 512, 3)
# imgcv.size = 786432
# imgcv.dtype = uint8


CompTrans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

# np.ascontiguousarray(image): 解决 numpy 内存连续的问题
# imgcv1 = transforms.ToTensor()(np.ascontiguousarray(imgcv[:,:,::-1]))
# or
imgcv1 = CompTrans(imgcv[:,:,::-1].copy())
print(f"\n\ntype(imgcv1) = {type(imgcv1)}")
print(f"imgcv1.shape = {imgcv1.shape}") # (h,w,c)
print(f"imgcv1.size = {imgcv1.size}") # 像素总数目
print(f"imgcv1.dtype = {imgcv1.dtype}")
# type(imgcv1) = <class 'torch.Tensor'>
# imgcv1.shape = torch.Size([3, 512, 512])
# imgcv1.size = <built-in method size of Tensor object at 0x7f24b6905ef0>
# imgcv1.dtype = torch.float32


print(f"np.allclose(img2, imgcv[:,:,::-1]) =  {np.allclose(img2, imgcv[:,:,::-1])}")
print(f"np.allclose(img1, imgcv1) =  {np.allclose(img1, imgcv1)}")
# np.allclose(img2, imgcv[:,:,::-1]) =  True
# np.allclose(img1, imgcv1) =  True




#======================================= imageio =================================================
import PIL
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import sys

# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

imgio = imageio.imread(source, )

### 查看图像实例的属性
print(f"\n\ntype(imgio) = {type(imgio)}")
print(f"imgio.shape = {imgio.shape}") # (h,w,c)
print(f"imgio.size = {imgio.size}") # 像素总数目
print(f"imgio.dtype = {imgio.dtype}")
# type(imgio) = <class 'numpy.ndarray'>
# imgio.shape = (512, 512, 3)
# imgio.size = 786432
# imgio.dtype = uint8


CompTrans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

imgio1 = CompTrans( imgio)
print(f"\n\ntype(imgio1) = {type(imgcv1)}")
print(f"imgio1.shape = {imgio1.shape}") # (h,w,c)
print(f"imgio1.size = {imgio1.size}") # 像素总数目
print(f"imgio1.dtype = {imgio1.dtype}")
# type(imgio1) = <class 'torch.Tensor'>
# imgio1.shape = torch.Size([3, 512, 512])
# imgio1.size = <built-in method size of Tensor object at 0x7f2462462590>
# imgio1.dtype = torch.float32


print(f"\n\nnp.allclose(img2, imgio) =  {np.allclose(img2, imgio)}")
print(f"np.allclose(img1, imgio1) =  {np.allclose(img1, imgio1)}\n")
# np.allclose(img2, imgio) =  True
# np.allclose(img1, imgio1) =  True



#======================================= skimage.io. =================================================
import PIL
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import sys

# 原图是 png 还是 jpg
source = '/home/jack/公共的/Python/PytorchTutor/Pytorch/torchvision_transforms/Figures/baby.png' # './Figures/lena.png'  ./Figures/flower.jpg'

# source = './Figures/flower.jpg'

io = skimage.io.imread(source, )

### 查看图像实例的属性
print(f"\n\ntype(io) = {type(io)}")
print(f"io.shape = {io.shape}") # (h,w,c)
print(f"io.size = {io.size}") # 像素总数目
print(f"io.dtype = {io.dtype}")
# type(io) = <class 'numpy.ndarray'>
# io.shape = (512, 512, 3)
# io.size = 786432
# io.dtype = uint8


CompTrans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean

io1 = CompTrans( io)
print(f"\n\ntype(io1) = {type(io1)}")
print(f"io1.shape = {io1.shape}") # (h,w,c)
print(f"io1.size = {io1.size}") # 像素总数目
print(f"io1.dtype = {io1.dtype}")
# type(io1) = <class 'torch.Tensor'>
# io1.shape = torch.Size([3, 512, 512])
# io1.size = <built-in method size of Tensor object at 0x7f24644c0180>
# io1.dtype = torch.float32


print(f"\n\nnp.allclose(img2, io) =  {np.allclose(img2, io)}")
print(f"np.allclose(img1, io1) =  {np.allclose(img1, io1)}\n")
# np.allclose(img2, io) =  True
# np.allclose(img1, io1) =  True



























