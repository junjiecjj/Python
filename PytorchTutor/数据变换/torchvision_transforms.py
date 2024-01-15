#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:36:13 2022

@author: jack

https://wxler.github.io/2020/11/24/213959/


"""




from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import imageio.v2 as imageio


#torchvision.transforms常用变换类

#首先使用PIL加载原始图片
img = Image.open("/home/jack/公共的/MLData/flower.jpg")
print(img.size)
plt.imshow(img)


img = imageio.imread("/home/jack/公共的/MLData/flower.jpg", mode = 'RGB')
print(img.size)
plt.imshow(img)


##transforms.Compose
#transforms.Compose类看作一种容器，它能够同时对多种数据变换进行组合。传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作。

transformer = transforms.Compose([                                
    transforms.Resize(256),
    transforms.transforms.RandomResizedCrop((224), scale = (0.5,1.0)),
    transforms.RandomHorizontalFlip(),
])
test_a = transformer(img)
plt.imshow(test_a)



#transforms.Normalize(mean, std)
#这里使用的是标准正态分布变换，这种方法需要使用原始数据的均值（Mean）和标准差（Standard Deviation）来进行数据的标准化，在经过标准化变换之后，数据全部符合均值为0、标准差为1的标准正态分布。计算公式如下：

#在这里插入图片描述
#一般来说，mean和std是实现从原始数据计算出来的，对于计算机视觉，更常用的方法是从样本中抽样算出来的或者是事先从相似的样本预估一个标准差和均值。如下代码，对三通道的图片进行标准化：



# 标准化是把图片3个通道中的数据整理到规范区间 x = (x - mean(x))/stddev(x)
# [0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])







#transforms.Resize(size)
#对载入的图片数据按照我们的需要进行缩放，传递给这个类的size可以是一个整型数据，也可以是一个类似于 (h ,w) 的序列。如果输入是个(h,w)的序列，h代表高度，w代表宽度，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸，相当于force。如果使用的是一个整型数据，则将图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。

# 等比缩放
test1 = transforms.Resize(224)(img)
print(test1.size)
plt.imshow(test1)



#transforms.Scale(size)
#对载入的图片数据我们的需要进行缩放，用法和torchvision.transforms.Resize类似。。传入的size只能是一个整型数据，size是指缩放后图片最小边的边长。举个例子，如果原图的height>width,那么改变大小后的图片大小是(size*height/width, size)。

# 等比缩放
test2 = transforms.scale(224)(img)
print(test2.size)
plt.imshow(test2)





# transforms.CenterCrop(size)
#以输入图的中心点为中心点为参考点，按我们需要的大小进行裁剪。传递给这个类的参数可以是一个整型数据，也可以是一个类似于(h,w)的序列。如果输入的是一个整型数据，那么裁剪的长和宽都是这个数值
test3 = transforms.CenterCrop((500,500))(img)
print(test3.size)
plt.imshow(test3)


test4 = transforms.CenterCrop(224)(img)
print(test4.size)
plt.imshow(test4)





# transforms.RandomCrop(size)
#用于对载入的图片按我们需要的大小进行随机裁剪。传递给这个类的参数可以是一个整型数据，也可以是一个类似于(h,w)的序列。如果输入的是一个整型数据，那么裁剪的长和宽都是这个数值
test5 = transforms.RandomCrop(224)(img)
print(test5.size)
plt.imshow(test5)


test6 = transforms.RandomCrop((300,300))(img)
print(test6.size)
plt.imshow(test6)





#transforms.RandomResizedCrop(size,scale)
#先将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为size的大小。即先随机采集，然后对裁剪得到的图像安装要求缩放，默认scale=(0.08, 1.0)。scale是一个面积采样的范围，假如是一个100*100的图片，scale = (0.5,1.0)，采样面积最小是0.5*100*100=5000，最大面积就是原图大小100*100=10000。先按照scale将给定图像裁剪，然后再按照给定的输出大小进行缩放。

test9 = transforms.RandomResizedCrop(224)(img)
print(test9.size)
plt.imshow(test9)



test9 = transforms.RandomResizedCrop(224,scale=(0.5,0.8))(img)
print(test9.size)
plt.imshow(test9)





#transforms.RandomHorizontalFlip
#用于对载入的图片按随机概率进行水平翻转。我们可以通过传递给这个类的参数自定义随机概率，如果没有定义，则使用默认的概率值0.5。
test7 = transforms.RandomHorizontalFlip()(img)
print(test7.size)
plt.imshow(test7)






#transforms.RandomVerticalFlip
#用于对载入的图片按随机概率进行垂直翻转。我们可以通过传递给这个类的参数自定义随机概率，如果没有定义，则使用默认的概率值0.5。

test8 = transforms.RandomVerticalFlip()(img)
print(test8.size)
plt.imshow(test8)





"""
transforms.RandomRotation
功能：按照degree随机旋转一定角度
degree：加入degree是10，就是表示在（-10，10）之间随机旋转，如果是（30，60），就是30度到60度随机旋转
resample是重采样的方法
center表示中心旋转还是左上角旋转

transforms.RandomRotation(
    degrees,
    resample=False,
    expand=False,
    center=None,
    fill=None,
)
"""

test10 = transforms.RandomRotation((30,60))(img)
print(test10.size)
plt.imshow(test10)



#transforms.ToTensor
#用于对载入的图片数据进行类型转换，将之前构成PIL图片的数据转换成Tensor数据类型的变量，让PyTorch能够对其进行计算和处理。

#transforms.ToPILImage
#用于将Tensor变量的数据转换成PIL图片数据，主要是为了方便图片内容的显示。





# RandomResizedCrop 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
print("原图大小：",img.size)
# Crop代表剪裁到某个尺寸
data1 = transforms.RandomResizedCrop(224)(img)
# data1、data2、data3尺寸一样，长宽都是224*224  size也可以是一个Integer，在这种情况下，切出来的图片的形状是正方形
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

# 放四个格，布局为2*2
plt.subplot(2,2,1),plt.imshow(img),plt.title("Original")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("Transform 1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("Transform 2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("Transform 3")
plt.show()


