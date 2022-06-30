#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:58:29 2022

@author: jack
"""
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#==============================================================================================
# https://www.cnblogs.com/skyfsm/p/8276501.html
#==============================================================================================

import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./lena.png')
cv2.imshow('src',img)
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()


import cv2
import numpy as np

#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./lena.png')
cv2.imshow('src',img[256:,256:,:])
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()

gray = cv2.imread('./lena.png',cv2.IMREAD_GRAYSCALE) #灰度图
cv2.imshow('gray',gray)
print(gray.shape)
print(gray.size)
#print(gray)
cv2.waitKey()
cv2.destroyAllWindows()

#也可以这么写，先读入彩色图，再转灰度图
src = cv2.imread('./lena.png')
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
print(gray.shape)
print(gray.size)
#print(gray)
cv2.waitKey()



import cv2
#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('./lena.png')
cv2.imshow('src',img)
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()


#注意到，opencv读入的图片的彩色图是一个channel last的三维矩阵（h,w,c），即（高度，宽度，通道）
#有时候在深度学习中用到的的图片矩阵形式可能是channel first，那我们可以这样转一下
print(img.shape)
img = img.transpose(2,0,1)
print(img.shape)

#有时候还要扩展维度，比如有时候我们需要预测单张图片，要在要加一列做图片的个数，可以这么做
img = np.expand_dims(img, axis=0)
print(img.shape)

# (1, 3, 512, 512)




#因为opencv读入的图片矩阵数值是0到255，有时我们需要对其进行归一化为0~1
img3 = cv2.imread('./lena.png')
img3 = img3.astype("float") / 255.0  #注意需要先转化数据类型为float
print(img3.dtype)
print(img3)

#存储图片
cv2.imwrite('test1.jpg',img3) #得到的是全黑的图片，因为我们把它归一化了
#所以要得到可视化的图，需要先*255还原
img3 = img3 * 255
cv2.imwrite('test2.jpg',img3)  #这样就可以看到彩色原图了

# opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！
#opencv读入的矩阵是BGR，如果想转为RGB，可以这么转
img4 = cv2.imread('./lena.png')
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)


#分离通道
img5 = cv2.imread('./lena.png')
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

b,g,r = cv2.split(img5)
#合并通道
img5 = cv2.merge((b,g,r))
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

#也可以不拆分
img5[:,:,2] = 0  #将红色通道值全部设0
cv2.imshow('src',img5)
cv2.waitKey()


#也可以不拆分
img5[:,:,0] = 0  #将红色通道值全部设0
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()


#也可以不拆分
img5[:,:,1] = 0  #将红色通道值全部设0
cv2.imshow('src',img5)
cv2.waitKey()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
import numpy as np
#与opencv结合使用
import cv2
im2 = cv2.imread('./lena.png')
plt.imshow(im2)
plt.axis('off')
plt.show()
#发现图像颜色怪怪的，原因当然是我们前面提到的RGB顺序不同的原因啦,转一下就好
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
plt.imshow(im2)
plt.axis('off')
plt.show()
#所以无论用什么库读进图片，只要把图片改为矩阵，那么matplotlib就可以处理了



import imageio
im2 = imageio.imread('./lena.png')
print(im2.dtype)
print(im2.size)
print(im2.shape)
plt.imshow(im2)
plt.show()
#print(im2)
imageio.imsave('imageio.png',im2)

# uint8
# 786432
# (512, 512, 3)


import cv2
import numpy as np

n=4   # n max  is 4 RGBA
img = np.random.randint(0,255,(512,512,n),dtype=np.uint8)

cv2.imshow('src',img)
print(img.shape) # (h,w,c)
print(img.size) # 像素总数目
print(img.dtype)
# (512, 512, 3)
# 786432
# print(img)
cv2.waitKey()
cv2.destroyAllWindows()
#==============================================================================================
#  
#==============================================================================================



# if __name__ =='__main__':

#修改的尺寸大小
size_=[320,480]
img_path='./origin.png'


w = size_[0]
h = size_[1]
img=cv2.imread(img_path)
cv2.imshow('origin', img)
if cv2.waitKey(0) & 0xFF == ord('\x1b'):
    print("I'm done")
    cv2.destroyAllWindows()

# if key == ord('s'): # wait  for 's' key to save and exit
#     #cv2.imwrite('1.png',img)
#     cv2.destroyAllWindows()
# else:
#     cv2.destroyAllWindows()
cv2.destroyAllWindows()

#label_im=cv2.imread(path,cv2.IMREAD_UNCHANGED)
#修改图像的尺寸大小
new_array = cv2.resize(img, (320, 480), interpolation=cv2.INTER_CUBIC)#注意指定大小的格式是(宽度,高度), 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
data=np.array(new_array,dtype='int32')
#修改B通道的像素值
for i in range(data[:,:,0].shape[0]):
    for j in range(data[:,:,0].shape[1]):
        if data[:,:,0][i][j]>155:
            data[:,:,0][i][j]=255
        else:
            data[:,:,0][i][j]=0
#修改G通道的像素值
for i in range(data[:,:,1].shape[0]):
    for j in range(data[:,:,1].shape[1]):
        if data[:,:,1][i][j]>155:
            data[:,:,1][i][j]=255
        else:
            data[:,:,1][i][j]=0
#修改R通道的像素值
for i in range(data[:,:,2].shape[0]):
    for j in range(data[:,:,2].shape[1]):
        if data[:,:,2][i][j]>155:
            data[:,:,2][i][j]=255
        else:
            data[:,:,2][i][j]=0


#修改后的尺寸和修改后的像素值保存下来
save_img='./change.png'
cv2.imwrite(save_img, data)
print(data[100])


#===================================================================================
#
#===================================================================================
# 图6-1中的矩阵
img1 = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
], dtype=np.uint8)

# 用matplotlib存储
plt.imsave('img_pyplot.jpg', img1)

# 用OpenCV存储
cv2.imwrite('img_cv2.jpg', img1)
img1_1 = cv2.imread('img_cv2.jpg')

img1_2 = np.array(img1_1)


#===================================================================================
# https://www.cnblogs.com/hanxiaosheng/p/9559996.html
#===================================================================================
# 读取一张四川大录古藏寨的照片
img3 = cv2.imread('./origin.png')

# 缩放成200x200的方形图像
img_200x200 = cv2.resize(img3, (200, 200))

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_200x300 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_NEAREST)

# 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
img_300x300 = cv2.copyMakeBorder(img3, 50, 50, 0, 0,
                                       cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))

# 对照片中树的部分进行剪裁
patch_tree = img[20:150, -180:-50]

cv2.imwrite('cropped_tree.png', patch_tree)
cv2.imwrite('resized_200x200.png', img_200x200)
cv2.imwrite('resized_200x300.png', img_200x300)
cv2.imwrite('bordered_300x300.png', img_300x300)


#===================================================================================
# https://www.cnblogs.com/hanxiaosheng/p/9559996.html
#===================================================================================
# 读取一张四川大录古藏寨的照片
img4 = cv2.imread('./origin.png')
print("img4.shape = {}, {}".format(img4.shape,5)) #img4.shape=(高度,宽度,通道数)
#cv2默认为 BGR顺序，而其他软件一般使用RGB，所以需要转换
img4_1 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序

h1, w1, _ = img4_1.shape #返回height，width，以及通道数，不用所以省略掉



#===================================================================================
# https://www.ryanxin.cn/archives/340
#===================================================================================
r = np.array([[243 for i in range(50)] for j in range(50)]).astype(int)
g = np.array([[67 for i in range(50)] for j in range(50)]).astype(int)
b = np.array([[24 for i in range(50)] for j in range(50)]).astype(int)
img6 = cv2.merge([r,g,b])  #合并一张50x50的纯色RGB图像
cv2.imwrite('save.png', img6)  #图像的保存




#===================================================================================
# http://www.4k8k.xyz/article/LaoYuanPython/111351840
#===================================================================================

img7 = cv2.imread('./origin.png',cv2.IMREAD_UNCHANGED)
print("img7.shape = {}".format(img7.shape))
imgBlue = img[:,:,0]
imgGreen = img[:, :, 1]
imgRed = img[:, :, 2]

#读取完毕，进行通道分离（四通道）：
B,G,R,A = cv2.split(img7)
#print("len(ret) = {}".format(len(ret)))

#B,G,R,A = ret
print("B.shape = {}".format(B.shape))

imgBGR = cv2.merge((B,G,R))
print("imgBGR.shape = {}".format(imgBGR.shape))




#===================================================================================
# http://www.juzicode.com/opencv-python-split-merge/
#===================================================================================
def show_img(win_name,img,wait_time=0,img_ratio=0.5,is_show=True):
    if is_show is not True:
        return
    rows = img.shape[0]
    cols = img.shape[1]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL )#cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(win_name,(int(cols*img_ratio),int(rows*img_ratio)))
    cv2.imshow(win_name,img)
    if wait_time >= 0:
        if cv2.waitKey(0) & 0xFF == ord('\x1b'):
            print("I'm done")
            cv2.destroyAllWindows()

img = cv2.imread('./lena.png')
#img = cv2.imread('..\\opencv-logo.png',cv2.IMREAD_UNCHANGED)
if img is not None and len(img.shape)==3: #彩色图像才可以做通道分离
    print('img.shape:',img.shape)
    show_img('img',img,-1)

    #如果是3通道返回结果用b,g,r= cv2.split(img)接收分离结果，如果是4通道用b,g,r,a = cv2.split(img)接收分离结果：
    if img.shape[2] == 3:                 #如果是3通道，分离出3个图像实例
        b,g,r = cv2.split(img)
        show_img('b',b,-1)
        show_img('g',g,-1)
        show_img('r',r,-1)
        if cv2.waitKey(0) & 0xFF == ord('\x1b'):
            print("I'm done")
            cv2.destroyAllWindows()
    elif img.shape[2] == 4:               #如果是4通道
        b,g,r,a = cv2.split(img)
        show_img('b',b,-1)
        show_img('g',g,-1)
        show_img('r',r,-1)
        show_img('a',a,-1)
        if cv2.waitKey(0) & 0xFF == ord('\x1b'):
            print("I'm done")
            cv2.destroyAllWindows()

    #另外一种方法是利用numpy数组的切片或索引操作，比如用img[:,:,0]分离出0通道或b通道，img[:,:,1]对应g通道，img[:,:,2]对应r通道，如果有img[:,:,3]则对应alpha通道。
    if img.shape[2] == 3:                 #如果是3通道，分离出3个图像实例
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
        show_img('b',b,-1)
        show_img('g',g,-1)
        show_img('r',r,-1)
        if cv2.waitKey(0) & 0xFF == ord('\x1b'):
            print("I'm done")
            cv2.destroyAllWindows()
    elif img.shape[2] == 4:               #如果是4通道
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
        a = img[:,:,3]
        show_img('b',b,-1)
        show_img('g',g,-1)
        show_img('r',r,-1)
        show_img('a',a,-1)
        if cv2.waitKey(0) & 0xFF == ord('\x1b'):
            print("I'm done")
            cv2.destroyAllWindows()
# 用已有的多个通道图像构造成一个元组传递给merge()，可以实现图像的合并。
#下面这个例子先分离出bgr通道再合并后显示合成图像：
img = cv2.imread('./lena.png')
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
img2 = cv2.merge((b,g,r)) #传入bgr构成的元组
cv2.imshow('merged',img2)
if cv2.waitKey(0) & 0xFF == ord('\x1b'):
    print("I'm done")
    cv2.destroyAllWindows()


# 下面这个例子特意将bgr通道顺序做调换再合并：

img = cv2.imread('./lena.png')
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
img2 = cv2.merge((b,g,r)) #传入bgr构成的元组
cv2.imshow('merged',img2)

img2 = cv2.merge((r,g,b))
cv2.imshow('merged-rgb',img2)
img2 = cv2.merge((r,b,g))
cv2.imshow('merged-rbg',img2)
img2 = cv2.merge((g,b,r))
cv2.imshow('merged-gbr',img2)
if cv2.waitKey(0) & 0xFF == ord('\x1b'):
    print("I'm done")
cv2.destroyAllWindows()


#和用索引方式进行通道分离一样，也可以用索引方式完成通道合并：
print('VX公众号: 桔子code / juzicode.com')
print('cv2.__version__:',cv2.__version__)

img = cv2.imread('./lena.png')
b,g,r = cv2.split(img)
rows,cols,channels = img.shape[0],img.shape[1],img.shape[2]
img2 = np.zeros((rows,cols,channels),np.uint8)  #创建全0的numpy数组
img2[:,:,0]=b  #填充各个通道
img2[:,:,1]=g
img2[:,:,2]=r
cv2.imshow('merged',img2)
if cv2.waitKey(0) & 0xFF == ord('\x1b'):
    print("I'm done")
cv2.destroyAllWindows()
#下面这个例子读入lena.jpg时转换为灰度图，再使用split()进行分离：
img = cv2.imread('./lena.png',cv2.IMREAD_GRAYSCALE)
print('img.shape:',img.shape)
res = cv2.split(img)
print(type(res))
print(res)


#===================================================================================
# https://www.jianshu.com/p/5134e90955e6
#===================================================================================
import cv2
img=cv2.imread("lena.png")
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
b,g,r,a=cv2.split(bgra)
a[:,:]=125
bgra125=cv2.merge([b,g,r,a])
a[:,:]=0
bgra0=cv2.merge([b,g,r,a])
cv2.imshow("img",img)
cv2.imshow("bgra",bgra)
cv2.imshow("bgra125",bgra125)
cv2.imshow("bgra0",bgra0)
if cv2.waitKey(0) & 0xFF == ord('\x1b'):
    print("I'm done")
cv2.destroyAllWindows()
cv2.imwrite("bgra.png", bgra)
cv2.imwrite("bgra125.png", bgra125)
cv2.imwrite("bgra0.png", bgra0)



print("ord('d') = {}".format(ord('d')))
print("chr(100) = {}".format(chr(100)))

print("ord('a') = {}".format(ord('a')))
print("chr(97) = {}".format(chr(97)))


print("ord('\x1b') = {}".format(ord('\x1b')))
print("chr(27) = {}".format(chr(27)))

#===================================================================================
# pickle dump load
#===================================================================================

import imageio,pickle
import torch
import numpy as np

img = '/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/baby.png'

x = imageio.imread(img)
x1 = np.ascontiguousarray(x.transpose((2, 0, 1)))

x2 =   torch.from_numpy(x1).float()
x2.mul_(255 / 255)

f = '299086.pt'
with open(f, 'wb') as _f:
     pickle.dump(imageio.imread(img), _f)


with open(f, 'rb') as _f:
     lr = pickle.load(_f)

print(f"lr.shape = {lr.shape}")








#===================================================================================
# pickle dump load
#===================================================================================






img=cv2.imread("lena.png")
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)


b,g,r,a=cv2.split(bgra)
a[:,:]=125
bgra125=cv2.merge([b,g,r,a])
a[:,:]=0
bgra0=cv2.merge([b,g,r,a])
cv2.imshow("img",img)
cv2.imshow("bgra",bgra)
cv2.imshow("bgra125",bgra125)
cv2.imshow("bgra0",bgra0)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("bgra.png", bgra)
cv2.imwrite("bgra125.png", bgra125)
cv2.imwrite("bgra0.png", bgra0)






















