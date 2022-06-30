#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:52:54 2022

@author: jack
"""

import cv2
import numpy as np



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
img2 = np.zeros((rows, cols, channels), np.uint8)  #创建全0的numpy数组
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




