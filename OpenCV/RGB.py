#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:58:29 2022

@author: jack
"""

def image2label(path,size_):
    w = size_[0]
    h = size_[1]
    label_im=cv2.imread(path)
    #label_im=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    #修改图像的尺寸大小
    new_array = cv2.resize(label_im, (w, h), interpolation=cv2.INTER_CUBIC)
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
    
    return data

# if __name__ =='__main__':
import cv2
import numpy as np
from PIL import Image
#修改的尺寸大小    
size_=[320,480]
img_path='/home/jack/图片/pngsucai_6729744_d83066.png'
label = image2label(img_path,size_)
#修改后的尺寸和修改后的像素值保存下来
save_img='./0002_c1s1_000451_03.png'
cv2.imwrite(save_img, label)
print(label[100])