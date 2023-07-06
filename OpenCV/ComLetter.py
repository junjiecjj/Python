#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:57:55 2023

@author: jack

 
1. cv2.imread 它默认认为图片存储的方式是RGB，而不管实际意义上的顺序，但是读取的时候反序了, 它读出的图片的通道是BGR。
2. cv2.imshow 它把被显示的图片当成BGR的顺序，而不管实际意义上的顺序，然后显示的时候变成RGB显示。
3. cv2.imwrite 它认为被保存的图片是BGR顺序，而不管实际意义上的顺序，保存的时候反序成RGB再保存。
4. plt默认认为被打开，保存，显示的图片的通道顺序为RGB。

所以将cv2和plt混用时需要注意顺序
 

"""


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio



img = cv2.imread('./lena.png')

# plt.imshow(img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# cv2.imshow('src',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# img1 = img[:,:,::-1]
# cv2.imshow('src',img1)
# cv2.waitKey()
# cv2.destroyAllWindows()




Img = imageio.imread('./lena.png')
imageio.imwrite('./lenaImageioWrite.png',Img)

H, W, C = Img.shape

with open("lena.txt","w") as f:
    f.write(f"{H} {W} {C}")  # 自带文件关闭功能，不需要再写f.close()


for c in range(C):
    for h in range(H):
        for w in range(W):
            
            









































