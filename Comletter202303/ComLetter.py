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

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio

#========================================================================================
#  Read the png picture as pixel (INT) and save INT in txt file.
#========================================================================================

 
imgfile = "./Incident_SNR=0.00.png"
basename = os.path.splitext(os.path.basename(imgfile))[0]
Img = imageio.v2.imread(imgfile)
imageio.imwrite('./IncidentImageioWrite.png',Img)

H, W, C = Img.shape

with open(basename+".txt","w") as f:
    f.write(f"Image:High--Wideth--Channel\n") 
    f.write(f"{H}  {W}  {C}\n")  # 自带文件关闭功能，不需要再写f.close()
    f.write(f"Image--Pixels\n")
    #f.write(f"{Img.flatten()}") 

with open(basename+".txt","a") as f:
    for c in range(C):
        for h in range(H):
            for w in range(W):
                f.write(f"{Img[h][w][c]}\n") 


RecoverTXT = basename+".txt"
PixelArray = []
with open(RecoverTXT, "r") as f:
    H, W, C = [int(s) for s in f.readline().strip().split()]
    for line in f.readlines():
        PixelArray.append(int(line.strip().split()[0]))
ImgRecoverd = np.zeros((H, W, C),dtype='uint8') 
cnt = 0
for c in range(C):
    for h in range(H):
        for w in range(W):
            ImgRecoverd[h][w][c] = PixelArray[cnt] 
            cnt += 1
imageio.imwrite(f'{basename}_recover.png', ImgRecoverd)



#========================================================================================
#  write the recovered pixel(txt file) to Image(.png)
#========================================================================================
def ReadResutTxtSavePng(filename, time):
    PixelArray = []
    with open(filename, "r") as f:
        H, W, C = [int(s) for s in f.readline().strip().split()]
        for line in f.readlines():
            PixelArray.append(int(line.strip().split()[0]))
    ImgRecoverd = np.zeros((H, W, C),dtype='uint8') 
    cnt = 0
    for c in range(C):
        for h in range(H):
            for w in range(W):
                ImgRecoverd[h][w][c] = PixelArray[cnt] 
                cnt += 1
    imageio.imwrite(basefile+'lena_%s_Mid_SNR=%4.2f.png'%(time, snr), ImgRecoverd)
    cv2.imshow('src',ImgRecoverd[:,:,::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return H, W, C, PixelArray



def AddTextToFig(filename, snr, idx):
    # 读入图片
    src = cv2.imread(filename)
    # cv2.imshow('text', src)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    label = ['(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','']
    
    # 调用cv.putText()添加文字
    text = f"{label[idx]} SNR=%4.2f(dB)"%snr
    
 
    patchHigh = 55
    plain = np.ones((patchHigh, src.shape[1], src.shape[2]),dtype='uint8')*255

    cv2.putText(plain, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    # cv2.imshow('src',plain)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    ## 将原图片和添加文字后的图片拼接起来
    res = np.vstack([src, plain])
    
    padsavefile = basefile+'lena_pad_SNR=%4.2f.png'%snr
    cv2.imwrite(padsavefile, res)
    
    # 显示拼接后的图片
    cv2.imshow('text', res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return
 



SnrMin = 0.0
SnrMax = 2.5
SnrInc = 0.5

SnrList = np.arange(SnrMin, SnrMax+SnrInc/2, SnrInc)
SNRList = np.around(SnrList, decimals=2, out=None)

pic = "Lena"
name = "lena"
basefile = f"/home/jack/公共的/编解码程序/5GLDPC_FreeRide_Picture/{pic}_AfDec_MiddleResult/"


for idx, snr in enumerate(SNRList):
    filename = basefile+f"{name}_SNR={snr:4.2f}.txt" 
    H, W, C, PixelArray = ReadResutTxtSavePng(filename)
    # AddTextToFig(basefile+'lena_SNR=%4.2f.png'%snr, snr, idx)
    





# with open(revoverfilename ,"a") as f:
#     for c in range(C):
#         for h in range(H):
#             for w in range(W):
#                 ImgRecoverd[h][w][c] = ImgReadAsLine[cnt] 
#                 cnt += 1



# cv2.imshow('src',ImgRecoverd[:,:,::-1])
# cv2.waitKey()
# cv2.destroyAllWindows()




 

























