
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
 
pip install imageio
pip install opencv-python


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


name = "Safe"
imgfile = f"./{name}.png"
basename = os.path.splitext(os.path.basename(imgfile))[0]
Img = imageio.v2.imread(imgfile)
imageio.imwrite(f'./{name}ImageioWrite.png',Img)

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
            



#========================================================================================
#  write the recovered pixel(txt file) to Image(.png)
#========================================================================================
def ReadResutTxtSaveMidPng(filename, name, time, SNR):
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
    imageio.imwrite(basefile+f'{name}_{time}_SNR={SNR:4.2f}.png', ImgRecoverd)
    cv2.imshow('src',ImgRecoverd[:,:,::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return H, W, C, PixelArray

def ReadResutTxtSavePng(filename, name, time, SNR):
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
    imageio.imwrite(basefile+f'{name}_SNR={SNR:4.2f}.png', ImgRecoverd)
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


#========================================================================================
pic = "Safe"
name = "Safe"
time = "BefDec"
basefile = f"/home/jack/公共的/编解码程序/ChenJunJie_ComLetter2023_03_Relate/5GLDPC_FreeRide_Picture_Auto/{pic}_Result/"


for idx, snr in enumerate(SNRList):
    filename = basefile+f"{name}_SNR={snr:4.2f}.txt"
    H, W, C, PixelArray = ReadResutTxtSavePng(filename, name, "", snr)
    #AddTextToFig(basefile+'lena_SNR=%4.2f.png'%snr, snr, idx)
    


#========================================================================================

pic = "Safe"
name = "Safe"
time = "AfEnc"
basefile = f"/home/jack/公共的/编解码程序/ChenJunJie_ComLetter2023_03_Relate/5GLDPC_FreeRide_Picture_Auto/{pic}_{time}_MiddleResult/"


for idx, snr in enumerate(SNRList):
    filename = basefile+f"{name}_{time}_Mid_SNR={snr:4.2f}.txt"
    H, W, C, PixelArray = ReadResutTxtSaveMidPng(filename, name, time, snr)
    #AddTextToFig(basefile+'lena_SNR=%4.2f.png'%snr, snr, idx)
    

#========================================================================================

pic = "Safe"
name = "Safe"
time = "BefDec"
basefile = f"/home/jack/公共的/编解码程序/ChenJunJie_ComLetter2023_03_Relate/5GLDPC_FreeRide_Picture_Auto/{pic}_{time}_MiddleResult/"


for idx, snr in enumerate(SNRList):
    filename = basefile+f"{name}_{time}_Mid_SNR={snr:4.2f}.txt"
    H, W, C, PixelArray = ReadResutTxtSaveMidPng(filename, name, time, snr)
    #AddTextToFig(basefile+'lena_SNR=%4.2f.png'%snr, snr, idx)
    


# Read SeedSetting.txt file.

# RandomIntl is built successfully

# RandomIntl is built successfully

# RandomIntl is built successfully
# read ./figures/Incident.txt.txt success!!!
#   H = 891, W = 891, C = 3, m_pixel_num = 4680423
#   CBTotal = 19502
#   m_len_total_bin = 37443384, m_len_total_bin_TB = 37443840, m_PatchLen = 456
#   m_pic_uu_mid_len = 74887680, m_pic_pix_mid_len = 9360960, MidH = 1260, MidW = 2476, MidC = 3
# mkdir ./Incident_AfEnc_MiddleResult success

# mkdir ./Incident_BefDec_MiddleResult success

 










