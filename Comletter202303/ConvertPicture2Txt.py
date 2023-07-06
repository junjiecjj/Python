#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Mar 13 10:57:55 2023
@author: jack

"""

import glob
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio

#========================================================================================
#  Read the png picture as pixel (INT) and save INT in txt file.
#========================================================================================

SourceRoot = "/home/jack/GlobeCom2023_04_16_SimuData/benchmark"
SaveRoot = "/home/jack/GlobeCom2023_04_16_SimuData/benchmark_txt"

alldir = os.listdir(SourceRoot)
for name in alldir:
    ReadDir = os.path.join(SourceRoot, name)
    SaveDir = os.path.join(SaveRoot, name+'_txt')
    print(f"\n\nReadDir = {ReadDir}\nSaveDir = {SaveDir}\n")
    try:
        os.makedirs(SaveDir)
    except Exception as e:
        print(e)
    # absfilelist = sorted( glob.glob(os.path.join(ReadDir, '*' + ".png")) )
    absfilelist = sorted( glob.glob(os.path.join(ReadDir, '*' + '*')) )
    for picname in  absfilelist:
        filename, ext = os.path.splitext(os.path.basename(picname))
        savefilename = os.path.join(SaveDir, filename + '.txt')
        print(f"picname = {picname}")
        print(f"savefilename = {savefilename}")
        
        Img = imageio.imread(picname)        
        H, W, C = Img.shape
        with open(savefilename, "w") as f:
            f.write(f"Image:High--Wideth--Channel\n") 
            f.write(f"{H}  {W}  {C}\n")  # 自带文件关闭功能，不需要再写f.close()
            f.write(f"Image--Pixels\n")
            #f.write(f"{Img.flatten()}") 

            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        f.write(f"{Img[h][w][c]}\n") 




def scan_file(sourceroot, saveroot):
    alldir = os.listdir(sourceroot)
    for name in alldir:
        ReadDir = os.path.join(sourceroot, name)
        SaveDir = os.path.join(saveroot, name+'_txt')
        try:
            os.makedirs(SaveDir)
        except Exception as e:
            print(e)
        absfilelist = sorted( glob.glob(os.path.join(ReadDir, '*' + '*')) )
        for picname in  absfilelist:
            filename, ext = os.path.splitext(os.path.basename(picname))
            savefilename = os.path.join(SaveDir, filename + '.txt')
            
            Img = imageio.imread(picname)
            H, W, C = Img.shape
            with open(savefilename, "w") as f:
                f.write(f"Image:High--Wideth--Channel\n") 
                f.write(f"{H}  {W}  {C}\n")  # 自带文件关闭功能，不需要再写f.close()
                f.write(f"Image--Pixels\n")
                #f.write(f"{Img.flatten()}") 

                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            f.write(f"{Img[h][w][c]}\n") 
     
    return 


scan_file(SourceRoot, SaveRoot)















