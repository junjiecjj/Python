#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:59:51 2023

@author: jack
"""

# import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
# from matplotlib.pyplot import MultipleLocator




#================================================================================================================================
##  使用plt 显示多个image的使用案例
#================================================================================================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#=================================================================================================
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

im1 = cv2.imread('./Figures/baby.png')
print(f"im1.shape = {im1.shape}")
im1 = ('baby', "Baby", im1)

im2 = cv2.imread('./Figures/flower.jpg')
print(f"im2.shape = {im2.shape}")
im2 = ('flower', "Flower", im2)

im3 = cv2.imread('./Figures/lena.png')
print(f"im3.shape = {im3.shape}")
im3 = ('lena', "Lena", im3)

im4 = cv2.imread('./Figures/bird.png')
print(f"im4.shape = {im4.shape}")
im4 = ('bird', "Bird", im4)

Image = []
Image.append(im1)
Image.append(im2)
Image.append(im3)
Image.append(im4)


cnt = 0
M = 2
N = 2
fig, axs = plt.subplots(M, N, figsize=(10, 10))
for i in range(M):
    for j in range(N):
        orig, adv, image = Image[cnt]
        axs[i, j].set_title("{} -> {}".format(orig, adv))
        axs[i, j].imshow(image[:,:,::-1] )
        axs[i, j].set_xticks([])  # #不显示x轴刻度值
        axs[i, j].set_yticks([] ) # #不显示y轴刻度值
        if j == 0:
            axs[i, j].set_ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        cnt += 1
plt.tight_layout()

plt.subplots_adjust(left=0, bottom=0, right=1, top=0.92 , wspace=0.0, hspace=0.1)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5, y=0.99, )

out_fig = plt.gcf()
filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'hh.eps',format='eps', bbox_inches = 'tight')

plt.show()

#=======================================================================================
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

im1 = cv2.imread('./Figures/baby.png')
print(f"im1.shape = {im1.shape}")
im1 = ('baby', "Baby", im1)

im2 = cv2.imread('./Figures/flower.jpg')
print(f"im2.shape = {im2.shape}")
im2 = ('flower', "Flower", im2)

im3 = cv2.imread('./Figures/lena.png')
print(f"im3.shape = {im3.shape}")
im3 = ('lena', "Lena", im3)

im4 = cv2.imread('./Figures/bird.png')
print(f"im4.shape = {im4.shape}")
im4 = ('bird', "Bird", im4)

Image = []
Image.append(im1)
Image.append(im2)
Image.append(im3)
Image.append(im4)



cnt = 0
M = 2
N = 2
plt.figure(figsize=(10, 10))
for i in range(M):
    for j in range(N):
        
        plt.subplot(M, N, cnt+1)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, image = Image[cnt]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(image[:,:,::-1])
        cnt += 1
plt.tight_layout()
plt.show()
































































































































































