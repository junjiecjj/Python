#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:07:50 2023

@author: jack
"""

import os
import glymur
import imageio
import skimage.data
import PIL
import numpy as np

figdir = '/home/jack/公共的/Python/OpenCV/Figures/0_3.bmp'   # bmp  jpg  jpg
savedir = '/home/jack/公共的/Python/OpenCV/Figures/0_3_1.jp2'
im = imageio.v2.imread(figdir )
imp = PIL.Image.fromarray(im )
size1 = os.path.getsize(figdir)

# m = 30
# jm = glymur.Jp2k(savedir, data = im,  cratios = [m])
# size2 = os.path.getsize(savedir)
# print(f"m = {m}, size1 = {size1}, size2 = {size2}")


for m in np.arange(0, 110, 10):
    jm = glymur.Jp2k(savedir, data = im,  cratios = [m])
    # jmp = PIL.Image.fromarray(jm)
    size2 = os.path.getsize(savedir)
    print(f"m = {m}, size1 = {size1}, size2 = {size2}")

# # jm = glymur.Jp2k(savedir)


# # jm = imageio.v2.imread(savedir )
# # jmp = PIL.Image.fromarray(jm )
