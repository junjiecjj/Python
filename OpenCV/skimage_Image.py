#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 22:22:44 2023

@author: jack

https://blog.csdn.net/hfuter2016212862/article/details/104763385

"""

import skimage
 
img = skimage.io.imread('./Figures/baby.jpg')
skimage.io.imshow(img)
 
skimage.io.imsave('./Figures/baby_skimage.jpg',img)