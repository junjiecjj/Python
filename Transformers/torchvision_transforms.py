#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 21:36:13 2022

@author: jack
"""




from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt




img = Image.open("./DSC_4352.jpg")
print(img.size)
plt.imshow(img)


































































































