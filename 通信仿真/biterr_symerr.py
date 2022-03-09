#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack

https://www.cxyzjd.com/article/qikaihuting/110562852

"""
import numpy  as np

def hammingDistance(x, y):
       """
       :type x: int
       :type y: int
       :rtype: int
       """
       print("x = {}, y = {}".format(x,y))
       return bin(x^y).count("1")
   
def hanmingDist( bitstream1, bitstream2):
       """
       input binarized string 
       output distance
       """
       arr_1 = np.asarray([int(i) for i in bitstream1])
       arr_2 = np.asarray([int(i) for i in bitstream2])
       count = np.count_nonzero(arr_1 != arr_2)
       return count


bit1= "01010101" # = 85
bit2= "01110110" # = 118
print(hanmingDist(bit1, bit2)) # 3
x1 = int(bit1,2) #转十进制
x2 = int(bit2,2) 
print(hammingDistance(x1,x2)) #3