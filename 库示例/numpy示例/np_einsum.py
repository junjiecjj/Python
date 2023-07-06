#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:53:43 2023

@author: jack
"""

import os
import sys
import math
import time
import datetime

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import numpy as np





a = np.arange(25).reshape(5,5)
b = np.arange(5)
c = np.arange(6).reshape(2,3)

#========================================================================================
#                            计算矩阵的迹
#========================================================================================

x = np.einsum('ii...->...i', a)

print(f"1 计算矩阵的迹 = {x}")


x = np.einsum('i...i', a)
print(f"2 计算矩阵的迹 = {x}")

x = np.trace(a)
print(f"3 计算矩阵的迹 = {x}")


