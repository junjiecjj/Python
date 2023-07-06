#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:17:50 2022

@author: jack
"""


import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from pylab import tick_params
import copy
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


N=1000000
a = []
for i in range(1,N+1):
    if ( i%4== 0 or i%6==0 or i%9==0):
        a.append(i)

print(f"resu = {len(a)/N}")
