#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack
"""

import numpy as np
 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = "/usr/share/fonts/truetype/arphic/uming.ttf", size=19) 

mu, sigma = 0.5, 1
x = mu + sigma * np.random.randn(1000000)

n, bins, patches = plt.hist(x,bins=100, facecolor='g',alpha=0.75,normed = 0)
a = []
for i in range(100):
    a.append((bins[i]+bins[i+1])/2)
plt.plot(a,n)
plt.xlabel('x',fontproperties = font)#,fontproperties = font
plt.ylabel('y',fontproperties = font)#,fontproperties = font
plt.legend(prop=font,loc='upper right',bbox_to_anchor=(1.4, 1), borderaxespad=0)#'loc='upper left', 
figure_fig = plt.gcf() 
#figure_fig.savefig('/home/jack/snap/hh.svg',format="svg",bbox_inches='tight')
plt.show()
