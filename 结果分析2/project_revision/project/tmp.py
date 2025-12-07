#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:51:57 2019

@author: jack
"""


import os
import numpy as np
import matplotlib.pyplot as plt
 
x = np.arange(0,6,0.1)
y = np.sin(x)
z = np.cos(x)
#host = plt.subplots()
fig,left_axis=plt.subplots()
#fig.subplots_adjust(right_axis=0,75)
 
right_axis = left_axis.twinx()
 
p1, = left_axis.plot(x, y, 'b.-')
p2, = right_axis.plot(x, z, 'r.-')
 
left_axis.set_xlim(-0.1,6.1)
left_axis.set_xticks(np.arange(1, 22,0.5))

#left_axis.set_ylim(0.265,0.355)
#left_axis.set_yticks(np.arange(0.27,0.36,0.01))
 
#right_axis.set_ylim(0,0.26)
#right_axis.set_yticks(np.arange(0,0.26,0.02))
 
left_axis.set_xlabel('Bar Number in One Edge')
left_axis.set_ylabel('Modulation Factor')
right_axis.set_ylabel('Efficiency')
 
left_axis.yaxis.label.set_color(p1.get_color())
right_axis.yaxis.label.set_color(p2.get_color())
 
tkw = dict(size=5, width=1.5)
left_axis.tick_params(axis='y', colors=p1.get_color(), **tkw)
right_axis.tick_params(axis='y', colors=p2.get_color(), **tkw)
left_axis.tick_params(axis='x', **tkw)
 
#plt.savefig('fig/abc.eps')
plt.show()
