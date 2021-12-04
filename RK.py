# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:28:21 2017

@author: 科
"""

import numpy as np
import matplotlib.pyplot as plt
y=np.linspace(0,0,100000)
y[0]=1
h=0.00001
t=np.arange(0,1,0.00001)
y1=1/(1-t)
for i in np.arange(99999):
    y[i+1]=(y[i]**2)*h+y[i]
    
plt.plot(t,y,'r-')
plt.plot(t,y1,'b-')
plt.show()