#!/usr/bin/env python
#-*-coding=utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab

x=np.arange(-6,6,0.001)
y=1/(1+np.exp(-x))

X=np.arange(7,8,0.001)
Y=1/(1+np.exp(-(X-(8-0.3))*20))

c = np.arange(7,8,0.001)
t1 = 7.5
t2 = 7.8
d = 1/(1+np.exp(-(c-(t1+t2)/2)*(10/(t2-t1))))
fig,axs = plt.subplots(3,1,figsize=(6,9))

axs[0].plot(x,y,'r',label='original sigmoid')
axs[0].grid()
axs[0].set_xlabel('x',fontsize=15)
axs[0].set_ylabel('y',fontsize=15)
axs[0].legend(loc='best')
axs[1].plot(X,Y)
axs[1].axvline(x=7.7,ls='--',color='b',label='center')
axs[1].grid()
axs[1].set_xlabel('x',fontsize=15)
axs[1].set_ylabel('y',fontsize=15)
axs[1].axvline(x=7,ls='--',color='lime',label=r'$t_{1}$')
axs[1].axvline(x=8,ls='--',color='fuchsia',label=r'$t_{2}$')
axs[1].legend(loc='best')
axs[2].plot(c,d)
axs[2].axvline(x=t1,ls='--',color='lime',label=r'$t_{min\_density}$')
axs[2].axvline(x=t2,ls='--',color='fuchsia',label=r'$t_{max\_density}$')
axs[2].grid()
axs[2].set_xlabel('x',fontsize=15)
axs[2].set_ylabel('y',fontsize=15)
axs[2].legend(loc='best')
fig.subplots_adjust(hspace=0.3)#调节两个子图间的距离
plt.savefig('/home/jack/snap/sigmoid.jpg',format='jpg',dpi=1000)
plt.show()
