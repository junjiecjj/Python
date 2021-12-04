#!/usr/bin/env python
#-*-coding=utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab

x=np.arange(-6,6,0.001)
y=1/(1+np.exp(-x))

e=np.arange(5,10,0.001)
f=1/(1+np.exp(-(e-7.5)*10))
 
X=np.arange(7,8,0.001)
Y=1/(1+np.exp(-(X-(8-0.3))*20))

c = np.arange(7,8,0.001)
t1 = 7.5
t2 = 7.8
d = 1/(1+np.exp(-(c-(t1+t2)/2)*(10/(t2-t1))))
fig,axs = plt.subplots(4,1,figsize=(6,10))

axs[0].plot(x,y,'r',label='original sigmoid')
axs[0].grid()
axs[0].set_xlabel('x',fontsize=15)
axs[0].set_ylabel('y',fontsize=15)
axs[0].legend(loc='best')
axs[1].plot(e,f,'r',label='tran and stret')
axs[1].grid()
axs[1].set_xlabel('x',fontsize=15)
axs[1].set_ylabel('y',fontsize=15)
axs[1].legend(loc='best')
axs[2].plot(X,Y)
axs[2].grid()
axs[2].set_xlabel('x',fontsize=15)
axs[2].set_ylabel('y',fontsize=15)
axs[2].axvline(x=7,ls='--',color='lime',label=r'$t_{1}$')
axs[2].axvline(x=8,ls='--',color='fuchsia',label=r'$t_{2}$')
axs[2].legend(loc='best')
axs[3].plot(c,d)
axs[3].axvline(x=7,ls='--',color='lime',label=r'$t_{1}$')
axs[3].axvline(x=8,ls='--',color='fuchsia',label=r'$t_{2}$')
axs[3].axvline(x=t1,ls='--',color='r',label=r'$t_{min\_density}$')
axs[3].axvline(x=t2,ls='--',color='b',label=r'$t_{max\_density}$')
axs[3].grid()
axs[3].set_xlabel('x',fontsize=15)
axs[3].set_ylabel('y',fontsize=15)
axs[3].legend(loc='best',fontsize=10)
fig.subplots_adjust(hspace=0.3)#调节两个子图间的距离
plt.savefig('/home/jack/snap/sigmoid.jpg',format='jpg',dpi=1500)
plt.show()
