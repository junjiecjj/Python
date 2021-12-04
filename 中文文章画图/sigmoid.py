#!/usr/bin/env python
#-*-coding=utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import math
import pylab

td = 10
fl = 7
sp = 0.001

x=np.arange(-10,10,0.001)
y=1/(1+np.exp(-x))

e=np.arange(fl,td+sp,sp)
f=1/(1+np.exp(-(e-(td-(td-fl)/2))*25))
 
X=np.arange(fl,td+sp,sp)
Y=1/(1+np.exp(-(X-(td-(td-fl)/3))*25))



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
#axs[2].axvline(x=7,ls='--',color='lime',label=r'$t_{1}$')
#axs[2].axvline(x=8,ls='--',color='fuchsia',label=r'$t_{2}$')
axs[2].legend(loc='best')


xx = [fl,td]
#group_labels = [r'$f_{l}$', r'$t_d$']
group_labels = ('a', 'b')



#axs[3].axes.set_xticks(XX)
#axs[3].axes.set_xticklabels(XX,)
axs[3].plot(X,Y)
axs[3].set_xlabel('x',fontsize=15)
axs[3].set_ylabel('y',fontsize=15)
axs[3].set_xticks(xx)
axs[3].set_xticklabels([r'$t_{flat}$',r'$t_d$'],fontsize=15)
axs[3].grid(linestyle='-.',)
#axs[3].legend(loc='best')

fig.subplots_adjust(hspace=0.5)#调节两个子图间的距离
#plt.savefig('/home/jack/snap/sigmoid.jpg',format='jpg',dpi=1500)
plt.show()
