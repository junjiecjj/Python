#/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

X=np.arange(0,10,0.001)
y=np.sin(X)
y1=0.5*np.ones(X.shape)

y2=abs(y-y1)
a=np.where(np.abs(y2)<0.0001)

fig,ax = plt.subplots(1,1)
ax.plot(X,y)
ax.plot(X,y1)

ax.axvline(x=X[np.where(y2==y2.min())],ls='--',color='#B22222')
ax.set_ylabel('electronic density')
ax.set_title(r'$t_{real\_disr}$=%.3f'%1.0,loc='left',color='#0000FF')
ax.set_title(r'$t_{pred\_disr}$=%.3f'%2.0,loc='right',color='#9400D3')
fig.suptitle('Shut:%d' % 1333)            
print('the result is:\n ')
print('suss_rate: %.3f\n'% 0.7)
print('miss_rate: %f\n'% 0.1)
print('late_rate: %f\n'% 0.3)
print('flase_rate: %f\n'% 0.4) 