# -*- coding: utf-8 -*-

'''
import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0,2,0.1)
s1=np.sin(2*np.pi*t)
s2=np.cos(2*np.pi*t)
fig=plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(t,s1,color='b',linestyle='-',linewidth=2,marker='*',markerfacecolor='red',markersize=12,label='sin(x)')
ax1.plot(t,s2,color='r',linestyle=':',linewidth=2,marker='*',markerfacecolor='red',markersize=12,label='cos(x)')
plt.axvline(x=1,ls='--',color='b',label='*',marker='_')
ax1.legend(loc='best')
plt.show()
'''
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

fig, ax = plt.subplots()

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = ax.plot(t2, np.exp(-t2))
l2, l3 = ax.plot(t2, np.sin(2 * np.pi * t2), '--o', t1, np.log(1 + t1), '.')
l4, = ax.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 's-.')

ax.legend((l2, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)
ax.set_xlabel('time')
ax.set_ylabel('volts')
ax.set_title('Damped oscillation')
plt.show()