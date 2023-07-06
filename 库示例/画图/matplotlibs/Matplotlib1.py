#!/usr/bin/env python3.6
#-*-coding=utf-8-*-
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 2, 0.1)
s1 = np.sin(2 * np.pi * t)
s2 = np.cos(2 * np.pi * t)
s3 = np.tan(2 * np.pi * t)

fig, axs = plt.subplots(3, 1)
A, = axs[0].plot(t,
                 s1,
                 color='b',
                 linestyle='-',
                 linewidth=2,
                 marker='*',
                 markerfacecolor='red',
                 markersize=12)
B, = axs[0].plot(t,
                 s3,
                 color='b',
                 linestyle='-',
                 linewidth=2,
                 marker='o',
                 markerfacecolor='c',
                 markersize=12)
C = axs[0].axvline(x=1, ls='--', color='b')
axs[0].legend((A, B, C), ('sin(x)', 'tan(x)', 'disruption time'),
              loc='best',
              shadow=True)
#axs[0].legend((A,B),('sin(x)','tan(x)'),loc='best',shadow=True)
axs[0].set_xlabel(r'time (s)')
axs[0].set_title('sin and tan')
axs[0].set_title('jack', loc='left', color='#0000FF')
axs[0].set_title('rose', loc='right', color='#9400D3')
D, = axs[1].plot(t,
                 s2,
                 color='r',
                 linestyle='--',
                 linewidth=2,
                 label=r'$cos(x)$',
                 markerfacecolor='k',
                 markersize=12)
E = axs[1].axvline(x=1, ls=':', color='b')
F = axs[1].axhline(y=0.5, ls=':', color='r')
axs[1].annotate(r'$t_{real\_disr}$',xy=(1,-1),xytext=(1.1,-0.7),\
arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
axs[1].legend((D, E, F), ('cos(x)', 'disruption', 'hh'),
              loc='best',
              shadow=True)
axs[1].set_xlabel(r'time (s)')
axs[1].set_title('cos(x)')
axs[1].set_title(r'$t_{disr}$=%d' % 1, loc='left')

fig.subplots_adjust(hspace=0.8)  #调节两个子图间的距离

plt.suptitle('cos and sin')

plt.show()
'''
linestyle是设置线头风格，-为实线，--为破折线，:为虚线，-.为点划线
linewidth设置线条宽度
label=r'$sin(x)$是给曲线打上标签，但是只当一副子图中画出多幅图时有用
marker='*'是设置标志字符
markerfacecolor='red'是设置标志颜色
markersize=12是设置标志大小
ax1.set_title('sin(x)')给子图打上标题
ax2.annotate()是给子图中某处打上箭头并给出描述
plt.suptitle('cos and sin')给整个画布打上大标题
'''

##################################################################
#这一段代码和上面一段作用一样
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 2, 0.1)
s1 = np.sin(2 * np.pi * t)
s2 = np.cos(2 * np.pi * t)
s3 = np.tan(2 * np.pi * t)

fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t,
         s1,
         color='b',
         linestyle='-',
         linewidth=2,
         marker='*',
         markerfacecolor='red',
         markersize=12,
         label='sin(x)')
ax0.plot(t,
         s3,
         color='b',
         linestyle='-',
         linewidth=2,
         marker='o',
         markerfacecolor='c',
         markersize=12,
         label='tan(x)')
ax0.axvline(x=1, ls='--', color='b', label='disr')
ax0.legend(loc='best', shadow=True)
ax0.set_xlabel(r'time (s)')
ax0.set_title('sin and tan')

ax1 = fig.add_subplot(212)
ax1.plot(t,
         s2,
         color='r',
         linestyle='--',
         linewidth=2,
         markerfacecolor='k',
         markersize=12,
         label='cos(x)')
ax1.axvline(x=1, ls=':', color='b', label='disruption')
ax1.annotate(r'$t_{disr}$',xy=(1,-1),xytext=(1.1,-0.7),\
arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
ax1.legend(loc='best', shadow=True)
ax1.set_xlabel(r'time (s)')
ax1.set_title('cos(x)')

fig.subplots_adjust(hspace=0.6)  #调节两个子图间的距离

plt.suptitle('cos and sin')
plt.show()

################################################














