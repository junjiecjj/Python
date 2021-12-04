#!/usr/bin/env python3.6
#-*-coding=utf-8-*-
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
#matplotlib.use('Agg')
#3matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/arphic/SimHei.ttf", size = 14)  


t=np.arange(0,2,0.1)
s1=np.sin(2*np.pi*t)
s2=np.cos(2*np.pi*t)
s3=np.tan(2*np.pi*t)
t1 = np.arange(-1,1.1,0.1)
s4 = np.arcsin(t1)


fig, axs=plt.subplots(3,1,figsize=(10,10))


axs[0].tick_params(labelsize=19 )
axs[0].plot(t,s1,color='b',linestyle='-',label='sin(x)学习',)
axs[0].plot(t,s3,color='r',linestyle='-',label='tan(x)',)
axs[0].axvline(x=1,ymin=0.4,ymax = 0.6,ls='-',linewidth=4,color='b',label='tan(x)',)

#axs[0].legend((A,B),('sin(x)','tan(x)'),loc='best',shadow=True)
axs[0].set_xlabel(r'time (s)',fontproperties = font)
axs[0].set_ylabel(r'dd达到',fontproperties = font)
axs[0].set_title('ssssss事实上',fontproperties = font)
axs[0].set_title('jack',loc='left',color='#0000FF',fontproperties = font)
axs[0].set_title('rose',loc='right',color='#9400D3',fontproperties = font)
axs[0].grid()
axs[0].legend(prop=font,)
#axs[0].xticks(fontsize=40)
axs[0].tick_params(labelsize=23)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
############################################################################################
axs[1].tick_params(labelsize=19 )
axs[1].plot(t,s2,color='r',linestyle='--',label=r'$cos(x)$',)
axs[1].axvline(x=1,ls=':',color='b',label = 'disruption')
axs[1].axhline(y=0.5,ls=':',color='r',label = 'hh')
axs[1].annotate(r'$t_{real\_disr}$实时',xy=(1,-1),xytext=(1.1,-0.7),\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
axs[1].legend(loc='best',shadow=True,prop = font)
axs[1].set_xlabel(r'time (s)',fontproperties = font)
axs[1].set_title('cos(x)',fontproperties = font)
axs[1].set_title(r'$t_{disr}$=%d达到'%1,loc='left',fontproperties = font)
axs[1].grid()
axs[1].tick_params(labelsize=23)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
#############################################################################################3
axs[2].tick_params(labelsize=19 )
axs[2].plot(t1,s4,color='blue',linestyle='-',linewidth=1,label='arcsin()')
axs[2].set_xlabel(r'time (s)实时',fontproperties = font)
axs[2].set_title('arcsin(x)',fontproperties = font)
axs[2].axvline(x=0,ymin=0.4,ymax = 0.6,ls='-',linewidth=4,color='b',)
axs[2].annotate(r'$t_{disr}$',
            xy=(0, 0), xycoords='data',
            xytext=(0.4, 0.2), textcoords='figure fraction',
            arrowprops=dict(arrowstyle = '->',connectionstyle = 'arc3'),fontproperties = font)
axs[2].legend(loc='best',shadow=False, prop = font)
axs[2].grid()

axs[2].tick_params(labelsize=23)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离

plt.suptitle('cos and sin and tan',fontproperties = font)

out_fig = plt.gcf()
out_fig.savefig(filepath2+'hh.eps',format='eps',dpi=1000, bbox_inches = 'tight')
out_fig.savefig(filepath2+'hh.svg',format='svg',dpi=1000, bbox_inches = 'tight')
out_fig.savefig(filepath2+'hh.pdf',format='pdf',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.jpg',format='jpg',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
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
ax.set_xticks([-4, -2, 0, 2])设置刻度值
ax.tick_params(labelcolor='r', labelsize='medium', width=3)设置刻度的大小颜色宽度等，但是不能设置刻度字体
'''

"""
##################################################################
#这一段代码和上面一段作用一样
#from __future__ import (absolute_import, division,print_function, unicode_literals)
import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0,2,0.1)
s1=np.sin(2*np.pi*t)
s2=np.cos(2*np.pi*t)
s3=np.tan(2*np.pi*t)

fig=plt.figure()
ax0=fig.add_subplot(211)
ax0.plot(t,s1,color='b',linestyle='-',linewidth=2,marker='*',markerfacecolor='red',markersize=12,label='sin(x)')
ax0.plot(t,s3,color='b',linestyle='-',linewidth=2,marker='o',markerfacecolor='c',markersize=12,label='tan(x)')
ax0.axvline(x=1,ls='--',color='b',label='disr')
ax0.legend(loc='best',shadow=True)
ax0.set_xlabel(r'time (s)')
ax0.set_title('sin and tan')

ax1=fig.add_subplot(212)
ax1.plot(t,s2,color='r',linestyle='--',linewidth=2,markerfacecolor='k',markersize=12,label='cos(x)')
ax1.axvline(x=1,ls=':',color='b',label='disruption')
ax1.annotate(r'$t_{disr}$',xy=(1,-1),xytext=(1.1,-0.7),\
arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
ax1.legend(loc='best',shadow=True)
ax1.set_xlabel(r'time (s)')
ax1.set_title('cos(x)')

fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离

plt.suptitle('cos and sin')
plt.show()

################################################
"""