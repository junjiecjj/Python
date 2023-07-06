#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:11:11 2022

@author: jack
"""
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)



fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size = 22)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 24)

#=====================================================================================
# https://blog.csdn.net/qq_29831163/article/details/90115647
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#高斯分布的概率分布直方图
mean = 1    #均值为0
sigma = 6  #标准差为1，反应数据集中还是分散的值
x=mean+sigma*np.random.randn(100000)
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) #绘制2行1列的子图

#第二个参数bins越大、则条形bar越窄越密，density=True则画出频率，否则次数
ax0.hist(x,50, density=True, histtype='bar',color='yellowgreen', alpha=0.75) #normed=True或1 表示频率图
##pdf概率分布图，一万个数落在某个区间内的数有多少个
ax0.set_title('pdf')

#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
ax1.hist(x,50, density=True, histtype='bar',facecolor='pink',alpha=0.75,cumulative=True, rwidth=0.8)
ax1.set_title("cdf")
fig.subplots_adjust(hspace=0.4)
plt.show()

#=====================================================================================
# 绘制一维正态分布直方图的三种方式
# coding=utf-8
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
 
sampleNo = 1000;
# 一维正态分布
# 下面三种方式是等效的
mu = 3
sigma = 0.1
np.random.seed(0)
s = np.random.normal(mu, sigma, sampleNo )
#s = np.random.rand(1, sampleNo )
plt.subplot(141)
plt.hist(s, 10, density=True)   #####bins=10
 
np.random.seed(0)
s = sigma * np.random.randn(sampleNo ) + mu
plt.subplot(142)
plt.hist(s, 30, density=True)   #####bins=30
 
np.random.seed(0)
s = sigma * np.random.standard_normal(sampleNo ) + mu
plt.subplot(143)
plt.hist(s, 30, density=True)   #####bins=30


#=====================================================================================
# 绘制二维正态分布

mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
s = np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.subplot(144)
# 注意绘制的是散点图，而不是直方图
plt.plot(s[:,0],s[:,1],'+')
plt.show()



#=====================================================================================
# https://www.matplotlib.org.cn/gallery/statistics/histogram_features.html

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()



#=====================================================================================
# https://www.matplotlib.org.cn/gallery/statistics/histogram_multihist.html
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 14)
ax0.set_title('bars with legend' , fontproperties = font1)
font2  = {'family':'Times New Roman','style':'normal','size':17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 10)
legend1 = ax0.legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明




ax1.hist(x, n_bins, density=True, histtype='bar',color=colors, stacked=True)
ax1.set_title('stacked bar' , fontproperties = font1)

ax2.hist(x, n_bins, histtype='step', stacked=True, color=colors, fill=False)
ax2.set_title('stack step (unfilled)' , fontproperties = font1)

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar', color=colors,)
ax3.set_title('different sample sizes' , fontproperties = font1)


fig.tight_layout()
plt.show()



fig, ax = plt.subplots()
ax.hist(x[:,0], n_bins, density=1, color='red')
ax.set_ylabel('x[:,0]')
ax.set_title(r'x[:,0]')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.hist(x[:,1], n_bins, density=1, color='tan')
ax.set_ylabel('x[:,1]')
ax.set_title(r'x[:,1]')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()



fig, ax = plt.subplots()
ax.hist(x[:,2], n_bins, density=1, color='lime')
ax.set_ylabel('x[:,2]')
ax.set_title(r'x[:,2]')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()


# Implementation of matplotlib function 
from matplotlib import colors 
from matplotlib.ticker import PercentFormatter 
import numpy as np 
import matplotlib.pyplot as plt 
  
   
N_points = 100000
x = np.random.randn(N_points) 
y = 4 * x + np.random.randn(100000) + 50
   
plt.hist2d(x, y, 
           bins = 100,  
           norm = colors.LogNorm(),  
           cmap ="gray") 
  
plt.title('matplotlib.pyplot.hist2d() function Example\n\n', fontweight ="bold") 
  
plt.show()

#Implementation of matplotlib function
from matplotlib import colors
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
  
    
result = np.vstack([
    multivariate_normal([10, 10],
            [[3, 2], [2, 3]], size=1000000),
    multivariate_normal([30, 20],
            [[2, 3], [1, 3]], size=100000)
])
  
plt.hist2d(result[:, 0],
           result[:, 1],
           bins = 100, 
           cmap = "Greens",
           norm = colors.LogNorm())
plt.title('matplotlib.pyplot.hist2d function \
Example')
plt.show()
  
plt.hist2d(result[:, 0], 
           result[:, 1],
           bins = 100, 
           cmap = "RdYlGn_r",
           norm = colors.LogNorm())
plt.show()


# 导入模块
import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.random.randn(1000)+2
y = np.random.randn(1000)+3

# 画图
plt.hist2d(x=x, y=y, bins=30)

# 展示
plt.show()












