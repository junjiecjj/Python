#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:39:32 2024

@author: jack
https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html

https://docs.scipy.org/doc/scipy/reference/stats.html
"""

import numpy as np
import pandas as pd
import scipy
# import scipy.special as sps
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###########    连续随机变量
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%% erf, erfc, Qfun %%%%%%%%%%%%%%%%%%%%%%%%

def Qfun(x):
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

def QfunUPbound1(x):
    return 1/6*np.exp(-2*x**2) + 1/12 * np.exp(-x**2) + 1/4 * np.exp(-x**2/2)

def QfunUPbound2(x):
    return np.exp(-x**2/2)
x = np.arange(-10, 10, 0.01)
Q = Qfun(x)
QupBound1 = QfunUPbound1(x)
QupBound2 = QfunUPbound2(x)

fig, ax = plt.subplots(figsize = (8, 6))
ax.plot(x, Q, 'b', lw = 2, alpha=0.6, label='Q(x)')
ax.plot(x, QupBound1, 'r', lw = 2, alpha=0.6, label='QupBound1(x)')
ax.plot(x, QupBound2, 'k', lw = 2, alpha=0.6, label='QupBound2(x)')
plt.grid()
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 27}
plt.xlabel('x')
plt.legend(loc='best', prop=font2,)
plt.show()

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
x = np.linspace(-3, 3)
plt.plot(x, special.erf(x))
plt.xlabel('$x$')
plt.ylabel('$erf(x)$')
plt.show()

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
x = np.linspace(-3, 3)
plt.plot(x, special.erfc(x))
plt.xlabel('$x$')
plt.ylabel('$erfc(x)$')
plt.show()


#%%%%%%%%%%%%%%%%%%%%%% 伽马分布 (Gamma Distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%>>>>>>>>>>>>>>>>>>>>>>>>  gamma function
import numpy as np
# from scipy.special import factorial

print(scipy.special.gamma([0, 0.5, 1, 5]))
z = 2.5 + 1j
print(scipy.special.gamma(z))


print(scipy.special.gamma(z+1), z*scipy.special.gamma(z))  # Recurrence property  for complex

print(scipy.special.gamma(0.5)**2,  np.pi )  # gamma(0.5) = sqrt(pi)

x = np.linspace(-3.5, 5.5, 2251)
y = scipy.special.gamma(x)

k = np.arange(1, 7)
y1 = scipy.special.factorial(k-1)

fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(x, y, 'b', lw = 2, alpha=0.6, label='gamma(x)')
ax.plot(k, y1, 'k*', ms = 10, alpha=0.6, label='(x-1)!, x = 1, 2, ...')

ax.set_xlim(-3.5, 5.5)
ax.set_ylim(-10, 25)
plt.grid()
plt.xlabel('x')
plt.legend(loc='lower right')
plt.show()


#%%>>>>>>>>>  gamma  distribution: Gamma分布即为多个独立且相同分布（iid）的指数分布变量的和的分布 %%%%%%%%%%%%%%%%%%%%%%%%%%
# 伽马分布（Gamma Distribution）是统计学中的一种连续概率函数，它包含两个参数α和β，其中α称为形状参数，β称为尺度参数。
# 1.定义与概念：假设随机变量X为等到第α件事发生所需之等候时间，且每个事件之间的等待时间是互相独立的，α为事件发生的次数，β代表事件发生一次的概率，那么这α个事件的时间之和服从伽马分布。

a = 1.99
mean, var, skew, kurt = scipy.stats.gamma.stats(a, moments='mvsk')

# α (shape) = 2.0, θ (scale) = 2.0
a_array = [0.8, 1, 2, 4 ]
scale_array = [0.1, 0.5, 1, np.sqrt(2), 4]
scale_array_, a_array_ = np.meshgrid(scale_array, a_array)

### PDF of Beta Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(scale_array), figsize=(len(scale_array)*4, len(a_array)*3))
for a_idx, scale_idx, ax in zip(a_array_.ravel(), scale_array_.ravel(), axs.ravel()):
    mean, var, skew, kurt = scipy.stats.gamma.stats(a_idx, scale = scale_idx, moments='mvsk')

    x = np.linspace(scipy.stats.gamma.ppf(0.01, a_idx, scale = scale_idx), scipy.stats.gamma.ppf(0.99, a_idx, scale = scale_idx), 100)

    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, scipy.stats.gamma.pdf(x, a_idx, scale = scale_idx), 'b', lw=2, label = 'gamma pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.gamma(a_idx, scale = scale_idx)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.gamma.rvs(a_idx, scale = scale_idx, size = 10000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.03, facecolor = "#FF3300", label= "rvs", zorder = 3)

    ## np
    s = np.random.gamma(a_idx, scale_idx, 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = bins**(a_idx-1)*(np.exp(-bins/scale_idx) / (scipy.special.gamma(a_idx)*scale_idx**a_idx))
    ax.plot(bins, y, lw=2, color='r', label = "np pdf")

    print(f"mean = {mean:.2f}/{a_idx*scale_idx:.2f}/{np.mean(s)}, var = {var:.2f}/{a_idx*scale_idx**2:.2f}/{np.var(s)}")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of gamma Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(scale_array), figsize=(len(scale_array)*4, len(a_array)*3))
for a_idx, scale_idx, ax in zip(a_array_.ravel(), scale_array_.ravel(), axs.ravel()):
    x = np.linspace(scipy.stats.gamma.cdf(0.01, a_idx, scale = scale_idx), scipy.stats.gamma.ppf(0.99999, a_idx, scale = scale_idx), 100)
    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, scipy.stats.gamma.cdf(x, a_idx, scale_idx), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import gamma

## 设置随机种子以确保可重复性
np.random.seed(42)

## 伽马分布的参数
shape, scale = 2.0, 2.0  # α (shape) = 2.0, θ (scale) = 2.0

## 生成伽马分布的随机样本
data = scipy.stats.gamma.rvs(a=shape, scale=scale, size=1000)

## 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

## 子图1：伽马分布的概率密度函数 (PDF)
x = np.linspace(0, 20, 1000)
pdf = scipy.stats.gamma.pdf(x, a=shape, scale=scale)
axs[0, 0].plot(x, pdf, 'r-', lw=2, label=f'Gamma PDF\nα={shape}, θ={scale}')
axs[0, 0].fill_between(x, pdf, color='red', alpha=0.3)
axs[0, 0].set_title('Probability Density Function (PDF)')
axs[0, 0].legend()

# 子图2：伽马分布样本的直方图
sns.histplot(data, kde=False, bins=30, color='blue', ax=axs[0, 1], stat="density")
axs[0, 1].set_title('Histogram of Gamma Distributed Data')
axs[0, 1].set_ylabel('Density')

# 子图3：伽马分布样本的核密度估计 (KDE)
sns.kdeplot(data, color='green', lw=2, ax=axs[1, 0])
axs[1, 0].set_title('Kernel Density Estimate (KDE)')
axs[1, 0].set_ylabel('Density')

# 子图4：样本的累积分布函数 (CDF)
cdf = scipy.stats.gamma.cdf(x, a=shape, scale=scale)
axs[1, 1].plot(x, cdf, 'b-', lw=2, label='CDF')
axs[1, 1].set_title('Cumulative Distribution Function (CDF)')
axs[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 验证 Gamma分布即为多个独立且相同分布（iid）的指数分布变量的和的分布 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = 4
scale = np.sqrt(2)
N = 100000

x = np.linspace(scipy.stats.gamma.ppf(0.0001, a, scale = scale), scipy.stats.gamma.ppf(0.9999, a, scale = scale), 1000)

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
axs.plot(x, scipy.stats.gamma.pdf(x, a, scale = scale), 'b', lw=2, label = 'gamma pdf', zorder = 1)

Y = np.zeros(N)
for i in range(a):
    Y += scipy.stats.expon.rvs(loc = 0, scale = scale, size = N)

    ## scipy Random variates.
    # r = scipy.stats.expon.rvs(loc = loc, scale = scale, size = 1000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label =  "rvs", zorder = 3)

axs.hist(Y, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label = f"sum of {a} exp rvs with {scale:.3f}", zorder = 3)

axs.set_xlabel( 'x',)
axs.set_ylabel('pdf',)
axs.set_title("验证 Gamma分布即为多个独立且相同分布（iid）的指数分布变量的和的分布")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 验证 chi2 分布是独立的k标准正态变量，则其平方和服从自由度为 k 的卡方分布， %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

df = 4
N = 100000

x = np.linspace(scipy.stats.chi2.ppf(0.0001, df), scipy.stats.chi2.ppf(0.9999, df), 1000)

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
title_idx = f"df = {df:.2f}"
axs.plot(x, scipy.stats.chi2.pdf(x, df), 'b', lw=2, label = 'chi2 pdf', zorder = 1)

Y = np.zeros(N)
for i in range(df):
    Y += np.random.randn(N)**2

axs.hist(Y, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label = f"sum of {a} tandard normal rvs", zorder = 3)

axs.set_xlabel( 'x',)
axs.set_ylabel('pdf',)
axs.set_title("验证 chi2 分布是独立的k标准正态变量的平方和")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 验证 chi 分布是独立的k标准正态变量，则其平方和开方服从自由度为 k 的卡分布， %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

df = 4
N = 100000

x = np.linspace(scipy.stats.chi.ppf(0.0001, df), scipy.stats.chi.ppf(0.9999, df), 1000)

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
title_idx = f"df = {df:.2f}"
axs.plot(x, scipy.stats.chi.pdf(x, df), 'b', lw=2, label = 'chi pdf', zorder = 1)

Y = np.zeros(N)
for i in range(df):
    Y += np.random.randn(N)**2
Y = np.sqrt(Y)
axs.hist(Y, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label = f"sum of {a} tandard normal rvs", zorder = 3)

axs.set_xlabel( 'x',)
axs.set_ylabel('pdf',)
axs.set_title("验证 chi 分布是独立的k标准正态变量的平方和开方")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 卡分布(chi distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

df = 55
mean, var, skew, kurt = scipy.stats.chi.stats(df, moments='mvsk')

df_array = [1, 2, 4, 5, 7, 9, 10 ]

### PDF of chi Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(df_array), figsize=(len(df_array)*4, 3))
for df, ax in zip(df_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.chi.stats(df, moments='mvsk')
    print(f"mean = {mean:.2f}/ , var = {var:.2f}/ ")

    x = np.linspace(scipy.stats.chi.ppf(0.01, df), scipy.stats.chi.ppf(0.99, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, scipy.stats.chi.pdf(x, df), 'b', lw=2, label = 'chi pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.chi(df)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.chi.rvs(df, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.03, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    # s = np.random.chi(df = df, size = 1000)
    # count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/2)**(df/2-1) * bins**(df-1) * np.exp(-bins**2/2) / scipy.special.gamma(df/2)
    ax.plot(bins, y, lw=2, color='r', label = "np pdf")

    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of gamma Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(df_array), figsize=(len(df_array)*4, 3))
for df, ax in zip(df_array, axs.ravel()):
    x = np.linspace(scipy.stats.chi.cdf(0.01, df), scipy.stats.chi.ppf(0.99999, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, scipy.stats.chi.cdf(x, df), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 卡方分布(chi2  distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# from scipy.stats import chi2

df = 55
mean, var, skew, kurt = scipy.stats.chi2.stats(df, moments='mvsk')

df_array = [1, 2, 4, 5, 7, 9, 10 ]

### PDF of chi2 Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(df_array), figsize=(len(df_array)*4, 3))
for df, ax in zip(df_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.chi2.stats(df, moments='mvsk')
    print(f"mean = {mean:.2f}/ , var = {var:.2f}/ ")

    x = np.linspace(scipy.stats.chi2.ppf(0.01, df), scipy.stats.chi2.ppf(0.99, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, scipy.stats.chi2.pdf(x, df), 'b', lw=2, label = 'chi2 pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.chi2(df)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.chi2.rvs(df, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.03, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.chisquare(df = df, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/2)**(df/2) * bins**(df/2-1) * np.exp(-bins/2) / scipy.special.gamma(df/2)
    ax.plot(bins, y, lw=2, color='r', label = "np pdf")

    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of gamma Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(df_array), figsize=(len(df_array)*4, 3))
for df, ax in zip(df_array, axs.ravel()):
    x = np.linspace(scipy.stats.chi2.cdf(0.01, df), scipy.stats.chi2.ppf(0.99999, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, scipy.stats.chi2.cdf(x, df), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%% 卡方分布是特殊的gamma分布,  with gamma parameters a = df/2, loc = 0 and scale = 2.

df = 5
a = df/2
scale = 2
x = np.linspace(scipy.stats.gamma.ppf(0.001, a, scale = scale), scipy.stats.gamma.ppf(0.99, a, scale = scale), 100)

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
title_idx = f"a = {a:.2f}, scale = {scale:.2f}"
axs.plot(x, scipy.stats.gamma.pdf(x, a, scale = scale), 'b', lw=2, label = 'gamma pdf', zorder = 1)

title_idx = f"df = {df}"
axs.plot(x, scipy.stats.chi2.pdf(x, df), 'r', lw=2,ls = '--', label = 'chi2 pdf', zorder = 1)

axs.set_xlabel( 'x',)
axs.set_ylabel('pdf',)
axs.set_title("The chi-squared distribution is a special case of the gamma distribution")
axs.legend(fontsize = 20)

plt.show()
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Beta分布 (Beta Distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# from scipy.stats import beta

a_array = [0.5, 0.8, 2, 4, 6]
b_array = [0.5, 0.8, 2, 4, 6 ]
b_array_, a_array_ = np.meshgrid(b_array, a_array)

### PDF of Beta Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(b_array), figsize=(len(b_array)*4, len(a_array)*3))
for a_idx, b_idx, ax in zip(a_array_.ravel(), b_array_.ravel(), axs.ravel()):
    mean, var, skew, kurt = scipy.stats.beta.stats(a = a_idx, b = b_idx, moments='mvsk')

    x = np.linspace(scipy.stats.beta.ppf(0.01, a = a_idx, b = b_idx), scipy.stats.beta.ppf(0.99, a = a_idx, b = b_idx), 100)
    title_idx = f"a = {a_idx:.2f}, b = {b_idx:.2f}"
    ax.plot(x, scipy.stats.beta.pdf(x, a = a_idx, b = b_idx), 'b', lw=2, label = 'gamma pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.beta(a = a_idx, b = b_idx)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.beta.rvs(a = a_idx, b = b_idx, size = 10000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.03, facecolor = "#FF3300", label= "rvs", zorder = 3)

    ## np
    s = np.random.beta(a = a_idx, b = b_idx, size = 10000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = bins**(a_idx-1) * (1 - bins)**(b_idx-1) * scipy.special.gamma(a_idx+b_idx)/ (scipy.special.gamma(a_idx) * scipy.special.gamma(b_idx))
    ax.plot(bins, y, lw=2, color='r',ls = '--', label = "np pdf")

    print(f"mean = {mean:.2f}/{np.mean(s):.2f}, var = {var:.2f}/{np.var(s):.2f}")

    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of gamma Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(b_array), figsize=(len(b_array)*4, len(a_array)*3))
for a_idx, b_idx, ax in zip(a_array_.ravel(), b_array_.ravel(), axs.ravel()):
    mean, var, skew, kurt = scipy.stats.beta.stats(a = a_idx, b = b_idx, moments='mvsk')

    x = np.linspace(scipy.stats.beta.ppf(0.01, a = a_idx, b = b_idx), scipy.stats.beta.ppf(0.99, a = a_idx, b = b_idx), 100)
    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, scipy.stats.beta.cdf(x, a = a_idx, b = b_idx), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import beta

# 参数设置
alpha_1, beta_1 = 2, 5
alpha_2, beta_2 = 5, 2

# 生成x值
x = np.linspace(0, 1, 1000)

# 计算PDF（概率密度函数）
pdf_1 = scipy.stats.beta.pdf(x, alpha_1, beta_1)
pdf_2 = scipy.stats.beta.pdf(x, alpha_2, beta_2)

# 计算CDF（累积分布函数）
cdf_1 = scipy.stats.beta.cdf(x, alpha_1, beta_1)
cdf_2 = scipy.stats.beta.cdf(x, alpha_2, beta_2)

# 生成随机样本数据
samples_1 = scipy.stats.beta.rvs(alpha_1, beta_1, size=1000)
samples_2 = scipy.stats.beta.rvs(alpha_2, beta_2, size=1000)

# 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘制PDF
axs[0, 0].plot(x, pdf_1, 'r-', label=f'Beta({alpha_1}, {beta_1}) PDF')
axs[0, 0].plot(x, pdf_2, 'b-', label=f'Beta({alpha_2}, {beta_2}) PDF')
axs[0, 0].set_title('Probability Density Function (PDF)')
axs[0, 0].legend()

# 绘制CDF
axs[0, 1].plot(x, cdf_1, 'r-', label=f'Beta({alpha_1}, {beta_1}) CDF')
axs[0, 1].plot(x, cdf_2, 'b-', label=f'Beta({alpha_2}, {beta_2}) CDF')
axs[0, 1].set_title('Cumulative Distribution Function (CDF)')
axs[0, 1].legend()

# 绘制直方图
axs[1, 0].hist(samples_1, bins=30, alpha=0.5, color='red', label=f'Beta({alpha_1}, {beta_1}) Samples')
axs[1, 0].hist(samples_2, bins=30, alpha=0.5, color='blue', label=f'Beta({alpha_2}, {beta_2}) Samples')
axs[1, 0].set_title('Histogram of Samples')
axs[1, 0].legend()

# 绘制随机样本散点图
axs[1, 1].scatter(range(1000), samples_1, alpha=0.5, color='red', label=f'Beta({alpha_1}, {beta_1}) Samples')
axs[1, 1].scatter(range(1000), samples_2, alpha=0.5, color='blue', label=f'Beta({alpha_2}, {beta_2}) Samples')
axs[1, 1].set_title('Scatter Plot of Samples')
axs[1, 1].legend()

# 设置总体图标题
fig.suptitle('Beta Distribution Analysis', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 瑞丽分布(rayleigh  distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 瑞利分布（Rayleigh Distribution）：当一个随机二维向量的两个分量呈独立的、均值为0，有着相同的方差的正态分布时，这个向量的模呈瑞利分布。
# loc一定要写明确，注意关键字参数和位置参数造成的错误.
loc = 0
scale = 2
mean, var, skew, kurt = scipy.stats.rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

scale_array = [1, 2, 3, 4, 5, 6, ]

### PDF of rayleigh Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.rayleigh.ppf(0.01, loc = loc, scale = scale), scipy.stats.rayleigh.ppf(0.99, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.rayleigh.pdf(x, loc = loc, scale = scale), 'b', lw=2, label = 'rayleigh pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.rayleigh(loc = loc, scale = scale)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.rayleigh.rvs(loc = loc, scale = scale, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.rayleigh(scale = scale, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = bins / scale**2 * np.exp(-bins**2/(2 * scale**2))
    ax.plot(bins, y, lw = 2, color = 'r', ls = '--', label = "np pdf")

    print(f"mean = {mean:.2f}/{np.sqrt(np.pi/2)*scale}/{np.mean(s)}, var = {var:.2f}/{(2-np.pi/2)*scale**2}/{np.var(s)} ")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of rayleigh Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.rayleigh.cdf(0.01, loc = loc, scale = scale), scipy.stats.rayleigh.ppf(0.99999, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.rayleigh.cdf(x, loc = loc, scale = scale), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 指数分布 (Exponential Distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# from scipy.stats import expon
# loc一定要写明确，注意关键字参数和位置参数造成的错误
loc = 0
scale = 2
mean, var, skew, kurt = scipy.stats.expon.stats(loc = loc, scale = scale, moments='mvsk')

scale_array = [0.1, 0.5, 1, 2, 3, ]

### PDF of exponential Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.expon.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.expon.ppf(0.01, loc = loc, scale = scale), scipy.stats.expon.ppf(0.99, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.expon.pdf(x, loc = loc, scale = scale), 'b', lw=2, label = 'expon pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.expon(loc = loc, scale = scale)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.expon.rvs(loc = loc, scale = scale, size = 1000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.exponential(scale = scale, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/scale) * np.exp(-bins/scale)
    ax.plot(bins, y, lw=2, color='r', label = "np pdf")

    print(f"mean = {mean:.2f}/{scale}/{np.mean(s)}, var = {var:.2f}/{scale**2}/{np.var(s)} ")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of exponential Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))
for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.expon.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.expon.cdf(0.01, loc = loc, scale = scale), scipy.stats.expon.ppf(0.99999, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.expon.cdf(x, loc = loc, scale = scale), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

##>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成虚拟数据集（指数分布）
lambda_param = 1.5  # 参数lambda
size = 1000  # 数据集大小
data = np.random.exponential(scale = 1/lambda_param, size = size)

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['Exponential'])

# 创建图形对象
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 图1：直方图 + KDE
sns.histplot(df['Exponential'], kde=True, ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title('Histogram + KDE of Exponential Distribution')

# 图2：累积分布函数（CDF）
sns.ecdfplot(df['Exponential'], ax=axs[0, 1], color='green')
axs[0, 1].set_title('Empirical CDF of Exponential Distribution')

# 图3：箱线图
sns.boxplot(data=df, ax=axs[1, 0], color='lightcoral')
axs[1, 0].set_title('Boxplot of Exponential Distribution')

# 图4：Q-Q 图
from scipy import stats
stats.probplot(df['Exponential'], dist="expon", plot=axs[1, 1])
axs[1, 1].set_title('Q-Q Plot of Exponential Distribution')

# 调整布局
plt.tight_layout()
plt.show()

#%%>>>>>>>>>> 循环对称高斯分布( X~CN(0, sigma^2)  distribution) %%%%%%%%%%%%%%%%%%%%%%%%
import scipy
# 验证由两个均值为0，方差相同的独立同分布的正太分布分别为实部和虚部的复数向量分布符合瑞丽分布, 且这个复数的模的平方服从指数分布,

# X = (x1+jx2) ~ CN(0, sigma^2)
# x1 ~ CN(0, sigma^2/2), x2 ~ CN(0, sigma^2/2)

# from scipy.stats import rayleigh
# from scipy.stats import expon

mu = 0
sigma2_array = [1, 2]

### PDF of sqrt(x1**2 + x2**2) Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(sigma2_array), figsize=(len(sigma2_array)*4, 3))
loc = 0
for sigma2, ax in zip(sigma2_array, axs.ravel()):
    x1 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    x2 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    X = np.sqrt(x1**2 + x2**2)
    ## np
    count, bins, ignored = ax.hist(X, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (2*bins/sigma2) * np.exp(-bins**2/sigma2)
    ax.plot(bins, y, lw=2, color='r', label = "theory pdf")

    ## scipy
    x = np.linspace(scipy.stats.rayleigh.ppf(0.01, loc = loc, scale = np.sqrt(sigma2/2)), scipy.stats.rayleigh.ppf(0.99, loc = loc, scale = np.sqrt(sigma2/2)), 100)
    ax.plot(x, scipy.stats.rayleigh.pdf(x, loc = loc, scale = np.sqrt(sigma2/2)), 'b', lw=2, label = 'rayleigh pdf', zorder = 1)

    print(f"mean = {np.sqrt(np.pi/2)*np.sqrt(sigma2/2):.2f}/{np.mean(X)}, var = {(2-np.pi/2)*sigma2/2:.2f}/{np.var(X)} ")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc = 'best', frameon = False)
    title_idx = f"sigma^2 = {sigma2}"
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis = "x", direction = 'in')
    ax.tick_params(axis = "y", direction = 'in')

### PDF of (x1**2 + x2**2) Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(sigma2_array), figsize = (len(sigma2_array)*4, 3))

for sigma2, ax in zip(sigma2_array, axs.ravel()):
    x1 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size = 10000)
    x2 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size = 10000)
    X = (x1**2 + x2**2)
    ## np
    count, bins, ignored = ax.hist(X, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/sigma2) * np.exp(-bins/sigma2)
    ax.plot(bins, y, lw=2, color='r', label = "theory pdf", zorder = 1)

    ## scipy
    x = np.linspace(scipy.stats.expon.ppf(0.01, loc = loc, scale = sigma2), scipy.stats.expon.ppf(0.99, loc = loc, scale = sigma2), 100)
    ax.plot(x, scipy.stats.expon.pdf(x, loc = loc, scale = sigma2), 'b--', lw=2, label = 'expon pdf', zorder = 2)

    print(f"mean = {sigma2:.2f}/{np.mean(X)}, var = {sigma2**2:.2f}/{np.var(X)} ")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    title_idx = f"sigma^2 = {sigma2}"
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dirichlet分布 (Dirichlet Distribution)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 生成Dirichlet分布数据
alpha = np.array([2, 5, 3, 7])  # 参数向量
data = np.random.dirichlet(alpha, size=1000) # (1000, 4)

# 计算每个组别的均值
mean_values = np.mean(data, axis=0)

# 图1：不同组别的分布（条形图）
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.barplot(x=np.arange(1, len(alpha)+1), y=mean_values, palette="Set2")
plt.title("Mean Proportion of Each Category")
plt.xlabel("Category")
plt.ylabel("Mean Proportion")

# 图2：数据的散点图（散点图）
plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, c=data[:, 2], cmap='viridis')
plt.colorbar(label="Category 3 Proportion")
plt.title("Scatter plot of Category 1 vs. Category 2")
plt.xlabel("Category 1 Proportion")
plt.ylabel("Category 2 Proportion")

# 图3：饼图展示一个样本的比例
plt.subplot(1, 3, 3)
sample_idx = np.random.choice(range(1000))
plt.pie(data[sample_idx], labels=[f'Category {i+1}' for i in range(len(alpha))], autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title(f"Sample {sample_idx} Proportion Distribution")

plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 均匀分布 (Uniform Distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform

# 生成虚拟数据集
np.random.seed(42)
data = uniform.rvs(size=1000, loc=0, scale=10)  # 在区间 [0, 10) 上生成 1000 个均匀分布的样本

# 创建图形对象
plt.figure(figsize=(12, 8))

# 子图1：直方图和KDE图
plt.subplot(2, 2, 1)
sns.histplot(data, kde=True, color='skyblue', bins=30, stat='density', edgecolor='black')
plt.title('Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Density')

# 子图2：累积分布函数（CDF）
plt.subplot(2, 2, 2)
sns.ecdfplot(data, color='green')
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Value')
plt.ylabel('CDF')

# 子图3：Q-Q图（与标准均匀分布对比）
plt.subplot(2, 2, 3)
uniform_samples = np.sort(uniform.rvs(size=1000, loc=0, scale=10))
plt.scatter(uniform_samples, np.sort(data), color='purple', edgecolor='black')
plt.plot([0, 10], [0, 10], 'r--')  # 参考线
plt.title('Q-Q Plot against Uniform Distribution')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

# 子图4：均匀分布样本的散点图（对比两个均匀分布样本）
plt.subplot(2, 2, 4)
data2 = uniform.rvs(size=1000, loc=0, scale=10)
plt.scatter(data, data2, color='orange', edgecolor='black', alpha=0.6)
plt.title('Scatter Plot of Two Uniform Distributions')
plt.xlabel('Data1')
plt.ylabel('Data2')

# 调整布局
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 拉普拉斯分布(laplace  distribution) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loc = 0
Lambda = [1, 2, 3, 4, 5, 6, ]

### PDF of laplace Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(Lambda), figsize=(len(Lambda)*6, 6))

for scale, ax in zip(Lambda, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.laplace.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.laplace.ppf(0.01, loc = loc, scale = scale), scipy.stats.laplace.ppf(0.99, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.laplace.pdf(x, loc = loc, scale = scale), 'b', lw=2, label = 'rayleigh pdf', zorder = 1)

    ## frozen
    rv = scipy.stats.laplace(loc = loc, scale = scale)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = scipy.stats.laplace.rvs(loc = loc, scale = scale, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.laplace(scale = scale, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = 1 / (2*scale) * np.exp(-np.abs(bins - loc)/scale)
    ax.plot(bins, y, lw = 2, color = 'r', ls = '--', label = "np pdf")

    print(f"mean = {mean:.2f}/{loc}/{np.mean(s)}, var = {var:.2f}/{2*scale**2}/{np.var(s)} ")
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,4)

    ax.legend(loc='best', frameon=False)
    ax.set_title(title_idx)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

### CDF of laplace Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = scipy.stats.laplace.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(scipy.stats.laplace.cdf(0.01, loc = loc, scale = scale), scipy.stats.laplace.ppf(0.99999, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, scipy.stats.laplace.cdf(x, loc = loc, scale = scale), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###########    离散随机变量
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 伯努利分布 (Bernoulli Distribution)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置随机种子
np.random.seed(42)

# 生成一个虚拟数据集，包含1000个伯努利分布的数据点
p = 0.3  # 成功的概率
n = 1000
data = np.random.binomial(1, p, n)

# 创建一个DataFrame
df = pd.DataFrame(data, columns=['Outcome'])

# 计算统计量
success_count = df['Outcome'].sum()
failure_count = n - success_count
probabilities = df['Outcome'].value_counts(normalize=True)
mean = df['Outcome'].mean()
variance = df['Outcome'].var()

# 设置画布大小
plt.figure(figsize=(15, 10))

# 图1：直方图（Histogram）
plt.subplot(2, 2, 1)
sns.histplot(df['Outcome'], bins=2, kde=False)
plt.title('Histogram of Bernoulli Distribution')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])

# 图2：概率质量函数（PMF）
plt.subplot(2, 2, 2)
sns.barplot(x=probabilities.index, y=probabilities.values, palette='viridis')
plt.title('Probability Mass Function (PMF)')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])

# 图3：箱型图（Box Plot）
plt.subplot(2, 2, 3)
sns.boxplot(x=df['Outcome'], palette='viridis')
plt.title('Box Plot of Outcomes')
plt.xlabel('Outcome')
plt.xticks([0, 1], labels=['Failure (0)', 'Success (1)'])

# 图4：统计量
plt.subplot(2, 2, 4)
stats_text = f"Sample Size: {n}\nSuccess Count: {success_count}\nFailure Count: {failure_count}\nMean: {mean:.2f}\nVariance: {variance:.2f}"
plt.text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center', bbox=dict(facecolor='lightgrey', alpha=0.5))
plt.axis('off')
plt.title('Summary Statistics')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二项分布 (Binomial Distribution)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 二项分布参数设置
n = 20  # 试验次数
p = 0.5  # 每次试验成功的概率

# 生成二项分布的虚拟数据集
data = np.random.binomial(n, p, size=1000)

# 计算概率质量函数（PMF）
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

# 创建图形
plt.figure(figsize=(14, 8))

# 子图1: 二项分布的PMF
plt.subplot(2, 2, 1)
plt.stem(x, pmf, basefmt=" ")
plt.title('Probability Mass Function (PMF)')
plt.xlabel('Number of successes')
plt.ylabel('Probability')
plt.grid()

# 子图2: 样本数据的直方图
plt.subplot(2, 2, 2)
sns.histplot(data, bins=n+1, kde=False, color='skyblue')
plt.title('Histogram of Sample Data')
plt.xlabel('Number of successes')
plt.ylabel('Frequency')
plt.grid()

# 子图3: 样本数据的累积分布函数（CDF）
plt.subplot(2, 2, 3)
sns.ecdfplot(data, color='red')
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Number of successes')
plt.ylabel('Cumulative Probability')
plt.grid()

# 调整子图间距
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  泊松分布 (Poisson Distribution)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# 设置泊松分布的参数
lambda_ = 5  # λ是泊松分布的均值，也就是事件发生的平均次数

# 生成泊松分布的随机样本
np.random.seed(42)
data = np.random.poisson(lambda_, 1000)

# 设置画布大小
plt.figure(figsize=(14, 10))

# 图1：数据的直方图
plt.subplot(2, 2, 1)
sns.histplot(data, bins=range(0, max(data)+2), kde=False, color='skyblue')
plt.title('Poisson Distribution Histogram')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')

# 图2：数据的折线图
plt.subplot(2, 2, 2)
unique_values, counts = np.unique(data, return_counts=True)
plt.plot(unique_values, counts, marker='o', linestyle='-', color='green')
plt.title('Frequency Line Plot')
plt.xlabel('Number of Events')
plt.ylabel('Frequency')

# 图3：泊松分布的概率质量函数(PMF)
plt.subplot(2, 2, 3)
x = np.arange(0, 20)
pmf = poisson.pmf(x, lambda_)
plt.bar(x, pmf, color='orange')
plt.title('Poisson Distribution PMF')
plt.xlabel('Number of Events')
plt.ylabel('Probability')

# 图4：理论PMF与样本数据的对比
plt.subplot(2, 2, 4)
plt.plot(x, pmf * len(data), marker='o', linestyle='-', color='red', label='Theoretical PMF')
plt.plot(unique_values, counts, marker='x', linestyle='--', color='blue', label='Sample Data')
plt.title('Theoretical PMF vs Sample Data')
plt.xlabel('Number of Events')
plt.ylabel('Count')
plt.legend()

# 调整布局并显示图像
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  多项分布 (Multinomial Distribution)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multinomial

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 定义多项分布的参数
n = 1000  # 实验总次数
p = [0.1, 0.3, 0.2, 0.15, 0.25]  # 每个类别的概率

# 生成多项分布的数据
data = multinomial.rvs(n=n, p=p, size=1)[0]

# 创建类别标签
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']

# 创建一个2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 绘制条形图
sns.barplot(x=categories, y=data, palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Bar Plot of Category Counts')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Counts')

# 绘制饼图
axes[0, 1].pie(data, labels=categories, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(categories)))
axes[0, 1].set_title('Pie Chart of Category Proportions')

# 绘制累积条形图
cumulative_data = np.cumsum(data)
sns.barplot(x=categories, y=cumulative_data, palette='viridis', ax=axes[1, 0])
axes[1, 0].set_title('Cumulative Bar Plot')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Cumulative Counts')

# 绘制各类别的分布情况
sns.histplot(data, bins=10, kde=True, color='purple', ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Counts Across Categories')
axes[1, 1].set_xlabel('Counts')
axes[1, 1].set_ylabel('Frequency')

# 调整布局
plt.tight_layout()
plt.show()















































































































































































































































































































