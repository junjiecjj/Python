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
import scipy as sy
import scipy.special as sps
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
    return 0.5 * sy.special.erfc(x / np.sqrt(2))

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


#%%%%%%%%%%%%%%%%%%%%%% Gamma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%>>>>>>>>>>>>>>>>>>>>>>>>  gamma function.
import numpy as np
from scipy.special import factorial

sy.special.gamma([0, 0.5, 1, 5])
z = 2.5 + 1j
sy.special.gamma(z)

sy.special.gamma(z+1), z*sy.special.gamma(z)  # Recurrence property

sy.special.gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
x = np.linspace(-3.5, 5.5, 2251)
y = sy.special.gamma(x)

fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(x, y, 'b', lw = 2, alpha=0.6, label='gamma(x)')
k = np.arange(1, 7)
ax.plot(k, factorial(k-1), 'k*', ms = 10, alpha=0.6, label='(x-1)!, x = 1, 2, ...')
ax.set_xlim(-3.5, 5.5)
ax.set_ylim(-10, 25)
plt.grid()
plt.xlabel('x')
plt.legend(loc='lower right')
plt.show()


#%%>>>>>>>>>  gamma  distribution %%%%%%%%%%%%%%%%%%%%%%%%%%

a = 1.99
mean, var, skew, kurt = sy.stats.gamma.stats(a, moments='mvsk')


a_array = [0.8, 1, 2, 4 ]
scale_array = [0.1, 0.5, 1, np.sqrt(2), 4]
scale_array_, a_array_ = np.meshgrid(scale_array, a_array)

### PDF of Beta Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(scale_array), figsize=(len(scale_array)*4, len(a_array)*3))
for a_idx, scale_idx, ax in zip(a_array_.ravel(), scale_array_.ravel(), axs.ravel()):
    mean, var, skew, kurt = sy.stats.gamma.stats(a_idx,scale = scale_idx, moments='mvsk')
    print(f"mean = {mean:.2f}/{a_idx*scale_idx:.2f}, var = {var:.2f}/{a_idx*scale_idx**2:.2f}")

    x = np.linspace(sy.stats.gamma.ppf(0.01, a_idx, scale = scale_idx), sy.stats.gamma.ppf(0.99, a_idx, scale = scale_idx), 100)
    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, sy.stats.gamma.pdf(x, a_idx, scale = scale_idx), 'b', lw=2, label = 'gamma pdf', zorder = 1)

    ## frozen
    rv = sy.stats.gamma(a_idx, scale = scale_idx)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = sy.stats.gamma.rvs(a_idx, scale = scale_idx, size = 10000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.03, facecolor = "#FF3300", label= "rvs", zorder = 3)

    ## np
    s = np.random.gamma(a_idx, scale_idx, 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = bins**(a_idx-1)*(np.exp(-bins/scale_idx) / (sps.gamma(a_idx)*scale_idx**a_idx))
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
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(scale_array), figsize=(len(scale_array)*4, len(a_array)*3))
for a_idx, scale_idx, ax in zip(a_array_.ravel(), scale_array_.ravel(), axs.ravel()):
    x = np.linspace(sy.stats.gamma.cdf(0.01, a_idx, scale = scale_idx), sy.stats.gamma.ppf(0.99999, a_idx, scale = scale_idx), 100)
    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, sy.stats.gamma.cdf(x, a_idx, scale_idx), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')

#%%>>>>>>>>>>>>>>>>>  beta distribution     %%%%%%%%%%%%%%%%%%%%%%%%%%%%
from scipy.stats import beta

a_array = [0.5, 0.8, 2, 4, 6]
b_array = [0.5, 0.8, 2, 4, 6 ]
b_array_, a_array_ = np.meshgrid(b_array, a_array)

### PDF of Beta Distributions
fig, axs = plt.subplots(nrows = len(a_array), ncols = len(b_array), figsize=(len(b_array)*4, len(a_array)*3))
for a_idx, b_idx, ax in zip(a_array_.ravel(), b_array_.ravel(), axs.ravel()):
    mean, var, skew, kurt = beta.stats(a = a_idx, b = b_idx, moments='mvsk')

    x = np.linspace(beta.ppf(0.01, a = a_idx, b = b_idx), beta.ppf(0.99, a = a_idx, b = b_idx), 100)
    title_idx = f"a = {a_idx:.2f}, b = {b_idx:.2f}"
    ax.plot(x, beta.pdf(x, a = a_idx, b = b_idx), 'b', lw=2, label = 'gamma pdf', zorder = 1)

    ## frozen
    rv = beta(a = a_idx, b = b_idx)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = beta.rvs(a = a_idx, b = b_idx, size = 10000)
    ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.03, facecolor = "#FF3300", label= "rvs", zorder = 3)

    ## np
    s = np.random.beta(a = a_idx, b = b_idx, size = 10000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    # y = bins**(a_idx-1) * (1 - bins)**(b_idx-1) *sps.gamma(a_idx+b_idx)/ (sps.gamma(a_idx) * sps.gamma(b_idx))
    # ax.plot(bins, y, lw=2, color='r', label = "np pdf")

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
    mean, var, skew, kurt = beta.stats(a = a_idx, b = b_idx, moments='mvsk')

    x = np.linspace(beta.ppf(0.01, a = a_idx, b = b_idx), beta.ppf(0.99, a = a_idx, b = b_idx), 100)
    title_idx = f"a = {a_idx:.2f}, scale = {scale_idx:.2f}"
    ax.plot(x, beta.cdf(x, a = a_idx, b = b_idx), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%>>>>>>>>>> chi2  distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from scipy.stats import chi2

df = 55
mean, var, skew, kurt = chi2.stats(df, moments='mvsk')

df_array = [1, 2, 4, 5, 7, 9, 10 ]

### PDF of chi2 Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(df_array), figsize=(len(df_array)*4, 3))
for df, ax in zip(df_array, axs.ravel()):
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
    print(f"mean = {mean:.2f}/ , var = {var:.2f}/ ")

    x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, chi2.pdf(x, df), 'b', lw=2, label = 'chi2 pdf', zorder = 1)

    ## frozen
    rv = chi2(df)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = chi2.rvs(df, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.03, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.chisquare(df = df, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/2)**(df/2) * bins**(df/2-1) * np.exp(-bins/2) / sps.gamma(df/2)
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
    x = np.linspace(chi2.cdf(0.01, df), chi2.ppf(0.99999, df), 100)
    title_idx = f"df = {df}"
    ax.plot(x, chi2.cdf(x, df), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')





#%%>>>>>>>>>>>>>>>>> rayleigh  distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from scipy.stats import rayleigh
# loc一定要写明确，注意关键字参数和位置参数造成的错误
loc = 0
scale = 2
mean, var, skew, kurt = rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

scale_array = [1, 2, 3, 4, 5, 6, ]

### PDF of rayleigh Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(rayleigh.ppf(0.01, loc = loc, scale = scale), rayleigh.ppf(0.99, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, rayleigh.pdf(x, loc = loc, scale = scale), 'b', lw=2, label = 'rayleigh pdf', zorder = 1)

    ## frozen
    rv = rayleigh(loc = loc, scale = scale)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = rayleigh.rvs(loc = loc, scale = scale, size = 10000)
    ax.hist(r, density = True, bins = 'auto', histtype = 'stepfilled', alpha = 0.1, facecolor = "#FF3300", label =  "rvs", zorder = 3)

    ## np
    s = np.random.rayleigh(scale = scale, size = 1000)
    count, bins, ignored = ax.hist(s, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = bins / scale**2 * np.exp(-bins**2/(2 * scale**2))
    ax.plot(bins, y, lw=2, color='r', label = "np pdf")

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
    mean, var, skew, kurt = rayleigh.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(rayleigh.cdf(0.01, loc = loc, scale = scale), rayleigh.ppf(0.99999, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, rayleigh.cdf(x, loc = loc, scale = scale), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%>>>>>>>>>>>>  指数分布 exponential  distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from scipy.stats import expon
# loc一定要写明确，注意关键字参数和位置参数造成的错误
loc = 0
scale = 2
mean, var, skew, kurt = expon.stats(loc = loc, scale = scale, moments='mvsk')

scale_array = [0.1, 0.5, 1, 2, 3, ]

### PDF of exponential Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(scale_array), figsize=(len(scale_array)*4, 3))

for scale, ax in zip(scale_array, axs.ravel()):
    mean, var, skew, kurt = expon.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(expon.ppf(0.01, loc = loc, scale = scale), expon.ppf(0.99, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, expon.pdf(x, loc = loc, scale = scale), 'b', lw=2, label = 'expon pdf', zorder = 1)

    ## frozen
    rv = expon(loc = loc, scale = scale)
    ax.plot(x, rv.pdf(x), 'k', lw=1, label = 'frozen pdf', zorder = 2)

    ## scipy Random variates.
    r = expon.rvs(loc = loc, scale = scale, size = 1000)
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
    mean, var, skew, kurt = expon.stats(loc = loc, scale = scale, moments='mvsk')

    x = np.linspace(expon.cdf(0.01, loc = loc, scale = scale), expon.ppf(0.99999, loc = loc, scale = scale), 100)
    title_idx = f"scale = {scale}"
    ax.plot(x, expon.cdf(x, loc = loc, scale = scale), 'b', lw=1)
    ax.set_title(title_idx)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')


#%%>>>>>>>>>>  X~CN(0, sigma^2)  distribution %%%%%%%%%%%%%%%%%%%%%%%%

# X = (x1+jx2) ~ CN(0, sigma^2)
# x1 ~ CN(0, sigma^2/2), x2 ~ CN(0, sigma^2/2)

from scipy.stats import rayleigh
from scipy.stats import expon

mu = 0
sigma2_array = [1, 2]

### PDF of sqrt(x1**2 + x2**2) Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(sigma2_array), figsize=(len(sigma2_array)*4, 3))

for sigma2, ax in zip(sigma2_array, axs.ravel()):
    x1 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    x2 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    X = np.sqrt(x1**2 + x2**2)
    ## np
    count, bins, ignored = ax.hist(X, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (2*bins/sigma2) * np.exp(-bins**2/sigma2)
    ax.plot(bins, y, lw=2, color='r', label = "theory pdf")

    ## scipy
    x = np.linspace(rayleigh.ppf(0.01, loc = loc, scale = np.sqrt(sigma2/2)), rayleigh.ppf(0.99, loc = loc, scale = np.sqrt(sigma2/2)), 100)

    ax.plot(x, rayleigh.pdf(x, loc = loc, scale = np.sqrt(sigma2/2)), 'b', lw=2, label = 'rayleigh pdf', zorder = 1)

    print(f"mean = {mean:.2f}/{np.sqrt(np.pi/2)*scale}/{np.mean(s)}, var = {var:.2f}/{(2-np.pi/2)*scale**2}/{np.var(s)} ")
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


### PDF of (x1**2 + x2**2) Distributions
fig, axs = plt.subplots(nrows = 1, ncols = len(sigma2_array), figsize=(len(sigma2_array)*4, 3))

for sigma2, ax in zip(sigma2_array, axs.ravel()):
    x1 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    x2 = np.random.normal(loc = mu, scale = np.sqrt(sigma2/2), size=10000)
    X = (x1**2 + x2**2)
    ## np
    count, bins, ignored = ax.hist(X, density=True, bins='auto', histtype='stepfilled', alpha=0.1, facecolor = "#0099FF", label= "np hist", zorder = 4)
    y = (1/sigma2) * np.exp(-bins/sigma2)
    ax.plot(bins, y, lw=2, color='r', label = "theory pdf", zorder = 1)

    ## scipy
    x = np.linspace(expon.ppf(0.01, loc = loc, scale = sigma2), expon.ppf(0.99, loc = loc, scale = sigma2), 100)
    ax.plot(x, expon.pdf(x, loc = loc, scale = sigma2), 'b--', lw=2, label = 'expon pdf', zorder = 2)

    print(f"mean = {mean:.2f}/{np.sqrt(np.pi/2)*scale}/{np.mean(s)}, var = {var:.2f}/{(2-np.pi/2)*scale**2}/{np.var(s)} ")
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




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###########    离散随机变量
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

















































































































































































































































































































































