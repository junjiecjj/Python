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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% erf, erfc, Qfun %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gamma %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%>>>>>>>>>>>>>>>>>>>>>>>>  gamma function.
import numpy as np
from scipy.special import gamma, factorial

gamma([0, 0.5, 1, 5])
z = 2.5 + 1j
gamma(z)

gamma(z+1), z*gamma(z)  # Recurrence property

gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
x = np.linspace(-3.5, 5.5, 2251)
y = gamma(x)

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


#%%>>>>>>>>>>>>>>>>>>>>>>>>  gamma continuous random variable distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








































































































































































































































































































































































