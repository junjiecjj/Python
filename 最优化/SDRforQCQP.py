#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:05:29 2024

@author: jack
https://blog.csdn.net/weixin_39274659/article/details/117627627
https://blog.csdn.net/weixin_39274659/article/details/121619619

https://blog.csdn.net/weixin_44655342/article/details/121175761

《Semidefinite Relaxation of Quadratic Optimization Problems》

"""


import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import cvxpy as cpy
from matplotlib.pyplot import MultipleLocator

import scipy.constants as CONSTANTS

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



Nt = 16
M = 4
L = 100  # number of Gaussian randomizations
G = sqrt(2) / 2 * (randn(M, Nt) + 1j * randn(M, Nt))
hr = sqrt(2) / 2 * (randn(M, 1) + 1j * randn(M, 1))
hd = sqrt(2) / 2 * (randn(Nt, 1) + 1j * randn(Nt, 1))
phi = diag(hr') * G

R = [phi * phi' phi * hd; hd' * phi' 0]

cvx_begin sdp quiet
    variable V(M+1, M+1) hermitian
    maximize(real(trace(R*V)));
    subject to
    diag(V) == 1;
    V >= 0;
cvx_end

#%% method 1
max_F = 0;
max_v = 0;
[U, Sigma] = eig(V);
for l = 1 : L
    r = sqrt(2) / 2 * (randn(M+1, 1) + 1j * randn(M+1, 1));
    v = U * Sigma^(0.5) * r;
    if v' * R * v > max_F
        max_v = v;
        max_F = v' * R * v;
    end
end

v = exp(1j * angle(max_v / max_v(end)));
v = v(1 : M);
v' * phi * phi' * v

#%% method 2
max_F = 0;
max_v = 0;
[U, Sigma] = eig(V);
for l = 1 : L
    r = sqrt(2) / 2 * (randn(M+1, 1) + 1j * randn(M+1, 1));
    v = U * Sigma^(0.5) * r;
    v = exp(1j * angle(v / v(end)));
    v = v(1 : M);
    if v' * phi * phi' * v > max_F
        max_v = v;
        max_F = v' * phi * phi' * v;
    end
end
max_v' * phi * phi' * max_v


#%% method 3  element iteration
T = phi * phi';
v = sqrt(2) / 2 * (randn(M, 1) + 1j * randn(M, 1));
for n = 1 : 10
    for i = 1 : M
        tmp = 0;
        for j = 1 : M
            if i~= j
                tmp = tmp + T(i,j) * v(j);
            end
        end
        v(i) = exp(1j * angle(tmp));
    end
end
v' * phi * phi' * v



































































