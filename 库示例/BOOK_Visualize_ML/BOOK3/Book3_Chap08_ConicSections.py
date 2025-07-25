#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:54:07 2025

@author: jack
"""

#%% Bk3_Ch8_01

from matplotlib import pyplot as plt
from sympy import plot_implicit, symbols, Eq
x1, x2 = symbols('x1 x2')

def plot_curve(Eq_sym):
    h_plot = plot_implicit(Eq_sym, (x1, -2.5, 2.5), (x2, -2.5, 2.5), xlabel = r'$\it{x_1}$', ylabel = r'$\it{x_2}$')
    h_plot.show()

#%% plot ellipses

plt.close('all')

# major axis on x1
Eq_sym = Eq(x1**2 + x2**2, 1)
plot_curve(Eq_sym)

# major axis on x1
Eq_sym = Eq(x1**2/4 + x2**2, 1)
plot_curve(Eq_sym)

# major axis on x2
Eq_sym = Eq(x1**2 + x2**2/4, 1)
plot_curve(Eq_sym)

# major axis on x1 with center (h,k)
Eq_sym = Eq((x1-0.5)**2/4 + (x2-0.5)**2, 1)
plot_curve(Eq_sym)

# major axis on x2 with center (h,k)
Eq_sym = Eq((x1-0.5)**2 + (x2-0.5)**2/4, 1)
plot_curve(Eq_sym)

# major axis rotated counter clockwise by pi/4, 逆时针旋转 θ = 45° = π/4 获得
Eq_sym = Eq(5*x1**2/8 -3*x1*x2/4 + 5*x2**2/8, 1)
plot_curve(Eq_sym)

# major axis rotated counter clockwise by 3*pi/4, 逆时针旋转 θ = 135° = 3π/4 获
Eq_sym = Eq(5*x1**2/8 +3*x1*x2/4 + 5*x2**2/8, 1)
plot_curve(Eq_sym)





#%%




#%%




#%%




#%%




