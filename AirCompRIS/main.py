#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 22:29:12 2024

@author: jack
"""

import sys
import numpy as np
# import scipy
import cvxpy as cp
import matplotlib.pyplot as plt
# import math
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS


from Solver import DC_F, DC_theta
from Utility import set_random_seed, set_printoption
set_random_seed(99999)


filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


BS_locate = np.array([0, 0, 25])
RIS_locate = np.array([50, 50, 40])

T0 = 30  # dB

alpha_UB = 3.5  # AP 和 User 之间的衰落因子
alpha_RB = 2.2  # RIS 和 AP 之间的衰落因子
alpha_UR = 2.8  # RIS 和 User 之间的衰落因子

K = 16 # users
M = 30 # RIS elements
N = 20 # AP antennas
rho = 5
epsion = 1e-3
epsion_dc = 1e-8


users_locate_x = np.random.rand(K, 1) * 100 - 50
users_locate_y = np.random.rand(K, 1) * 100 + 50
users_locate_z = np.zeros((K, 1))
users_locate = np.hstack((users_locate_x, users_locate_y, users_locate_z))

d_UR = np





























































































































































































