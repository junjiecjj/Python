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

from Tools import  ULA2UPA_Los, RIS2UserSteerVec
sys.path.append("../")
from Solver import DC_F, DC_theta
from Utility import set_random_seed, set_printoption

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


BS_local = [0, 0, 25]
RIS_local = [50, 50, 40]

T0 = 30  # dB

alpha_UB = 3.5
alpha_RB = 2.2
alpha_UR = 2.8

K = 16
M = 30
N = 20



































































































































































































