#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:11:50 2025

@author: jack
"""

from typing import Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy


A = np.array([[1.0, 2.0], [3.0, 4.0]])
r = scipy.linalg.sqrtm(A)

print(f"A = {A}")
print(f"r = {r}")
print(f"r@r = {r@r}")


print(f"A^(1/2) = {A**(1/2)}") # != r







