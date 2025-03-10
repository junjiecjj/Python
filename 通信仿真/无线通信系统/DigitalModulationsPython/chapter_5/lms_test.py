"""
Script: DigiCommPy.chapter_5.lms_test.py
Verify LMS filter update equations (adaptive filter design)
System identification using LMS algorithm

@author: Mathuranathan Viswanathan
Created on Sep 4, 2019
"""
import numpy as np
from numpy.random import randn
from numpy import convolve
import sys
sys.path.append("..")


N = 5 # length of the desired filter
mu=0.1 # step size for LMS algorithm
r=randn(10000) # random input sequence of length 10000
h=randn(N)+1j*randn(N) # random complex system
a=convolve(h,r) # reference signal


from DigiCommPy.equalizers import LMSEQ
lms_eq = LMSEQ(N) #initialize the LMS filter object
lms_eq.design(mu,r,a) # design using input and reference sequences
print('System impulse response (h): {}'.format(h))
print('LMS adapted filter (w): {}'.format(lms_eq.w))
