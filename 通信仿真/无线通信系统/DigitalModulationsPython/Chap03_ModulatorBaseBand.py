#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:37:52 2025

@author: jack
"""
import numpy as np # for numerical computing
import sys
sys.path.append('..')
from DigiCommPy.modem import PSKModem


M = 16 #16 points in the constellation
pskModem = PSKModem(M) #create a 16-PSK modem object
pskModem.plotConstellation() #plot ideal constellation for this modem

nSym = 10 #10 symbols as input to PSK modem
inputSyms = np.random.randint(low=0, high = M, size=nSym) # uniform random

modulatedSyms = pskModem.modulate(inputSyms) #modulate


detectedSyms = pskModem.demodulate(modulatedSyms) #demodulate

