#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:16 2024

@author: jack
"""

import numpy as np
import scipy.io as sio


class SCMA(object):
    def __init__(self, args):
        ## code parameters
        self.args = args
        self.J = 0
        self.K = 0

        self.M = 0

    def readCB(self):
        CB = sio.loadmat('DE_rayleigh.mat')['CB']
        (self.K, self.M, self.J) = CB.shape

        for k in range(CB.shape[0]):
            CB[k] /= np.sum(np.abs(CB[k])**2) / self.K
        return

    def encoder(self, uu):

        return

    def MPA_detector_hard(self, yy):

        return


    def MPA_detector_soft(self, yy):

        return

