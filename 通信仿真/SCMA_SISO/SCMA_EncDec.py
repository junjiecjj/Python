#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:16 2024

@author: jack
"""

import numpy as np
import scipy.io as sio


class SCMA(object):
    def __init__(self, codebookfile = 'DE_rayleigh.mat'):
        ## code parameters
        # self.args = args
        self.J = 0  # user num
        self.K = 0  # resource block num
        self.M = 0  # codeword num
        self.Init(codebookfile)

    ## init & normalize CB;
    def Init(self, codebookfile = 'DE_rayleigh.mat'):
        CB = sio.loadmat(codebookfile)['CB']
        (self.K, self.M, self.J) = CB.shape

        for k in range(CB.shape[0]):
            tmp = np.sum(np.abs(CB[k])**2) / self.M
            CB[k] /= np.sqrt(tmp)
        self.CB = CB
        return


    def encoder(self, uu, h, CB):

        return

    def MPAdetector_hard(self, yy, h, CB, sigma2):

        return


    def MPAdetector_soft(self, yy, h, CB, sigma2):

        return


scma = SCMA()
