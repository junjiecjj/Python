#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:00:16 2024
@author: Junjie Chen
@email: 2716705056@qq.com
"""

import numpy as np
import scipy.io as sio


class SCMA(object):
    ## code parameters
    def __init__(self, codebookfile = 'DE_rayleigh.mat'):
        # self.args = args
        self.J = 0  # user num
        self.K = 0  # resource block num
        self.M = 0  # codeword num
        self.Init(codebookfile)

    ## init & normalize CB;
    def Init(self, codebookfile = 'DE_rayleigh.mat'):
        CB = sio.loadmat(codebookfile)['CB']
        (self.K, self.M, self.J) = CB.shape
        ## normlized CodeBook
        for k in range(CB.shape[0]):
            tmp = np.sum(np.abs(CB[k])**2) / self.M
            CB[k] /= np.sqrt(tmp)
        self.CB = CB
        ## factor Graph
        F = np.zeros((self.K, self.J), dtype = np.int8)
        for j in range(self.J):
            F[np.where(CB[:,1,j] != 0)[0], j] = 1
        self.FG = F
        return

    def encoder(self, uu, h, CB):

        return

    def MPAdetector_hard(self, yy, h, CB, sigma2, Nit = 6):

        return

    def MPAdetector_soft(self, yy, h, CB, sigma2, Nit = 6):

        return


scma = SCMA()











































































































































































