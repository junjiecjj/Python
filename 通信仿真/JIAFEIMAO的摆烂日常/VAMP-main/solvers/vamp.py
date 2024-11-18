# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:35:02 2021

@author: Parthe
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from .base import BaseSolver

class Vamp(BaseSolver):
    
    def __init__(self, in_out_shape, denoisers, damp=1., damp_var=1., eps=1e-32):
        super().__init__(['xhat_in','eta_in','r_in','gamma_in',\
                                            'xhat_out','eta_out','r_out','gamma_out'])
        self.input_size, self.output_size  = in_out_shape
        self.damp = damp
        self.damp_var = damp_var
        self.eps = eps
        self.input_denoiser = denoisers[0]
        self.output_denoiser = denoisers[-1]
    
    def fit(self, r_in=None, gamma_in=None, n_iterations=10):
        
        if r_in is None:
            r_in = randn(self.input_size,1)*.01
        if gamma_in is None:
            gamma_in = 100
        
        self.n_iterations = n_iterations
        
        for iteration in range(n_iterations):
    
            # forward pass
            xhat_in, eta_in, _ = self.input_denoiser(r_in, gamma_in)
            if iteration>0:
                xhat_in = self.damp*xhat_in + (1-self.damp)*self.xhat_in[-1]
                eta_in = self.damp_var*eta_in + (1-self.damp_var)*self.eta_in[-1]
            eta_in =  max(self.eps, eta_in)
            gamma_out =  max(self.eps, eta_in - gamma_in)
            r_out = (eta_in * xhat_in - gamma_in * r_in)/ gamma_out
            
            # backward pass
            xhat_out, eta_out, _ = self.output_denoiser(r_out, gamma_out)
            eta_out =  max(self.eps, eta_out)
            gamma_in =  max(self.eps, eta_out - gamma_out)
            r_in = (eta_out * xhat_out - gamma_out * r_out)/ gamma_in
            
            self.update([xhat_in, eta_in, r_in, gamma_in, \
                         xhat_out, eta_out, r_out, gamma_out])