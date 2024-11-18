# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:27:41 2021

@author: Parthe
"""

import numpy as np
from numpy.linalg import svd, norm, inv
from numpy.random import randn
import matplotlib.pyplot as plt
from denoisers import prox_l1, prox_ridge, Track_variables, LMMSE_denoiser
from functools import partial
from numpy import sqrt, log10, abs, count_nonzero, array, where, eye








#%% VAMP - State Evolution




vamp = Track_variables(['Xhat_in','eta_in','R_in','gamma_in','Xhat_out','eta_out','R_out','gamma_out','mse'])
r_in = r0
gamma_in = gamma0
n_iterations = 20
rho = 1





for iteration in range(n_iterations):
    
    # forward pass
    xhat_in, eta_in, _ = input_denoiser(r_in, gamma_in)
    try:
        xhat_in = rho*xhat_in + (1-rho)*vamp.xhat_in[-1]
    except IndexError:
        pass
    eta_in =  max(eps, eta_in)
    gamma_out =  max(eps, eta_in - gamma_in)
    r_out = (eta_in * xhat_in - gamma_in * r_in)/ gamma_out
    
    # backward pass
    xhat_out, eta_out, _ = output_denoiser(r_out, gamma_out)
    eta_out =  max(eps, eta_out)
    gamma_in =  max(eps, eta_out - gamma_out)
    r_in = (eta_out * xhat_out - gamma_out * r_out)/ gamma_in
    
    vamp.update([xhat_in, eta_in, r_in, gamma_in, xhat_out, eta_out, r_out, gamma_out, norm(xhat_in-x_true)**2/p])

plt.figure()
plt.plot(range(n_iterations),10*np.log10(np.array(vamp.mse)))
plt.ylim([-10,5])
plt.hlines(MSE_analytic,0,n_iterations,'m','dashed','Analytic')
plt.ylabel('MSE (dB)')
plt.xlabel('iteration')