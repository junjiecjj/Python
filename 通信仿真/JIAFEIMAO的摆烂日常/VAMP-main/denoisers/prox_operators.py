# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:38:57 2021

@author: Parthe
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
eps = 1e-16


def prox_l1(r, gamma, Lambda=1.):
    # returns  argmin_x gamma/2 * ||x-r||^2 + Lambda * ||x||_1
    
    xhat = np.where(r>0,np.maximum(0,r-Lambda/gamma), np.minimum(0, r+Lambda/gamma))
    alpha = (np.count_nonzero(xhat)/len(xhat))
    eta = gamma/(eps+alpha)
    
    return xhat, eta, alpha

def prox_lp(r, gamma, Lambda=1, p=2, max_iter=3):
    
    """
    Returns argmin_x gamma/2 * ||x-r||^2 + Lambda / p * ||x||_p^p for p>1
    using Newtons method for ``max_iter`` iterations starting from 0
    For p=1 use ``prox_l1``
    For p=2 use ``prox_ridge`` for a faster implementation
    """
    xhat = np.zeros_like(r)
    for i in range(max_iter):
        grad = (gamma*(xhat-r)+Lambda*np.power(abs(xhat),(p-1)) *np.sign(xhat))
        if p>=2:
            hess_inv = 1/(gamma+Lambda*(p-1)*np.power(abs(xhat),(p-2)))
        else:
            hess_inv = 1/(gamma+Lambda*(p-1)/(eps+np.power(abs(xhat),(2-p))))
        xhat = xhat - grad*hess_inv # Newton iteration
        hess_inv_mean = hess_inv.mean()
    alpha = gamma * hess_inv_mean
    eta = gamma/alpha
    return xhat, eta, alpha

def prox_elastic_net(r, gamma, Lambda=1., l1_ratio=.5):
    """
    Returns argmin_x gamma/2* ||x-r||^2 + l1_ratio*Lambda* ||x||_1 + (1-l1_ratio)*Lambda/2* ||x||^2
    """
    xhat, _ , alpha = prox_l1(r, gamma, Lambda*l1_ratio)
    xhat = xhat * 1/(1 + Lambda/gamma * (1-l1_ratio))
    alpha = alpha * 1/(1 + Lambda/gamma * (1-l1_ratio))
    eta = gamma/(eps+alpha)
    return xhat, eta, alpha

def prox_ridge(r, gamma, Lambda=1):
    # returns  argmin_x gamma/2 * ||x-r||^2 + 1/2 * Lambda * ||x||^2
    alpha = (gamma/(Lambda+gamma))
    eta = gamma/alpha
    xhat = alpha * r 
    return xhat, eta, alpha


    
