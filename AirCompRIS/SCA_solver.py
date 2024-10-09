
# -*- coding: utf-8 -*-
"""
Created on 2024/08/15

@author: Junjie Chen

"""

import os, sys

import scipy
import numpy as np

import copy
from scipy.optimize import minimize


def SCA(N, L, K, h_d, G, f, theta, Imax, tau, threshold, P0, verbose, RISON = 1):
    if not RISON:
        # print("No RIS!!!")
        theta = np.zeros(L, dtype = complex)
    MSE_recod = np.zeros(Imax + 1)
    h = np.zeros([N, K], dtype = complex)
    for i in range(K):
        h[:,i] = h_d[:,i] + G[:,:,i]@theta
    mse = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / P0
    MSE_recod[0] = mse
    for it in range(Imax):
        obj_mse = mse
        a = np.zeros((N, K), dtype = complex)
        b = np.zeros((L, K), dtype = complex)
        c = np.zeros((1, K), dtype = complex)
        F_cro = np.outer(f, np.conjugate(f))
        for i in range(K):
            a[:,i] = tau*f + np.outer(h[:, i], np.conjugate(h[:, i]))@f
            if RISON:
                b[:, i] = tau * theta + G[:,:,i].conj().T @ F_cro @ h[:,i]
                c[:, i] = np.abs(f.conj() @ h[:, i])**2 + 2*tau*(L+1) + 2*np.real((theta.conj().T)@(G[:,:,i].conj().T)@F_cro@h[:,i])
            else:
                c[:, i] = np.abs(f.conj() @ h[:,i])**2 + 2*tau

        fun = lambda mu: np.real(2*np.linalg.norm(a@mu, ord = 2) + 2*np.linalg.norm(b@mu, ord = 1) - c@mu)
        cons = ({'type': 'eq', 'fun': lambda mu: np.sum(mu)-1})
        bnds = ((0, None) for i in range(K))
        res = minimize(fun, [1]*K, bounds = tuple(bnds), constraints = cons)
        if ~res.success:
            print("    minimize fail")
            pass
        fn = a@res.x
        fn = fn/np.linalg.norm(fn)
        f = fn
        thetan = b@res.x
        ## thetan=thetan/np.abs(thetan)
        if RISON:
            # print(" RIS ON")
            thetan = thetan/np.abs(thetan)
            theta = thetan
        h = np.zeros([N, K], dtype = complex)
        for i in range(K):
            h[:,i] = h_d[:,i] + G[:,:,i]@theta
        mse = np.linalg.norm(f, ord = 2)**2 / min(np.abs(f.conj()@h)**2) / P0
        MSE_recod[it + 1] = mse
        # if  verbose >= 1:
        #     if (it + 1) % 50 == 0:
        #         print(f'    Iteration {it} MSE {mse:.6f} Opt Obj {res.fun:.6f}' )
        if np.abs(mse - obj_mse)/ abs(mse) <= threshold:
            break
    # if  verbose >= 1:
    #     print(f'    SCA Take {it+1} iterations with final obj {MSE_recod[it+1]:.6f}')
    MSE_recod = MSE_recod[0 : it + 2]
    return f, theta, MSE_recod

def SCA_RIS(N, L, K, h_d, G, threshold, P0, Imax, tau, verbose, RISON = 1):
    h = np.zeros([N, K], dtype = complex)
    if RISON:
        # print("ON")
        # theta0 = np.ones([L], dtype = complex)
        theta0 = np.random.randn(L, ) + 1j * np.random.randn(L, )
        theta0 = theta0/np.abs(theta0)
    else:
        # print("OFF")
        theta0 = np.zeros(L, dtype = complex)
    for i in range(K):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta0
    # f0 = h[:,0]/np.linalg.norm(h[:, 0])
    f0 = np.random.randn(N, ) + 1j * np.random.randn(N, )
    f0 = f0 / np.linalg.norm(f0, ord = 2)

    f, theta, MSE_log = SCA(N, L, K, h_d, G, f0, theta0, Imax, tau, threshold, P0, verbose, RISON)
    h = np.zeros([N, K], dtype=complex)
    for i in range(K):
        h[:,i] = h_d[:,i] + G[:,:,i]@theta

    return f, theta, MSE_log
















































