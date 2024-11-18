# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:33:44 2021

@author: Parthe
"""

import numpy as np
from numpy.linalg import svd, norm, inv
from numpy.random import randn
import matplotlib.pyplot as plt
from VAMP.denoisers.map_denoisers import L1, L2, Lp, ElasticNet
from VAMP.denoisers.lmmse_denoisers import Lmmse
from VAMP.solvers.vamp import Vamp
from numpy import sqrt, log10, abs, count_nonzero, eye
from sklearn import linear_model

def generate_true_signal(p, prior):
    x_true = randn(p,1)
    if prior == 'sparse':
        return np.where(abs(x_true)>2, x_true, 0)
    elif prior.lower() == 'gaussian':
        return x_true
    elif prior == 'elastic_net':
        return x_true + np.where(abs(x_true)>2, x_true, 0)*9

def generate_data(n,p,snr=30, *args, **kwargs):
    
    unscaled_A = randn(n, p)
    A = unscaled_A / sqrt(n)
    x_true = generate_true_signal(p,*args, **kwargs)
    y_true = A @ x_true
    
    unscaled_noise = randn(n,1)
    sigma_w = norm(y_true)/(10**(snr/20.))/norm(unscaled_noise)
    noise = unscaled_noise * sigma_w
    y = y_true + noise
    
    try:
        if verbose:
            print(f'norm^2(x) = {norm(x_true)**2}')
            print(f'norm^2(y) = {norm(y_true)**2}')
            print(10*log10((norm(y_true)**2)/(norm(noise)**2)))
            print(f'snr = {snr:.2f} dB')
    except NameError: pass
    
    return x_true, A, y, sigma_w 
    
def gaussian_demo():
    np.random.seed(1)
    n, p = 160, 200
    x_true, A, y, sigma_w =generate_data(n,p,snr=30,prior='gaussian')
    noise_precision = (1/sigma_w)**2
    Lambda = sigma_w**2
    xhat_analytic = inv(A.T @ A + Lambda *  eye(A.shape[1])) @ (A.T @ y)
    MSE_analytic = 10*log10(norm(xhat_analytic-x_true)**2/p)
    
    vamp = Vamp(in_out_shape=(p,n),\
                        denoisers=(L2(Lambda=Lambda), Lmmse(A=svd(A), y=y, noise_precision=noise_precision)))
    vamp.fit()
    plt.plot(range(vamp.n_iterations),[10*log10(norm(x-x_true)**2/p) for x in vamp.xhat_in],label='VAMP')
    plt.xlabel('iteration')
    plt.ylabel('MSE(dB)')
    plt.ylim([-10,5])
    plt.hlines(MSE_analytic, 0, vamp.n_iterations-1, 'm', 'dashed', label='Analytic')
    plt.title('Gaussian parameter')
    
def sparse_demo():
    np.random.seed(1)
    n, p = 100, 200
    x_true, A, y, sigma_w =generate_data(n,p,snr=30,prior='sparse')
    noise_precision = (1/sigma_w)**2
    Lambda = (1- count_nonzero(x_true)/n)*sigma_w**2
    
    xhat_ridge = inv(A.T @ A + Lambda *  eye(A.shape[1])) @ (A.T @ y)
    MSE_ridge = 10*log10(norm(xhat_ridge-x_true)**2/p)
    print(f'MSE_ridge = {MSE_ridge:.3f}dB')


    # sklearn
    model = linear_model.Lasso(alpha = Lambda/n, fit_intercept=False, max_iter=10000)
    model.fit(A,y)
    xhat_sklearn = model.coef_[:,np.newaxis]
    MSE_sklearn = 10*log10(norm(xhat_sklearn-x_true)**2/p)
    print(f'MSE_sklearn = {MSE_sklearn:.3f}dB')
    
    in_denoiser = L1(Lambda=Lambda)
    out_denoiser = Lmmse(A=svd(A), y=y, noise_precision=noise_precision)
    vamp = Vamp(in_out_shape=(p,n), denoisers=(in_denoiser, out_denoiser))
    vamp.fit(r_in=randn(p,1)*.1, gamma_in=.1, n_iterations=20)
    plt.plot(range(vamp.n_iterations),[10*log10(norm(x-x_true)**2/p) for x in vamp.xhat_in],label='VAMP')
    plt.xlabel('iteration')
    plt.ylabel('MSE(dB)')
    plt.ylim([-50,30])
    plt.hlines(MSE_ridge, 0, vamp.n_iterations-1, 'm', 'dashed', label='Ridge')
    plt.hlines(MSE_sklearn,0, vamp.n_iterations-1,'r','dashed',label='sklearn')
    plt.title('Sparse parameter')
    plt.legend()
    
    
    
def lp_demo():
    lp_norm = 1.2
    np.random.seed(0)
    n, p = 160, 200
    x_true, A, y, sigma_w =generate_data(n,p,snr=30,prior='gaussian')
    noise_precision = (1/sigma_w)**2
    Lambda = sigma_w**2
    
    xhat_ridge = inv(A.T @ A + Lambda *  eye(A.shape[1])) @ (A.T @ y)
    MSE_ridge = 10*log10(norm(xhat_ridge-x_true)**2/p)
    print(f'MSE_ridge = {MSE_ridge:.3f} dB')


    # sklearn
    model = linear_model.Lasso(alpha = Lambda/n, fit_intercept=False, max_iter=10000)
    model.fit(A,y)
    xhat_sklearn = model.coef_[:,np.newaxis]
    MSE_sklearn = 10*log10(norm(xhat_sklearn-x_true)**2/p)
    print(f'MSE_sklearn = {MSE_sklearn:.3f} dB')
    
    in_denoiser = Lp(p=lp_norm,Lambda=Lambda, max_iter=10)
    out_denoiser = Lmmse(A=svd(A), y=y, noise_precision=noise_precision)
    vamp = Vamp(in_out_shape=(p,n), denoisers=(in_denoiser, out_denoiser))
    vamp.fit(r_in=randn(p,1)*10, gamma_in=100, n_iterations=20)
    print(f'MSE_vamp = {10*log10(norm(vamp.xhat_in[-1]-x_true)**2/p):.3f} dB')
    plt.plot(range(vamp.n_iterations),[10*log10(norm(x-x_true)**2/p) for x in vamp.xhat_in],label='VAMP')
    plt.xlabel('iteration')
    plt.ylabel('MSE(dB)')
    plt.ylim([-10,5])
    plt.hlines(MSE_ridge, 0, vamp.n_iterations-1, 'm', 'dashed', label='analytic Ridge')
    plt.hlines(MSE_sklearn,0, vamp.n_iterations-1,'r','dashed',label='sklearn LASSO')
    plt.title('Lp regression')
    plt.legend()
    
def elastic_net_demo():
    np.random.seed(0)
    n, p = 160, 200
    l1_ratio = .8
    x_true, A, y, sigma_w =generate_data(n,p,snr=30,prior='elastic_net')
    noise_precision = (1/sigma_w)**2
    Lambda = sigma_w**2
    
    xhat_ridge = inv(A.T @ A + Lambda *  eye(A.shape[1])) @ (A.T @ y)
    MSE_ridge = 10*log10(norm(xhat_ridge-x_true)**2/p)
    print(f'MSE_ridge = {MSE_ridge:.3f} dB')

    # sklearn
    model = linear_model.ElasticNet(alpha = Lambda/n, l1_ratio = l1_ratio, fit_intercept=False, max_iter=10000)
    model.fit(A,y)
    xhat_sklearn = model.coef_[:,np.newaxis]
    MSE_sklearn = 10*log10(norm(xhat_sklearn-x_true)**2/p)
    print(f'MSE_sklearn = {MSE_sklearn:.3f} dB')
    
    in_denoiser = ElasticNet(l1_ratio=l1_ratio, Lambda=Lambda)
    out_denoiser = Lmmse(A=svd(A), y=y, noise_precision=noise_precision)
    vamp = Vamp(in_out_shape=(p,n), denoisers=(in_denoiser, out_denoiser))
    vamp.fit(r_in=randn(p,1)*.1, gamma_in=.1, n_iterations=20)
    print(f'MSE_vamp = {10*log10(norm(vamp.xhat_in[-1]-x_true)**2/p):.3f} dB')
    plt.plot(range(vamp.n_iterations),[10*log10(norm(x-x_true)**2/p) for x in vamp.xhat_in],label='VAMP')
    plt.xlabel('iteration')
    plt.ylabel('MSE(dB)')
    plt.ylim([-10,5])
    plt.hlines(MSE_ridge, 0, vamp.n_iterations-1, 'm', 'dashed', label='analytic Ridge')
    plt.hlines(MSE_sklearn,0, vamp.n_iterations-1,'r','dashed',label='sklearn E-Net')
    plt.title('Elastic Net')
    plt.legend()
    
if __name__ == '__main__':
    
    None
    # gaussian_demo()
    # sparse_demo()
    # lp_demo()
    elastic_net_demo()
