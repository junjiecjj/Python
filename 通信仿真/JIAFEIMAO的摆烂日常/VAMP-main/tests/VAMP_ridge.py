# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:44:37 2021

@author: Parthe
"""

import numpy as np
from numpy.linalg import svd, norm, inv
from numpy.random import randn
import matplotlib.pyplot as plt
from denoisers import prox_l1, prox_ridge, Track_variables, LMMSE_denoiser
from functools import partial
from numpy import sqrt, log10, abs, count_nonzero, array, where, eye

# Vector Approximate Message Passing


np.random.seed(0)
n, p = 160, 200
unscaled_A = randn(n, p)
A = unscaled_A / sqrt(n)
# A = unscaled_A/ norm(unscaled_A, axis=0)
# U, S, Vh = svd(unscaled_A, full_matrices=False)
# A = U* np.ones_like(S)*.9 @ Vh 
x_true = randn(p,1)
y_true = A @ x_true
print(f'norm^2(x) = {norm(x_true)**2}')
print(f'norm^2(y) = {norm(y_true)**2}')

snr = 30. # dB
unscaled_noise = randn(n,1)
sigma_w = norm(y_true)/(10**(snr/20.))/norm(unscaled_noise)
noise_precision = sqrt(1/sigma_w)
noise = unscaled_noise * sigma_w

print(10*log10((norm(y_true)**2)/(norm(noise)**2)))
y = y_true + noise
print(f'snr = {snr:.2f} dB')

Lambda = sigma_w**2

#%%
## Sparse case
# # x_true = where(abs(x_true)>2,x_true,0)
# # Lambda = (1- count_nonzero(x_true)/n)
# # print(f'Lambda = {Lambda:.2f}')

## Analytic solution for Ridge regression
# mse_analytic_array = []
# Lambda_array = [10**x for x in range(-5,5)]
# for Lambda in Lambda_array:
#     xhat_analytic = inv(A.T @ A + Lambda* n/p * sigma_w**2 *  eye(A.shape[1])) @ (A.T @ y)
#     mse_analytic_array.append(norm(xhat_analytic - x_true)**2/p)
# plt.plot(10* log10(Lambda_array), 10* log10( array(mse_analytic_array)), 'r')
# plt.xlabel('Lambda')
# plt.ylabel('MSE (dB)')
xhat_analytic = inv(A.T @ A + Lambda *  eye(A.shape[1])) @ (A.T @ y)
MSE_analytic = 10*log10(norm(xhat_analytic-x_true)**2/p)
print(f'MSE_ridge = {MSE_analytic:.3f}dB')


#%% VAMP



gamma0 = 100
r0 = randn(*x_true.shape)*0.

Lambda = sigma_w**2
input_denoiser = partial(prox_ridge, Lambda=Lambda)
U, S, VT = svd(A, full_matrices=True)
output_denoiser = partial(LMMSE_denoiser, noise_precision=noise_precision, A= (U, S, VT), y=y)
eps = 1e-10


vamp = Track_variables(['xhat_in','eta_in','r_in','gamma_in','xhat_out','eta_out','r_out','gamma_out','mse'])
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




