#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:26:35 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzI1NzU4ODgwMg==&mid=2247493413&idx=1&sn=e5684fdc3d3702044fab049221337f0b&chksm=eb8009311c337d5b56413cf16aceb96182bb24c7d0256d5f4412003dd2207d5d3468f67f3b76&mpshare=1&scene=1&srcid=1114TM1nqlR3cxR5gvrMqjYQ&sharer_shareinfo=809873a389d090a424e6dce2f28e2131&sharer_shareinfo_first=809873a389d090a424e6dce2f28e2131&exportkey=n_ChQIAhIQV%2FTXtsIwO6SqTQPWEI3NXRKfAgIE97dBBAEAAAAAAD7KBWQ4Fw0AAAAOpnltbLcz9gKNyK89dVj0Ki86ZeXSFQhbzFzmgFEC03uh0mFCHNWy%2FZWd8NbjACcev%2BWC8GsJIBnW%2FQl%2F5WQ11klrG%2F4Mec6qgaSmyZfVEZJrc0m5UygF4u%2Fen16Qshe%2BjyaBG5BefNxO7iMQGC0G7FwmYIayr1zS3ESiv54A%2B%2FOse9GN6mfJTFmctyjA%2F%2FXxsrCEFTQUKOVA4nyV8XZuyvIYvsnXfIytCqkdfduVy0eNDD0xNwsVf3MIBwn5y8gJfdmHQ4vQA%2B8r5dsh9pV2lE3cGOHnhpm5bZIHxeoBAf%2F%2Bd9cfhx1qnrvzAjdYe19NjD6eQ665D0hb%2FFtL%2BkBt8wkoNE%2BnQvK7&acctmode=0&pass_ticket=t%2F1O97V7GS%2FQyoqphF16TxoP7Y24Vxbbv%2BSH8vRBlNOlxKaai1rztcMj1k5fhrsY&wx_header=0#rd


"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as op
from scipy.stats import norm


def denoise(v,var_x,var_z,epsilon):
    term1 = (1-epsilon)*norm.pdf(v,0,np.sqrt(var_z))
    term2 = epsilon*norm.pdf(v,0,np.sqrt(var_x+var_z))
    xW = var_x/(var_x+var_z)*v # Wiener filter
    added_term = term1+term2
    div_term = np.divide(term2, added_term)
    xhat = np.multiply(div_term, xW) # denoised version, x(t+1)

    # empirical derivative
    Delta = 0.0000000001 # perturbation
    term1_d = (1-epsilon)*norm.pdf((v+Delta),0,np.sqrt(var_z))
    term2_d = epsilon*norm.pdf((v+Delta),0,np.sqrt(var_x+var_z))
    xW2 = var_x/(var_x+var_z)*(v+Delta) # Wiener filter

    added_term = term1_d+term2_d
    mul_term = np.multiply(xW2,term2_d)
    xhat2 = np.divide(mul_term,added_term)
    d = (xhat2-xhat)/Delta
    return xhat,d

# y = np.sqrt(gamma)*A*x + z
# solve for x given y,A
N = 2000 # length of signal
M = 500 # number of measurements
delta = M/N # measurement rate
gamma=20 # SNR


# signal parameters
var_x=1
epsilon = 0.05 # probability of nonzero signal

# AMP parameters
max_iter = 40 # number of AMP iterations
lamda = 0.3 # damping parameter can't use 'lambda' since that is a python keyword


# generate signal
# f_X(x) = epsilon*N(0,1)+(1-epsilon)*delta(x)
x = (np.random.rand(N,1)<epsilon)*np.random.randn(N,1)

# matrix
A = 1/np.sqrt(M)*np.random.randn(M,N) # unit norm columns
print(A)
Atrans = np.transpose(A)# transpose of A


# Mark each data value and customize the linestyle:
plt.plot(x[1:200], marker = "o", linestyle = "--")
plt.show()


# measurements
# y = np.sqrt(gamma)*A*x + z
y = np.sqrt(gamma)*np.dot(A,x) + np.random.randn(M,1)# measurements

# normalize differently with gamma
y=y/np.sqrt(gamma)

# AMP algorithm
# initialization
mse = np.zeros((max_iter,1)) # store mean square error
xt = np.zeros((N,1))# estimate of signal
dt = np.zeros((N,1))# derivative of denoiser
rt = np.zeros((M,1))# residual

for iter in range(0,max_iter):
    # update residual
    rt = y - np.dot(A,xt) + 1/delta*np.mean(dt)*rt
    # compute pseudo-data
    vt = xt + np.dot(Atrans,rt)
    # estimate scalar channel noise variance estimator is due to Montanari
    var_t = np.mean(rt**2)
    # denoising
    xt1, dt = denoise(vt,var_x,var_t,epsilon)
    # damping step
    xt = lamda*xt1 + (1-lamda)*xt
    mse[iter] = np.mean((xt-x)**2)

## plot result
#figure
plt.plot(mse,'o-')
plt.xlabel('Iteration')
plt.ylabel('MSE')
print('AMP error = {}\n'.format(min(mse)))

plt.plot(xt1[1:200], marker = "o", linestyle = "--")
plt.show()

# Plot with Varying MSE vs SNR values
mse_snr = [] #np.zeros((max_iter,1)) # store mean square error
xt_1 = np.zeros((N,1))# estimate of signal
dt_1 = np.zeros((N,1))# derivative of denoiser
rt_1 = np.zeros((M,1))# residual

SNR=[10,20,30,40,50,60,70,80,90,100]

for snr in SNR:
    y = np.sqrt(snr)*np.dot(A,x) + np.random.randn(M,1)# measurements

    # normalize differently with gamma
    y=y/np.sqrt(snr)
    # update residual
    rt_1 = y - np.dot(A,xt_1) + 1/delta*np.mean(dt_1)*rt_1
    # compute pseudo-data
    vt = xt_1 + np.dot(Atrans,rt_1)
    # estimate scalar channel noise variance estimator is due to Montanari
    var_t = np.mean(rt_1**2)
    # denoising
    xt1,dt_1 = denoise(vt,var_x,var_t,epsilon)
    # damping step
    xt_1=lamda*xt1+(1-lamda)*xt_1
    mse_snr.append(np.mean((xt_1-x)**2))
    #np.append(mse_snr, np.mean((xt_1-x)**2) )

## plot result
#figure
plt.plot(SNR, mse_snr,'o-')
plt.xlabel('SNR')
plt.ylabel('MSE')
print('AMP error = {}\n'.format(min(mse_snr)))








