#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:59:06 2024

@author: jack

https://github.com/shx-lyu/AMP-in-MIMO

https://github.com/parthe/VAMP

https://github.com/sphinxteam/tramp

https://github.com/GAMPTeam/vampyre

https://github.com/shuai-huang/1Bit-CS/tree/master


https://mp.weixin.qq.com/s?__biz=MzI1NzU4ODgwMg==&mid=2247493413&idx=1&sn=e5684fdc3d3702044fab049221337f0b&chksm=eb8009311c337d5b56413cf16aceb96182bb24c7d0256d5f4412003dd2207d5d3468f67f3b76&mpshare=1&scene=1&srcid=1114TM1nqlR3cxR5gvrMqjYQ&sharer_shareinfo=809873a389d090a424e6dce2f28e2131&sharer_shareinfo_first=809873a389d090a424e6dce2f28e2131&exportkey=n_ChQIAhIQV%2FTXtsIwO6SqTQPWEI3NXRKfAgIE97dBBAEAAAAAAD7KBWQ4Fw0AAAAOpnltbLcz9gKNyK89dVj0Ki86ZeXSFQhbzFzmgFEC03uh0mFCHNWy%2FZWd8NbjACcev%2BWC8GsJIBnW%2FQl%2F5WQ11klrG%2F4Mec6qgaSmyZfVEZJrc0m5UygF4u%2Fen16Qshe%2BjyaBG5BefNxO7iMQGC0G7FwmYIayr1zS3ESiv54A%2B%2FOse9GN6mfJTFmctyjA%2F%2FXxsrCEFTQUKOVA4nyV8XZuyvIYvsnXfIytCqkdfduVy0eNDD0xNwsVf3MIBwn5y8gJfdmHQ4vQA%2B8r5dsh9pV2lE3cGOHnhpm5bZIHxeoBAf%2F%2Bd9cfhx1qnrvzAjdYe19NjD6eQ665D0hb%2FFtL%2BkBt8wkoNE%2BnQvK7&acctmode=0&pass_ticket=t%2F1O97V7GS%2FQyoqphF16TxoP7Y24Vxbbv%2BSH8vRBlNOlxKaai1rztcMj1k5fhrsY&wx_header=0#rd

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as op
from scipy.stats import norm
from sklearn.linear_model import OrthogonalMatchingPursuit

def denoise(v, var_x, var_z, epsilon):
    term1 = (1-epsilon)*norm.pdf(v, 0, np.sqrt(var_z))
    term2 = epsilon*norm.pdf(v, 0, np.sqrt(var_x+var_z))
    xW = var_x / (var_x + var_z)*v # Wiener filter
    added_term = term1 + term2
    div_term = np.divide(term2, added_term)
    xhat = np.multiply(div_term, xW) # denoised version, x(t+1)

    # empirical derivative
    Delta = 0.0000000001 # perturbation
    term1_d = (1-epsilon) * norm.pdf((v+Delta), 0, np.sqrt(var_z))
    term2_d = epsilon * norm.pdf((v + Delta), 0, np.sqrt(var_x + var_z))
    xW2 = var_x / (var_x + var_z)*(v + Delta) # Wiener filter

    added_term = term1_d + term2_d
    mul_term = np.multiply(xW2, term2_d)
    xhat2 = np.divide(mul_term, added_term)
    d = (xhat2 - xhat)/Delta
    return xhat, d

# AMP parameters
max_iter = 40 # number of AMP iterations
lamda = 0.3 # damping parameter can't use 'lambda' since that is a python keyword
var_x = 1
## AMP algorithm
def AMPforCS(A, y, real_x, max_iter = max_iter, lamda = lamda, var_x = var_x, epsilon = 0.2 ):
    M, N = A.shape
    delta = M/N          # measurement rate
    # initialization
    mse = np.zeros((max_iter,1)) # store mean square error
    xt = np.zeros((N,1))# estimate of signal
    dt = np.zeros((N,1))# derivative of denoiser
    rt = np.zeros((M,1))# residual
    for iter in range(0, max_iter):
        # update residual
        rt = y - A @ xt + 1 / delta * np.mean(dt) * rt
        # compute pseudo-data
        vt = xt + A.T @ rt
        # estimate scalar channel noise variance estimator is due to Montanari
        var_t = np.mean(rt**2)
        # denoising
        xt1, dt = denoise(vt, var_x, var_t, epsilon)
        # damping step
        xt = lamda*xt1 + (1-lamda)*xt
        mse[iter] = np.mean((xt - real_x)**2)
    return xt , mse

def OMP1(phi, y, sparsity):
    """
    OMP算法的Python实现
        参数：
        A: 测量矩阵，形状为(m, n)
        y: 观测向量，形状为(m,)
        k: 稀疏度，即信号的非零元素个数
    返回： x: 重构的稀疏信号，形状为(n, 1)
    """
    N = phi.shape[1]
    y = y.flatten()
    residual = y.copy()
    index_set = []
    theta = np.zeros(N)

    for _ in range(sparsity):
        correlations = phi.T @ residual
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)
        phi_selected = phi[:, index_set]
        theta_selected, _, _, _ = np.linalg.lstsq(phi_selected, y, rcond=None)
        for i, idx in enumerate(index_set):
            theta[idx] = theta_selected[i]
        residual = y - phi @ theta
        if np.linalg.norm(residual) < 1e-10:
            break
    return theta


# y = np.sqrt(snr)*A*x + z
# solve for x given y,A
snr = 90 # SNR
N = 2000 # length of signal
M = 1000 # number of measurements
K = 10

# matrix
A = 1/np.sqrt(M)*np.random.randn(M, N) # unit norm columns

x = np.zeros((N, 1))
x[np.random.choice(N, K, replace=False)] = np.random.randn(K, 1)
epsilon = K/M # probability of nonzero signal

# signal parameters
# generate signal
# f_X(x) = epsilon*N(0,1)+(1-epsilon)*delta(x)
# epsilon = 0.1 # probability of nonzero signal
# x = (np.random.rand(N,1) < epsilon)*np.random.randn(N,1)

# measurements
# y = np.sqrt(snr)*A*x + z
y = np.dot(A, x) + np.random.randn(M, 1)/np.sqrt(snr)# measurements
# normalize differently with snr
# y = y/np.sqrt(snr)

###### AMP
xt_AMP, mse = AMPforCS(A, y, x, epsilon = epsilon)
# print('AMP error = {}\n'.format(min(mse)))

###### OMP
xt_OMP = OMP1(A, y, sparsity = K)

###### OMP
omp = OrthogonalMatchingPursuit(n_nonzero_coefs = K)
omp.fit(A, y)
reconstructed_signal = omp.coef_

##>>>>>>>>>>>  AMP plot result
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(mse,'o-')
axs.set_xlabel('Iteration')
axs.set_ylabel('MSE')
plt.show()
plt.close()

##>>>>>>>>>>>  Origin signal
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(x, ls = '--', color = 'b', marker = "o",  )
axs.set_title('Original Signal')
plt.show()
plt.close()

##>>>>>>>>>>>  AMP signal
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(xt_AMP, ls = '--', color = 'orange', marker = "o", )
axs.set_title(f'AMP, recover Signal in {snr}')
plt.show()
plt.close()

##>>>>>>>>>>>  OrthogonalMatchingPursuit signal
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(reconstructed_signal, ls = '--', color = 'orange', marker = "o", )
axs.set_title(f'Scipy OMP, recover Signal in {snr}')
plt.show()
plt.close()

##>>>>>>>>>>>  OMP signal
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
axs.plot(xt_OMP, ls = '--', color = 'orange', marker = "o", )
axs.set_title(f'OMP, recover Signal in {snr}')
plt.show()
plt.close()


#%%
# Plot with Varying MSE vs SNR values
mse_snr = [] #np.zeros((max_iter,1)) # store mean square error
SNR = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for snr in SNR:
    y = np.dot(A,x) + np.random.randn(M,1)/np.sqrt(snr) # measurements
    x_amp, mse = AMPforCS(A, y, x, epsilon = epsilon)
    mse_snr.append(np.mean((x_amp - x)**2))

    ## plot result
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
    axs.plot(mse,'o-')
    axs.set_xlabel('Iteration')
    axs.set_ylabel('MSE')
    axs.set_title(f'snr = {snr}')
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
    axs.plot(x_amp, ls = '--', color = 'orange', marker = "o", )
    axs.set_title(f'recover under snr = {snr}')
    plt.show()
    plt.close()

print('AMP error = {}\n'.format(min(mse_snr)))
## plot result
#figure
fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
plt.plot(SNR, mse_snr,'o-')
axs.set_xlabel('SNR')
axs.set_ylabel('MSE')
plt.show()
plt.close()



























































































































