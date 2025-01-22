


# AMP for signal recovery in compressed sensing.
# CS model: y = Ax + noise
# Goal    : recover x given y and A
#
# Copyright (c) 2021 Kuan Hsieh

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.fftpack import dct, idct # DCT
import scipy
# Need to fix the partial fourier matrix thing.
# Shouldn't sample first row. And also can do DCT matrix,
# shouldn't sample row 1 and row N/2.

##### Compressed Sensing #####
def initialise_CS(N, M, K, sigma = 0, x_choice = 0, A_choice = 0):
    ''' Initialises compressed sensing (CS) problem, y = Ax + noise.
    Inputs:
      N: Signal dimension
      M: Number of measurements
      K: Sparsity, number of non-zero entries in signal vector
      sigma: noise standard deviation
      A_choice: type of sensing matrix
          0: With iid normal entries N(0,1/M)
          1: With iid entries in {+1/sqrt(M), -1/sqrt(M)} with uniform probability
          2: Random partial Fourier matrix
      x_choice: type of signal
          0: Elements in {+1, 0, -1}, with K +-1's with uniform probability of +1 and -1's
          1: K non-zero entries drawn from the standard normal distribution
    Outputs:
        y: measurment vector
        A: sensing matrix
        x: signal vector
    '''
    # Generate signal x
    if x_choice == 0:
        x = 1*(np.random.rand(K)<0.5)
        x[np.logical_not(x)] = -1
        x = np.concatenate((x, np.zeros((N-K))))
    elif x_choice == 1:
        x = np.random.randn(K)
        x = np.concatenate((x, np.zeros((N-K))))
    else:
        print('x_choice must be either 0 or 1')
    np.random.shuffle(x)

    # Generate sensing matrix A
    if A_choice == 0:
        A = np.random.randn(M,N)/np.sqrt(M)
    elif A_choice == 1:
        A = np.random.rand(M,N)
        A = (A<0.5)/np.sqrt(M)
        A[np.logical_not(A)] = -1/np.sqrt(M)
    elif A_choice == 2:
        A = scipy.fftpack.fft(np.eye(N)) # N-by-N DFT matrix
        rows = np.random.choice(N,size=M,replace=False) # Without replacement, include first row
        A = A[rows,:]/np.sqrt(M) # Normalize so that L2 norm of each column = 1
    else:
        # With odd number of rows cannot normaize columns to have unit norm
        print('M must be an even number for random partial Fourier sensing matrix ')
    # print(LA.norm(A,axis=0)) # Checks the norm of the columns are 1
    # Calculate measurement vector y
    y = np.dot(A,x)
    if sigma != 0:
        y += sigma * np.random.randn(M)
    return y, A, x

##### AMP #####
### AMP with soft-thresholding denoiser
def plus_op(x):
    '''Plus operator.
    plus_op(x) = x if x>0, = 0 otherwise'''
    return x*(x>0)

def soft_thresh(x, L):
    '''Soft-thresholding function.

    x is the signal, L is the threshold lambda
    x = sign(x)(abs(x)-lambda)_+ ()_+ is the element-wise plus operator which equals the +ve part of x if x>0, otherwise = 0.'''
    return np.multiply(np.sign(x), plus_op(np.absolute(x)-L) )

def opt_tuning_param(eps, limit=3):
    '''Find optimal tuning parameter for given sparsity ratio (epsilon) for the AMP algorithm.
    Equation on p8 of "Graphical Models Concepts in Compressed Sensing"
    Inputs
        eps: epsilon = K/N = delta*rho
        limit: Limits the search for optimal tuning parameter alpha
    '''
    res   = minimize_scalar(M_pm, bracket=(0, limit), args = (eps))
    alpha = res.x
    return alpha

def M_pm(alpha, eps):
    return eps*(1+alpha**2) + (1-eps)*(2*(1+alpha**2)*norm.cdf(-alpha)-2*alpha*norm.pdf(alpha))

def amp(y, A, x, z, alpha):
    '''Approximate message passing (AMP) iteration with soft-thresholding denoiser.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        alpha: threshold tuning parameter
    Outputs
        x: signal estimate
        z: residual
    Note
        Need to initialise AMP iteration with
        x = np.zeros(N)
        z = y
    '''
    M = len(y)

    # Estimate vector
    theta = alpha*np.sqrt(LA.norm(z)**2/M) # alpha*tau
    x     = soft_thresh(x + np.dot(A.T,z), theta)

    # Calculate residual with the Onsager term
    # print(x.shape)
    b = LA.norm(x,0)/M
    z = y - np.dot(A,x) + b*z

    # L = theta*(1 - b) # The last L is the actual lambda of the LASSO we're minimizing

    return (x, z)

### AMP with Bayes-optimal denoiser for different signals
def amp_3pt(y, A, x, z, eps, c = 1):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE) denoiser for signals with iid entries drawn from the 3-point distribution: probability (1-eps) equal to 0 and probability eps/2 equal to each +c and -c.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
        c: the values of the non-zero entries of the signal equal to +c and -c
    Outputs
        x: signal estimate
        z: residual
    Note
        Need to initialise AMP iteration with
        x = np.zeros(N)
        z = y
    '''
    M,N = np.shape(A)
    tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation

    # Estimate vector
    s = x + np.dot(A.T, z) # Effective (noisy) observation of signal x
    u = s*c / tau**2       # Temporary variable
    top = c * eps*np.sinh(u,dtype=np.float128)
    bot = eps*np.cosh(u,dtype=np.float128) + (1-eps)*np.exp(c**2/(2*tau**2),dtype=np.float128)
    x   = (top / bot).astype(np.float128)

    # Calculate residual with the Onsager term
    eta_der = x * (c/np.tanh(u) - x) / tau**2
    b = (N/M) * np.mean(eta_der)
    z = y - np.dot(A,x) + b*z

    return (x, z)

def amp_bg(y, A, x, z, eps, v=1):
    '''Approximate message passing (AMP) iteration with Bayes-optimal (MMSE) denoiser for signals with iid entries drawn from the Bernoulli-Gaussian distribution: probability (1-eps) equal to 0 and probability eps drawn from a Gaussian distribution with standard deviation v.
    Inputs
        y: measurement vector (length M 1d np.array)
        A: sensing matrix     (M-by-N 2d np.array)
        x: signal estimate    (length N 1d np.array)
        z: residual           (length M 1d np.array)
        eps: sparsity ratio (fraction of non-zero entries)
        v: the standard deviation of the non-zero entries of the signal which are drawn from a Gaussian distribution
    Outputs
        x: signal estimate
        z: residual
    Note
        Need to initialise AMP iteration with
        x = np.zeros(N)
        z = y
    '''
    M, N = np.shape(A)
    tau = np.sqrt(np.mean(z**2)) # Estimate of effective noise std deviation

    # Estimate vector
    s = x + np.dot(A.T, z)   # Effective (noisy) observation of signal x
    u = v**2 / tau**2        # Temporary variable
    term1 = 1 + tau**2/v**2
    term2 = 1 + (1-eps)/eps * np.sqrt(1+u) * np.exp(-(s**2/(2*tau**2))*u/(1+u))
    denom = term1 * term2
    x = s / denom

    # Calculate residual with the Onsager term
    eta_der = (1/denom) + (x/tau**2) * (s/(1+tau**2/v**2) - x)
    b = (N/M) * np.mean(eta_der)
    z = y - np.dot(A,x) + b*z

    return (x, z)

##### State evolution #####
def se(eps, delta, sigma, iter_max, nsamples=10000):
    '''State evolution (SE) for AMP decoder with soft-thresholding denoiser and signal entries drawn from 3-point distribution with probability (1-eps) equal to 0 and probability eps/2 equal to either +1 or -1.
    Inputs
        eps  : sparsity ratio (num of non-zero entries / signal dimension)
        delta: sensing matrix (num of measurements / signal dimension)
        sigma: measurement noise standard deviation
        iter_max: number of iterations to run SE
        nsamples: num of Monte Carlo samples used to evaluate expectation
    Outputs
        psi: SE variable that tracks AMP's per-iteration MSE
        tau: SE variable that tracks per-iteration effective noise variance.
    '''
    alpha = opt_tuning_param(eps) # Soft-threshold threshold parameter
    psi = np.ones(iter_max) * eps # Variance of signal entries
    tau = np.zeros(iter_max)
    for t in range(iter_max-1):
        tau[t] = sigma**2 + psi[t]/delta
        X = np.random.choice([-1,0,1], nsamples, p=[eps/2, 1-eps, eps/2])
        Z = np.random.randn(nsamples) # noise
        S = X + np.sqrt(tau[t]) * Z
        theta    = alpha * np.sqrt(tau[t])
        psi[t+1] = np.mean((soft_thresh(S, theta) - X)**2)
    return psi, tau

##### Subsampled DCT transform #####
def sub_dct(m, n, seed=0, order0=None, order1=None):
    """
    Returns functions to compute the sub-sampled Discrete Cosine Transform, i.e., matrix-vector multiply with subsampled rows from the DCT matrix.

    This is a direct modification of Adam Greig's pyfht source code which can be found at https://github.com/adamgreig/pyfht/blob/master/pyfht.py

    [Inputs]
    m: number of rows
    n: number of columns
    m < n
    Most efficient (but not required) for max(m+1,n+1) to be a power of 2.
    seed:   determines choice of random matrix
    order0: optional m-long array of row indices in [1, max(m+1,n+1)] to implement subsampling of rows; generated by seed if not specified.
    order1: optional n-long array of row indices in [1, max(m+1,n+1)] to implement subsampling of columns; generated by seed if not specified.
    [Outputs]
        Ax(x):    computes A.x (of length m), with x having length n
        Ay(y):    computes A*.y (of length n), with y having length m
    [Notes]
        There is a scaling of 1/sqrt(m) in the outputs of Ax() and Ay().
    """

    assert type(m)==int and m>0
    assert type(n)==int and n>0
    w = 2**int(np.ceil(np.log2(max(m+1,n+1))))

    if order0 is not None and order1 is not None:
        assert order0.shape == (m,)
        assert order1.shape == (n,)
    else:
        rng = np.random.RandomState(seed)
        idxs0 = np.arange(1, w, dtype=np.uint32)
        idxs1 = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs0)
        rng.shuffle(idxs1)
        order0 = idxs0[:m]
        order1 = idxs1[:n]

    def Ax(x):
        assert x.size == n, "x must be n long"
        x_ext = np.zeros(w)
        x_ext[order1] = x.reshape(n)
        y = np.sqrt(w)*dct(x_ext, norm='ortho')
        return y[order0]/np.sqrt(m)

    def Ay(y):
        assert y.size == m, "input must be m long"
        y_ext = np.zeros(w)
        y_ext[order0] = y
        x_ext = np.sqrt(w)*idct(y_ext, norm='ortho')
        return x_ext[order1]/np.sqrt(m)

    return Ax, Ay








































