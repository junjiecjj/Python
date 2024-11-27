# Algorithms for signal recovery in compressed sensing.
# CS model: y = Ax + noise
# Goal    : recover x given y and A
#
# Copyright (c) 2021 Kuan Hsieh

import numpy as np
from numpy import linalg as LA
from amp4cs import soft_thresh

def ist(y, A, x, z, alpha):
    '''Iterative soft-thresholding (IST) iteration.
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
        Need to initialise IST iteration with
        x = np.zeros(N)
        z = y
    '''
    M = len(y)
    # Estimate vector
    theta = alpha*np.sqrt(LA.norm(z)**2/M) # alpha*tau
    x     = soft_thresh(x + np.dot(A.T,z), theta)

    # Calculate residual
    z = y - np.dot(A,x)

    return (x, z)

def prox_grad(y, A, x, t, L):
    '''Proximal gradient descent iteration.
    See Section 5.3.3 of "Statistical Learning with Sparsity" by
    Hastie, Tibshirani and Wainwright for more details.
    Inputs
        y: measurement vector
        A: sensing matrix
        x: signal estimate
        t: step size
        L: lambda, soft-threshold threshold
    Outputs:
        x: signal estimate
    '''

    # Step 1: take gradient step
    z = x + t*np.dot(A.T, y - np.dot(A,x))

    # Step 2: perform element-wise soft-thresholding
    x = soft_thresh(z, t*L)

    return x

def nesterov(y, A, x, t, L, it, theta):
    '''Proximal gradient descent with Nesterov momentum iteration.
    See Section 5.3.4 of "Statistical Learning with Sparsity" by
    Hastie, Tibshirani and Wainwright for more details.
    Inputs
        y: measurement vector
        A: sensing matrix
        x: signal estimate
        t: step size
        L: lambda, soft-threshold threshold
        it: the current iteration number
        theta: extrapolated signal estimate calculated at the previous iteration
    Outputs:
        x: signal estimate
        theta: extrapolated signal estimate
    '''

    x_prev = x

    # Step 1: take gradient step
    z = theta + t*np.dot(A.T, y - np.dot(A,theta))

    # Step 2: perform element-wise soft-thresholding
    x = soft_thresh(z, t*L)

    # Step 3: update theta
    theta = x + it*(x-x_prev)/(it+3) # with Nesterov momentum

    return (x, theta)

def omp(y, A, x, z, Omega):
    '''Orthogonal matching pursuit iteration.
    Inputs
        y: measurement vector
        A: sensing matrix
        x: signal estimate
        z: residual vector
        Omega: index selections
    Outputs
        x: updated signal estimate
        z: updated residual vector
        Omega: updated index selections
    '''

    # Select
    # Find index with max inner product with columns of A
    n_l = np.argmax(abs(np.dot(A.T, z)) / LA.norm(A, axis=0))
    Omega.append(n_l)

    # Orthogonalize
    A_Omega         = A[:,Omega]
    coef_vals,_,_,_ = LA.lstsq(A_Omega, y, rcond=None) # Able to parallelize using LA.lstsq?
    x[Omega]        = coef_vals

    # Update residual
    z = y - np.dot(A_Omega, coef_vals)

    return x, z, Omega

def cosamp(Phi, u, s, a, v):
    '''CoSaMP algorithm iteration.
    See "CoSaMP: Iterative Signal Recovery from Incomplete and Inaccurate
    Samples" by Needell and Tropp for more details. Notation follows paper.
    Inputs
        u  : measurement vector
        Phi: sensing matrix
        s  : estimated sparsity
        a  : signal estimate
        v  : residual vector
    Outputs
        a: updated signal estimate
        v: updated residual vector
    '''

    y     = np.dot(Phi.T, v)          # Form signal proxy
    Omega = np.argsort(abs(y))[-2*s:] # Identify large components
    T     = np.union1d(Omega, np.nonzero(a)[0]) # Merge supports

    b = np.zeros_like(a)
    b[T],_,_,_ = LA.lstsq(Phi[:,T], u, rcond=None) # Might have to transpose the Phi after slicing...

    b[np.argsort(abs(b))[:-s]] = 0 # Prune to obtain next approximation
    a = np.copy(b)

    v = u - np.dot(Phi, a) # Update current samples

    return a, v
































