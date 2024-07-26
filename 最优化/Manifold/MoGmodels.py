#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:57:25 2024

@author: jack
"""


import autograd.numpy as np


np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


# %matplotlib inline

# Number of data points
N = 1000

# Dimension of each data point
D = 2

# Number of clusters
K = 3

pi = [0.1, 0.6, 0.3]
mu = [np.array([-4, 1]), np.array([0, 0]), np.array([2, -1])]
Sigma = [
    np.array([[3, 0], [0, 1]]),
    np.array([[1, 1.0], [1, 3]]),
    0.5 * np.eye(2),
]

components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
# For each component, generate all needed samples
for k in range(K):
    # indices of current component in X
    indices = k == components
    # number of those occurrences
    n_k = indices.sum()
    if n_k > 0:
        samples[indices, :] = np.random.multivariate_normal( mu[k], Sigma[k], n_k )

colors = ["r", "g", "b", "c", "m"]
for k in range(K):
    indices = k == components
    plt.scatter( samples[indices, 0], samples[indices, 1], alpha=0.4, color=colors[k % K], )
plt.axis("equal")
plt.show()


import sys
sys.path.insert(0, "../..")

from autograd.scipy.special import logsumexp

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Product, SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent


# (1) Instantiate the manifold
manifold = Product([SymmetricPositiveDefinite(D + 1, k=K), Euclidean(K - 1)])

# (2) Define cost function
# The parameters must be contained in a list theta.
@pymanopt.function.autograd(manifold)
def cost(S, v):
    # Unpack parameters
    nu = np.append(v, 0)

    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0)

    # Calculate log_q
    y = np.expand_dims(y, 0)

    # 'Probability' of y belonging to each cluster
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)
    alpha = np.expand_dims(alpha, 1)

    loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)
    return -np.sum(loglikvec)


problem = Problem(manifold, cost)

# (3) Instantiate a Pymanopt optimizer
optimizer = SteepestDescent(verbosity=1)

# let Pymanopt do the rest
Xopt = optimizer.run(problem).point

mu1hat = Xopt[0][0][0:2, 2:3]
Sigma1hat = Xopt[0][0][:2, :2] - mu1hat @ mu1hat.T
mu2hat = Xopt[0][1][0:2, 2:3]
Sigma2hat = Xopt[0][1][:2, :2] - mu2hat @ mu2hat.T
mu3hat = Xopt[0][2][0:2, 2:3]
Sigma3hat = Xopt[0][2][:2, :2] - mu3hat @ mu3hat.T
pihat = np.exp(np.concatenate([Xopt[1], [0]], axis=0))
pihat = pihat / np.sum(pihat)



print(mu[0])
print(Sigma[0])
print(mu[1])
print(Sigma[1])
print(mu[2])
print(Sigma[2])
print(pi[0])
print(pi[1])
print(pi[2])


print(mu1hat)
print(Sigma1hat)
print(mu2hat)
print(Sigma2hat)
print(mu3hat)
print(Sigma3hat)
print(pihat[0])
print(pihat[1])
print(pihat[2])






