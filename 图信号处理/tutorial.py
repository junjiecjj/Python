#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:15:15 2025

@author: jack

pip install pygsp


"""

#%%
from pygsp import graphs, filters
G = graphs.Logo()
G.estimate_lmax()
g = filters.Heat(G, tau=100)



import numpy as np
DELTAS = [20, 30, 1090]
s = np.zeros(G.N)
s[DELTAS] = 1
s = g.filter(s)
G.plot_signal(s, highlight=DELTAS, backend='matplotlib')







#%% Introduction to the PyGSP
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (10, 5)
### Graphs

rs = np.random.RandomState(42)  # Reproducible results.
W = rs.uniform(size=(30, 30))  # Full graph.
W[W < 0.93] = 0  # Sparse graph.
W = W + W.T  # Symmetric graph.
np.fill_diagonal(W, 0)  # No self-loops.
G = graphs.Graph(W)
print('{} nodes, {} edges'.format(G.N, G.Ne))

G.is_connected()
# True
G.is_directed()
# False

(G.W == W).all()
# True
type(G.W)
# <class 'scipy.sparse.lil.lil_matrix'>


G.L.shape
(30, 30)

G.compute_fourier_basis()
G.U.shape
(30, 30)

G.compute_differential_operator()
G.D.shape
(60, 30)

G.set_coordinates('ring2D')
G.plot()

# Fourier basis
G = graphs.Logo()
G.compute_fourier_basis()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for i, ax in enumerate(axes):
    G.plot_signal(G.U[:, i+1], vertex_size=30, ax=ax)
    _ = ax.set_title('Eigenvector {}'.format(i+2))
    ax.set_axis_off()
fig.tight_layout()



G2 = graphs.Ring(N=50)
G2.compute_fourier_basis()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
G2.plot_signal(G2.U[:, 4], ax=axes[0])
G2.set_coordinates('line1D')
G2.plot_signal(G2.U[:, 1:4], ax=axes[1])
fig.tight_layout()


# Filters
tau = 1
def g(x):
    return 1. / (1. + tau * x)
g = filters.Filter(G, g)

fig, ax = plt.subplots()
g.plot(plot_eigenvalues=True, ax=ax)
_ = ax.set_title('Filter frequency response')

# Graph signal: each letter gets a different value + additive noise.
s = np.zeros(G.N)
s[G.info['idx_g']-1] = -1
s[G.info['idx_s']-1] = 0
s[G.info['idx_p']-1] = 1
s += rs.uniform(-0.5, 0.5, size=G.N)

s2 = g.filter(s)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
G.plot_signal(s, vertex_size=30, ax=axes[0])
_ = axes[0].set_title('Noisy signal')
axes[0].set_axis_off()
G.plot_signal(s2, vertex_size=30, ax=axes[1])
_ = axes[1].set_title('Cleaned signal')
axes[1].set_axis_off()
fig.tight_layout()

#%% Introduction to spectral graph wavelets
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils

G = graphs.Bunny()

##>>>>>>> Simple filtering: heat diffusion
taus = [10, 25, 50]
g = filters.Heat(G, taus)

s = np.zeros(G.N)
DELTA = 20
s[DELTA] = 1

s = g.filter(s, method='chebyshev')

fig = plt.figure(figsize=(10, 3))
for i in range(g.Nf):
    ax = fig.add_subplot(1, g.Nf, i+1, projection='3d')
    G.plot_signal(s[:, i], colorbar=False, ax=ax)
    title = r'Heat diffusion, $\tau={}$'.format(taus[i])
    ax.set_axis_off()
fig.tight_layout()



##>>>>>>> Visualizing wavelets atoms
g = filters.MexicanHat(G, Nf=6)  # Nf = 6 filters in the filter bank.
fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of mexican hat wavelets')

s = g.localize(DELTA)

fig = plt.figure(figsize=(10, 2.5))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    _ = ax.set_title('Wavelet {}'.format(i+1))
    ax.set_axis_off()
fig.tight_layout()

s = G.coords
s = g.filter(s)
s = np.linalg.norm(s, ord=2, axis=1)

fig = plt.figure(figsize=(10, 7))
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    title = 'Curvature estimation (scale {})'.format(i+1)
    _ = ax.set_title(title)
    ax.set_axis_off()
fig.tight_layout()



#%% Optimization problems: graph TV vs. Tikhonov regularization

import numpy as np
from pygsp import graphs, plotting

# Create a random sensor graph
G = graphs.Sensor(N=256, distribute=True, seed=42)
G.compute_fourier_basis()

# Create label signal
label_signal = np.copysign(np.ones(G.N), G.U[:, 3])

G.plot_signal(label_signal)

rs = np.random.RandomState(42)

# Create the mask
M = rs.rand(G.N)
M = (M > 0.6).astype(float)  # Probability of having no label on a vertex.

# Applying the mask to the data
sigma = 0.1
subsampled_noisy_label_signal = M * (label_signal + sigma * rs.standard_normal(G.N))

G.plot_signal(subsampled_noisy_label_signal)


import pyunlocbox

# Set the functions in the problem
gamma = 3.0
d = pyunlocbox.functions.dummy()
r = pyunlocbox.functions.norm_l1()
f = pyunlocbox.functions.norm_l2(w=M, y=subsampled_noisy_label_signal, lambda_=gamma)

# Define the solver
G.compute_differential_operator()
L = G.D.toarray()
step = 0.999 / (1 + np.linalg.norm(L))
solver = pyunlocbox.solvers.mlfbf(L=L, step=step)
# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob1 = pyunlocbox.solvers.solve([d, r, f], solver=solver, x0=x0, rtol=0, maxit=1000)
# Solution found after 1000 iterations:
#     objective function f(sol) = 2.024594e+02
#     stopping criterion: MAXIT
# >>>
G.plot_signal(prob1['sol'])




# Set the functions in the problem
r = pyunlocbox.functions.norm_l2(A=L, tight=False)

# Define the solver
step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma * np.diag(M), 2)
solver = pyunlocbox.solvers.gradient_descent(step=step)

# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob2 = pyunlocbox.solvers.solve([r, f], solver=solver, x0=x0, rtol=0, maxit=1000)
# Solution found after 1000 iterations:
#     objective function f(sol) = 9.555135e+01
#     stopping criterion: MAXIT

G.plot_signal(prob2['sol'])







#%%



















#%%









































