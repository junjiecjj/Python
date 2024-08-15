


# Bk4_Ch11_01.py

import numpy as np
import seaborn as sns
from scipy.linalg import lu
from matplotlib import pyplot as plt

A = np.array([[ 5,  2, -2,  3],
              [-2,  5, -8,  7],
              [ 7, -5,  1, -6],
              [-5,  4, -4,  8]])

P,L,U = lu(A, permute_l = False)
# P, permutation matrix
# L, lower triangular with unit diagonal elements
# U, upper triangular
# Default do not perform the multiplication P*L

fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('A')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(P,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('P')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(L,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('L')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(U,cmap='RdBu_r', vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('U')





# Bk4_Ch11_02.py

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# generate original data matrix, X
np.random.default_rng()
X = np.random.randn(9, 5)

#%% QR decomposition， complete version

Q_complete, R_complete = np.linalg.qr(X, mode = 'complete')

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('X')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Rc')

# properties of Q (reduced)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(Q_complete.T@Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc.T@Qc')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q_complete.T,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc.T')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc')

#%% QR decomposition， reduced version

Q, R = np.linalg.qr(X)
# default: reduced

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = 2.5,vmin = -2.5, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('X')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('R')

# properties of Q (reduced)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(Q.T@Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q.T@Q')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q.T,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q.T')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q')

































































































































































































































































































