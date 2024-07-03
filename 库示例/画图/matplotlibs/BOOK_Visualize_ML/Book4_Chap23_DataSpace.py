

#%% 23.4 几何视角说空间
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'
import seaborn as sns
PRECISION = 3

def svd(X):
    full_matrices = True
    U, s, Vt = np.linalg.svd(X, full_matrices = full_matrices)

    # Put the vector singular values into a padded matrix
    if full_matrices:
        S = np.zeros(X.shape)
        np.fill_diagonal(S, s)
    else:
        S = np.diag(s)
    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)

def visualize_svd(X, title_X, title_U, title_S, title_V, fig_height=5):
    # Run SVD, as defined above
    U, S, V = svd(X)
    all_ = np.r_[X.flatten(order='C'), U.flatten(order='C'), S.flatten(order='C'), V.flatten(order='C')]

    # all_max = max(all_.max(),all_.min())
    # all_min = -max(all_.max(),all_.min())
    all_max = 6
    all_min = -6
    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(12, fig_height))

    plt.sca(axs[0])
    ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(U, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_U)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(S, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_S)

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(V.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_V)
    return X, U, S, V


X = np.array([[1, -1],[-np.sqrt(3), 3], [2, -1]])

full_matrices = True
U, S, VT = np.linalg.svd(X, full_matrices = full_matrices)
S = np.diag(S)
# X = U@S@VT
print(f"X = \n{X}")
print(f"U = \n{U}")
print(f"S = \n{S}")
print(f"VT = \n{VT}")
X, U, S, V = visualize_svd(X,'$X$','$U$','$S$','$V^T$', fig_height=3)


































































































































































































































































































