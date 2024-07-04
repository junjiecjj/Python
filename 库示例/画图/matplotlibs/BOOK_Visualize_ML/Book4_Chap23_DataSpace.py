

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
    plt.title('=')
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


X = np.array([[1, -1],[-np.sqrt(3), np.sqrt(3)], [2, -2]])
X, U, S, V = visualize_svd(X,'$X$','$U$','$S$','$V^T$', fig_height=3)

full_matrices = True
U, s, VT = np.linalg.svd(X, full_matrices = full_matrices)
S = np.zeros(X.shape)
np.fill_diagonal(S, s)
# X = U@S@VT
print(f"X = \n{X}")
print(f"U = \n{U}")
print(f"S = \n{S}")
print(f"VT = \n{VT}")

## U的第一列独立张成列空间 C(X)。顺藤摸瓜,有意思的是 SVD 分解中,我们顺路还得到了 u2 和 u3,它俩张起了左零空间Null(XT)。规范正交基 [u1, u 2, u3] 则是张成R^3无数规范正交基中的一个。
print(X.T@U)
# array([[-2.82842712e+00,  3.33066907e-16,  0.00000000e+00],
#        [ 2.82842712e+00, -3.33066907e-16,  0.00000000e+00]])

## V的第一列独立张成列空间 R(X),。顺藤摸瓜,有意思的是 SVD 分解中,我们顺路还得到了 V2, 它张起了左零空间Null(X)。规范正交基 [v1, v2] 则是张成R^2无数规范正交基中的一个。
print(X@V)
# array([[-1.41400000e+00,  0.00000000e+00],
#        [ 2.44911984e+00,  3.63959394e-17],
#        [-2.82800000e+00,  0.00000000e+00]])



# 计算协方差矩阵
cov_x = np.cov(X)
cov_xt = np.cov(X.T)

# 计算相关性系数矩阵
corr_x = np.corrcoef(X)
corr_xt = np.corrcoef(X.T)



























































































































































































































































































