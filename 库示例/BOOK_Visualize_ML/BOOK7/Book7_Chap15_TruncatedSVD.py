





# 截断型SVD分解，照片


import matplotlib.pyplot as plt
import numpy as np
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


from skimage import color
from skimage import io



X = color.rgb2gray(io.imread('iris_photo.jpg'))

X.min()

X.max()

X.shape




np.linalg.matrix_rank(X)


# Run SVD on Image
U, S, V = np.linalg.svd(X)

n_components = len(S)
component_idx = range(1,  n_components + 1)

lambda_i = np.square(S)/(X.shape[0] - 1)
# approximation, given that X is not centered


# Visualizations
fig, axs = plt.subplots()

## Raw Image, X （down-sampled)
plt.imshow(X, cmap='gray')

## Singular values
fig, ax = plt.subplots()

### Raw singular values
plt.plot(component_idx, S)
plt.grid()
ax.set_xscale('log')
plt.xlabel("Principal component")
plt.ylabel('Singular value')

## Eigen value
fig, ax = plt.subplots()

### Raw singular values
plt.plot(component_idx, lambda_i)
plt.grid()
ax.set_xscale('log')
plt.xlabel("Principal component")
plt.ylabel('Eigen value')

# Calculate the cumulative variance explained
variance_explained = 100 * np.cumsum(lambda_i) / lambda_i.sum()


fig, ax = plt.subplots()
plt.plot(component_idx, variance_explained)
ax.set_xscale('log')
plt.xlabel("Principal component")
plt.grid()
plt.ylabel('Cumulative variance explained (%)')



#%% Image Reconstruction

# Reconstruct image with increasing number of singular vectors/values
for rank in [1, 2, 4, 8, 16, 32, 64]:

    # Reconstructed Image
    X_reconstruction = U[:, :rank] * S[:rank] @ V[:rank,:]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(X_reconstruction, cmap='gray')
    axs[0].set_title('X_reproduced with ' + str(rank) + ' PCs')

    ## Reconstruction error

    axs[1].imshow(X - X_reconstruction, cmap='gray')
    axs[1].set_title('Error')

# U[:, order].shape




# 秩一矩阵
for order in np.arange(0,16):

    # Reconstructed Image
    X_rank_1 = S[order] * U[:, [order]] @ V[[order],:]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(X_rank_1, cmap='gray')
    title = 'Rank-1 matrix, order = ' + str(order)
    ax.set_title(title)
    plt.savefig(title + '.svg')

rank = 1
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=16)

# 降维后的结果
X_reduced = svd.fit_transform(X)
print(X_reduced.shape)
# 结果为(2990, 16)

# 反变换，获取近似数据
X_approx = svd.inverse_transform(X_reduced)

print(X_approx.shape)
# 结果为(2990, 2714)

print(np.linalg.matrix_rank(X_approx))
# 结果为16


# 可视化
fig, axs = plt.subplots()

plt.imshow(X_approx, cmap='gray')







































































































































































































































































































































