#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:00:43 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488019&idx=1&sn=23903ce599082f1fa29c8a7644cd39b4&chksm=c106c7917a44e4e18934f76519027524d9c2c291f5df514e49a2b692f2667d4260d71ddaf3cb&mpshare=1&scene=1&srcid=0104fS5QKBqt1fxgfRezvXhn&sharer_shareinfo=50eaf452ab5c7c1ec727c58b5a367941&sharer_shareinfo_first=50eaf452ab5c7c1ec727c58b5a367941&exportkey=n_ChQIAhIQ7ixRZ3uXs64INTSGX5ZfaBKfAgIE97dBBAEAAAAAAIjgANjcEdAAAAAOpnltbLcz9gKNyK89dVj0IrIqS5D5BrOfWpqyiS63Pm9njjcH4l%2FommmDhIXe3kTGc76Pm3Ovsp4oU2kyqIjS1q1gh65LBClPzZJl7Hxaoc7Ce%2Ftl82LLb3Axs6DFdLTSyjDgtpvVmlt7ho0ep79Gl39JMcEhdAKMjF2P%2FOWN2VCu3Pdtgx951uC1Xo97Szqse0dvSrDNTYOWiLnFxZ36e6oXtt%2FdJilw2dlXEQ8LqMZENjEMX0X%2BNoZxt1o%2BA6db1XBvjeM4Zfz7lKnnRH8WlWFo5%2BWDWzs%2BoLuo27AP8dlCjjXo98POcK%2FwTj3kWGBjfbATPChsPGas9txeMtqOMRLcfR6tanzP&acctmode=0&pass_ticket=A2I%2Fxr1k0Mrwa6Jl7O1epsjZ7FvUYLmAgqs7Pva2A6oae%2BYIYk%2BV%2FbJ%2FjMIruNw9&wx_header=0#rd



"""


#%%>>>>>>>>>>>>>> 1. 主成分分析 (PCA)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成一个三维数据集
mean = [0, 0, 0]
cov = [[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]]  # 协方差矩阵，定义数据的相关性
data = np.random.multivariate_normal(mean, cov, 5000)

# 使用PCA将数据降维到2D
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# 创建一个图形并绘制多个子图
fig = plt.figure(figsize=(16, 8))

# 第一个子图：三维散点图（原始数据）
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o', alpha=0.6)
ax1.set_title('3D Scatter Plot of Original Data', fontsize=15)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

# 第二个子图：二维散点图（PCA降维后的数据）
ax2 = fig.add_subplot(222)
ax2.scatter(data_2d[:, 0], data_2d[:, 1], c='b', marker='o', alpha=0.6)
ax2.set_title('2D Scatter Plot after PCA', fontsize=15)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 第三个子图：解释方差比例的柱状图
explained_variance = pca.explained_variance_ratio_
ax3 = fig.add_subplot(224)
ax3.bar(range(1, len(explained_variance) + 1), explained_variance, color='g', alpha=0.7)
ax3.set_title('Explained Variance Ratio of Principal Components', fontsize=15)
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Variance Explained')

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 2. 线性判别分析 (LDA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 进行LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 绘制降维后的数据分布图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_train_lda[y_train == i, 0], X_train_lda[y_train == i, 1], label=target_name)
plt.title('LDA of IRIS dataset (Training set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.subplot(1, 2, 2)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test_lda[y_test == i, 0], X_test_lda[y_test == i, 1], label=target_name)
plt.title('LDA of IRIS dataset (Test set)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.tight_layout()
plt.show()

# 使用训练好的模型进行预测
y_pred = lda.predict(X_test)

# 计算并绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 打印分类报告
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))



#%%>>>>>>>>>>>>>> 3. 核主成分分析 (KPCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles

# 生成非线性数据
X, y = make_circles(n_samples=800, factor=0.3, noise=0.1, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 核主成分分析
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_scaled)

# 可视化原始数据
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Original Data')

# 可视化核主成分分析后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Kernel PCA Transformed Data')
plt.show()

# 额外图形：核PCA前后的特征分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.hist(X_scaled[:, 0], bins=30, alpha=0.5, label='Feature 1')
ax1.hist(X_scaled[:, 1], bins=30, alpha=0.5, label='Feature 2')
ax1.set_title('Original Features Distribution')
ax1.legend()

ax2.hist(X_kpca[:, 0], bins=30, alpha=0.5, label='Principal Component 1')
ax2.hist(X_kpca[:, 1], bins=30, alpha=0.5, label='Principal Component 2')
ax2.set_title('Transformed Principal Components Distribution')
ax2.legend()

plt.show()





#%%>>>>>>>>>>>>>> 4. 独立成分分析 (ICA)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 设置随机种子
np.random.seed(0)

# 生成虚拟信号数据
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成3个独立的信号：正弦波、方波和噪声
s1 = np.sin(2 * time)  # 正弦波
s2 = np.sign(np.sin(3 * time))  # 方波
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 将信号组合成矩阵
S = np.c_[s1, s2, s3]

# 将信号标准化到范围内
S /= S.std(axis=0)

# 生成混合数据（线性混合）
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2.5]])  # 混合矩阵
X = np.dot(S, A.T)  # 生成混合信号

# 使用FastICA从混合信号中分离独立成分
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # 重构后的信号
A_ = ica.mixing_  # 分离出的混合矩阵

# 绘制原始信号、混合信号和分离出的信号
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(3, 1, 1)
plt.title("Original Signals")
colors = ['red', 'blue', 'green']
for i, signal in enumerate(S.T):
    plt.plot(time, signal, color=colors[i], label=f"Signal {i+1}")
plt.legend(loc='upper right')

# 混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
colors = ['orange', 'purple', 'brown']
for i, signal in enumerate(X.T):
    plt.plot(time, signal, color=colors[i], label=f"Mixed {i+1}")
plt.legend(loc='upper right')

# 分离出的信号
plt.subplot(3, 1, 3)
plt.title("ICA Recovered Signals")
colors = ['cyan', 'magenta', 'yellow']
for i, signal in enumerate(S_.T):
    plt.plot(time, signal, color=colors[i], label=f"Recovered {i+1}")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>> 5. 因子分析 (Factor Analysis)

import pandas as pd
import numpy as np

# 创建虚拟数据集
np.random.seed(0)
n_samples = 1000
n_features = 5

# 生成数据集
data = np.random.randint(1, 11, size=(n_samples, n_features))
df = pd.DataFrame(data, columns=['Screen Size', 'Camera Quality', 'Battery Life', 'Performance', 'Design'])

# 添加一些噪音列
df['other factory'] = np.random.randint(1, 11, size=n_samples)

# 显示数据集的前几行
print(df.head())
# 使用因子分析来提取潜在因子，并使用图形来展示分析结果。

from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# 因子分析
fa = FactorAnalysis(n_components=3)
fa.fit(df)

# 提取因子负荷
factors = pd.DataFrame(fa.components_, columns=df.columns)

# 绘制因子负荷热图
plt.figure(figsize=(10, 6))
plt.imshow(factors, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=45)
plt.yticks(range(len(factors)), ['Factor {}'.format(i+1) for i in range(len(factors))])
plt.title('Factor Loading Heatmap')
plt.show()

# 绘制因子得分散点图
factors_scores = fa.transform(df)
plt.figure(figsize=(8, 6))
plt.scatter(factors_scores[:, 0], factors_scores[:, 1])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Scores Scatter Plot')
plt.grid(True)
plt.show()





#%%>>>>>>>>>>>>>> 6. t-SNE


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 生成虚拟数据集
n_samples = 1000
X1, y1 = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42)
X2, y2 = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# 将两个数据集结合起来
X = np.vstack([X1, X2])
y = np.hstack([y1, y2 + 4])

# 使用PCA进行初步降维
pca = PCA(n_components=2)  # 将组件数改为2
X_pca = pca.fit_transform(X)

# 使用t-SNE降维到2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

# 绘图
plt.figure(figsize=(14, 7))

# t-SNE降维结果图
plt.subplot(1, 2, 1)
palette = sns.color_palette("hsv", len(np.unique(y)))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette, s=60, legend='full')
plt.title('t-SNE visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# 原始数据集的PCA降维前后的对比图
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=60, alpha=0.8, edgecolors='k')
plt.title('Original vs t-SNE reduced')
plt.xlabel('Feature 1 / t-SNE Component 1')
plt.ylabel('Feature 2 / t-SNE Component 2')

# 调整图例
plt.legend(title='Class')
plt.tight_layout()
plt.show()









#%%>>>>>>>>>>>>>> 7. 多维尺度分析 (MDS)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

# 生成高维数据
n_samples = 1000
n_features = 5
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=5, random_state=42)

# 计算原始数据的距离矩阵
distances = squareform(pdist(X))

# 使用MDS进行降维
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
X_mds = mds.fit_transform(distances)

# 可视化原始高维数据（选取前3个特征）
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.get_cmap('viridis', 5))
ax1.set_title('Original High-dimensional Data (First 3 Features)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')

# 可视化降维后的二维数据
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', 5))
ax2.set_title('MDS Reduced 2D Data')
ax2.set_xlabel('MDS Dimension 1')
ax2.set_ylabel('MDS Dimension 2')
legend1 = ax2.legend(*scatter.legend_elements(), title="Classes")
ax2.add_artist(legend1)

plt.show()

# 计算降维前后的距离矩阵差异
original_distances = pdist(X)
reduced_distances = pdist(X_mds)
correlation = np.corrcoef(original_distances, reduced_distances)[0, 1]

print(f"Correlation between original and reduced distances: {correlation:.4f}")


#%%>>>>>>>>>>>>>> 8. 自编码器 (Autoencoder)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 加载MNIST数据集
(x_train, _), (x_test, _) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)
# 定义自编码器
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

# 编译模型
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练模型
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 构建编码器模型
encoder = Model(input_img, encoded)

# 使用编码器对测试集进行降维
encoded_imgs = encoder.predict(x_test)

# 使用自编码器对测试集进行重建
decoded_imgs = autoencoder.predict(x_test)


# 绘制原始图像与重建图像的对比
n = 10  # 要显示的图像数量
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 显示重建图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 使用t-SNE算法对降维后的数据进行二维可视化
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)
encoded_imgs_2d = tsne.fit_transform(encoded_imgs)

plt.figure(figsize=(10, 10))
plt.scatter(encoded_imgs_2d[:, 0], encoded_imgs_2d[:, 1], c=_[:10000], cmap='tab10')
plt.colorbar()
plt.show()




#%%>>>>>>>>>>>>>> 9. 局部线性嵌入 (LLE)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# 生成瑞士卷数据
n_samples = 1500
X, color = make_swiss_roll(n_samples)

# 进行局部线性嵌入 (LLE)
n_neighbors = 12
n_components = 2
lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
X_r = lle.fit_transform(X)

# 绘制原始瑞士卷数据
fig = plt.figure(figsize=(15, 7))

ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("Original Swiss Roll Data")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

# 绘制降维后的数据
ax = fig.add_subplot(122)
sc = ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.get_cmap('Spectral'))
ax.set_title("2D LLE Projection")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
plt.colorbar(sc)
plt.show()

# 进一步分析和可视化
# 分析降维后的数据密度分布
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# KDE plot of the first component
plt.subplot(1, 2, 1)
sns.kdeplot(X_r[:, 0], shade=True, color="r")
plt.title("Density plot of Component 1")

# KDE plot of the second component
plt.subplot(1, 2, 2)
sns.kdeplot(X_r[:, 1], shade=True, color="b")
plt.title("Density plot of Component 2")

plt.show()

# Pairplot to see the relationships between the components and the original features
import pandas as pd

df = pd.DataFrame(X_r, columns=["Component 1", "Component 2"])
df["Color"] = color

sns.pairplot(df, vars=["Component 1", "Component 2"], hue="Color", palette="Spectral")
plt.show()


#%%>>>>>>>>>>>>>> 10. 奇异值分解 (SVD)

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成一个随机矩阵（比如：50x50）
original_matrix = np.random.rand(50, 50)

# 对矩阵执行 SVD 分解
U, S, VT = np.linalg.svd(original_matrix, full_matrices=False)

# 定义重构矩阵的奇异值数量
k_values = [5, 10, 20, 50]

# 创建图像
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(20, 5))

# 显示原始矩阵
axes[0].imshow(original_matrix, cmap='viridis')
axes[0].set_title("Original Matrix", fontsize=14)
axes[0].axis('off')

# 根据不同的奇异值数量重构矩阵并绘制热力图
for i, k in enumerate(k_values):
    # 使用前 k 个奇异值重构矩阵
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])

    # 重构矩阵
    reconstructed_matrix = U[:, :k] @ S_k @ VT[:k, :]

    # 显示重构的矩阵
    axes[i + 1].imshow(reconstructed_matrix, cmap='viridis')
    axes[i + 1].set_title(f"Reconstructed Matrix (k={k})", fontsize=14)
    axes[i + 1].axis('off')

# 调整布局
plt.tight_layout()
plt.show()










#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>












#%%>>>>>>>>>>>>>>
















