
"""

最强总结，十大降维算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484483&idx=1&sn=758cac475ccbd1d542573cf2e30e25c9&chksm=c0e5de85f7925793e5092c57d479a146403bacf3c3d4eeff48bb2dc9586285e35ebfe8af94fe&mpshare=1&scene=1&srcid=0726BRSLjsDMD7D7sMwhSURW&sharer_shareinfo=c77ff2867fa87083cab20f479226fed5&sharer_shareinfo_first=c77ff2867fa87083cab20f479226fed5&exportkey=n_ChQIAhIQNJ71rQv%2F2dHjiPUe9doSCBKfAgIE97dBBAEAAAAAAANjMMEGmLoAAAAOpnltbLcz9gKNyK89dVj0F1Q1ihNQLRnM1Ny07eqj2Rsnr0ajrwYNCwq5R5pd%2FqBqM1V%2F128bMJJYK1yvSNX1V5e%2Fa2ukk0gSRm6J27qX55%2Bcz7umHH2jXlgBFbHw%2BUrUXQqKsWQWfIcjRqsNENPO6ubBneN9za5eVeH2bh7D51Ij4KTO7S2%2BeZEKhEzGPvteVxs4e3gp8xPmxN9p1SqeEXx9Yu5tBexzDPlOAKKy3TATZwiyM5506bCje6hZNi8%2Bi5tMLedtWAWd6LSH6PaKGROGr02KvmabbMmTTQFfgcmoLEfoNXVNjcRH4W6TzieU75oKI5dWPWbSAiPw6BYXfsubgdzDddYD&acctmode=0&pass_ticket=8i8k1gbi4GtOFvUuUNbT8k2asxM6xlXj5ZVscQ%2BYYmGv6J5vPnbL%2FEN8oXPQNrwj&wx_header=0#rd



"""

#%%  1. 主成分分析 (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA进行降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolor='k', s=20)
plt.colorbar(label='Digit', ticks=range(10))
plt.title('PCA of MNIST Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()




#%% 2. 线性判别分析 (LDA)

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




#%% 3. 核主成分分析 (KPCA)
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




#%% 4. 独立成分分析 (ICA)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 生成两个原始音频信号
np.random.seed(0)
t = np.linspace(0, 1, 1000, endpoint=False)  # 时间轴
s1 = np.sin(2 * np.pi * 5 * t)              # 原始信号1：5Hz正弦波
s2 = np.sign(np.sin(2 * np.pi * 3 * t))    # 原始信号2：3Hz方波

# 绘制原始信号图形
plt.figure(figsize=(10, 3))
plt.subplot(2, 1, 1)
plt.plot(t, s1, label='Signal 1 (5 Hz)')
plt.title('Original Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, s2, label='Signal 2 (3 Hz)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# 将原始信号线性混合成观测信号
A = np.array([[0.5, 0.5],
              [0.3, 0.7]])  # 混合矩阵
X = np.dot(A, np.vstack([s1, s2]))

# 绘制观测信号图形
plt.figure(figsize=(6, 3))
plt.plot(t, X[0], label='Mixed Signal 1')
plt.plot(t, X[1], label='Mixed Signal 2')
plt.title('Mixed Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.decomposition import FastICA

# 对观测信号进行独立成分分析
ica = FastICA(n_components=2)
S_estimated = ica.fit_transform(X.T).T

# 绘制分离后的信号图形
plt.figure(figsize=(10, 3))
plt.subplot(2, 1, 1)
plt.plot(t, S_estimated[0], label='Estimated Signal 1')
plt.title('Estimated Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, S_estimated[1], label='Estimated Signal 2')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()







#%% 5. 因子分析 (Factor Analysis)


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






#%% 6. t-分布随机邻居嵌入 (t-SNE)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# 只选择前1000个样本进行降维
X_subset = X[:1000]
y_subset = y[:1000].astype(int)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化降维结果
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of MNIST Data')
plt.grid(True)
plt.show()

# 使用t-SNE降维至3维
tsne_3d = TSNE(n_components=3, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

# 3D可视化降维结果
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y_subset, cmap='tab10', alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
ax.add_artist(legend1)
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('3D t-SNE Visualization of MNIST Data')
plt.show()



#%% 7. 多维尺度分析 (MDS)

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








#%% 8. 自编码器 (Autoencoder)












#%% 9. 局部线性嵌入 (LLE)

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








#%% 10. Isomap (Isometric Mapping)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import Isomap

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 使用Isomap进行降维
n_neighbors = 10
n_components = 2
isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
X_transformed = isomap.fit_transform(X)

# 绘制降维后的数据
plt.figure(figsize=(12, 6))

# 子图1：二维散点图，按类别着色
plt.subplot(1, 2, 1)
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_transformed[y == label, 0], X_transformed[y == label, 1], label=iris.target_names[label])
plt.title('2D scatter plot by Isomap')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

# 子图2：二维散点图，按原始特征着色
plt.subplot(1, 2, 2)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
plt.title('2D scatter plot with color')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar()

plt.show()

# 子图3：与原始数据进行对比，降维前后的散点图
plt.figure(figsize=(12, 6))

# 原始数据的前两个特征的散点图
plt.subplot(1, 2, 1)
for i, label in enumerate(np.unique(y)):
    plt.scatter(X[y == label, 0], X[y == label, 1], label=iris.target_names[label])
plt.title('Original Data (First 2 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 降维后的数据
plt.subplot(1, 2, 2)
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_transformed[y == label, 0], X_transformed[y == label, 1], label=iris.target_names[label])
plt.title('Isomap Transformed Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.show()









































































































































































































































































































































































