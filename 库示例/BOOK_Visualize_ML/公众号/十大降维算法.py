
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
X_train_lda = lda.fit_transform(X_train, y_train) # (105, 2)
X_test_lda = lda.transform(X_test) # (45, 2)

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
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.get_cmap('viridis', 5))
ax1.set_title('Original High-dimensional Data (First 3 Features)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')

# 可视化降维后的二维数据
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap=plt.get_cmap('viridis', 5))
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





# https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247489491&idx=1&sn=01253c364d06f02f9b8b29b8e83171dd&chksm=c368ccd15f272790bbfeb53f46c7aa1c2f68c3c6727331e3b07536eabbb67b7e92acad225562&mpshare=1&scene=1&srcid=0825WhILEaMRAoODqt1XWLNm&sharer_shareinfo=19022c0f36e86fb5249c76a56af3578f&sharer_shareinfo_first=19022c0f36e86fb5249c76a56af3578f&exportkey=n_ChQIAhIQAvsIK%2BeFE5L9N0WQDEyyzhKfAgIE97dBBAEAAAAAAGgBINZQRmUAAAAOpnltbLcz9gKNyK89dVj0c3N6oyuH6I%2BxuRuYjd10BnzKrjxBEm6q7h%2BNv6xudjIyrPerqOLiHHUnGOqQZLBfngPUaSwElbfltmsAdQKQuDy3PM4R1KKOHWMI1RhdDzOO71lCjGvRvUhfmq4azy%2FfW4beHtaVD0hAnWcHCLoloOKXH8KJ4mwFDCKFN5ovz789oXltTm%2BQ%2FcWQofAbuVnvmYYlrt7MfrUb4ix7lMtEJTRtbAo0CeMJvfB%2BmuG3Bowl%2FIZhGr5m2zmAE2kaq9shqdXfIrRIT%2FXMO63glqMl%2Fk5bXPvhNdD6B%2FS7W4wkIr9N3naDapUcyf%2Fu%2BEKx84Inig6PM6Ov8Eu1&acctmode=0&pass_ticket=EKcg1AfhUKBNCHPsnDGBWERO28gz3bIoR%2B%2F8Obi06po2brDCZaAAwLiQEq8FYUoT&wx_header=0#rd



#%% 1. 主成分分析 (PCA, Principal Component Analysis)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import seaborn as sns

# 加载数据集
iris = load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

# 创建DataFrame
df = pd.DataFrame(data, columns=feature_names)
df['target'] = target

# 进行PCA降维
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
final_df = pd.concat([principal_df, df[['target']]], axis=1)

# 图1: PCA后的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=final_df, x='Principal Component 1', y='Principal Component 2', hue='target', palette='Set1')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(target_names)
plt.show()

# 图2: 原始特征的散点矩阵
sns.pairplot(df, hue='target', palette='Set1', diag_kind='kde')
plt.suptitle('Pairplot of Original Features', y=1.02)
plt.show()

# 图3: PCA后的解释方差比率
plt.figure(figsize=(10, 6))
explained_variance = pca.explained_variance_ratio_
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7, align='center', label='individual explained variance')
plt.step(range(len(np.cumsum(explained_variance))), np.cumsum(explained_variance), where='mid', linestyle='--', label='cumulative explained variance')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.show()

# 图4: PCA后成分的热力图
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2']), annot=True, cmap='coolwarm')
plt.title('PCA Component Heatmap')
plt.show()

#%% 2. 线性判别分析 (LDA, Linear Discriminant Analysis)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 创建LDA模型
lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# 创建一个DataFrame，用于可视化
df = pd.DataFrame(X_r2, columns=['LD1', 'LD2'])
df['target'] = y
df['target_name'] = df['target'].apply(lambda x: target_names[x])

# 绘制LDA结果的散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(x='LD1', y='LD2', hue='target_name', data=df, palette='Set1')
plt.title('LDA of Iris dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True)
plt.show()

# 计算每个类别的LDA投影的均值
means = df.groupby('target_name').mean()

# 绘制每个类别的LDA投影的均值的图形
plt.figure(figsize=(10, 8))
sns.scatterplot(x=means['LD1'], y=means['LD2'], hue=means.index, palette='Set1', s=100, marker='D', edgecolor='black')
for i, txt in enumerate(means.index):
    plt.annotate(txt, (means['LD1'][i], means['LD2'][i]), fontsize=12, weight='bold')
plt.title('Class Means in LDA Space')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True)
plt.show()

# 绘制LDA各线性判别向量的权重图
plt.figure(figsize=(12, 6))
components_df = pd.DataFrame(lda.coef_, columns=iris.feature_names, index=target_names)
components_df.plot(kind='bar', ax=plt.gca())
plt.title('LDA Coefficients')
plt.xlabel('Features')
plt.ylabel('Coefficient value')
plt.legend(loc='best')
plt.grid(True)
plt.show()


#%% 3. 奇异值分解 (SVD, Singular Value Decomposition)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# 生成一个合成高维数据集
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.rand(n_samples, n_features)

# 在高维数据中添加一些结构
for i in range(n_samples):
    if i % 2 == 0:
        X[i, :10] += 5
    else:
        X[i, 10:20] += 5

# 进行奇异值分解
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X)

# 可视化原始数据的特征分布（随机选择两组特征进行对比）
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7, c='blue', edgecolors='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature Distribution in Original High-dimensional Space')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 10], X[:, 11], alpha=0.7, c='red', edgecolors='k', s=50)
plt.xlabel('Feature 11')
plt.ylabel('Feature 12')
plt.title('Feature Distribution in Original High-dimensional Space')

plt.tight_layout()
plt.show()

# 可视化降维后的数据分布
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, c='green', edgecolors='k', s=50)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Data Distribution after SVD Dimensionality Reduction')
plt.show()

# 计算并可视化每个奇异值对应的方差解释比例
explained_variance = svd.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

plt.figure(figsize=(8, 6))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Different Principal Components')
plt.show()

#%% 4. 独立成分分析 (ICA, Independent Component Analysis)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from sklearn.decomposition import FastICA
from sklearn.datasets import make_blobs

# 生成样本数据
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成独立信号
s1 = np.sin(2 * time)  # 正弦波
s2 = np.sign(np.sin(3 * time))  # 方波
s3 = sawtooth(2 * np.pi * time)  # 锯齿波，使用sawtooth函数

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # 添加噪声

# 混合数据
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # 混合矩阵
X = np.dot(S, A.T)  # 观测信号

# ICA 分解
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # 重构信号
A_ = ica.mixing_  # 重构混合矩阵

# 结果可视化
plt.figure(figsize=(12, 8))

# 原始信号
plt.subplot(3, 1, 1)
plt.title("Original Signals")
for i, sig in enumerate(S.T):
    plt.plot(sig, label=f"Signal {i+1}")
plt.legend()

# 混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
for i, sig in enumerate(X.T):
    plt.plot(sig, label=f"Mixed Signal {i+1}")
plt.legend()

# 分离信号
plt.subplot(3, 1, 3)
plt.title("Separated Signals (after ICA)")
for i, sig in enumerate(S_.T):
    plt.plot(sig, label=f"Separated Signal {i+1}")
plt.legend()

plt.tight_layout()
plt.show()

# 对比原始信号和分离信号的相关系数矩阵
def plot_correlation_matrix(S, S_):
    correlation = np.corrcoef(S.T, S_.T)[:3, 3:]
    plt.figure(figsize=(6, 6))
    plt.imshow(correlation, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Correlation Matrix between Original and Separated Signals")
    plt.xlabel("Separated Signals")
    plt.ylabel("Original Signals")
    plt.xticks(range(3), ["Separated 1", "Separated 2", "Separated 3"])
    plt.yticks(range(3), ["Original 1", "Original 2", "Original 3"])
    plt.show()

plot_correlation_matrix(S, S_)


#%% 5. 非负矩阵分解 (NMF, Non-negative Matrix Factorization)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 应用NMF进行降维
nmf = NMF(n_components=2, random_state=42)
X_nmf = nmf.fit_transform(X_scaled)

plt.figure(figsize=(14, 6))

# 可视化原始数据的分布
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="viridis")
plt.title("Distribution of Original Data")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# 可视化NMF降维后的数据
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_nmf[:, 0], y=X_nmf[:, 1], hue=y, palette="viridis")
plt.title("Distribution of Data after NMF Dimensionality Reduction")
plt.xlabel("NMF Component 1")
plt.ylabel("NMF Component 2")
plt.show()

# 查看NMF组件
components = nmf.components_

# 可视化NMF组件
plt.figure(figsize=(10, 4))
for i, comp in enumerate(components):
    plt.subplot(1, 2, i+1)
    plt.bar(iris.feature_names, comp)
    plt.title(f"NMF Component {i+1}")
plt.tight_layout()
plt.show()

# 查看每个样本在NMF组件上的权重
plt.figure(figsize=(10, 6))
sns.heatmap(X_nmf, cmap="viridis", cbar=True, xticklabels=[f"NMF Component {i+1}" for i in range(2)])
plt.title("Weights of Each Sample on NMF Components")
plt.xlabel("NMF Component")
plt.ylabel("Sample")
plt.show()


#%% 6. 核PCA (KPCA, Kernel PCA)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用KPCA降维到二维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca_2d = kpca.fit_transform(X_scaled)

# 应用KPCA降维到三维
kpca_3d = KernelPCA(n_components=3, kernel='rbf', gamma=15)
X_kpca_3d = kpca_3d.fit_transform(X_scaled)

# 绘制原始数据的二维散点图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 绘制KPCA降维后的二维散点图
plt.subplot(1, 2, 2)
plt.scatter(X_kpca_2d[:, 0], X_kpca_2d[:, 1], c=y, cmap='viridis')
plt.title('KPCA Reduced Data (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()

# 绘制KPCA降维后的三维散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_kpca_3d[:, 0], X_kpca_3d[:, 1], X_kpca_3d[:, 2], c=y, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
ax.set_title('KPCA Reduced Data (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()


#%% 7. 多维尺度分析 (MDS, Multidimensional Scaling)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用MDS进行降维，将数据从4维降至2维
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_scaled)

# 绘制降维后的数据
plt.figure(figsize=(10, 6))

# 绘制不同类别的散点图
for i in range(len(np.unique(y))):
    plt.scatter(X_mds[y == i, 0], X_mds[y == i, 1], label=f'Class {i}', alpha=0.7)

plt.title('MDS Visualization of Iris Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.grid(True)
plt.show()


#%% 8. t-分布邻域嵌入 (t-SNE, t-Distributed Stochastic Neighbor Embedding)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target

# 使用t-SNE进行降维，降到2维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 绘制原始数据的散点图
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.title('Original Data')
plt.colorbar()

# 绘制t-SNE降维后的散点图
plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.title('t-SNE Visualization')
plt.colorbar()

plt.tight_layout()
plt.show()


#%% 9. 局部线性嵌入 (Locally Linear Embedding, LLE)
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

# 生成一个瑞士卷状数据集
X, _ = make_swiss_roll(n_samples=3000, noise=0.2, random_state=42)

# 使用LLE进行降维，目标是将3维数据降到2维
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
X_lle = lle.fit_transform(X)

# 绘制原始数据和降维后的数据
fig = plt.figure(figsize=(12, 6))

# 原始数据的散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap=plt.cm.Spectral)
ax1.set_title('Original 3D Swiss Roll')

# LLE降维后的散点图
ax2 = fig.add_subplot(122)
ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=X[:, 2], cmap=plt.cm.Spectral)
ax2.set_title('LLE Projection')

plt.tight_layout()
plt.show()


#%% 10. UMAP

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from umap import UMAP

# 加载手写数字数据集（MNIST）
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# 只选择一部分数据进行演示
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.1, random_state=42)

# 使用UMAP进行降维
umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X_train)

# 绘制降维后的数据点
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_train, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP Projection of MNIST')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



















































































































































































































































































































































