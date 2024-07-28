
"""


最强总结，十大异常检测算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484638&idx=1&sn=c932a8c68d417899f8f7c9e4317d274c&chksm=c0e5de18f792570e443968444663dc836d7e81aafef4d5064022c1c1a6516d1b265736661745&mpshare=1&scene=1&srcid=0726QVNYAJnEpc4tTVq9x9hg&sharer_shareinfo=9f15115037c971c9851926caf4cff23b&sharer_shareinfo_first=9f15115037c971c9851926caf4cff23b&exportkey=n_ChQIAhIQ1%2FYrQA%2BbQJ%2BXZo8HIQ4rcxKfAgIE97dBBAEAAAAAAB8vCWZPAGIAAAAOpnltbLcz9gKNyK89dVj0k9Fza3wxn4zS75cqztuF4H6mlESrZJ78iXkt1195HTTr9fDxXURoYLC5UFQ3x5TkdW7cK63C63Al%2BDkLibyJ8gqFZZ7MvrIyoBsYtDE9Hg8cy%2FjZq78WCDa0BmoZrpVMIhdjNrqpo%2BnQZJLAGKaX7sWOkQJ8S6TcOqVJfwGfq834Ui5tGUx7%2BRC7qW6eMITXv2nbOtrJkFbYyWloutxe1ELbBYKe1uK4UXdkXJeG80%2Fmk%2B0trWydT%2F0HTgh2ZbhbvoOJbk7nxZBZYxoHm6XnCJ4EHK1j8uxghymvx4F%2BG2Gg49VL2H3wFF5PQGOZB9YR7gPZ01hF0SHf&acctmode=0&pass_ticket=XSjDFTKHyYA%2BnNDZ9ZilAOC%2FmlDQ9Yl%2Fwt8fbismG02IPM4%2BPX8uvV6GUXJTmDMh&wx_header=0#rd





"""
#%%  1. 孤立森林 (Isolation Forest)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# 生成正常点
X, _ = make_blobs(n_samples=300, centers=[[0, 0], [3, 3]], cluster_std=0.5, random_state=42)

# 生成一些异常点
np.random.seed(42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))

# 合并数据
X = np.concatenate([X, X_outliers], axis=0)

# 训练孤立森林模型
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# 预测
y_pred = clf.predict(X)

# 将预测结果转换为二进制标签
normal = X[y_pred == 1]
anomalies = X[y_pred == -1]

# 图1：数据分布与检测到的异常点
plt.figure(figsize=(10, 6))
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly')
plt.title('Data Distribution with Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 图2：孤立森林决策边界
xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# 绘制数据点
plt.scatter(normal[:, 0], normal[:, 1], c='blue', label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly')

plt.title('Isolation Forest Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


#%% 2. 高斯混合模型 (Gaussian Mixture Model, GMM)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(42)

# 生成正常数据
normal_data = np.random.normal(loc=0, scale=1, size=500).reshape(-1, 1)

# 生成异常数据
outlier_data = np.random.normal(loc=5, scale=1, size=20).reshape(-1, 1)

# 合并数据集
data = np.vstack((normal_data, outlier_data))

# 初始化GMM模型
gmm = GaussianMixture(n_components=2, random_state=42)

# 拟合模型
gmm.fit(data)

# 预测每个样本的概率密度值
probs = gmm.score_samples(data)

# 标记异常点
threshold = np.percentile(probs, 1)  # 选择1%分位数作为异常阈值
outliers = data[probs < threshold]

# 绘制数据分布图和异常点识别结果
plt.figure(figsize=(12, 6))

# 绘制数据分布
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', label='Data Distribution')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 绘制异常点识别结果
plt.subplot(1, 2, 2)
plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', label='Data Distribution')
plt.scatter(outliers, np.zeros_like(outliers), color='red', label='Detected Outliers')
plt.title('Outlier Detection using GMM')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


#%% 3. 单类支持向量机 (One-Class SVM)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 生成训练数据
np.random.seed(42)
X_train = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X_train + 2, X_train - 2]

# 生成一些异常点
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 拟合One-Class SVM模型
clf = svm.OneClassSVM(kernel="rbf", gamma=0.1, nu=0.1)
clf.fit(X_train)

# 预测训练数据和异常点
y_pred_train = clf.predict(X_train)
y_pred_outliers = clf.predict(X_outliers)

# 找到支持向量
svm_sv = clf.support_vectors_

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 图1: 数据和检测到的异常点
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Data and Outliers")
plt.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolor='k', s=50, label='Training data')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', edgecolor='k', s=50, label='Outliers')
plt.legend()
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.grid(True)

# 图2: 决策边界和支持向量
plt.subplot(1, 2, 2)
plt.title("Decision Boundary and Support Vectors")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolor='k', s=50, label='Training data')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', edgecolor='k', s=50, label='Outliers')
plt.scatter(svm_sv[:, 0], svm_sv[:, 1], s=100, facecolors='none', edgecolors='k', label='Support vectors')
plt.legend()
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.grid(True)

plt.tight_layout()
plt.show()

#%% 4. 局部离群因子 (Local Outlier Factor, LOF)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 生成数据
np.random.seed(42)
n_inliers = 200
n_outliers = 20

# 生成二维正态分布的正常数据
X_inliers = 0.3 * np.random.randn(n_inliers, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# 生成均匀分布的异常数据
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

# 合并数据集
X = np.r_[X_inliers, X_outliers]

# 应用LOF算法
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
scores_pred = clf.negative_outlier_factor_

# 绘制数据分布图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Data Distribution")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data Points')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='r', s=20, label='Anomalies')
plt.legend()

# Plotting LOF scores
plt.subplot(1, 2, 2)
plt.title("LOF Score")
radius = (scores_pred.max() - scores_pred) / (scores_pred.max() - scores_pred.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r', facecolors='none', label='LOF Scores')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='r', s=20, label='Anomalies')
plt.legend()

plt.tight_layout()
plt.show()

#%% 5. 基于 k-最近邻 (k-Nearest Neighbors, k-NN) 的异常检测
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 生成随机数据
np.random.seed(42)
n_samples = 300
n_outliers = 20

# 正常数据点
X_inliers = 0.3 * np.random.randn(n_samples, 2)
# 异常数据点
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

# 合并数据
X = np.r_[X_inliers, X_outliers]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-NN模型，选择k为5
k = 5
nbrs = NearestNeighbors(n_neighbors=k)
nbrs.fit(X_scaled)

# 计算每个点到其k个最近邻的距离的平均值作为异常评分
distances, _ = nbrs.kneighbors(X_scaled)
anomaly_scores = distances.mean(axis=1)

# 绘制原始数据点和异常点
plt.figure(figsize=(12, 6))

# 图1：原始数据点
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', label='Inliers')
plt.scatter(X_scaled[-n_outliers:, 0], X_scaled[-n_outliers:, 1], c='red', label='Outliers')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 图2：异常评分
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=anomaly_scores, cmap='coolwarm', label='Anomaly Score')
plt.colorbar(label='Anomaly Score')
plt.title('Anomaly Detection using k-NN')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

#%% 6. 自编码器 (Autoencoder)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Normalize the amount column
data['NormalizedAmount'] = MinMaxScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

# Split the data into training and test sets
train_data = data[data['Class'] == 0].sample(frac=0.8)
test_data = data.drop(train_data.index)

# Separate the features and labels
X_train = train_data.drop(['Class'], axis=1)
X_test = test_data.drop(['Class'], axis=1)
y_test = test_data['Class']

input_dim = X_train.shape[1]
encoding_dim = 14  # The number of nodes in the hidden layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
decoder = Dense(input_dim, activation="relu")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.summary()

epochs = 50
batch_size = 128

history = autoencoder.fit(X_train, X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_split=0.2,
                          verbose=1).history

predictions = autoencoder.predict(X_test)
mse = mean_squared_error(X_test, predictions, multioutput='raw_values')

# Convert the MSE to a DataFrame
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})


import seaborn as sns

# Plot reconstruction error distribution for normal and fraudulent transactions
fig, ax = plt.subplots()
sns.histplot(error_df[error_df['true_class'] == 0]['reconstruction_error'], bins=50, kde=True, color='blue', label='Normal', ax=ax)
sns.histplot(error_df[error_df['true_class'] == 1]['reconstruction_error'], bins=50, kde=True, color='red', label='Fraudulent', ax=ax)
ax.set_yscale('log')
ax.legend()
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction error')
plt.ylabel('Frequency')
plt.show()

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(error_df['true_class'], error_df['reconstruction_error'])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#%% 7. 基于概率的异常检测 (Probabilistic Models)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import seaborn as sns

# 设置图形样式
sns.set(style='whitegrid')

# 生成数据
n_samples = 1000
n_outliers = 50

# 正常数据
X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)

# 添加异常点
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-10, high=10, size=(n_outliers, 2))

# 合并正常数据和异常点
X = np.concatenate([X, X_outliers], axis=0)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练高斯混合模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# 计算每个点的对数概率
log_probs = gmm.score_samples(X_scaled)

# 设置阈值
threshold = np.percentile(log_probs, 3)

# 异常点
outliers = X_scaled[log_probs < threshold]

# 正常点
inliers = X_scaled[log_probs >= threshold]

# 绘制正常点和异常点
plt.figure(figsize=(12, 6))

plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', label='Inliers')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.title('Gaussian Mixture Model for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

def plot_ellipse(ax, mean, cov, label, color):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180.0 * angle / np.pi  # 转换为角度

    ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, edgecolor=color, facecolor=color, alpha=0.5)
    ell.set_clip_box(ax.bbox)
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')
    ax.plot(mean[0], mean[1], 'o', color=color, label=label)

# 绘制高斯分布
plt.figure(figsize=(12, 6))
ax = plt.gca()

for i in range(gmm.n_components):
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    plot_ellipse(ax, mean, cov, f'Gaussian {i+1}', color=np.random.rand(3,))

# 绘制数据点
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=10, color='gray', label='Data points')
plt.title('Gaussian Mixture Model Ellipses')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

#%% 8. 基于统计的异常检测 (Statistical Methods)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以获得可重复的结果
np.random.seed(42)

# 生成正态分布数据
data = np.random.normal(loc=0, scale=1, size=1000)

# 插入一些异常值
data_with_outliers = np.concatenate([data, np.random.normal(loc=0, scale=10, size=10)])

# 计算Z-score
mean = np.mean(data_with_outliers)
std = np.std(data_with_outliers)
z_scores = (data_with_outliers - mean) / std

# 设置阈值
threshold = 3

# 找到异常值
outliers = np.where(np.abs(z_scores) > threshold)[0]

# 绘制原始数据和异常值
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(data_with_outliers, 'b+', label='Data points')
plt.plot(outliers, data_with_outliers[outliers], 'ro', label='Outliers')
plt.title('Data with Outliers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 绘制Z-score分布
plt.subplot(1, 2, 2)
sns.histplot(z_scores, bins=30, kde=True)
plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
plt.axvline(x=-threshold, color='r', linestyle='--')
plt.title('Z-score Distribution')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

#%% 9. 基于距离的异常检测 (Distance-Based Methods)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 生成数据
np.random.seed(42)
n_samples = 200
n_outliers = 20

# 生成正常数据点
X_inliers = 0.3 * np.random.randn(n_samples, 2)
# 生成异常数据点
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

# 合并数据集
X = np.concatenate([X_inliers, X_outliers], axis=0)

# 使用Local Outlier Factor进行异常检测
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
scores = -clf.negative_outlier_factor_

# 绘制数据点和异常点
plt.figure(figsize=(12, 6))

# 子图1：数据点及异常点标识
plt.subplot(1, 2, 1)
plt.title("Data Points and Outliers")
plt.scatter(X[:, 0], X[:, 1], color='b', s=50, label="Inliers")
plt.scatter(X[y_pred == -1][:, 0], X[y_pred == -1][:, 1], color='r', s=50, label="Outliers")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 子图2：异常得分热图
plt.subplot(1, 2, 2)
plt.title("Outlier Scores Heatmap")
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='hot', s=50)
plt.colorbar()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

#%% 10. 基于分位数的异常检测 (Quantile-Based Methods)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据
np.random.seed(42)
sales_data = np.random.normal(loc=1000, scale=200, size=100)  # 生成一些销售额数据

# 添加几个异常值
sales_data[5] = 5000  # 异常值1
sales_data[15] = 200  # 异常值2

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(sales_data, vert=False)
plt.title('Box Plot of Daily Sales')
plt.xlabel('Sales Amount')
plt.show()

# 计算分位数
Q1 = np.percentile(sales_data, 25)
Q3 = np.percentile(sales_data, 75)
IQR = Q3 - Q1

# 计算异常值阈值
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
outliers = [x for x in sales_data if x < lower_bound or x > upper_bound]

print(f"异常值: {outliers}")











