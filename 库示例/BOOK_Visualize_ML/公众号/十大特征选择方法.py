
"""

最强总结，十大特征选择方法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484603&idx=1&sn=077e11b50e9c9ab1c03fe4d23869c6a0&chksm=c0e5de7df792576b6e38896b08836a1d1ad2e219a771f34eff611d676be2429918de05ff4b49&mpshare=1&scene=1&srcid=0726h91uOCuZt7OaV0ejoew2&sharer_shareinfo=d9c7e12cf92d858c8a040c4b75391420&sharer_shareinfo_first=d9c7e12cf92d858c8a040c4b75391420&exportkey=n_ChQIAhIQxDCKq0UKli31Vxsl2lJkgRKfAgIE97dBBAEAAAAAAFF9JB4jdOoAAAAOpnltbLcz9gKNyK89dVj0EhlwCcbwszQdC7ZfuJMK5Al4I4VDOWP%2F1PScsfgOtn4QNVo%2BjqQx88imf5AgaLcmUZ716RaMMkYAbczxE2n2DYb0zVXrf1qllJ0iod6xE64N9x25wOM%2FFXV6h17pHYaNPx5ZmzjW0yIPaMayIh9210ivjnyB8u%2BXsUhsQ4KaVRSegvFqdQsankuJwVqZPb1jk7PkylHmH9Ip9wf4MP%2FWKwtUphZSJHw9tmWTNWHUmuSDzVB8%2FgVuSPJdN9VMSBPUEIbexwjI8f3yqZFSj3Sx3ssnshCMDbfVNqjhPdr4acGI7qj5VtRzlKr5ge%2BuAq8VrQX019mz8C3e&acctmode=0&pass_ticket=24XIzY%2Fd%2FLEOstz7LBX79eJrg6%2BnS1Rd%2F1%2F2fK8iK8SXIu9Pdo11glPeaxFVYpfM&wx_header=0#rd


"""
#%% 1. 方差阈值法 (Variance Threshold)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

# 生成示例数据
np.random.seed(0)
n_samples = 1000
n_features = 5

# 高方差特征
high_variance_data = np.random.normal(loc=0, scale=5, size=(n_samples, 2))
# 低方差特征
low_variance_data = np.random.normal(loc=0, scale=0.5, size=(n_samples, 3))

# 合并数据集
data = np.hstack((high_variance_data, low_variance_data))
# 添加噪声特征
noise = np.random.normal(loc=0, scale=0.1, size=(n_samples, 1))
data_with_noise = np.hstack((data, noise))

# 转换为DataFrame
columns = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'noise']
df = pd.DataFrame(data_with_noise, columns=columns)

# 方差阈值法选择特征
threshold = 0.1  # 设定方差阈值
selector = VarianceThreshold(threshold=threshold)
selected_features = selector.fit_transform(df)

# 打印选择的特征
selected_columns = df.columns[selector.get_support()]
print("选择的特征:", selected_columns)

# 画出数据分析图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 盒图展示各特征的分布
df.boxplot(ax=ax1)
ax1.set_title('Boxplot of Features')

# 相关性矩阵展示
corr_matrix = df.corr()
im = ax2.matshow(corr_matrix, cmap='coolwarm')
fig.colorbar(im, ax=ax2)
ax2.set_xticklabels([''] + list(df.columns))
ax2.set_yticklabels([''] + list(df.columns))
ax2.set_title('Correlation Matrix')

plt.tight_layout()
plt.show()


#%% 2. 单变量特征选择 (Univariate Feature Selection)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# 使用SelectKBest进行单变量特征选择
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

# 获取选择的特征的索引
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

# 将数据转换为DataFrame以便于可视化
df = pd.DataFrame(X, columns=feature_names)
df_selected = df.iloc[:, selected_indices]
df_selected['target'] = y

# 可视化1: 选择特征的盒须图（Boxplot）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='target', y=selected_features[0], data=df_selected)
plt.title(f'Boxplot of {selected_features[0]} by Target')
plt.subplot(1, 2, 2)
sns.boxplot(x='target', y=selected_features[1], data=df_selected)
plt.title(f'Boxplot of {selected_features[1]} by Target')
plt.tight_layout()
plt.show()

# 可视化2: 选择特征的散点图（Scatter plot）
plt.figure(figsize=(10, 6))
sns.scatterplot(x=selected_features[0], y=selected_features[1], hue='target', palette='Set1', data=df_selected)
plt.title(f'Scatter Plot of {selected_features[0]} vs {selected_features[1]}')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()

# 可视化3: 选择特征的联合分布图（Joint plot）
sns.jointplot(x=selected_features[0], y=selected_features[1], hue='target', palette='Set1', data=df_selected)
plt.show()




#%% 3. 递归特征消除 (Recursive Feature Elimination, RFE)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# 设置图形的风格
sns.set(style="whitegrid")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用支持向量机作为基模型进行递归特征消除
svc = SVC(kernel="linear", random_state=42)
rfe = RFE(estimator=svc, n_features_to_select=2)
rfe.fit(X_train, y_train)

# 输出被选中的特征
selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
print("Selected Features: ", selected_features)

# 绘制特征重要性图
ranking = rfe.ranking_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), ranking, color='blue', align='center')
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Ranking")
plt.title("Feature Importance Ranking using RFE")
plt.show()

# 记录特征数目与模型准确性的关系
num_features = list(range(1, len(feature_names) + 1))
accuracies = []

for n in num_features:
    rfe = RFE(estimator=svc, n_features_to_select=n)
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 绘制特征数目与模型准确性的关系图
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o', color='green')
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Number of Features Selected")
plt.show()

# 最终模型的性能
y_pred = rfe.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy with Selected Features: {:.2f}%".format(final_accuracy * 100))




#%% 4. 基于树模型的特征选择 (Feature Importance from Tree-Based Models)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# 使用随机森林进行特征选择
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
feature_importances = pd.DataFrame(rf.feature_importances_, index=feature_names, columns=['Importance']).sort_values(by='Importance', ascending=False)

# 特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['Importance'])
plt.xticks(rotation=90)
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()
# 选择重要性最高的前5个特征
top_features = feature_importances.index[:5]
pairplot_df = df[top_features.tolist() + ['Target']]

# 散点图矩阵
sns.pairplot(pairplot_df, hue='Target', diag_kind='kde', markers=["o", "s"])
plt.suptitle('Pairplot of Top 5 Important Features', y=1.02)
plt.show()

#%% 5. L1 正则化 (Lasso Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 数据预处理：标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型，使用 L1 正则化进行特征选择
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
logreg_l1.fit(X_train, y_train)

# 获取选择的特征
model = SelectFromModel(logreg_l1, prefit=True)
X_train_selected = model.transform(X_train)
X_test_selected = model.transform(X_test)

# 获取被选择的特征的索引
selected_features = model.get_support(indices=True)

# 打印选择的特征
print("选择的特征索引:", selected_features)

# 创建一个普通的逻辑回归模型（用于对比，不使用正则化）
logreg_plain = LogisticRegression(penalty=None, random_state=42)
logreg_plain.fit(X_train, y_train)

# 绘制特征选择前后的系数对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
coefficients_l1 = logreg_l1.coef_.ravel()
coefficients_plain = logreg_plain.coef_.ravel()

axes[0].bar(np.arange(len(coefficients_plain)), np.abs(coefficients_plain), color='b')
axes[0].set_title('Coefficients without regularization')
axes[0].set_xlabel('Feature Index')
axes[0].set_ylabel('Coefficient Magnitude')

axes[1].bar(np.arange(len(coefficients_l1)), np.abs(coefficients_l1), color='r')
axes[1].set_title('Coefficients with L1 regularization')
axes[1].set_xlabel('Feature Index')
axes[1].set_ylabel('Coefficient Magnitude')

plt.tight_layout()
plt.show()



#%% 6. 嵌入法 (Embedded Methods)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=10000)

# 使用SelectFromModel来进行特征选择
sfm = SelectFromModel(estimator=model, threshold=None)
sfm.fit(X_train_scaled, y_train)

# 获取选择的特征
selected_features = np.array(data.feature_names)[sfm.get_support()]

# 打印选择的特征
print("Selected features:", selected_features)

# 绘制一个柱状图展示特征的重要性
plt.figure(figsize=(10, 6))
plt.bar(selected_features, np.abs(sfm.estimator_.coef_[0][sfm.get_support()]))
plt.xlabel('Feature')
plt.ylabel('Coefficient Magnitude')
plt.title('Coefficient Magnitude of Selected Features in Logistic Regression')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



#%% 7. 主成分分析 (Principal Component Analysis, PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# 生成随机数据集
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)

# 进行PCA分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 原始数据的散点图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# PCA降维后的散点图
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.title('PCA Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# 输出PCA的主成分解释方差比例
print(f"Explained variance ratio (first two components): {pca.explained_variance_ratio_}")



#%% 8. 相关系数法 (Correlation Coefficient)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 生成模拟数据
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# 计算相关系数矩阵
correlation_matrix = data.corr()

# 选择与目标变量相关性较高的特征（绝对值大于0.5）
selected_features = correlation_matrix['target'].apply(lambda x: abs(x)).sort_values(ascending=False)
selected_features = selected_features[selected_features > 0.5]
selected_feature_names = selected_features.index.drop('target')

# 绘制相关系数热图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# 绘制散点图矩阵
sns.pairplot(data[selected_feature_names.to_list() + ['target']])
plt.suptitle('Pairplot of Selected Features and Target', y=1.02)
plt.show()

# 显示相关系数
print("Selected Features and Their Correlation with Target:")
print(selected_features)



#%% 9. 信息增益 (Information Gain)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# 生成一个玩具数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Target'] = y

# 查看数据集
print(df.head())

from sklearn.feature_selection import mutual_info_classif

# 计算每个特征的信息增益
info_gain = mutual_info_classif(X, y)
info_gain_df = pd.DataFrame(info_gain, index=['Feature1', 'Feature2'], columns=['Information Gain'])

# 显示信息增益
print(info_gain_df)

# 散点图
plt.figure(figsize=(12, 6))

# 绘制数据集的散点图
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Target', palette='viridis')
plt.title('Feature1 vs Feature2')

# 训练决策树分类器
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 绘制决策树图
plt.subplot(1, 2, 2)
plot_tree(clf, feature_names=['Feature1', 'Feature2'], class_names=['Class 0', 'Class 1'], filled=True)
plt.title('Decision Tree')

plt.tight_layout()
plt.show()


#%% 10. 互信息法 (Mutual Information)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据集（鸢尾花数据集）
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 计算各个特征与目标变量之间的互信息
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)

# 排序并选择前几个互信息最高的特征
selected_features = mi_scores.sort_values(ascending=False).index[:2]

# 绘制互信息分数条形图
plt.figure(figsize=(10, 6))
mi_scores.plot(kind='bar')
plt.title('Mutual Information Scores for Iris Features')
plt.ylabel('MI Score')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.show()

# 使用选定特征训练分类模型并评估性能
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with selected features: {accuracy:.2f}')








# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486189&idx=1&sn=a408a8de874183e60a46f4d920574110&exportkey=n_ChQIAhIQgA1p6nRxnES3QQEu5YqchhKfAgIE97dBBAEAAAAAAO62AJXlC5sAAAAOpnltbLcz9gKNyK89dVj0E5sRlpMzNA9W2PHKMi9ezCzZhss6yKV3Cy7JarGE%2BnJji4kUKqsmXYz38A%2FdZDs6v09ljC%2FUqXtC7ZGbiBEAcU7MLmG077EfyyLZkLkzzfN9wZpvH17AF%2F3V2TtIRyx4wyTG1KKcW8dkyeH%2BropBtdnW8zsd%2Fh%2FJACHJaW7u7CaeGETjfUD8wZCG4ZRHdvIIycv7dAzkMAN0iTy16nro8NXWE66iH7adxoGhA%2BBiC9OFi0KBO1SiEp6qvKLaQXae8L3FwRzpBfq4Kg%2BA3Sm4UfMycaaaJioZibG9bWEuNbwsl8Zeac574ns5EX2bNHMNeIoCA%2F8VsB%2Fe&acctmode=0&pass_ticket=xHur5%2F9ICMpVKawBDusJafDJG5qkVPmHJpkhiqzFwSrqLfkBIYCXXEQffOZ14cLQ&wx_header=0



#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 方差阈值法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 生成虚拟数据集
n_samples = 1000
n_features = 5

# 数据分布：0~1均匀分布
X = np.random.rand(n_samples, n_features)

# 人为地将某些特征的方差设置为接近0（即无变化）
X[:, 1] = 0.5  # 第二个特征方差为0
X[:, 3] = X[:, 3] * 0.01  # 第四个特征方差非常小

# 创建DataFrame以便查看数据
columns = [f'Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)

# 方差阈值法
threshold = 0.01  # 设置阈值
selector = VarianceThreshold(threshold=threshold)
X_selected = selector.fit_transform(X)

# 保留的特征
retained_features = df.columns[selector.get_support()]
dropped_features = df.columns[~selector.get_support()]

# PCA降维到2D以便可视化
pca = PCA(n_components=2)
X_pca_before = pca.fit_transform(X)
X_pca_after = pca.fit_transform(X_selected)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
fig.suptitle('Variance Threshold Feature Selection', fontsize=16)

# 原始数据的散点图 (PCA降维到2D)
axes[0, 0].scatter(X_pca_before[:, 0], X_pca_before[:, 1], c='blue', edgecolor='k', s=50, alpha=0.7)
axes[0, 0].set_title('Original Data (PCA 2D)', fontsize=14)
axes[0, 0].set_xlabel('PCA Component 1')
axes[0, 0].set_ylabel('PCA Component 2')

# 方差阈值法后的数据散点图 (PCA降维到2D)
axes[0, 1].scatter(X_pca_after[:, 0], X_pca_after[:, 1], c='green', edgecolor='k', s=50, alpha=0.7)
axes[0, 1].set_title('After Variance Threshold (PCA 2D)', fontsize=14)
axes[0, 1].set_xlabel('PCA Component 1')
axes[0, 1].set_ylabel('PCA Component 2')

# 原始数据的热力图
im1 = axes[1, 0].imshow(np.corrcoef(X.T), cmap='viridis', aspect='auto')
axes[1, 0].set_title('Original Data Correlation Heatmap', fontsize=14)
axes[1, 0].set_xticks(np.arange(n_features))
axes[1, 0].set_yticks(np.arange(n_features))
axes[1, 0].set_xticklabels(columns, rotation=45)
axes[1, 0].set_yticklabels(columns)
fig.colorbar(im1, ax=axes[1, 0])

# 方差阈值法后的数据热力图
im2 = axes[1, 1].imshow(np.corrcoef(X_selected.T), cmap='viridis', aspect='auto')
axes[1, 1].set_title('After Variance Threshold Correlation Heatmap', fontsize=14)
axes[1, 1].set_xticks(np.arange(len(retained_features)))
axes[1, 1].set_yticks(np.arange(len(retained_features)))
axes[1, 1].set_xticklabels(retained_features, rotation=45)
axes[1, 1].set_yticklabels(retained_features)
fig.colorbar(im2, ax=axes[1, 1])

# 显示图形
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 相关系数法

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from scipy.stats import pearsonr

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成虚拟数据集
n_samples = 100
n_features = 6

# 创建一个具有相关性的回归数据集
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)

# 将数据集转换为DataFrame以便于处理
columns = [f'Feature_{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df['Target'] = y

# 计算每个特征与目标变量的相关系数
correlations = {}
for col in columns:
    corr, _ = pearsonr(df[col], df['Target'])
    correlations[col] = corr

# 将相关系数结果转换为DataFrame
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# 绘制图形
plt.figure(figsize=(16, 10))

# 图1：相关系数矩阵的热图
plt.subplot(2, 1, 1)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix Heatmap', fontsize=16)

# 图2：特征与目标变量的散点图
plt.subplot(2, 1, 2)
colors = sns.color_palette('husl', n_features)
for i, col in enumerate(columns):
    plt.scatter(df[col], df['Target'], color=colors[i], alpha=0.7, label=f'{col} (Corr: {correlations[col]:.2f})')

plt.xlabel('Feature Value', fontsize=12)
plt.ylabel('Target Value', fontsize=12)
plt.title('Scatter Plot of Features vs. Target', fontsize=16)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 递归特征消除

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化分类器
svc = SVC(kernel="linear")

# 初始化RFE，选择最优特征数
rfe = RFE(estimator=svc, n_features_to_select=5)
rfe.fit(X_train, y_train)

# 绘制特征排序
ranking = rfe.ranking_
plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.bar(range(X.shape[1]), ranking, color='dodgerblue')
plt.xlabel("Feature Index")
plt.ylabel("Ranking")
plt.title("Feature Ranking using RFE")
plt.xticks(range(X.shape[1]), labels=range(1, X.shape[1]+1))

# 计算不同特征数量下的模型性能
scores = []
num_features = range(1, X.shape[1] + 1)
for n in num_features:
    rfe = RFE(estimator=svc, n_features_to_select=n)
    rfe.fit(X_train, y_train)
    score = np.mean(cross_val_score(rfe, X_train, y_train, cv=5))
    scores.append(score)

# 绘制特征数量与模型性能的关系
plt.subplot(1, 2, 2)
plt.plot(num_features, scores, marker='o', color='crimson', linestyle='-', linewidth=2, markersize=8)
plt.xlabel("Number of Selected Features")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Model Performance vs. Number of Features")
plt.grid(True)

# 显示图像
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 4. L1正则化

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(0)
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=0)

# 添加一些无关特征
X = np.hstack([X, np.random.randn(X.shape[0], 5)])

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 设置Lasso回归模型
alphas = [0.01, 0.1, 1, 10, 100]
coefs = []
mse_train = []
mse_test = []

# 训练Lasso回归模型并记录系数和均方误差
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    coefs.append(model.coef_)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

# 绘制结果图
plt.figure(figsize=(14, 7))

# 系数图
plt.subplot(1, 2, 1)
for i in range(len(alphas)):
    plt.plot(coefs[i], label=f'Alpha={alphas[i]}', marker='o')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients vs. Feature Index')
plt.legend()
plt.grid(True)

# 均方误差图
plt.subplot(1, 2, 2)
plt.plot(alphas, mse_train, label='Train MSE', marker='o', color='blue')
plt.plot(alphas, mse_test, label='Test MSE', marker='o', color='red')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Alpha')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 基于树模型的特征选择

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# 生成虚拟数据集
X, y = make_classification(n_samples=500, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用随机森林分类器进行特征选择
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# 获取特征重要性
importances = clf.feature_importances_

# 基于特征重要性选择特征
selector = SelectFromModel(clf, threshold="mean", prefit=True)
X_selected = selector.transform(X_scaled)

# 绘制特征重要性图和选择后特征图

plt.figure(figsize=(14, 6))

# 特征重要性图
plt.subplot(1, 2, 1)
indices = np.argsort(importances)[::-1]
plt.bar(range(X_scaled.shape[1]), importances[indices], color='skyblue', align='center')
plt.xticks(range(X_scaled.shape[1]), indices + 1)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance from Random Forest')

# 选择后的特征图
plt.subplot(1, 2, 2)
selected_features = np.sum(selector.get_support())
plt.bar([0], [selected_features], color='salmon', align='center')
plt.xticks([0], ['Selected Features'])
plt.xlabel('Selected Features')
plt.ylabel('Count')
plt.title('Number of Selected Features')

# 显示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 卡方检验

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest

# 生成虚拟数据集
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, n_informative=5, n_redundant=0, random_state=42)

# 将特征转化为数据框
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y, name='Target')

# 合并特征和目标变量
data = pd.concat([X_df, y_df], axis=1)

# 转换目标变量为类别标签
le = LabelEncoder()
y_encoded = le.fit_transform(y_df)

# 卡方检验
X_new = SelectKBest(chi2, k='all').fit_transform(X, y_encoded)
chi2_values, p_values = chi2(X, y_encoded)

# 创建图形
plt.figure(figsize=(14, 10))

# 特征卡方检验值的条形图
plt.subplot(2, 2, 1)
sns.barplot(x=feature_names, y=chi2_values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Chi-Square Values for Each Feature')
plt.xlabel('Feature')
plt.ylabel('Chi-Square Value')

# 特征p值的条形图
plt.subplot(2, 2, 2)
sns.barplot(x=feature_names, y=p_values, palette='plasma')
plt.xticks(rotation=90)
plt.title('P-Values for Each Feature')
plt.xlabel('Feature')
plt.ylabel('P-Value')

# 特征值分布直方图
plt.subplot(2, 2, 3)
for feature in feature_names:
    sns.histplot(data[feature], kde=True, label=feature, bins=30)
plt.legend()
plt.title('Distribution of Feature Values')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')

# 目标变量的频率分布直方图
plt.subplot(2, 2, 4)
sns.histplot(y_df, kde=True, bins=3, palette='coolwarm')
plt.title('Distribution of Target Variable')
plt.xlabel('Target Class')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 互信息法

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算互信息
mi = mutual_info_classif(X_scaled, y)

# 将结果转为数据框
mi_df = pd.DataFrame({'Feature': np.arange(X_scaled.shape[1]), 'Mutual Information': mi})

# 对特征按互信息排序
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# 计算PCA用于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 子图1: 特征与互信息的关系
axs[0].bar(mi_df['Feature'].astype(str), mi_df['Mutual Information'], color='royalblue', edgecolor='k')
axs[0].set_title('Feature vs Mutual Information')
axs[0].set_xlabel('Feature')
axs[0].set_ylabel('Mutual Information')
axs[0].grid(True)

# 子图2: PCA后的数据分布
scatter = axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
axs[1].set_title('PCA Projection')
axs[1].set_xlabel('PCA Component 1')
axs[1].set_ylabel('PCA Component 2')
axs[1].legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].grid(True)

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 主成分分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 生成虚拟数据集
n_samples = 1000
n_features = 5
n_classes = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.0, random_state=42)

# 创建PCA对象
pca = PCA(n_components=2)

# 进行PCA降维
X_pca = pca.fit_transform(X)

# 画出原始数据散点图
plt.figure(figsize=(12, 8))

# 原始数据散点图
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Original Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 降维后的数据散点图
plt.subplot(2, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA Reduced Data Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# PCA主成分载荷图
plt.subplot(2, 2, 3)
components = pca.components_.T
plt.bar(range(components.shape[0]), components[:, 0], color='red', alpha=0.6, label='PC1')
plt.bar(range(components.shape[0]), components[:, 1], color='blue', alpha=0.6, label='PC2')
plt.xticks(range(components.shape[0]), [f'Feature {i+1}' for i in range(components.shape[0])])
plt.title('PCA Component Loadings')
plt.xlabel('Features')
plt.ylabel('Loading')
plt.legend()

# PCA主成分解释方差比例图
plt.subplot(2, 2, 4)
explained_variance_ratio = pca.explained_variance_ratio_
plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, color='orange', alpha=0.8)
plt.title('Explained Variance Ratio of PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 顺序特征选择

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义分类模型
model = LogisticRegression()

# 顺序特征选择（向前选择）
selector_forward = SequentialFeatureSelector(model, n_features_to_select='auto', direction='forward', cv=5)
selector_forward.fit(X_train, y_train)
selected_features_forward = selector_forward.get_support()

# 顺序特征选择（向后选择）
selector_backward = SequentialFeatureSelector(model, n_features_to_select='auto', direction='backward', cv=5)
selector_backward.fit(X_train, y_train)
selected_features_backward = selector_backward.get_support()

# 训练并评估模型
model.fit(selector_forward.transform(X_train), y_train)
y_pred_forward = model.predict(selector_forward.transform(X_test))
accuracy_forward = accuracy_score(y_test, y_pred_forward)

model.fit(selector_backward.transform(X_train), y_train)
y_pred_backward = model.predict(selector_backward.transform(X_test))
accuracy_backward = accuracy_score(y_test, y_pred_backward)

# 绘图
plt.figure(figsize=(14, 6))

# 绘制特征选择过程中的特征重要性
plt.subplot(1, 2, 1)
plt.title('Feature Selection Process')
plt.bar(range(X.shape[1]), np.mean(X_train, axis=0), color='skyblue', label='Feature Importance')
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in range(X.shape[1])], rotation=90)
plt.ylabel('Feature Importance')
plt.legend()

# 绘制特征选择后的结果
plt.subplot(1, 2, 2)
plt.title('Feature Selection Results')
plt.plot(np.arange(1, len(selected_features_forward) + 1), [accuracy_forward] * len(selected_features_forward), 'o-', color='red', label='Forward Selection Accuracy')
plt.plot(np.arange(1, len(selected_features_backward) + 1), [accuracy_backward] * len(selected_features_backward), 'o-', color='blue', label='Backward Selection Accuracy')
plt.xlabel('Number of Selected Features')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 极限学习机特征选择

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 设置随机种子以便结果可复现
np.random.seed(42)

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y, name='Target')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)

# 训练随机森林分类器
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 获取特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性图
plt.figure(figsize=(14, 7))

# 特征重要性条形图
plt.subplot(1, 2, 1)
sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')

# 随机森林模型的预测准确率
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 显示预测结果的混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>





