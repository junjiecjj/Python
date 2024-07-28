
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
logreg_plain = LogisticRegression(penalty='none', random_state=42)
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



#%%











