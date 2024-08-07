
"""


最强总结，十大分类算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484390&idx=1&sn=2531e69610a73cc7d3692ff371242c63&chksm=c0e5d920f79250365bcb8fb1e9ed699a01f924f570d0a6ce4f42417ab0175c6a764d6abd6be0&mpshare=1&scene=1&srcid=0726VUC3ieR7ysfMvfpZHNWC&sharer_shareinfo=05063e795f8bc9c4fec372e21d53bc01&sharer_shareinfo_first=05063e795f8bc9c4fec372e21d53bc01&exportkey=n_ChQIAhIQTUoK%2BiA1keLNYJNkzFAWpxKfAgIE97dBBAEAAAAAAI9fB72UXXAAAAAOpnltbLcz9gKNyK89dVj09OHtl0Vz7COs%2BKDdHdlxbfsEmZQ%2BTyP7B%2B5lq9ha6Nvw8VMkOTKGFGcm0I1%2FRWJfBI0ZAvDmhBpGdwjl8K%2BqltNkzZJb9qI34Y6k1XBLdmWL7gmwntwhi6WvwkhZervBTFMTZfBdAZCXGyHp3ZuGXVCKU%2BYHdm4adCJk4tlOu%2FdXjgpEuekFDAe3pvV%2FCczWBanQuUMwFbkmG1JbkwU07qzf86od8qCe4U4cBzH9l6s%2Fl1Lilfnh%2FEKgRRrhPNw3KzfC5CrgWQx4R3mXLDz2E2G7srN5m%2F4VE9E%2FXwDqeMUV4Bf%2BaiAIqeTkqUweMzJ%2BEHUJ%2Bkt3vISL&acctmode=0&pass_ticket=TalTZhN5A1s0vpdJIlN%2FcpeN%2FuhQdIiSMSn6G0hoSGPeWD%2BN8mtbGP1BPbRJ0MTx&wx_header=0#rd




"""


#%% 1. 逻辑回归 (Logistic Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 绘制数据
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.75, edgecolors='k')
plt.title('Generated Data')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 绘制决策边界函数
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.75)
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制决策边界
plot_decision_boundary(model, X, y)




#%% 2. 决策树 (Decision Tree)

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 加载数据集
iris = load_iris()
X = iris.data[:, :2]  # 仅使用前两个特征，便于可视化
y = iris.target

# 训练决策树模型
clf = DecisionTreeClassifier(max_depth=5)  # 可以调整 max_depth 参数来增加树的深度
clf.fit(X, y)

# 绘制决策树图像
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.title("Decision Tree Classifier on Iris Dataset")
plt.show()





#%% 3. 随机森林 (Random Forest)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 创建一些随机数据作为示例
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))  # 添加一些噪音到目标变量中

# 训练随机森林回归模型
n_estimators = 100
model = RandomForestRegressor(n_estimators=n_estimators)
model.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = model.predict(X_test)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, y_pred, c="g", label="prediction", linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()




#%% 4. 支持向量机 (Support Vector Machine, SVM)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 生成数据集
np.random.seed(42)
X, y = datasets.make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, class_sep=1.0)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 创建SVM分类器并进行训练
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(X_train, y_train)

# 绘制决策边界
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 确定坐标轴的范围
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

# 创建网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
zz = np.zeros_like(xx)  # 创建与xx相同形状的零数组
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        # 对于每个点(xx[i, j], yy[i, j])，预测其类别
        zz[i, j] = svm_clf.predict([[xx[i, j], yy[i, j], 0]])

# 绘制决策边界和数据点
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
ax.plot_surface(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D SVM Decision Boundary with RBF Kernel')

plt.show()


#%% 5. K近邻 (K-Nearest Neighbors, KNN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 生成分类数据集
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 绘制决策边界
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Nearest Neighbors Decision Boundary')

def complex_visualization(X_train, X_test, y_train, y_test, y_pred):
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='viridis', s=100, edgecolor='k')
    plt.title('Training Data')

    plt.subplot(2, 2, 2)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette='coolwarm', s=100, edgecolor='k')
    plt.title('Test Data Prediction')

    plt.subplot(2, 2, 3)
    ConfusionMatrixDisplay(conf_matrix, display_labels=knn.classes_).plot(cmap='Blues', ax=plt.gca())
    plt.title('Confusion Matrix')

    plt.subplot(2, 2, 4)
    plot_decision_boundary(knn, X_train, y_train)
    plt.title('Decision Boundary')

    plt.tight_layout()
    plt.show()

# 打印分类报告
print("Classification Report:\n", class_report)

# 调用绘图函数
complex_visualization(X_train, X_test, y_train, y_test, y_pred)



#%% 6. 朴素贝叶斯 (Naive Bayes)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# 生成一个二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练朴素贝叶斯分类器
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# 预测
y_pred = nb_classifier.predict(X_test)

# 混淆矩阵和分类报告
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Classification Report:\n", class_report)

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# 绘制决策边界（仅对二维数据有效）
def plot_decision_boundary(clf, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend1)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# 生成二维数据
X_2d, y_2d = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# 训练朴素贝叶斯分类器
nb_classifier_2d = GaussianNB()
nb_classifier_2d.fit(X_2d, y_2d)

# 绘制决策边界
fig, ax = plt.subplots(figsize=(10, 7))
plot_decision_boundary(nb_classifier_2d, X_2d, y_2d, ax)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Naive Bayes Decision Boundary')
plt.show()


#%% 7. 梯度提升 (Gradient Boosting)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 生成示例数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练梯度提升分类器
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)

# 预测测试集结果
y_pred = gbc.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 绘制ROC曲线
y_score = gbc.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制特征重要性
importances = gbc.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(14, 7))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), indices)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()



#%% 8. XGBoost (Extreme Gradient Boosting)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb

# 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建XGBoost模型
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 绘制特征重要性图
xgb.plot_importance(model, height=0.5, max_num_features=20, importance_type='weight', show_values=False)
plt.title('Feature Importance (Weight)')
plt.show()

xgb.plot_importance(model, height=0.5, max_num_features=20, importance_type='gain', show_values=False)
plt.title('Feature Importance (Gain)')
plt.show()

xgb.plot_importance(model, height=0.5, max_num_features=20, importance_type='cover', show_values=False)
plt.title('Feature Importance (Cover)')
plt.show()



#%% 9. 神经网络 (Neural Networks)





#%%  10. LightGBM (Light Gradient Boosting Machine)

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report
import shap

# 加载数据
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names) # (569, 30)
y = pd.Series(data.target, name='target')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 训练模型
gbm = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], )

# 预测
y_pred_proba = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = (y_pred_proba >= 0.5).astype(int)

# 计算并打印性能指标
print('Accuracy:', accuracy_score(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_pred_proba))
print('Classification Report:\n', classification_report(y_test, y_pred))

# 绘制特征重要性
plt.figure(figsize=(12, 8))
lgb.plot_importance(gbm, max_num_features=10, importance_type='gain', title='Feature Importance (Top 10)', grid=False)
plt.show()

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 使用 SHAP 解释模型
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_test)

# 绘制 SHAP 总体重要性图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns)
plt.show()

# 绘制 SHAP 值分布图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.show()

# 绘制 SHAP 依赖图（选择特征 'mean radius'）
plt.figure(figsize=(12, 8))
shap.dependence_plot("mean radius", shap_values, X_test, feature_names=X.columns)
plt.show()






# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485576&idx=1&sn=87a50cccc1dadc06ee47bfaaffd6b039&chksm=c0e5d24ef7925b58bb806332a0bd92cb44de60c4aa840b0d2bf1ddaceeaf52333c345d57554f&mpshare=1&scene=1&srcid=0807S4tATGiNbK9FP6CLeIBn&sharer_shareinfo=44e099a7f09b6d83cffae9deba10bd72&sharer_shareinfo_first=44e099a7f09b6d83cffae9deba10bd72&exportkey=n_ChQIAhIQM4qWRGdTRgnKtv26blSt5hKfAgIE97dBBAEAAAAAAHvKLnj8zdgAAAAOpnltbLcz9gKNyK89dVj0rVI31kdSpTbXhW5%2FT8fC22%2BAVKiYaseulBAY0epx2GUObH8wA0%2B%2BmSeUhAalCeg6eUxbTeXSsGqZD0QzS76eo8pvQ2qNuDUogW%2BX4emxLvnmMbvJBD2P%2FTDQqmhLTUSetkLry5pU6b3IIdBO1J3jMI6Cb4zvElV89nhDZL1vdz%2B%2Frn5lJB8%2FBXZ%2Fisv8SxE6Jcb54W%2ByUZjUAP%2Fl3RhoxkWpwsag6%2B56krVzAnypzO3kSUsVJJlcSUK06bu5kYSFwMEXjn2KE3iATvQiBuGZbztHd1y3vDiEl1c18TQq9GgpZDPuvf0g3EcaqXdwN6bB6REzPv2bj%2Fsu&acctmode=0&pass_ticket=Bi24r%2BnG61mfeWyttWcmS2TWmztifWlFRhPdkpDChXqyP9BS%2BzhRDz4O3SBasccd&wx_header=0#rd


# 1. 逻辑回归 (Logistic Regression)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 绘制决策边界
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

# 调用绘图函数
plot_decision_boundary(X, y, model)

# 2. 决策树 (Decision Tree)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 决策树模型训练
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 绘制决策树
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'], rounded=True)
plt.title("Decision Tree")
plt.show()

# 绘制决策边界
def plot_decision_boundary(clf, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

fig, ax = plt.subplots(figsize=(10, 6))
plot_decision_boundary(clf, X_test, y_test, ax)
plt.title("Decision Boundary")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
# plot_confusion_matrix(clf, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 3. 随机森林 (Random Forest)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=3, random_state=42)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 使用seaborn设置风格
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest Classification Boundaries')
plt.show()

# 计算特征重要性
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# 绘制特征重要性条形图
plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [f'Feature {i+1}' for i in indices])
plt.xlim([-1, X.shape[1]])
plt.show()





# 4. 支持向量机 (Support Vector Machine, SVM)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 生成线性可分数据集
X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=2, random_state=42)

# 定义SVM分类器
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 绘制决策边界和支持向量
def plot_svc_decision_function(clf, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 绘制决策边界和间隔
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 绘制数据点和决策边界
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.title("Linear SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 2. 非线性可分数据的SVM (使用核方法)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 生成非线性可分数据集
X, y = datasets.make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)

# 定义RBF核SVM分类器
clf = SVC(kernel='rbf', C=1.0, gamma=0.5)
clf.fit(X, y)

# 绘制决策边界和支持向量
def plot_svc_decision_function(clf, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 绘制决策边界和间隔
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# 绘制数据点和决策边界
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.title("Non-linear SVM Decision Boundary (RBF Kernel)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# 5. k-近邻算法 (k-Nearest Neighbors, k-NN)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# 生成虚拟数据集
np.random.seed(0)
X, y = make_moons(1000, noise=0.3)

# 训练k-NN模型
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# 创建网格以绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格点上的值
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=ListedColormap(('red', 'blue')))
plt.title(f'2D Classification with k-NN (k={k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# 6. 朴素贝叶斯分类器 (Naive Bayes Classifier)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成虚拟数据
np.random.seed(42)
n_samples = 1000
X = np.zeros((n_samples, 2))
y = np.zeros(n_samples)

# 类别0
X[:50] = np.random.normal(loc=[5, 3], scale=[0.5, 0.5], size=(50, 2))
y[:50] = 0

# 类别1
X[50:100] = np.random.normal(loc=[6, 5], scale=[0.5, 0.5], size=(50, 2))
y[50:100] = 1

# 类别2
X[100:] = np.random.normal(loc=[8, 3], scale=[0.5, 0.5], size=(50, 2))
y[100:] = 2

# 训练朴素贝叶斯分类器
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 打印分类准确率
print(f'分类准确率: {accuracy_score(y_test, y_pred):.2f}')

# 绘制数据和分类边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('朴素贝叶斯分类器的分类边界')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.colorbar()
plt.show()


# 7. 梯度提升树 (Gradient Boosting Trees)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

# 1. 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1, class_sep=1.0, random_state=42)

# 2. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 训练梯度提升树模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. 决策边界可视化
def plot_decision_boundary(model, X, y, ax):
    DecisionBoundaryDisplay.from_estimator(
        model, X, response_method="predict", ax=ax, grid_resolution=1000, cmap="coolwarm", alpha=0.6
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
    ax.set_xlim(X[:, 0].min(), X[:, 0].max())
    ax.set_ylim(X[:, 1].min(), X[:, 1].max())
    ax.set_title('Decision Boundary')

# 5. 特征重要性可视化
def plot_feature_importance(model, ax):
    importances = model.feature_importances_
    feature_names = ['Feature 1', 'Feature 2']
    indices = np.argsort(importances)

    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')

# 6. 学习曲线可视化
def plot_learning_curve(model, X_train, y_train, ax):
    train_sizes = np.arange(100, len(X_train), 100)  # 修改为整数
    train_scores = []
    test_scores = []

    for size in train_sizes:
        X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
        model.fit(X_sub, y_sub)
        train_scores.append(model.score(X_sub, y_sub))
        test_scores.append(model.score(X_test, y_test))

    ax.plot(train_sizes, train_scores, label='Train score', marker='o')
    ax.plot(train_sizes, test_scores, label='Test score', marker='o')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve')
    ax.legend()



# 7. 绘制图形
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 决策边界
plot_decision_boundary(model, X_test, y_test, ax=axs[0])

# 特征重要性
plot_feature_importance(model, ax=axs[1])

# 学习曲线
plot_learning_curve(model, X_train, y_train, ax=axs[2])

plt.tight_layout()
plt.show()



# 9. 线性判别分析 (Linear Discriminant Analysis, LDA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

# 生成虚拟数据集
np.random.seed(0)
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_classes=3, n_clusters_per_class=1)

# 训练LDA模型
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 绘制LDA结果
plt.figure(figsize=(12, 8))

# 2D散点图
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA 2D Projection')
plt.colorbar(scatter, label='Class')

# 3D散点图
plt.subplot(1, 2, 2, projection='3d')
ax = plt.axes(projection='3d')
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], np.zeros_like(X_lda[:, 0]), c=y, cmap='viridis', edgecolor='k', s=50)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3 (Fixed)')
ax.set_title('LDA 3D Projection')

plt.tight_layout()
plt.show()





# 10. 高斯混合模型 (Gaussian Mixture Model, GMM)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

# 生成虚拟数据集
np.random.seed(0)
X, _ = make_moons(n_samples=1000, noise=0.05)

# 创建 GMM 模型并训练
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# 绘制数据点
plt.figure(figsize=(12, 6))

# 绘制数据点颜色根据GMM的标签
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.5)
plt.title('GMM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 绘制每个高斯成分的等高线
plt.subplot(1, 2, 2)
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X1, Y1 = np.meshgrid(x, y)
XY = np.vstack([X1.ravel(), Y1.ravel()]).T
Z = np.exp(gmm.score_samples(XY))
Z = Z.reshape(X1.shape)

plt.contourf(X1, Y1, Z, cmap='viridis', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', edgecolor='k')
plt.title('GMM Contour')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

























