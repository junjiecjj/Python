#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:24:37 2024

@author: jack
"""
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484392&idx=1&sn=9ed6c9d59812ed59da156cf823978c48&chksm=c0e5d92ef792503878b0e609c3b744bec03daef537627055006e16237005348fc0c13bf8cd1e&cur_album_id=3445855686331105280&scene=189#wechat_redirect



#%% 1. 袋装法 (Bagging, Bootstrap Aggregating)

# 导入必要的库和模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# 创建一个具有噪声的月亮形状的数据集
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义一个函数来绘制决策边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# 创建一个决策树模型
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# 创建一个袋装法集成决策树模型
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)

# 绘制单独决策树的决策边界
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(tree_clf, X, y)
plt.title('Decision Tree')

# 绘制袋装法集成决策树的决策边界
plt.subplot(1, 2, 2)
plot_decision_boundary(bagging_clf, X, y)
plt.title('Bagging Decision Trees')

plt.tight_layout()
plt.show()

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (Bagging Decision Trees)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(bagging_clf, title, X, y, cv=cv, n_jobs=4)

plt.show()





#%% 2. 提升法 (Boosting)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

# 创建一个月亮型数据集
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 初始化AdaBoost分类器，基础分类器使用决策树
base_estimator = DecisionTreeClassifier(max_depth=1)
ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)

# 训练模型
ada_boost.fit(X_train, y_train)

# 预测
y_pred = ada_boost.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), [accuracy_score(y_test, y_pred) for y_pred in ada_boost.staged_predict(X_test)], marker='o')
plt.xlabel('Number of Trees in AdaBoost Ensemble')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees in AdaBoost')
plt.grid(True)
plt.show()

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, ada_boost.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247488733&idx=1&sn=1379da8b231fed0200ba18b98e08a0a8&chksm=9b146b34ac63e222f2be322e171ebfe8273bceba042abb50b242eb1718d88717160e86234a31&cur_album_id=3256084713219047427&scene=190#rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 生成模拟数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练Adaboost模型
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# 绘制特征重要性图
features = model.feature_importances_
plt.bar(range(len(features)), features)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in AdaBoost')
plt.show()
#%% 3. 堆叠 (Stacking)

# 导入所需库
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 仅选取两个类别进行二分类
X = X[y != 2]
y = y[y != 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基础模型
base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svm', SVC(probability=True))
]

# 定义元学习器
meta_learner = LogisticRegression()

# 训练基础模型
for name, model in base_learners:
    model.fit(X_train, y_train)

# 构造元学习器的训练数据
meta_features = np.column_stack([
    model.predict_proba(X_train)[:, 1] for name, model in base_learners
])

# 训练元学习器
meta_learner.fit(meta_features, y_train)

# 评估基础模型
plt.figure(figsize=(18, 12))  # 调整高度以适应更多子图

for i, (name, model) in enumerate(base_learners):
    # 混淆矩阵
    plt.subplot(3, len(base_learners), i+1)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1], rotation=45)
    plt.yticks(tick_marks, [0, 1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # ROC曲线
    plt.subplot(3, len(base_learners), i+len(base_learners)+1)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {name}')
    plt.legend(loc="lower right")

# 评估堆叠模型
meta_features_test = np.column_stack([
    model.predict_proba(X_test)[:, 1] for name, model in base_learners
])

# 混淆矩阵
plt.subplot(3, len(base_learners), len(base_learners)*2+1)
y_pred_meta = meta_learner.predict(meta_features_test)
cm_meta = confusion_matrix(y_test, y_pred_meta)
plt.imshow(cm_meta, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Stacking Model Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1], rotation=45)
plt.yticks(tick_marks, [0, 1])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# ROC曲线
plt.subplot(3, len(base_learners), len(base_learners)*3)
y_score_meta = meta_learner.predict_proba(meta_features_test)[:, 1]
fpr_meta, tpr_meta, _ = roc_curve(y_test, y_score_meta)
roc_auc_meta = auc(fpr_meta, tpr_meta)
plt.plot(fpr_meta, tpr_meta, label=f'AUC: {roc_auc_meta:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stacking Model ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()




#%% 4. 投票法 (Voting)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# 加载数据集并选择前两个特征
iris = load_iris()
X, y = iris.data[:, :2], iris.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义三种不同的分类器
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)  # SVM需要设置probability=True用于软投票

# 创建投票分类器
voting_clf = VotingClassifier(estimators=[
    ('knn', knn_clf),
    ('dt', dt_clf),
    ('svm', svm_clf)
], voting='soft')  # 使用软投票更稳健

# 训练投票分类器
voting_clf.fit(X_train, y_train)

# 预测
y_pred = voting_clf.predict(X_test)

# 定义函数绘制决策边界
def plot_decision_boundary(clf, X, y, title):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制投票法分类器的决策边界
plot_decision_boundary(voting_clf, X_train, y_train, 'Voting Classifier Decision Boundary')





#%% 5. 袋装提升 (Bagged Boosting)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions  # 需要安装mlxtend库

# 加载数据集
iris = load_iris()
X = iris.data[:, :2]  # 只使用前两个特征，便于可视化
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k', alpha=0.8)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Scatter plot of Iris dataset')
plt.colorbar(label='Species', ticks=[0, 1, 2], orientation='vertical')
plt.show()

# 绘制决策边界
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=clf, legend=2)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Decision Boundary of Random Forest on Iris dataset')
plt.show()





#%% 6. 极端随机森林 (Extra Trees, Extremely Randomized Trees)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import plot_confusion_matrix

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化极端随机森林分类器
clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 绘制特征重要性
feature_importance = clf.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(range(4), feature_importance, align='center')
plt.yticks(range(4), iris.feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Extra Trees Classifier')
plt.show()

# 绘制混淆矩阵
disp = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion Matrix')
plt.show()






#%% 7. 梯度提升 (Gradient Boosting)
# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载糖尿病数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化梯度提升回归模型
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 评估模型
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# 绘制特征重要性图
feature_importance = model.feature_importances_
feature_names = diabetes.feature_names

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, align='center')
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Regression - Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# 绘制预测结果对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, model.predict(X_test), color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gradient Boosting Regression - Actual vs Predicted')
plt.show()





#%% 8. 随机森林 (Random Forests)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_clf.fit(X_train, y_train)

# 在测试集上做预测
y_pred = rf_clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 获取特征的重要性
feature_importances = rf_clf.feature_importances_

# 将特征重要性进行排序并绘制条形图
sorted_indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), np.array(iris.feature_names)[sorted_indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()




#%% 9. Adaboost (Adaptive Boosting)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# 加载数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义 Adaboost 分类器
base_classifier = DecisionTreeClassifier(max_depth=1)
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)

# 拟合模型
adaboost_classifier.fit(X_train, y_train)

# 预测
y_pred = adaboost_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Adaboost 分类器在测试集上的准确率: {accuracy:.2f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Adaboost Classifier')
plt.colorbar()
plt.xticks([0, 1], ['Benign', 'Malignant'], rotation=45)
plt.yticks([0, 1], ['Benign', 'Malignant'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 计算 ROC 曲线和 AUC
y_score = adaboost_classifier.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Adaboost Classifier')
plt.legend(loc='lower right')
plt.show()




#%% 10. XGBoost (eXtreme Gradient Boosting)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost分类器
model = XGBClassifier()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 绘制特征重要性图
plt.figure(figsize=(8, 6))
plot_importance(model, height=0.5, importance_type='gain', max_num_features=len(feature_names))
plt.title('Feature Importance')
plt.show()



# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485677&idx=1&sn=9e07d096b3d8e1edd964af056ae01168&chksm=c0e5d22bf7925b3d523d755a334cffd66b126e60dc933b0c32310835f5660d3b154a3034808d&mpshare=1&scene=1&srcid=0812vttNelBQkbHVhLHx1KET&sharer_shareinfo=da94674af7e7ddf339ca42768ac42cd2&sharer_shareinfo_first=da94674af7e7ddf339ca42768ac42cd2&exportkey=n_ChQIAhIQfPE3l0gNN%2FiJ%2FFrJ8ztWfhKfAgIE97dBBAEAAAAAAI%2FlBrB81UEAAAAOpnltbLcz9gKNyK89dVj0cK%2F8F%2BFrgMQt%2B7bDFd8AdFEIKQcj0qq%2BiR2I2rcm39lYvfmchP5g%2Bm%2FDb%2F6ANCuksIwAi%2BsSJWGU5kQfaWaGpp1NKZ7X%2FavGlgmU3SItdHGd5fTkZmmxMDjl4WCX7w3w%2FFBDKYEEft0mklB%2F0TE2KUjGz70pTsaa2dg4Zj8OAXMOsErUs9PL0BpAWjJO95zNVwlakIbavxRjL5O98bOCwoDGnzKVad9LUOv7EnEryuU4hxQmo6gWvWbWe%2B%2B9kJ%2FIF3%2B5sT95U7TWTvvHoWbWUkB%2B%2BGulgZXrT5J%2Beq4oAk4M%2FLN%2BIW3MX6hmzcj2i%2F%2Bqh6dUqE8o45tM&acctmode=0&pass_ticket=2%2BN7Rv%2FjYLoPtFo%2BSFfX7z4VCSSUC1NfJJ5YsOR1J2BkvMF8d0Qstwzi4j2%2BIaiC&wx_header=0#rd


#%%>>>>>>>>>>>>>> 1. 随机森林 (Random Forest)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# 生成虚拟分类数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=2, n_classes=3, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化并训练随机森林模型（使用所有特征）
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 绘制特征重要性图
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 绘制决策边界（仅限于前两个特征）
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

h = .02  # 网格步长
X_train_2d = X_train[:, :2]  # 仅使用前两个特征
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 使用前两个特征训练新的随机森林模型
rf_2d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_2d.fit(X_train_2d, y_train)

Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# 训练集点
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()



#%%>>>>>>>>>>>>>> 2. 梯度提升决策树 (Gradient Boosting Decision Trees, GBDT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# 设置绘图风格
sns.set(style="whitegrid")

# 生成一个虚拟的回归数据集
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

# 转换为 DataFrame 以便查看和处理
feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练 GBDT 模型
gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbdt.fit(X_train, y_train)

# 预测
y_pred = gbdt.predict(X_test)

# 获取特征重要性
importance = gbdt.feature_importances_

# 创建特征重要性 DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title('Feature Importance')
plt.show()

# 绘制预测值与实际值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

# 计算残差
residuals = y_test - y_pred

# 绘制残差分布图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color="blue", bins=30)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()





#%%>>>>>>>>>>>>>> 3. XGBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_clusters_per_class=2, random_state=42)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建XGBoost分类器并训练
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 绘制特征重要性图
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=10, importance_type='weight', title='Feature Importance')
plt.show()

# 预测概率
y_score = model.predict_proba(X_test)[:, 1]

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 预测结果
y_pred = model.predict(X_test)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()



#%%>>>>>>>>>>>>>> 4. LightGBM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import shap

# 生成虚拟数据集
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 100000, n_samples),
    'loan_amount': np.random.randint(1000, 50000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'years_with_bank': np.random.randint(1, 30, n_samples),
    'num_of_credit_cards': np.random.randint(1, 10, n_samples),
})

# 生成目标变量（是否违约）
y = (X['income'] < 50000) & (X['credit_score'] < 600) & (X['loan_amount'] > 20000)
y = y.astype(int)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM模型训练
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42
}

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

# 预测与评估
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred)}")

# 特征重要性可视化
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10, importance_type='gain', title='Feature Importance by Gain')
plt.show()

plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10, importance_type='split', title='Feature Importance by Split')
plt.show()

# SHAP值分析
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP值总体影响力图
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.show()

# 单个特征的SHAP值分布图
plt.figure(figsize=(10, 6))
shap.dependence_plot('income', shap_values, X_test, interaction_index='age')
plt.show()





#%%>>>>>>>>>>>>>> 5. Extra Trees (Extremely Randomized Trees)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 生成虚拟数据集
np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 将目标变量转换为非线性形式
y = np.sin(y) + np.log1p(np.abs(y))

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练Extra Trees回归模型
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测并评估模型
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f'Training MSE: {train_mse:.4f}')
print(f'Testing MSE: {test_mse:.4f}')
print(f'Training R2: {train_r2:.4f}')
print(f'Testing R2: {test_r2:.4f}')

# 绘制预测值与真实值的对比图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.6, edgecolor='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--', color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train Set: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test Set: Actual vs Predicted')

plt.tight_layout()
plt.show()

# 绘制特征重要性图
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()



#%%>>>>>>>>>>>>>> 6. AdaBoost (Adaptive Boosting)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# 1. 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=3, random_state=42)

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 使用 AdaBoost 进行分类
# 使用 DecisionTreeClassifier 作为弱学习器
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 3. 可视化训练数据和分类结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='Set1')
plt.title("Training Data")

# 预测测试集
y_pred = model.predict(X_test)

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette='Set1')
plt.title("Test Data with Predicted Labels")
plt.show()

# 4. 可视化决策边界
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', ax=ax)
    ax.set_title(title)

fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(model, X_train, y_train, ax, "Decision Boundary (Training Data)")
plt.show()

# 5. 绘制模型在不同弱学习器数量下的准确率变化
estimators = range(1, 101)
train_acc = []
test_acc = []

for n in estimators:
    model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(estimators, train_acc, label='Train Accuracy')
plt.plot(estimators, test_acc, label='Test Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Estimators')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>> 7. 袋装法 (Bagging, Bootstrap Aggregating)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 单一决策树回归
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X, y)
y_pred_tree = tree_reg.predict(X)

# 使用袋装法的决策树回归
bagging_reg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
bagging_reg.fit(X, y)
y_pred_bagging = bagging_reg.predict(X)

# 绘制单一决策树回归和袋装法回归的比较图
plt.figure(figsize=(14, 6))

# 图1: 数据和模型的预测结果
plt.subplot(1, 2, 1)
plt.scatter(X, y, c='blue', label="Training Data")
plt.plot(X, y_pred_tree, color="red", label="Decision Tree Prediction")
plt.plot(X, y_pred_bagging, color="green", label="Bagging Prediction")
plt.title("Decision Tree vs Bagging Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# 图2: 模型误差比较
plt.subplot(1, 2, 2)
error_tree = y - y_pred_tree
error_bagging = y - y_pred_bagging
plt.hist(error_tree, bins=20, alpha=0.6, color="red", label="Decision Tree Errors")
plt.hist(error_bagging, bins=20, alpha=0.6, color="green", label="Bagging Errors")
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

# 打印均方误差 (MSE)
mse_tree = mean_squared_error(y, y_pred_tree)
mse_bagging = mean_squared_error(y, y_pred_bagging)
print(f"Decision Tree MSE: {mse_tree:.4f}")
print(f"Bagging Regressor MSE: {mse_bagging:.4f}")





#%%>>>>>>>>>>>>>> 8. 堆叠 (Stacking)
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基学习器
base_learners = [
    ('lr', LogisticRegression(solver='liblinear')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, kernel='linear'))
]

# 定义元学习器
meta_learner = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 构建堆叠分类器
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# 拟合模型
stacking_clf.fit(X_train, y_train)

# 预测并计算性能
y_pred = stacking_clf.predict(X_test)
y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制特征重要性图
meta_learner.fit(X_train, y_train)
feature_importances = meta_learner.feature_importances_

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=[f'Feature {i}' for i in range(X.shape[1])])
plt.title('Feature Importances from Meta-Learner')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()





#%%>>>>>>>>>>>>>> 9. 投票法 (Voting)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 1. 数据集生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 模型定义
clf1 = LogisticRegression(solver='liblinear', random_state=42)
clf2 = SVC(kernel='linear', probability=True, random_state=42)
clf3 = RandomForestClassifier(n_estimators=100, random_state=42)

# 硬投票
voting_clf_hard = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3)], voting='hard')

# 软投票
voting_clf_soft = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3)], voting='soft')

# 模型训练
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)

# 3. 模型预测与ROC曲线
y_pred1 = clf1.predict_proba(X_test)[:, 1]
y_pred2 = clf2.predict_proba(X_test)[:, 1]
y_pred3 = clf3.predict_proba(X_test)[:, 1]
y_pred_voting_soft = voting_clf_soft.predict_proba(X_test)[:, 1]

fpr1, tpr1, _ = roc_curve(y_test, y_pred1)
fpr2, tpr2, _ = roc_curve(y_test, y_pred2)
fpr3, tpr3, _ = roc_curve(y_test, y_pred3)
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_voting_soft)

plt.figure(figsize=(10, 6))
plt.plot(fpr1, tpr1, label='Logistic Regression (AUC = {:.2f})'.format(auc(fpr1, tpr1)))
plt.plot(fpr2, tpr2, label='SVM (AUC = {:.2f})'.format(auc(fpr2, tpr2)))
plt.plot(fpr3, tpr3, label='Random Forest (AUC = {:.2f})'.format(auc(fpr3, tpr3)))
plt.plot(fpr_voting, tpr_voting, label='Voting Classifier (AUC = {:.2f})'.format(auc(fpr_voting, tpr_voting)))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# 4. 混淆矩阵比较
y_pred_hard = voting_clf_hard.predict(X_test)
y_pred_soft = voting_clf_soft.predict(X_test)

cm_hard = confusion_matrix(y_test, y_pred_hard)
cm_soft = confusion_matrix(y_test, y_pred_soft)
cm_rf = confusion_matrix(y_test, clf3.predict(X_test))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ConfusionMatrixDisplay(cm_rf, display_labels=['Class 0', 'Class 1']).plot(ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
ConfusionMatrixDisplay(cm_hard, display_labels=['Class 0', 'Class 1']).plot(ax=axes[1])
axes[1].set_title('Hard Voting Confusion Matrix')
ConfusionMatrixDisplay(cm_soft, display_labels=['Class 0', 'Class 1']).plot(ax=axes[2])
axes[2].set_title('Soft Voting Confusion Matrix')
plt.show()

# 5. 特征重要性分析（仅适用于随机森林）
importances = clf3.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()





#%%>>>>>>>>>>>>>> 10. 加权平均 (Weighted Averaging)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# 1. 生成虚拟房价数据集
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 1) * 10  # 房屋面积 (10 ~ 100 平方米)
y = 2 * X.flatten() + np.random.randn(n_samples) * 2 + 3  # 房价 (线性关系 + 噪声)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 模型训练与预测
# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 随机森林回归
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 支持向量回归
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# 3. 加权平均预测
weights = [0.3, 0.4, 0.3]  # 假设的权重
y_pred_weighted = (weights[0] * y_pred_lr +
                   weights[1] * y_pred_rf +
                   weights[2] * y_pred_svr)

# 计算误差
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_svr = mean_squared_error(y_test, y_pred_svr)
mse_weighted = mean_squared_error(y_test, y_pred_weighted)

# 4. 数据分析图形
plt.figure(figsize=(14, 6))

# 图1：各模型预测 vs 真实房价
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='black', label='Real Price')
plt.scatter(X_test, y_pred_lr, color='blue', label=f'Linear Regression (MSE: {mse_lr:.2f})')
plt.scatter(X_test, y_pred_rf, color='green', label=f'Random Forest (MSE: {mse_rf:.2f})')
plt.scatter(X_test, y_pred_svr, color='red', label=f'SVR (MSE: {mse_svr:.2f})')
plt.scatter(X_test, y_pred_weighted, color='purple', marker='x', s=100, label=f'Weighted Avg (MSE: {mse_weighted:.2f})')
plt.xlabel('House Size (square meters)')
plt.ylabel('Price')
plt.title('Model Predictions vs Real Prices')
plt.legend()

# 图2：预测误差分析
plt.subplot(1, 2, 2)
plt.bar(['Linear Regression', 'Random Forest', 'SVR', 'Weighted Avg'],
        [mse_lr, mse_rf, mse_svr, mse_weighted], color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Prediction Error Comparison')

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>
























