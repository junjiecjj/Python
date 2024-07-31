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











































