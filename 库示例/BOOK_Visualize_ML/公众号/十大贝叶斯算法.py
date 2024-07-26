
"""

最强总结，十大贝叶斯算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484679&idx=1&sn=789522244120467c83b298fe90e86b10&exportkey=n_ChQIAhIQitTAf1FLYIHCaVO%2FvI8MPRKfAgIE97dBBAEAAAAAADpwMoVdpJsAAAAOpnltbLcz9gKNyK89dVj0YFO9Qc0gpA4LxgGcaE88syT78YwDmNDQqdGLZfGVFZ3OVCe48BF0RLb7kY0r4WoBkP865X9Kzy5PFgGT9Y7iLzkilHH90wghfq2U8RbFqOXwCahPBLBeSvWDNnLD2hML8Lb773eSBbWJGjR7p8FkD6YZxbLQjtUC67KIhKeVSlYlp2u9kCmcchBnaqBZjaNzm16gpAgc7xYm7xBqzQJTLzm1wKMpCqPf0LV7K%2BqJD0XCMdpZ63Wi2W276hIp0pdM5vS7nKesgW6DqA0yhRZcMVoheT41F53K5hzwWX8zU7qyTDZn9we1UvMJFCGZexidfpxtE8OHpog%2F&acctmode=0&pass_ticket=%2BEMPsVAURy%2F8H628HeinbwUZpzF%2Bic9DzIenzb6U3EsAb3zSE0W6X%2FJDVz793ZD5&wx_header=0


"""

#%% 1. 朴素贝叶斯分类器（Naive Bayes Classifier）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# 加载数据集：这里使用20个新闻组数据集的一个子集
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# 定义一个朴素贝叶斯分类器的pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 在训练集上训练模型
model.fit(train.data, train.target)

# 在测试集上进行预测
predicted = model.predict(test.data)

# 计算混淆矩阵
cm = confusion_matrix(test.target, predicted)

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix for Naive Bayes Classifier')
plt.xticks(np.arange(len(categories)), categories, rotation=45)
plt.yticks(np.arange(len(categories)), categories)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 计算ROC曲线和AUC
probs = model.predict_proba(test.data)

# 为每个类别计算ROC曲线和ROC面积
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(test.target == i, probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'purple']  # 可以根据需要修改颜色
for i, category in enumerate(categories):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(category, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes Classifier')
plt.legend(loc='lower right')
plt.show()



#%% 2. 贝叶斯线性回归（Bayesian Linear Regression）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, X.shape[0])

# 将数据转换为适用于sklearn的形状
X = X[:, np.newaxis]

# 拟合贝叶斯线性回归模型
model = BayesianRidge()
model.fit(X, y)

# 预测
X_test = np.linspace(0, 10, 1000)[:, np.newaxis]
y_mean, y_std = model.predict(X_test, return_std=True)

# 绘制原始数据和模型预测结果
plt.figure(figsize=(14, 7))

# 绘制训练数据
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_test, y_mean, color='red', label='Prediction')
plt.fill_between(X_test.ravel(), y_mean - 2 * y_std, y_mean + 2 * y_std, color='red', alpha=0.2, label='Confidence Interval')
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Bayesian Linear Regression')
plt.legend()

# 绘制参数分布
plt.subplot(1, 2, 2)
n_iter = model.n_iter_
coefs = np.zeros((n_iter, X.shape[1]))
intercepts = np.zeros(n_iter)
for i in range(n_iter):
    model = BayesianRidge(n_iter=i+1)
    model.fit(X, y)
    coefs[i, :] = model.coef_
    intercepts[i] = model.intercept_

plt.plot(range(1, n_iter + 1), coefs, label='Coefficients')
plt.plot(range(1, n_iter + 1), intercepts, label='Intercept', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Parameter Convergence')
plt.legend()

plt.tight_layout()
plt.show()


#%% 3. 贝叶斯逻辑回归（Bayesian Logistic Regression）

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pystan
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成样本数据
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 斯坦模型定义
stan_code = """
data {
  int<lower=0> N;  // 样本数
  int<lower=0> D;  // 特征数
  matrix[N, D] X;  // 特征矩阵
  int<lower=0, upper=1> y[N];  // 标签
}
parameters {
  vector[D] beta;  // 回归系数
  real alpha;  // 截距
}
model {
  y ~ bernoulli_logit(alpha + X * beta);  // 逻辑回归模型
}
"""

# 准备数据
stan_data = {
    'N': X_train.shape[0],
    'D': X_train.shape[1],
    'X': X_train,
    'y': y_train
}

# 编译并拟合模型
sm = pystan.StanModel(model_code=stan_code)
fit = sm.sampling(data=stan_data, iter=2000, chains=4, seed=42)

# 抽取样本
samples = fit.extract()

# 画出回归系数的后验分布
az.plot_posterior(samples, var_names=['beta', 'alpha'], credible_interval=0.95)
plt.show()

# 使用训练数据的均值进行预测
X_test_ = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
beta_mean = np.mean(samples['beta'], axis=0)
alpha_mean = np.mean(samples['alpha'])
y_pred_prob = 1 / (1 + np.exp(-(alpha_mean + np.dot(X_test, beta_mean))))
y_pred = (y_pred_prob >= 0.5).astype(int)

# 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, y_pred))

# 决策边界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 500), np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = 1 / (1 + np.exp(-(alpha_mean + np.dot(grid, beta_mean))))
probs = probs.reshape(xx.shape)

plt.contourf(xx, yy, probs, 25, cmap="RdBu", alpha=0.6)
plt.colorbar()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, edgecolor="k", linewidth=1, cmap="RdBu", alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()





















#%%























#%%























#%%























#%%























#%%














#%%














#%%


















