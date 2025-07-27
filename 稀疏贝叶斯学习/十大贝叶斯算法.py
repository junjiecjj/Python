
"""

最强总结，十大贝叶斯算法 ！
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484679&idx=1&sn=789522244120467c83b298fe90e86b10&exportkey=n_ChQIAhIQitTAf1FLYIHCaVO%2FvI8MPRKfAgIE97dBBAEAAAAAADpwMoVdpJsAAAAOpnltbLcz9gKNyK89dVj0YFO9Qc0gpA4LxgGcaE88syT78YwDmNDQqdGLZfGVFZ3OVCe48BF0RLb7kY0r4WoBkP865X9Kzy5PFgGT9Y7iLzkilHH90wghfq2U8RbFqOXwCahPBLBeSvWDNnLD2hML8Lb773eSBbWJGjR7p8FkD6YZxbLQjtUC67KIhKeVSlYlp2u9kCmcchBnaqBZjaNzm16gpAgc7xYm7xBqzQJTLzm1wKMpCqPf0LV7K%2BqJD0XCMdpZ63Wi2W276hIp0pdM5vS7nKesgW6DqA0yhRZcMVoheT41F53K5hzwWX8zU7qyTDZn9we1UvMJFCGZexidfpxtE8OHpog%2F&acctmode=0&pass_ticket=%2BEMPsVAURy%2F8H628HeinbwUZpzF%2Bic9DzIenzb6U3EsAb3zSE0W6X%2FJDVz793ZD5&wx_header=0


"""

#%% 1. 朴素贝叶斯分类器（Naive Bayes Classifier）

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from sklearn.model_selection import train_test_split

# # 加载数据集：这里使用20个新闻组数据集的一个子集
# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# train = fetch_20newsgroups(subset='train', categories=categories)
# test = fetch_20newsgroups(subset='test', categories=categories)

# # 定义一个朴素贝叶斯分类器的pipeline
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# # 在训练集上训练模型
# model.fit(train.data, train.target)

# # 在测试集上进行预测
# predicted = model.predict(test.data)

# # 计算混淆矩阵
# cm = confusion_matrix(test.target, predicted)

# # 绘制混淆矩阵图
# plt.figure(figsize=(8, 6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title('Confusion Matrix for Naive Bayes Classifier')
# plt.xticks(np.arange(len(categories)), categories, rotation=45)
# plt.yticks(np.arange(len(categories)), categories)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

# # 计算ROC曲线和AUC
# probs = model.predict_proba(test.data)

# # 为每个类别计算ROC曲线和ROC面积
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(len(categories)):
#     fpr[i], tpr[i], _ = roc_curve(test.target == i, probs[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # 绘制ROC曲线
# plt.figure(figsize=(8, 6))
# colors = ['blue', 'red', 'green', 'purple']  # 可以根据需要修改颜色
# for i, category in enumerate(categories):
#     plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(category, roc_auc[i]))

# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Naive Bayes Classifier')
# plt.legend(loc='lower right')
# plt.show()



#%% 2. 贝叶斯线性回归（Bayesian Linear Regression）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, X.shape[0])
# (100,)

# 将数据转换为适用于sklearn的形状
X = X[:, np.newaxis] # (100, 1)

# 拟合贝叶斯线性回归模型
model = BayesianRidge()
model.fit(X, y)

# 预测
X_test = np.linspace(0, 10, 1000)[:, np.newaxis] # (1000, 1)
y_mean, y_std = model.predict(X_test, return_std = True)
# (1000,), (1000,)


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
    model = BayesianRidge(max_iter=i+1)
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






#%% 4. 贝叶斯网络（Bayesian Networks）


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 创建贝叶斯网络结构
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# 定义条件概率表（CPD）
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.2], [0.8]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.5], [0.5]])
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9, 0.7, 0.8, 0.1], [0.1, 0.3, 0.2, 0.9]], evidence=['A', 'B'], evidence_card=[2, 2])

# 添加CPD到模型
model.add_cpds(cpd_a, cpd_b, cpd_c)

# 检查模型是否有效
assert model.check_model()

# 使用变量消去法进行推理
infer = VariableElimination(model)

# 查询草地湿的概率
prob_C = infer.query(variables=['C'])
print(prob_C)

# 查询在草地湿的情况下下雨的概率
prob_A_given_C = infer.query(variables=['A'], evidence={'C': 1})
print(prob_A_given_C)

# 生成一些样本数据
samples = model.simulate(int(1e4))

# 绘制贝叶斯网络结构图
plt.figure(figsize=(8, 6))
nx.draw(model, with_labels=True, node_size=2000, node_color='skyblue', font_size=20, font_weight='bold')
plt.title('Bayesian Network Structure')
plt.show()

# 绘制样本数据分布图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A的分布
samples['A'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of A (Rain)')
axes[0].set_xlabel('A')
axes[0].set_ylabel('Frequency')

# B的分布
samples['B'].value_counts().plot(kind='bar', ax=axes[1], color='skyblue')
axes[1].set_title('Distribution of B (Sprinkler)')
axes[1].set_xlabel('B')
axes[1].set_ylabel('Frequency')

# C的分布
samples['C'].value_counts().plot(kind='bar', ax=axes[2], color='skyblue')
axes[2].set_title('Distribution of C (Wet Grass)')
axes[2].set_xlabel('C')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 绘制联合概率分布
pd.plotting.scatter_matrix(samples, alpha=0.2, figsize=(12, 12), diagonal='hist')
plt.suptitle('Joint Probability Distribution')
plt.show()




#%% 5. 马尔可夫链蒙特卡洛（Markov Chain Monte Carlo, MCMC）

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

# 1. 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_intercept = 1
true_slope = 2
sigma = 1
Y = true_intercept + true_slope * X + np.random.normal(0, sigma, size=len(X))

# 2. 定义贝叶斯模型
with pm.Model() as model:
    # 先验
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    slope = pm.Normal('Slope', mu=0, sigma=10)
    sigma = pm.HalfNormal('Sigma', sigma=1)

    # 线性模型
    mu = intercept + slope * X

    # 观测
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

    # 3. 使用MCMC进行采样
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# 4. 绘制数据和拟合的回归直线
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, label='Data')
pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(0, 10, 100), label='Posterior predictive regression lines', lw=1, alpha=0.3)
plt.plot(X, true_intercept + true_slope * X, label='True regression line', lw=2., c='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Data and Posterior Predictive Regression Lines')
plt.show()

# 5. 绘制回归系数的后验分布
pm.plot_posterior(trace, var_names=['Intercept', 'Slope', 'Sigma'])
plt.show()

# 6. 绘制预测值的后验分布
pred_x = np.linspace(0, 10, 100)
pred_y = trace.posterior['Intercept'].mean() + trace.posterior['Slope'].mean() * pred_x

plt.figure(figsize=(10, 5))
plt.plot(X, Y, 'o', label='Data')
plt.plot(pred_x, pred_y, label='Mean of posterior predictive', lw=2)
plt.fill_between(pred_x,
                 trace.posterior['Intercept'].mean() + trace.posterior['Slope'].mean() * pred_x - 1.96 * trace.posterior['Sigma'].mean(),
                 trace.posterior['Intercept'].mean() + trace.posterior['Slope'].mean() * pred_x + 1.96 * trace.posterior['Sigma'].mean(),
                 color='gray', alpha=0.2, label='95% prediction interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Posterior Predictive Mean and 95% Prediction Interval')
plt.show()





#%% 6. 变分贝叶斯方法（Variational Bayesian Methods）

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, dirichlet
from sklearn.mixture import BayesianGaussianMixture


# 生成数据
np.random.seed(42)

# 设定高斯混合模型参数
true_means = [-3, 0, 3]
true_variances = [0.5, 1, 2]
true_weights = [0.3, 0.4, 0.3]
true_components = len(true_means)

# 生成数据
n_samples = 1000
components = np.random.choice(np.arange(true_components), size=n_samples, p=true_weights)
data = np.zeros(n_samples)

for i in range(n_samples):
    data[i] = np.random.normal(true_means[components[i]], np.sqrt(true_variances[components[i]]))

# 绘制生成的数据分布
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='b', label='Generated Data')
plt.title('Generated Data from Gaussian Mixture Model')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()



# 使用变分贝叶斯高斯混合模型
bgm = BayesianGaussianMixture(n_components=3, covariance_type='full', max_iter=1000, random_state=42)
bgm.fit(data.reshape(-1, 1))

# 绘制变分贝叶斯估计的分布
x = np.linspace(-10, 10, 1000)
logprob = bgm.score_samples(x.reshape(-1, 1))
responsibilities = bgm.predict_proba(x.reshape(-1, 1))
pdf = np.exp(logprob)
plt.figure(figsize=(12, 6))
plt.hist(data, 30, density=True, alpha=0.5, color='b')
plt.plot(x, pdf, '-k', label='VBGMM')
plt.title('Variational Bayesian Gaussian Mixture Model')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()



#%% 7. 拉普拉斯近似（Laplace Approximation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

# 生成模拟数据
np.random.seed(0)
N = 100  # 数据点数
X = np.linspace(0, 10, N)
true_slope = 2.5
true_intercept = 1.0
true_noise_std = 1.0
y = true_slope * X + true_intercept + np.random.normal(0, true_noise_std, N)

# 绘制模拟数据
plt.scatter(X, y, label='Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated Data')
plt.legend()
plt.show()

# 定义线性模型和负对数后验
def linear_model(theta, X):
    return theta[0] * X + theta[1]

def negative_log_posterior(theta, X, y, prior_mean, prior_cov):
    model_predictions = linear_model(theta, X)
    residuals = y - model_predictions
    log_likelihood = -0.5 * np.sum(residuals**2)
    prior_term = -0.5 * np.dot((theta - prior_mean).T, np.linalg.solve(prior_cov, (theta - prior_mean)))
    return - (log_likelihood + prior_term)

# 先验分布参数
prior_mean = np.array([0, 0])
prior_cov = np.eye(2) * 10

# 优化找到MAP估计
initial_guess = np.array([0, 0])
result = minimize(negative_log_posterior, initial_guess, args=(X, y, prior_mean, prior_cov))
map_estimate = result.x
hessian_inv = result.hess_inv if result.hess_inv is not None else np.linalg.inv(result.jac)

# 计算拉普拉斯近似的协方差矩阵
laplace_cov = np.linalg.inv(hessian_inv)

# 绘制拟合结果
plt.scatter(X, y, label='Data')
plt.plot(X, linear_model(map_estimate, X), label='MAP Fit', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('MAP Fit')
plt.legend()
plt.show()

# 绘制后验分布等高线图
theta0 = np.linspace(map_estimate[0] - 3*np.sqrt(laplace_cov[0, 0]), map_estimate[0] + 3*np.sqrt(laplace_cov[0, 0]), 100)
theta1 = np.linspace(map_estimate[1] - 3*np.sqrt(laplace_cov[1, 1]), map_estimate[1] + 3*np.sqrt(laplace_cov[1, 1]), 100)
Theta0, Theta1 = np.meshgrid(theta0, theta1)
pos = np.dstack((Theta0, Theta1))
rv = multivariate_normal(map_estimate, laplace_cov)

plt.contourf(Theta0, Theta1, rv.pdf(pos), levels=50, cmap='viridis')
plt.scatter(map_estimate[0], map_estimate[1], color='red', label='MAP Estimate')
plt.xlabel('Slope')
plt.ylabel('Intercept')
plt.title('Posterior Distribution (Laplace Approximation)')
plt.legend()
plt.colorbar()
plt.show()





#%% 8. 高斯过程（Gaussian Processes）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 生成数据
np.random.seed(1)
X = np.linspace(0, 10, 100)
y = np.sin(X) + 0.5 * np.random.randn(100)

# 训练数据
X_train = X[::5][:, np.newaxis]
y_train = y[::5]

# 定义高斯过程模型
kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合模型
gp.fit(X_train, y_train)

# 预测
X_pred = np.linspace(0, 10, 1000)[:, np.newaxis]
y_pred, sigma = gp.predict(X_pred, return_std=True)

# 画图
plt.figure(figsize=(14, 6))

# 原始数据和训练数据
plt.subplot(1, 2, 1)
plt.scatter(X, y, c='k', label='Data')
plt.scatter(X_train, y_train, c='r', label='Train Data')
plt.title('Data and Train Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 高斯过程回归预测
plt.subplot(1, 2, 2)
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k', label='95% Confidence Interval')
plt.scatter(X_train, y_train, c='r', label='Train Data')
plt.title('Gaussian Process Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()





#%% 9. 贝叶斯优化（Bayesian Optimization）


import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence

# 定义Branin函数
def branin(x):
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

# 定义搜索空间
space = [(-5.0, 10.0), (0.0, 15.0)]

# 使用贝叶斯优化最小化Branin函数
res = gp_minimize(branin, space, n_calls=50, random_state=0)

# 图1：绘制优化过程中的收敛情况
plt.figure(figsize=(10, 6))
plot_convergence(res)
plt.title('Convergence Plot')
plt.xlabel('Number of calls')
plt.ylabel('Minimum value')
plt.show()

# 图2：绘制优化过程中的点
plt.figure(figsize=(10, 6))
plt.plot([x[0] for x in res.x_iters], [x[1] for x in res.x_iters], marker='o')
plt.title('Gaussian Process Fit')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 图3：绘制Branin函数的3D图形
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 10, 100)
y = np.linspace(0, 15, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([branin([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.scatter([x[0] for x in res.x_iters], [x[1] for x in res.x_iters], [branin(x) for x in res.x_iters], color='r', marker='o', label='Optimization Points')
ax.set_title('Branin Function Surface')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.legend()
plt.show()




#%% 10. 层次贝叶斯模型（Hierarchical Bayesian Models）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm

# 模拟数据
np.random.seed(42)

# 学校A
num_students_A = 30
num_classes_A = 3
math_scores_A = np.random.normal(70, 10, num_students_A)
class_indices_A = np.random.randint(0, num_classes_A, num_students_A)

# 学校B
num_students_B = 35
num_classes_B = 4
math_scores_B = np.random.normal(75, 8, num_students_B)
class_indices_B = np.random.randint(0, num_classes_B, num_students_B)

# 整合数据
data = pd.DataFrame({
    'school': ['A'] * num_students_A + ['B'] * num_students_B,
    'class': np.concatenate([class_indices_A, class_indices_B]),
    'math_score': np.concatenate([math_scores_A, math_scores_B])
})

# 绘制箱线图比较不同学校的数学成绩
plt.figure(figsize=(10, 6))
sns.boxplot(x='school', y='math_score', data=data)
plt.title('Math Scores by School')
plt.xlabel('School')
plt.ylabel('Math Score')
plt.show()

# 绘制每个班级的数学成绩分布
plt.figure(figsize=(12, 8))
sns.violinplot(x='school', y='math_score', hue='class', data=data, split=True, inner='quartile')
plt.title('Math Scores Distribution by School and Class')
plt.xlabel('School')
plt.ylabel('Math Score')
plt.legend(title='Class', loc='best')
plt.show()
















