#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:31:32 2024

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485495&idx=1&sn=ccf89a1e8368e7f727175f2cdaa52ce9&chksm=c0e5d2f1f7925be7782db27be0e6df84479d1089c454c49b0c68a1a04d3f3c26a2bb4ffea4fc&mpshare=1&scene=1&srcid=0810HG4P4hNke2PLC3gM8dh4&sharer_shareinfo=d5014d8cae2730ed78f5045e1111225b&sharer_shareinfo_first=d5014d8cae2730ed78f5045e1111225b&exportkey=n_ChQIAhIQk1XmjTKNxq8v8qopnkZc6hKfAgIE97dBBAEAAAAAAGrEFCl%2BGncAAAAOpnltbLcz9gKNyK89dVj0m%2FmYNBhDcEPexAAikgxnhe9vjsbwh3ALpSGi5lJwqvhCA2jOi%2Fp6MrPznLy1V%2FY7BusSybhehk5KjiV%2F4wjDgTOYX1KQn9PZxC5f1YieB21EP4T1p7kAYnHvFti85rJ482Sjd%2F2e1EsxDNvGKQEjxFva%2BEWFUOyZygHryQZtc4AfobvOw5Hs5%2Bh0z63M1%2FnUaWfYt5N97d4%2B8H9yxhqOePgH6gXMYLJ0RYW4suhN71Ho%2FpwBCsCc27rXOtdG0tKIPtNEE3ZO2rUJsEkwy0mSaDl6ZCCVfF%2BeT6MLHPQpHxNJT7TCCIzcmoDbB4E5YB76o7sxCbjfqjOv&acctmode=0&pass_ticket=HJbuTZU%2B4yy%2BO4Vyq5dSnE9mpGPq%2ByXGhFnFlE5RPOPKFuZv1jReqjhgixr7AP3K&wx_header=0#rd


#%%>>>>>>>>>>>>>>  1. 熵 (Entropy)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# 生成一个随机的100x100灰度图像
np.random.seed(0)
original_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# 定义不同的压缩级别，模拟通过降低分辨率来实现
compression_levels = [1, 2, 4, 8, 16]  # 1 表示无压缩，16 表示最大压缩

# 保存不同压缩级别的图像和熵值
compressed_images = []
entropies = []

for level in compression_levels:
    compressed_image = (original_image // level) * level
    compressed_images.append(compressed_image)

    # 计算熵
    pixel_counts = np.bincount(compressed_image.flatten(), minlength=256)
    probs = pixel_counts / np.sum(pixel_counts)
    entropies.append(entropy(probs, base=2))

# 绘制原始图像和不同压缩级别的图像
plt.figure(figsize=(12, 6))
for i, img in enumerate(compressed_images):
    plt.subplot(1, len(compression_levels), i+1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Compression: {compression_levels[i]}x')
    plt.axis('off')
plt.suptitle('Compressed Images at Different Levels')
plt.show()

# 绘制压缩级别与熵的关系
plt.figure(figsize=(8, 5))
plt.plot(compression_levels, entropies, marker='o')
plt.title('Compression Level vs Entropy')
plt.xlabel('Compression Level')
plt.ylabel('Entropy (bits)')
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>  2. 相对熵 / Kullback-Leibler 散度 (KL Divergence)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import kl_div

# 设定随机种子以确保结果可重复
np.random.seed(0)

# 生成两个不同的正态分布数据集
mu1, sigma1 = 0, 1  # 平均值和标准差
mu2, sigma2 = 2, 1.5  # 平均值和标准差

# 生成样本数据
data1 = np.random.normal(mu1, sigma1, 1000)
data2 = np.random.normal(mu2, sigma2, 1000)

# 计算每个分布的概率密度函数（PDF）
x = np.linspace(-5, 5, 1000)
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)

# 计算KL散度
def kl_divergence(p, q):
    return np.sum(kl_div(p, q))

# 归一化PDF
pdf1_normalized = pdf1 / np.sum(pdf1)
pdf2_normalized = pdf2 / np.sum(pdf2)

# 计算KL散度
kl_div = kl_divergence(pdf1_normalized, pdf2_normalized)

# 绘制数据直方图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data1, kde=True, color='blue', label='Data1', stat='density')
sns.histplot(data2, kde=True, color='red', label='Data2', stat='density')
plt.title('Histograms of Data1 and Data2')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# 绘制PDF曲线
plt.subplot(1, 2, 2)
plt.plot(x, pdf1, color='blue', label='PDF1')
plt.plot(x, pdf2, color='red', label='PDF2')
plt.fill_between(x, pdf1, alpha=0.3, color='blue')
plt.fill_between(x, pdf2, alpha=0.3, color='red')
plt.title(f'Probability Density Functions (KL Divergence: {kl_div:.4f})')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>  3. 互信息 (Mutual Information)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

# 生成虚拟数据集
np.random.seed(0)
n_samples = 1000
x = np.random.uniform(-5, 5, size=n_samples)
y = np.sin(x) + np.random.normal(scale=0.5, size=n_samples)

# 散点图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.5, c='blue', edgecolors='w', s=20)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 计算互信息
# 为了计算互信息，我们需要离散化连续数据
discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
x_binned = discretizer.fit_transform(x.reshape(-1, 1)).flatten()
y_binned = discretizer.fit_transform(y.reshape(-1, 1)).flatten()

mi = mutual_info_classif(x_binned.reshape(-1, 1), y_binned, discrete_features=True)

# 热力图
plt.subplot(1, 2, 2)
mi_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        mi_matrix[i, j] = mi[0] if i == j else 0

sns.heatmap(mi_matrix, annot=True, cmap='viridis', cbar=True)
plt.title('Mutual Information Heatmap')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  4. 交叉熵 (Cross-Entropy)
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
num_samples = 1000
num_classes = 3

# 真实标签（one-hot 编码）
true_labels = np.eye(num_classes)[np.random.choice(num_classes, num_samples)]

# 模型1的预测概率
model1_preds = true_labels * 0.8 + (1 - true_labels) * 0.2
model1_preds = model1_preds + np.random.normal(0, 0.05, model1_preds.shape)
model1_preds = np.clip(model1_preds, 0.01, 0.99)  # 防止概率为0或1

# 模型2的预测概率
model2_preds = true_labels * 0.6 + (1 - true_labels) * 0.4
model2_preds = model2_preds + np.random.normal(0, 0.05, model2_preds.shape)
model2_preds = np.clip(model2_preds, 0.01, 0.99)

# 计算交叉熵损失
def cross_entropy_loss(preds, labels):
    return -np.sum(labels * np.log(preds)) / labels.shape[0]

loss_model1 = cross_entropy_loss(model1_preds, true_labels)
loss_model2 = cross_entropy_loss(model2_preds, true_labels)

# 打印损失
print(f"Model 1 Loss: {loss_model1:.4f}")
print(f"Model 2 Loss: {loss_model2:.4f}")

# 图1：模型1和模型2的预测概率分布直方图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(model1_preds.flatten(), bins=30, alpha=0.7, label='Model 1')
plt.hist(model2_preds.flatten(), bins=30, alpha=0.7, label='Model 2')
plt.title('Prediction Probability Distribution Histogram')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.legend()

# 图2：每个样本的交叉熵损失分布
individual_losses_model1 = -np.sum(true_labels * np.log(model1_preds), axis=1)
individual_losses_model2 = -np.sum(true_labels * np.log(model2_preds), axis=1)

plt.subplot(1, 2, 2)
plt.hist(individual_losses_model1, bins=30, alpha=0.7, label='Model 1')
plt.hist(individual_losses_model2, bins=30, alpha=0.7, label='Model 2')
plt.title('Cross-Entropy Loss Distribution per Sample')
plt.xlabel('Cross-Entropy Loss')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  5. 信息增益 (Information Gain)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# 创建虚拟数据集
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'Feature_1': np.random.rand(n_samples),
    'Feature_2': np.random.rand(n_samples),
    'Feature_3': np.random.randint(0, 2, n_samples),
    'Feature_4': np.random.randint(0, 3, n_samples)
})
y = np.random.randint(0, 2, n_samples)

# 训练决策树分类器
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 绘制决策树结构图
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Class 0', 'Class 1'])
plt.title('Decision Tree Structure')
plt.show()

# 提取和绘制特征重要性图
importances = clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()




#%%>>>>>>>>>>>>>>  6. 贝叶斯信息准则 (BIC, Bayesian Information Criterion)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# 生成虚拟数据集
np.random.seed(42)
X = 2 - 3 * np.random.normal(0, 1, 1000)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 1000)

# 将X变为二维数组
X = X[:, np.newaxis]

# 线性回归模型
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# 二次回归模型
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# 计算BIC
def calculate_bic(y, y_pred, num_params):
    n = len(y)
    mse = mean_squared_error(y, y_pred)
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

bic_linear = calculate_bic(y, y_pred_linear, linear_model.coef_.shape[0] + 1)
bic_poly = calculate_bic(y, y_pred_poly, poly_model.coef_.shape[0] + 1)

# 可视化数据和拟合结果
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, s=10)
plt.plot(X, y_pred_linear, color='r', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, s=10)
plt.plot(X, y_pred_poly, color='g', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()

plt.show()

# 可视化BIC值
models = ['Linear', 'Polynomial']
bic_values = [bic_linear, bic_poly]

plt.figure(figsize=(8, 6))
plt.bar(models, bic_values, color=['red', 'green'])
plt.xlabel('Model')
plt.ylabel('BIC')
plt.title('BIC for Different Models')
plt.show()

print(f"BIC for Linear Model: {bic_linear}")
print(f"BIC for Polynomial Model: {bic_poly}")



#%%>>>>>>>>>>>>>>  7. 最小描述长度 (MDL, Minimum Description Length)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 生成数据
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 1000)
y = 2 * np.sin(x) + 0.5 * x + np.random.normal(0, 0.5, x.shape)

# 模型定义
def linear_model(x, a, b):
    return a * x + b

def sine_model(x, a, b, c):
    return a * np.sin(b * x) + c

# 拟合线性模型
params_linear, _ = curve_fit(linear_model, x, y)
y_linear_fit = linear_model(x, *params_linear)

# 拟合正弦模型
params_sine, _ = curve_fit(sine_model, x, y)
y_sine_fit = sine_model(x, *params_sine)

# 计算残差
residuals_linear = y - y_linear_fit
residuals_sine = y - y_sine_fit

# 计算MDL
def mdl(y_true, y_pred, num_params):
    residual = y_true - y_pred
    residual_sum_of_squares = np.sum(residual**2)
    mdl_value = 0.5 * np.log(residual_sum_of_squares / len(y_true)) + (num_params / len(y_true))
    return mdl_value

mdl_linear = mdl(y, y_linear_fit, len(params_linear))
mdl_sine = mdl(y, y_sine_fit, len(params_sine))

# 绘制图形
plt.figure(figsize=(15, 10))

# 数据与模型拟合
plt.subplot(3, 1, 1)
plt.scatter(x, y, label='Data', color='black')
plt.plot(x, y_linear_fit, label='Linear Fit', color='blue')
plt.plot(x, y_sine_fit, label='Sine Fit', color='red')
plt.legend()
plt.title('Data and Model Fits')

# 残差分布
plt.subplot(3, 1, 2)
plt.scatter(x, residuals_linear, label='Linear Residuals', color='blue')
plt.scatter(x, residuals_sine, label='Sine Residuals', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.title('Residuals')

# MDL 比较
plt.subplot(3, 1, 3)
plt.bar(['Linear Model', 'Sine Model'], [mdl_linear, mdl_sine], color=['blue', 'red'])
plt.title('MDL Comparison')
plt.ylabel('MDL Value')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  8. Fano’s Inequality

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import entropy

# 生成虚拟数据集
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=2)

# 计算分类器的错误率（这里假设有不同的错误率）
error_rates = np.linspace(0.01, 0.5, 50)

# 计算分类器信息熵
def compute_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

entropies = compute_entropy(error_rates)

# 计算 Fano’s Inequality
def fano_inequality(error_rate):
    return (1 + error_rate) * np.log2(1 + error_rate) - error_rate * np.log2(error_rate) - (1 - error_rate) * np.log2(1 - error_rate)

fano_values = fano_inequality(error_rates)

# 创建图形
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# 图形 1: 分类器的错误率与信息熵的关系
axs[0].plot(error_rates, entropies, marker='o', linestyle='-', color='b')
axs[0].set_title('Classifier Error Rate vs. Information Entropy')
axs[0].set_xlabel('Error Rate')
axs[0].set_ylabel('Information Entropy')
axs[0].grid(True)

# 图形 2: 分类器的错误率与 Fano’s Inequality 计算值的关系
axs[1].plot(error_rates, fano_values, marker='o', linestyle='-', color='r')
axs[1].set_title('Classifier Error Rate vs. Fano’s Inequality')
axs[1].set_xlabel('Error Rate')
axs[1].set_ylabel('Fano’s Inequality Value')
axs[1].grid(True)

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  9. Fisher信息 (Fisher Information)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保可重复性
np.random.seed(42)

def generate_data(mu, sigma, n):
    return np.random.normal(mu, sigma, n)

def calculate_fisher_information(n, sigma):
    I_mu = n / (sigma ** 2)
    I_sigma2 = n / (2 * sigma ** 4)
    return I_mu, I_sigma2

# 参数设置
mu = 0
sigma = 1
sample_sizes = [30, 100, 500]
num_samples = 10  # 每个样本大小下生成10个样本

# 创建图形和坐标轴
fig, axes = plt.subplots(len(sample_sizes), 2, figsize=(14, 5 * len(sample_sizes)))
axes = axes.reshape(-1, 2)

for i, n in enumerate(sample_sizes):
    # 生成数据
    data_samples = [generate_data(mu, sigma, n) for _ in range(num_samples)]

    # 绘制数据分布图
    sns.histplot(data_samples, bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Data Distribution with Sample Size {n}')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    # 计算Fisher信息
    I_mu, I_sigma2 = calculate_fisher_information(n, sigma)

    # 绘制Fisher信息图
    axes[i, 1].bar(['I_mu', 'I_sigma2'], [I_mu, I_sigma2], color=['blue', 'green'])
    axes[i, 1].set_title(f'Fisher Information with Sample Size {n}')
    axes[i, 1].set_ylabel('Fisher Information')

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  10. Shannon’s Noisy Channel Coding Theorem
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置随机种子
np.random.seed(0)

# 定义函数生成信号和噪声
def generate_signal(n_samples):
    return np.random.randint(0, 2, n_samples)  # 二进制信号

def add_noise(signal, snr_dB):
    noise_std = 10**(-snr_dB / 20)
    noise = noise_std * np.random.randn(len(signal))
    return signal + noise

def calculate_ber(original_signal, received_signal):
    return np.mean(original_signal != (received_signal > 0.5))

# 参数设置
n_samples = 2000
snr_dB_range = np.arange(0, 21, 2)  # 从0到20的SNR值，每次递增2
ber_values = []

# 计算不同SNR下的误码率
for snr_dB in snr_dB_range:
    signal = generate_signal(n_samples)
    noisy_signal = add_noise(signal, snr_dB)
    ber = calculate_ber(signal, noisy_signal)
    ber_values.append(ber)

# 绘制SNR与BER的关系图
plt.figure(figsize=(12, 6))

# 图1: SNR vs BER
plt.subplot(1, 2, 1)
plt.semilogy(snr_dB_range, ber_values, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER')
plt.grid(True)

# 图2: 实际接收信号与理论接收信号的对比
snr_dB_example = 10
signal_example = generate_signal(n_samples)
noisy_signal_example = add_noise(signal_example, snr_dB_example)

plt.subplot(1, 2, 2)
plt.hist(signal_example, bins=2, alpha=0.6, label='Original Signal')
plt.hist(noisy_signal_example, bins=30, alpha=0.6, label='Noisy Signal')
plt.xlabel('Signal Value')
plt.ylabel('Frequency')
plt.title(f'Signal Distribution (SNR = {snr_dB_example} dB)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()












































































































































































































































































































































































