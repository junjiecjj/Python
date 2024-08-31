
"""




"""

#%%>>>>>>>>>>>>>>>>>>>>>>
# 逻辑回归，一个强大算法模型！
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247483792&idx=1&sn=a7d4c8b718d3ae979569aa1fd161e70b&chksm=c0e5db56f7925240579336264b761e8d35faa1b0f6549a7a5d9adc6db498d3cdf8e9162c1770&mpshare=1&scene=1&srcid=0726br4rnsI37hbe7jmfVkY6&sharer_shareinfo=b04350edfa67ac3fb84559d456879741&sharer_shareinfo_first=b04350edfa67ac3fb84559d456879741&exportkey=n_ChQIAhIQ%2BfV4SY5T2H6IHFbYowLoFhKfAgIE97dBBAEAAAAAAGyiMHuQBmcAAAAOpnltbLcz9gKNyK89dVj0%2Bkt3TDhNtFBSKM2eGzpzFc8h3%2B0zRakVT1S81gmNxt5VgDUzF3KEcjBRSH3TxnJagcDIg8k2RiQvy3YcC9VuZvJ7UGOcI7eh%2B6U16cCjHqGpg8KE1Y6cbTKz6WU3p2YQ4W1lMpMN90XK5mqKgYA0l7rGBCz5hKQTtX1DYte1%2BIxBpQ%2BCs%2F40qtGNPBkG3DTPqaaLQFZG4ZHOiePynzjPVsa8XN5jzvWoF0nRGTdXTNTBTouZR%2Bg1kJy%2FKsH%2BDXhPHBBUa76myBPjuNkt%2F4%2BrYbsVOShZ59jRuTHR8N7xFD6NBTHKbpa5NcbKi2hFQClBipDmRDmyYaKK&acctmode=0&pass_ticket=HhCVQdbJnetdzTglynwVoxMHpNLj99qTW4wcHxvVnwBxaSRmD6CfdO%2Bl2rM%2BA00J&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt

# 定义逻辑函数
def logistic_function(x):
    return 1 / (1 + np.exp(-x))

# 生成一些随机数据
x = np.linspace(-6, 6, 100)
y = logistic_function(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Logistic Function', color='blue')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Logistic Regression Function')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>> 通透！逻辑回归 7 个核心点！！

# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247485474&idx=1&sn=8c1138feaf02d5bbb70250abe6910398&chksm=9a996de958b26b4a707562fedd73b9339735583ed00407d93c3458f67c37fa6a0c9bb354d387&mpshare=1&scene=1&srcid=0829buF9Vqr2ivWIxqdMmnTD&sharer_shareinfo=96c2e17a4e9a0410e4ab0b21ea1b7543&sharer_shareinfo_first=96c2e17a4e9a0410e4ab0b21ea1b7543&exportkey=n_ChQIAhIQ7l4GtlsTWO7dC%2Fg1q30mxxKfAgIE97dBBAEAAAAAAKS4Krm1wwEAAAAOpnltbLcz9gKNyK89dVj0%2FP5s8oyed2Hl%2BddtX63V5A4%2Bxoj0Mp98xCOrR6I3CVvrWJCsOOpvmWXMHa%2FPs3%2ByDqwWwcdPKcp62NFpzbTaNGloAq5rgSaIW%2BVZhikus6yNLeicGt1%2BXXSFpINonQcwjjVvF%2BmEB%2BBbdMDwnSOzAxAsfc8P07vaVmDRggUAVhgRH00%2F4zX1KtnyeXaKHTOnZfdi0sH%2B4Rnm00%2FzGJB%2FEn7Ks%2Fb4K%2FansJDTF5%2ByhUQST40bdTBO0UFHyyNH%2BpbOI4kZQReVfEj8DO7Xtef%2BUhX5Wu6S3QzfcqtH4uOFOLtZyBlSTL71%2F8nzAwwsZRXlAD7ILSTGj38E&acctmode=0&pass_ticket=cBooBtud0mGnd93wvD%2FK1n79keOlIZj2o21qOLtMZlSHFevScPxJsLHPgxm9r8bI&wx_header=0#rd


# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 创建虚拟的二分类数据集
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放（使用Z-score标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归模型
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')

# 绘制决策边界
plt.figure(figsize=(8, 6))

# 绘制训练集样本
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.Paired, marker='o', edgecolors='k')

# 绘制测试集样本
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=plt.cm.Paired, marker='x', edgecolors='k', s=200)

# 绘制决策边界
h = .02  # 步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))

# 将决策边界绘制出来
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', linewidths=1)

plt.title('Logistic')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>> 深挖一个强大算法模型，逻辑回归！！
# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247486597&idx=1&sn=ffbd8b859ec39683ecad28e93d1ee622&chksm=9a61feb47982b887644ae8471b8f5257e23fd8acd1fd487f6be95baa311a26149a71aa47ae55&mpshare=1&scene=1&srcid=0829gWCF4zswiaUUxa5oy5Qk&sharer_shareinfo=54aa34de806177099b348c1a6cdd357e&sharer_shareinfo_first=54aa34de806177099b348c1a6cdd357e&exportkey=n_ChQIAhIQPMhH8umMjZBrWqLFtJ%2F8dRKYAgIE97dBBAEAAAAAAK1VAHkC%2FPYAAAAOpnltbLcz9gKNyK89dVj0i7ESUzltSX8a0SWnYzmeDZTVgoaMWhD8EaQ2B1dJH1t9TEcesN8aiL%2BRfmE%2F033H%2FLgi%2B2kqHr9Ms%2FMpmrLx2VVMWzbXJJ8ceKX7agC7kUh0B7rdC60UVmQMzJgRLyERIom2HdHjZiLLhbmT3FIepgmkVoiS7cOZ0XxsXdpXouFGvLBmt2aBg9ANzIlDlUj0l8kkMEqcJDX7uxfYlCNXKUgi77L4CYKiopvsTuelp%2FDS%2F3zr9bpTa%2F2fbpituIedpiu1u4LZ8f2P20C6e%2BW8zRbGHDyREDm%2F52AvxG7o1wwvOlwPhmeOvWSnnQhlJqoH2G4%3D&acctmode=0&pass_ticket=DSS2wypQKqOhb2sVujLJ53rKq%2BkLV373Cc76ciLrlNl9xT2dvwiHykw%2F%2FvAeLEoc&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成一些示例数据
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.random.choice([0, 1], 100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用逻辑回归进行分类（添加L2正则化）
model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 绘制sigmoid函数的图像
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_values, y_values)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')

# 绘制逻辑回归的决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>> 逻辑回归，一个强大的算法模型！！

# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247485724&idx=1&sn=c2db8b754eb20a92c562ff7b6f3b644a&chksm=9a9f93c5c556831d7fe74621ebd9ac1f964d47ffd3ba840059e06ca69ffca72b9622d20cb6b5&mpshare=1&scene=1&srcid=0829LqtvF4VyGmcduCixzjEs&sharer_shareinfo=fea8cca8deea9fafc222970179539eb8&sharer_shareinfo_first=fea8cca8deea9fafc222970179539eb8&exportkey=n_ChQIAhIQXfo6Imk38MsBWd0owYj7aBKfAgIE97dBBAEAAAAAAI2KDodEvJkAAAAOpnltbLcz9gKNyK89dVj0E7zctjU1Huk4Q5Ljmb3viVzf07IYKEfRC9rhkuIhc8idUlGH3IVOdYNqgSjeWKKhs0Bg3aEVP47FYkIVI%2FZaWt3dPvDVAqm1%2FixuewfiMfjp8lDyt5a%2Biu3qm%2BZecB7ybSr5skFUSewzvOtiht0iWgHvJ399LjPu31XsmyHIqx0f1NK9BlbXSnS%2BxhGstZ%2FBEMC0e9L9J7yZ3rjjwregSn541WvFRhP2w6pwycfwwqMaAih7qrRGF%2BJApnHop4TTMg0qdi%2FqwD7ifqiao3fzuKD17azo0Gx3hL3JDrCGDC4hh%2F4jnKrb5BNj3m5HScs5RpSP6wNEPb4J&acctmode=0&pass_ticket=XqCiIEL3jAetL1t82V6FQBQ%2Bk7Zwb0iOsU7xm%2F0BF%2Buy472YCUy%2Bb4nrduDoQgGP&wx_header=0#rd
#


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成分类数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# 创建逻辑回归分类器
logreg = LogisticRegression()

# 使用数据集训练模型
logreg.fit(X, y)

# 绘制数据点和决策边界
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)

# 生成网格点来绘制决策边界
h = 0.02  # 步长
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.colorbar()
plt.show()




















































































































