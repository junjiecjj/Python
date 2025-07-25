




#%% 8. 高斯过程（Gaussian Processes）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 生成数据
np.random.seed(1)
X = np.linspace(0, 10, 100) # (100,)
y = np.sin(X) + 0.5 * np.random.randn(100) # (100,)

# 训练数据
X_train = X[::5][:, np.newaxis] # (20, 1)
y_train = y[::5] #  (20,)

# 定义高斯过程模型
kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 拟合模型
gp.fit(X_train, y_train)

# 预测
X_pred = np.linspace(0, 10, 1000)[:, np.newaxis] # (1000, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True) # (1000,), (1000,)

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

#%% https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-iris-py
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)

h = 0.02  # step size in the mesh

kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
kernel = 1.0 * RBF([1.0, 1.0])
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

titles = ["Isotropic RBF", "Anisotropic RBF"]
plt.figure(figsize=(10, 5))
for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
    # Plot the predicted probabilities. For that, we will assign a color to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(
        "%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta))
    )

plt.tight_layout()
plt.show()


#%% https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_xor.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-xor-py
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct

xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# fit the model
plt.figure(figsize=(10, 5))
kernels = [1.0 * RBF(length_scale=1.15), 1.0 * DotProduct(sigma_0=1.0) ** 2]
for i, kernel in enumerate(kernels):
    clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X, Y)

    # plot the decision function for each datapoint on the grid
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)

    plt.subplot(1, 2, i + 1)
    image = plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=["k"])
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.colorbar(image)
    plt.title(
        "%s\n Log-Marginal-Likelihood:%.3f"
        % (clf.kernel_, clf.log_marginal_likelihood(clf.kernel_.theta)),
        fontsize=12,
    )

plt.tight_layout()
plt.show()


#%% https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247492478&idx=1&sn=4faf3572374c6c8c0274f64c362fc7cd&chksm=9a1df3f99e0b89662f0d93895942f4dde3f61f2ba5435768e98197a41adcd3a10fcccfbb1531&mpshare=1&scene=1&srcid=0224K5QFb3pn7veOQnZZTZVI&sharer_shareinfo=5d89d6fb55da7773bf826f150124b95d&sharer_shareinfo_first=ddb1ad27516d45facaf51e5f1db6dd5c&exportkey=n_ChQIAhIQ7lmrHdoDhCdcqwLSv29ChhKfAgIE97dBBAEAAAAAAEoTB3SByDwAAAAOpnltbLcz9gKNyK89dVj0YAhQdTjX9B2GpdARf28%2F1gEZTPwP6bMMWYyBHvbjN1K2ESBGqF9O1LV9Q6lV0qnNaNCzjYMgvPh6%2BvNYsbxATN%2Bur2xMEaHhU8BW%2FZEVQoKcS39tSZcc2kxhAQHKzQYyfsZB1%2Fwi9tyRNkIdAOQpciGzPlylVfRd0OpHd7%2BhUVxMgLuKqL3mvhVZnPVVPvP1nG7NldaWkB%2BCwIuDL%2BV3TTm3zAGFLZRJcjnQ1%2FPA0VRWuBZn%2Ba9IYYNe3nke2pXWRikj0SZ%2FpcT0Wh%2F6mn%2FN43tByoi95EL6m2AxrDu7N%2BXaf21Ku6nkid6U%2BJ9tOGYlBb8HJaZpIpQ2&acctmode=0&pass_ticket=ysYJxrDKREyWalk6uWKhXi3SKrbRDnwdlYxbl4%2F1Q8svVPKcE2FAuvThE5OjtzpE&wx_header=0#rd



import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义真实函数（正弦函数）
def true_function(x):
    return torch.sin(2 * np.pi * x)

# 生成虚拟训练数据：在 [0, 1] 区间均匀取样，并加入噪声
n_train = 200
train_x = torch.linspace(0, 1, n_train)
noise_std = 0.2
train_y = true_function(train_x) + noise_std * torch.randn(train_x.size())

# 生成测试数据：用于预测和绘图
n_test = 500
test_x = torch.linspace(0, 1, n_test)

# 定义高斯过程模型（使用 gpytorch 的 ExactGP）
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # 均值函数：采用常数均值
        self.mean_module = gpytorch.means.ConstantMean()
        # 协方差函数：采用 RBF 核，并用 ScaleKernel 包装
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 初始化似然函数与模型
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# 将模型设置为训练模式
model.train()
likelihood.train()

# 定义优化器（Adam）以及边缘对数似然损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# 记录训练过程中的损失
training_loss = []
n_iter = 100  # 训练迭代次数

# 训练过程
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)  # 我们最小化负对数边缘似然
    loss.backward()
    optimizer.step()
    training_loss.append(loss.item())
    if (i+1) % 10 == 0:
        print(f'Iteration {i+1}/{n_iter} - Loss: {loss.item():.3f}')

# 训练完成，切换到评估模式
model.eval()
likelihood.eval()

# 预测：在测试数据上计算后验分布
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_pred = likelihood(model(test_x))
    pred_mean = test_pred.mean
    pred_var = test_pred.variance
    lower, upper = test_pred.confidence_region()

# 绘制图形：4个子图放在一幅图中
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Gaussian Process Regression Example', fontsize=16)

# 子图1：训练数据与真实函数
axes[0, 0].scatter(train_x.numpy(), train_y.numpy(), color='red', s=50, label='Training Data')
axes[0, 0].plot(test_x.numpy(), true_function(test_x).numpy(), color='blue', linewidth=2, label='True Function')
axes[0, 0].set_title('Training Data and True Function', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# 子图2：GP预测均值与置信区间
axes[0, 1].plot(test_x.numpy(), pred_mean.numpy(), color='magenta', linewidth=2, label='Predicted Mean')
axes[0, 1].fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='magenta', alpha=0.3, label='95% Confidence Interval')
axes[0, 1].scatter(train_x.numpy(), train_y.numpy(), color='red', s=40, label='Training Data', zorder=10)
axes[0, 1].set_title('GP Predicted Mean and Confidence Interval', fontsize=12)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# 子图3：预测方差（不确定性）
axes[1, 0].plot(test_x.numpy(), pred_var.numpy(), color='green', linewidth=2, label='Predicted Variance')
axes[1, 0].set_title('Predicted Variance (Uncertainty)', fontsize=12)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# 子图4：训练过程中负对数边缘似然损失曲线
axes[1, 1].plot(range(1, n_iter+1), training_loss, color='orange', linewidth=2, label='Training Loss')
axes[1, 1].set_title('Negative Log Marginal Likelihood Loss Curve', fontsize=12)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Negative Log Marginal Likelihood')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()











































































































































































































































































