




#%% https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html#sphx-glr-auto-examples-gaussian-process-plot-gpc-iris-py
##
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


#%% 8. 高斯过程（Gaussian Processes）
# 最强总结，十大贝叶斯算法 ！
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484679&idx=1&sn=789522244120467c83b298fe90e86b10&exportkey=n_ChQIAhIQitTAf1FLYIHCaVO%2FvI8MPRKfAgIE97dBBAEAAAAAADpwMoVdpJsAAAAOpnltbLcz9gKNyK89dVj0YFO9Qc0gpA4LxgGcaE88syT78YwDmNDQqdGLZfGVFZ3OVCe48BF0RLb7kY0r4WoBkP865X9Kzy5PFgGT9Y7iLzkilHH90wghfq2U8RbFqOXwCahPBLBeSvWDNnLD2hML8Lb773eSBbWJGjR7p8FkD6YZxbLQjtUC67KIhKeVSlYlp2u9kCmcchBnaqBZjaNzm16gpAgc7xYm7xBqzQJTLzm1wKMpCqPf0LV7K%2BqJD0XCMdpZ63Wi2W276hIp0pdM5vS7nKesgW6DqA0yhRZcMVoheT41F53K5hzwWX8zU7qyTDZn9we1UvMJFCGZexidfpxtE8OHpog%2F&acctmode=0&pass_ticket=%2BEMPsVAURy%2F8H628HeinbwUZpzF%2Bic9DzIenzb6U3EsAb3zSE0W6X%2FJDVz793ZD5&wx_header=0

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
X_pred = np.linspace(0, 10, 1000)[:, np.newaxis]      # (1000, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)   # (1000,), (1000,)

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
plt.close()


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
    pred_mean = test_pred.mean # 500
    pred_var = test_pred.variance # 500
    lower, upper = test_pred.confidence_region() # 500, 500

# 绘制图形：4个子图放在一幅图中
fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout = True)
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

plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8. 高斯过程回归插值
# 十大数据差值方法
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488088&idx=1&sn=95c453fb33771526bf80adb49115b74a&chksm=c0e5c89ef792418893e3f14de1b2abc7dde28f8c8ea3c5d4e14ab4d177eab7efc0e3434ebc31&cur_album_id=3445855686331105280&scene=190#rd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. 生成虚拟数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(30, 1), axis=0)  # 输入数据
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # 真实值 + 噪声

# 测试点
X_test = np.linspace(0, 5, 1000).reshape(-1, 1)

# 2. 配置高斯过程回归模型
# 使用常数核和RBF核的乘积
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# 3. 拟合数据
gp.fit(X, y)

# 4. 预测
y_pred, sigma = gp.predict(X_test, return_std=True)

# 5. 计算残差
y_train_pred, _ = gp.predict(X, return_std=True)
residuals = y - y_train_pred

# 6. 绘图
plt.figure(figsize=(12, 8))

# 第一张图：原始数据与预测插值曲线
plt.subplot(2, 1, 1)
plt.plot(X_test, y_pred, 'b-', label='Prediction (mean)', lw=2)  # 预测均值曲线
plt.fill_between(X_test.ravel(), y_pred - 2*sigma, y_pred + 2*sigma, color='lightgray', alpha=0.5, label='Confidence interval (±2σ)')  # 置信区间
plt.scatter(X, y, c='r', s=50, zorder=10, label='Observed data')  # 真实数据点
plt.title('Gaussian Process Regression with Confidence Interval')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend(loc='upper left')

# 第二张图：残差分析
plt.subplot(2, 1, 2)
plt.scatter(X, residuals, c='purple', s=50, zorder=10)
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.title('Residuals Analysis')
plt.xlabel('Input')
plt.ylabel('Residuals (True - Predicted)')

plt.tight_layout()
plt.show()



#%% Gaussian Processes regression: basic introductory example

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


X = np.linspace(start=0, stop=10, num=1000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))
fig, ax = plt.subplots( figsize=(8, 6), constrained_layout = True)
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("True generative process")
plt.show()
plt.close()

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
fig, ax = plt.subplots( figsize=(8, 6), constrained_layout = True)
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.scatter(X_train, y_train, label="Observations")
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("Gaussian process regression on noise-free dataset")
plt.show()
plt.close()


# Example with noisy targets
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)


gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

fig, ax = plt.subplots( figsize=(8, 6), constrained_layout = True)
ax.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
ax.errorbar( X_train, y_train_noisy, noise_std, linestyle="None", color="tab:blue", marker=".", markersize=10, label="Observations", )
ax.plot(X, mean_prediction, label="Mean prediction")
ax.fill_between( X.ravel(), mean_prediction - 1.96 * std_prediction, mean_prediction + 1.96 * std_prediction, color="tab:orange", alpha=0.5, label=r"95% confidence interval",)
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("Gaussian process regression on a noisy dataset")
plt.show()
plt.close()


#%% Illustration of prior and posterior Gaussian process for different kernels


import matplotlib.pyplot as plt
import numpy as np


def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(x, single_prior, linestyle="--", alpha=0.7, label=f"Sampled function #{idx + 1}",)
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.1, color="black", label=r"$\pm$ 1 std. dev.",)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])
    return

# Dataset and Gaussian process generation
rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5

############# Radial Basis Function kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, constrained_layout = True, figsize=(10, 12))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Radial Basis Function kernel", fontsize=18)

print(f"Kernel parameters before fit:\n{kernel})")
print(f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")


#%% Rational Quadratic kernel
from sklearn.gaussian_process.kernels import RationalQuadratic

kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 12))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Rational Quadratic kernel", fontsize=18)
plt.tight_layout()

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

#%% Exp-Sine-Squared kernel
from sklearn.gaussian_process.kernels import ExpSineSquared

kernel = 1.0 * ExpSineSquared(
    length_scale=1.0,
    periodicity=3.0,
    length_scale_bounds=(0.1, 10.0),
    periodicity_bounds=(1.0, 10.0),
)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Exp-Sine-Squared kernel", fontsize=18)
plt.tight_layout()

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)


#%% Dot-product kernel
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct

kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
    DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Dot-product kernel", fontsize=18)
plt.tight_layout()

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

#%% Matérn kernel
from sklearn.gaussian_process.kernels import Matern

kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Matérn kernel", fontsize=18)
plt.tight_layout()

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)






































































































































































































