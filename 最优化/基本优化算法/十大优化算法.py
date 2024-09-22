#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:04:55 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484520&idx=1&sn=c2c25b442734b44ddfa412a1129f8dca&chksm=c1fa6683ca294107ef4816e6add6a6422cc0bbb593e0c64b5c64adecafd09abe82e05939450a&mpshare=1&scene=1&srcid=0812uv9NtQrWpJQ6prUy2C6J&sharer_shareinfo=39a4c9769f55194efb3a702d8cb1281d&sharer_shareinfo_first=6bc86de1b2f94b09b94707b820ffe899&exportkey=n_ChQIAhIQKrh0y7hnY1su%2FG0qKUy7YRKfAgIE97dBBAEAAAAAAJlBAfWZUzkAAAAOpnltbLcz9gKNyK89dVj0ARjnCEJReDN%2BekOxv9jldjPmruLo8eMl4ISbYnjrBD54DjpPj6pPM3eufDEONXWO7dWBQlqEgZgul99Yh7vsovX3APoqj1gQLYjVSO0%2FXHbvbA71b96Igkv87KoHI%2BtG3oEvjoM2NH0wpSTHBHmpJsjin%2FJp1vP0DfGhip%2FV52b%2BkLxyiHQdQiQTflKAWV%2F9%2FL3WeOm8etKlUzFwOdZZ2Q51LKjUfBFxPPI86cZwDrqK1OLwr6lpfg3JV5mhOQA8%2BxtTTm8TU79AcnoivDARJwvs1DflypQd04%2BsFaVrgVmTDEJndKDXwLf3sbPgebp1sOnNYlON8Pk5&acctmode=0&pass_ticket=3qZUWMNvP47Z0UY45NrWgr2ouMyo0YE60mjzpevzs4uA7ojRXiuA04r%2FlCWjD%2B%2Fh&wx_header=0#rd
"""

#%%>>>>>>>>>>>>>> 1. 梯度下降法 (Gradient Descent)

import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

# 定义梯度下降函数
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    m = len(y)
    history = {'cost': []}

    for iteration in range(iterations):
        gradients = 2/m * X.T @ (X@theta  - y)
        theta = theta - learning_rate * gradients
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = gradient_descent(X_b, y, theta_initial, learning_rate=0.1, iterations=100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()


#%%>>>>>>>>>>>>>> 2. 随机梯度下降 (Stochastic Gradient Descent, SGD)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# 生成回归数据集
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化SGD回归模型
sgd = SGDRegressor(max_iter=100, tol=10000, warm_start=True, learning_rate='constant', eta0=0.01, random_state=42)

# 记录训练过程中的信息
n_epochs = 50
train_errors, test_errors = [], []
coef_updates = []

# 训练模型
for epoch in range(n_epochs):
    sgd.fit(X_train, y_train)
    y_train_predict = sgd.predict(X_train)
    y_test_predict = sgd.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    test_errors.append(mean_squared_error(y_test, y_test_predict))
    coef_updates.append(sgd.coef_.copy())

# 绘制损失函数的收敛图
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Train Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 绘制预测结果与实际结果的比较图
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_test_predict, color='red', label='Predicted')
plt.xlabel('Input Feature')
plt.ylabel('Target')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid()
plt.show()

# 绘制参数更新的轨迹图
coef_updates = np.array(coef_updates)
plt.figure(figsize=(10, 6))
plt.plot(coef_updates, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Updates Over Epochs')
plt.grid()
plt.show()

#%%>>>>>>>>>>>>>> 3. 动量法 (Momentum)

import numpy as np
import matplotlib.pyplot as plt

# 定义二次函数及其梯度
def func(x):
    return 0.5 * x**2

def grad(x):
    return x

# 初始化参数
x = 10  # 初始点
learning_rate = 0.1
momentum = 0.9 # 动量系数
velocity = 0
num_iterations = 50

# 存储优化过程中的值
x_values = []
func_values = []

# 动量法优化
for i in range(num_iterations):
    grad_val = grad(x)
    velocity = momentum * velocity + learning_rate * grad_val
    x -= velocity

    x_values.append(x)
    func_values.append(func(x))

# 绘制优化过程中的函数值变化
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(func_values, 'b-o')
plt.title('Function Value during Optimization')
plt.xlabel('Iteration')
plt.ylabel('Function Value')

plt.subplot(1, 2, 2)
plt.plot(x_values, 'r-o')
plt.title('x Value during Optimization')
plt.xlabel('Iteration')
plt.ylabel('x Value')

plt.tight_layout()
plt.show()

# 绘制函数和优化轨迹
x_range = np.linspace(-10, 10, 400)
y_range = func(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Function: $0.5x^2$')
# plt.scatter(x_values, func_values, color='red', label='Optimization Path', zorder=5)
plt.plot(x_values, func_values, 'r-o', label='Optimization Path', zorder=5)
plt.title('Optimization Path on the Function')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.legend()
plt.show()


########## 机器学习例子
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

# 定义梯度下降函数
def gradient_descentMomentum(X, y, theta, learning_rate=0.01, iterations=100):
    # 初始化参数
    momentum = 0.9 # 动量系数
    velocity = np.array([0, 0]).reshape(-1,1)

    m = len(y)
    history = {'cost': []}
    for iteration in range(iterations):
        grad_val = 2/m * X.T @ (X@theta  - y)
        velocity = momentum * velocity + learning_rate * grad_val
        theta -= velocity

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = gradient_descentMomentum(X_b, y, theta_initial, learning_rate = 0.01, iterations = 100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>> 4. AdaGrad (Adaptive Gradient Algorithm)

import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 初始化参数
theta = np.random.randn(2, 1)
m = len(X)
X_b = np.c_[np.ones((m, 1)), X]  # 添加 x0 = 1 的偏置项

# AdaGrad 参数
eta = 0.1  # 学习率
n_iterations = 1000
epsilon = 1e-8  # 防止除零
G = np.zeros((2, 1))

# 存储每次迭代的损失值
losses = []

# 训练模型
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T @ (X_b@theta - y)
    G += gradients**2
    adjusted_gradients = eta / (np.sqrt(G) + epsilon) * gradients
    theta -= adjusted_gradients
    loss = (1 / m) * np.sum((X_b@theta - y)**2)
    losses.append(loss)

# 画出损失值下降曲线
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(n_iterations), losses, label="Training loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs. Iterations")
plt.legend()

# 可视化拟合结果
plt.subplot(1, 2, 2)
plt.plot(X, y, "b.")
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Data and Model Predictions")
plt.legend()

plt.tight_layout()
plt.show()

print("Final parameters (theta):", theta)


#%%>>>>>>>>>>>>>> 5. RMSProp (Root Mean Square Propagation)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# # 设置随机种子，以确保结果可复现
# torch.manual_seed(42)

# # 加载和预处理数据集
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# # 定义简单的神经网络模型
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, 1)
#         self.conv2 = nn.Conv2d(16, 32, 3, 1)
#         self.fc1 = nn.Linear(32 * 6 * 6, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 实例化模型和 RMSProp 优化器
# model = SimpleCNN()
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# # 定义损失函数
# criterion = nn.CrossEntropyLoss()

# # 训练模型
# epochs = 10
# losses = []

# for epoch in range(epochs):
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     epoch_loss = running_loss / len(train_loader)
#     losses.append(epoch_loss)
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# # 绘制损失变化图
# plt.figure(figsize=(8, 6))
# plt.plot(losses, label='Training Loss')
# plt.title('Training Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()


########## 机器学习例子
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

def rmsprop(X, y, theta, alpha, beta, epsilon=1e-8, iterations=100):
    m = len(y)
    history = {'cost': []}

    E_g2 = np.zeros_like(theta)
    for i in range(iterations):
        gradient = 2/m * X.T @ (X @ theta - y)
        E_g2 = beta * E_g2 + (1 - beta) * gradient ** 2
        theta = theta - alpha / (np.sqrt(E_g2 + epsilon)) * gradient
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta, history


# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

alpha = 0.1
beta = 0.2

# 运行梯度下降算法
theta_best, history = rmsprop(X_b, y, theta_initial, alpha, beta, epsilon=1e-8, iterations=100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()


#%%>>>>>>>>>>>>>> 6. Adam (Adaptive Moment Estimation)
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

def adam(X, y, theta, alpha, beta1, beta2, epsilon=1e-8, iterations=100):
    m = len(y)
    history = {'cost': []}

    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    for t in range(1, iterations + 1):
        gradient = 2/m * X.T @ (X @ theta - y)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        theta = theta - alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)
    return theta, history


# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

alpha = 0.1
beta1 = 0.2
beta2 = 0.2
# 运行梯度下降算法
theta_best, history = adam(X_b, y, theta_initial, alpha, beta1, beta2, epsilon=1e-8, iterations=100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np

# # 定义卷积神经网络
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 数据预处理和加载
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # 实例化网络和定义损失函数及优化器
# net = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# # 训练网络
# num_epochs = 10
# train_losses = []
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 2000 == 1999:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#     train_losses.append(running_loss / len(trainloader))

# print('Finished Training')

# # 保存模型
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# # 加载模型
# net = Net()
# net.load_state_dict(torch.load(PATH))

# # 在测试集上测试网络
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# # 数据分析和可视化
# # 1. 绘制训练损失
# plt.figure(figsize=(10,5))
# plt.title("Training Loss")
# plt.plot(train_losses, label="train")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # 2. 可视化一些测试图像及其预测
# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # 打印图像
# def imshow(img):
#     img = img / 2 + 0.5     # 非标准化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # 打印若干测试图像及其预测
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# # 3. 每个类的准确率
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

# # 绘制每个类的准确率
# plt.figure(figsize=(10,5))
# plt.title("Accuracy per Class")
# plt.bar(classes, [100 * class_correct[i] / class_total[i] for i in range(10)])
# plt.xlabel("Classes")
# plt.ylabel("Accuracy (%)")
# plt.show()


#%%>>>>>>>>>>>>>> 7. 牛顿法 (Newton's Method)
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数和其一阶、二阶导数
def f(x):
    return x**2 + 3*x + 2

def f_prime(x):
    return 2*x + 3

def f_double_prime(x):
    return 2

# 牛顿法优化函数
def newtons_method(x0, tol=1e-6, max_iter=100):
    x = x0
    iters = 0
    history = [x]

    while iters < max_iter:
        x_new = x - f_prime(x) / f_double_prime(x)
        history.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
        iters += 1

    return x, history

# 初始值和优化
x0 = 10
optimal_x, history = newtons_method(x0)

# 生成x值和对应的函数值用于绘图
x_vals = np.linspace(-10, 10, 400)
y_vals = f(x_vals)

# 图形1：目标函数和优化路径
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='f(x) = x^2 + 3x + 2')
plt.scatter(history, f(np.array(history)), color='red')
for i, x in enumerate(history):
    plt.text(x, f(x), f'{i}', fontsize=12, color='red')
plt.title('Newton\'s Method Optimization Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

# 图形2：迭代次数 vs 函数值
plt.subplot(1, 2, 2)
iter_nums = np.arange(len(history))
function_values = f(np.array(history))
plt.plot(iter_nums, function_values, marker='o')
plt.title('Iteration vs Function Value')
plt.xlabel('Iteration')
plt.ylabel('f(x)')

plt.tight_layout()
plt.show()


########## 机器学习例子
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)

# 绘制原始数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

# 定义梯度下降函数
def gradient_descentNewton(X, y, theta, tol=1e-6, iterations=100):
    # 初始化参数
    m = len(y)
    history = {'cost': []}
    for iteration in range(iterations):
        theta_new = theta - np.linalg.inv(2/m * X.T @ X) @ (2/m * X.T @ (X@theta  - y) )
        if np.abs(theta_new - theta).sum() < tol:
            break
        theta = theta_new
        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = gradient_descentNewton(X_b, y, theta_initial, tol=1e-6, iterations = 100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()


#%%>>>>>>>>>>>>>> 8. 共轭梯度法 (Conjugate Gradient Method)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二次函数
def quadratic_function(x):
    return 0.5 * (x[0]**2 + 10 * x[1]**2)

# 定义二次函数的梯度
def quadratic_gradient(x):
    return np.array([x[0], 10 * x[1]])

# 共轭梯度法实现
def conjugate_gradient(f, grad_f, x0, iterations=50):
    x = x0
    trajectory = [x]
    gradient = grad_f(x)
    direction = -gradient

    for _ in range(iterations):
        step_size = 0.1  # 步长
        x = x + step_size * direction
        next_gradient = grad_f(x)
        beta = np.dot(next_gradient, next_gradient) / np.dot(gradient, gradient)
        direction = -next_gradient + beta * direction
        gradient = next_gradient
        trajectory.append(x)
    return np.array(trajectory)

# 初始点和优化
x0 = np.array([3.0, 4.0])
trajectory = conjugate_gradient(quadratic_function, quadratic_gradient, x0)

# 绘制函数曲面和优化轨迹
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = quadratic_function([X, Y])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('Quadratic Function')

ax1.plot(trajectory[:, 0], trajectory[:, 1], quadratic_function(trajectory.T), color='r', marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(X, Y)')

# 绘制优化过程中的收敛曲线
iterations = len(trajectory)
steps = np.arange(iterations)

ax2 = fig.add_subplot(122)
ax2.plot(steps, trajectory[:, 0], label='X')
ax2.plot(steps, trajectory[:, 1], label='Y')
ax2.set_title('Convergence of Variables')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Variable Value')
ax2.legend()

plt.tight_layout()
plt.show()



########## 机器学习例子
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
X = 2 * np.random.rand(1000, 1)
y = 4 + 3 * X + np.random.randn(1000, 1)
# 添加偏置项 x0 = 1 到 X 中
X_b = np.c_[np.ones((1000, 1)), X]

# 共轭梯度法实现
def ConjugateGradientMethod(X, y, theta_initial, iterations=100):
    # 初始化参数
    m = len(y)
    history = {'cost': []}

    theta = theta_initial
    # trajectory = [theta]
    gradient = 2/m * X.T @ (X @ theta - y)
    direction = -gradient

    for _ in range(iterations):
        step_size = 0.1  # 步长
        theta = theta + step_size * direction
        next_gradient = 2/m * X.T @ (X @ theta - y)
        beta = ((next_gradient.T @ next_gradient) / (gradient.T @ gradient) )[0,0]
        print(f"beta = {beta}")
        direction = - next_gradient + beta * direction
        gradient = next_gradient
        # trajectory.append(theta)

        cost = np.mean((X @ theta  - y) ** 2)
        history['cost'].append(cost)

    return theta, history

# 初始化参数 theta
theta_initial = np.random.randn(2, 1)

# 运行梯度下降算法
theta_best, history = ConjugateGradientMethod(X_b, y, theta_initial, iterations = 100)

# 绘制损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(history['cost'], c='r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 绘制拟合曲线和数据散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='Original data')
plt.plot(X, X_b@theta_best , c='r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Fitted Line')
plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>> 9. BFGS (Broyden-Fletcher-Goldfarb-Shanno Algorithm)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 生成非线性数据
np.random.seed(42)
x_data = np.linspace(-10, 10, 100)
y_data = 5 * x_data**2 + 3 * x_data + 10 + np.random.normal(0, 10, size=x_data.shape)

# 定义二次多项式模型
def model(x, params):
    return params[0] * x**2 + params[1] * x + params[2]

# 定义损失函数（均方误差）
def loss(params, x, y):
    return np.mean((model(x, params) - y)**2)

# 初始参数猜测
initial_params = [1, 1, 1]

# 使用BFGS算法进行优化
result = minimize(loss, initial_params, args=(x_data, y_data), method='BFGS')

# 优化后的参数
optimized_params = result.x
print(f'Optimized parameters: {optimized_params}')

# 绘制原始数据与拟合曲线
plt.figure(figsize=(14, 7))

# 原始数据
plt.scatter(x_data, y_data, color='blue', label='Original Data')

# 拟合曲线
y_fit = model(x_data, optimized_params)
plt.plot(x_data, y_fit, color='red', label='Fitted Curve', linewidth=2)

plt.title('Original Data and Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 绘制损失函数值的收敛过程
plt.figure(figsize=(14, 7))
plt.plot(result.hess_inv.diagonal(), label='Loss Value')
plt.title('Convergence of Loss Function')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>> 10. 粒子群优化 (Particle Swarm Optimization, PSO)

import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数（最小化问题）
def objective_function(x):
    return np.sum(x**2)

# 粒子群优化函数
def pso(objective_function, dimensions, population_size=50, iterations=100, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
    # 初始化粒子群
    particles_position = np.random.uniform(-10, 10, size=(population_size, dimensions))
    particles_velocity = np.zeros((population_size, dimensions))
    personal_best_position = particles_position.copy()
    personal_best_value = np.array([float('inf')] * population_size)
    global_best_position = np.zeros(dimensions)
    global_best_value = float('inf')

    # 迭代指定次数
    for t in range(iterations):
        # 更新个体最佳
        for i in range(population_size):
            fitness = objective_function(particles_position[i])
            if fitness < personal_best_value[i]:
                personal_best_value[i] = fitness
                personal_best_position[i] = particles_position[i].copy()

        # 更新全局最佳
        for i in range(population_size):
            if personal_best_value[i] < global_best_value:
                global_best_value = personal_best_value[i]
                global_best_position = personal_best_position[i].copy()

        # 更新粒子的速度和位置
        for i in range(population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_velocity = cognitive_weight * r1 * (personal_best_position[i] - particles_position[i])
            social_velocity = social_weight * r2 * (global_best_position - particles_position[i])
            particles_velocity[i] = inertia_weight * particles_velocity[i] + cognitive_velocity + social_velocity
            particles_position[i] = particles_position[i] + particles_velocity[i]

    return global_best_position, global_best_value

# 示例用法
if __name__ == "__main__":
    dimensions = 2
    population_size = 50
    iterations = 100
    best_solution, best_value = pso(objective_function, dimensions, population_size, iterations)

    print(f"全局最优解: {best_solution}")
    print(f"全局最优值: {best_value}")

    # 可视化优化过程（如果需要）
    # 示例中，你可以绘制每次迭代中粒子的移动情况

    # 举例绘图（你可以根据具体分析需求进行定制）
    plt.figure(figsize=(8, 6))
    plt.scatter(best_solution[0], best_solution[1], color='red', marker='*', label='全局最优解')
    plt.scatter(0, 0, color='green', marker='o', label='真实最小值')
    plt.title('粒子群优化')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()



# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247488869&idx=1&sn=da676c439e96fb942ae75ecdc82411b4&chksm=9b146a8cac63e39a74b05aecd70a924ebe9b0497aa8498d253edf9e4318ce9474e0dba37e73f&mpshare=1&scene=1&srcid=0812Tr5fId9u8wgQbbX3SiXK&sharer_shareinfo=798d0355173f43ae25290b16a6097c5a&sharer_shareinfo_first=798d0355173f43ae25290b16a6097c5a&exportkey=n_ChQIAhIQguY0iS3SgPaz3FJbQupgjhKfAgIE97dBBAEAAAAAAOGeFCykjdoAAAAOpnltbLcz9gKNyK89dVj0DjWM6S7ks9jOHZc3rIQ5oDMH9nuo7jht6uXLLOGgOdC9Ep11fj%2FEtVcJ73iBrMKIEI1GjTM%2Fxemx5tN5FbsU%2BB%2Bg%2F7Aotx7R4NRCd4u7qGeGZxj0wF%2BvsqVSBbiJppxwPhwmJzv6a%2B2zQuW6Gc%2FAyv38H03xVMKh%2FpGH%2Bk8I9czN93uwmjvf8t53x7%2FXccea8SfbjErRKKmrsAiq5zWnQbbCMf3JRYqNdOcujOwBBb%2BRFZInXltgtyWR4KXBnGbVsb9VifedRxIG6RWn3XLAG674iZ8Hp86sYMmYTm%2FOLPw8A%2FZSalKnsouXXr2Che7nev2kgyT6HPKU&acctmode=0&pass_ticket=P9leDvK3%2F0jK4eblZprUySBUXaP8CFx6uJW9JynSi0N1zoJQZ%2FSBIQ1AkKMdhPbl&wx_header=0#rd



#%%>>>>>>>>>>>>>> 梯度下降算法（Gradient Descent）

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for i in range(num_iters):
        gradient = X.T @ (X @ theta - y) / m
        theta = theta - alpha * gradient
    return theta

#%%>>>>>>>>>>>>>> 动量梯度下降（Gradient Descent with Momentum）
def momentum_gradient_descent(X, y, theta, alpha, beta, num_iters):
    m = len(y)
    v = np.zeros_like(theta)
    for i in range(num_iters):
        gradient = X.T @ (X @ theta - y) / m
        v = beta * v + (1 - beta) * gradient
        theta = theta - alpha * v
    return theta

#%%>>>>>>>>>>>>>> Nesterov加速梯度（Nesterov Accelerated Gradient, NAG）
def nesterov_accelerated_gradient(X, y, theta, alpha, beta, num_iters):
    m = len(y)
    v = np.zeros_like(theta)
    for i in range(num_iters):
        temp_theta = theta - alpha * beta * v
        gradient = X.T @ (X @ temp_theta - y) / m
        v = beta * v + (1 - beta) * gradient
        theta = theta - alpha * v
    return theta

#%%>>>>>>>>>>>>>> AdaGrad（Adaptive Gradient Algorithm）
def adagrad(X, y, theta, alpha, num_iters, epsilon=1e-8):
    m = len(y)
    G = np.zeros_like(theta)
    for i in range(num_iters):
        gradient = X.T @ (X @ theta - y) / m
        G += gradient ** 2
        theta = theta - alpha / (np.sqrt(G + epsilon)) * gradient
    return theta

#%%>>>>>>>>>>>>>> RMSProp（Root Mean Square Propagation）

def rmsprop(X, y, theta, alpha, beta, num_iters, epsilon=1e-8):
    m = len(y)
    E_g2 = np.zeros_like(theta)
    for i in range(num_iters):
        gradient = X.T @ (X @ theta - y) / m
        E_g2 = beta * E_g2 + (1 - beta) * gradient ** 2
        theta = theta - alpha / (np.sqrt(E_g2 + epsilon)) * gradient
    return theta

#%%>>>>>>>>>>>>>> Adam（Adaptive Moment Estimation）
def adam(X, y, theta, alpha, beta1, beta2, num_iters, epsilon=1e-8):
    m = len(y)
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    for t in range(1, num_iters + 1):
        gradient = X.T @ (X @ theta - y) / m
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        theta = theta - alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
    return theta

#%%>>>>>>>>>>>>>> AdaDelta

def adadelta(X, y, theta, rho, num_iters, epsilon=1e-8):
    m = len(y)
    E_g2 = np.zeros_like(theta)
    E_delta_theta2 = np.zeros_like(theta)
    for i in range(num_iters):
        gradient = X.T @ (X @ theta - y) / m
        E_g2 = rho * E_g2 + (1 - rho) * gradient ** 2
        delta_theta = - (np.sqrt(E_delta_theta2 + epsilon) / np.sqrt(E_g2 + epsilon)) * gradient
        theta = theta + delta_theta
        E_delta_theta2 = rho * E_delta_theta2 + (1 - rho) * delta_theta ** 2
    return theta


#%%>>>>>>>>>>>>>> Nadam（Nesterov-accelerated Adaptive Moment Estimation）
def nadam(X, y, theta, alpha, beta1, beta2, num_iters, epsilon=1e-8):
    m = len(y)
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    for t in range(1, num_iters + 1):
        gradient = X.T @ (X @ theta - y) / m
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        m_t_hat = m_t / (1 - beta1 ** t)
        v_t_hat = v_t / (1 - beta2 ** t)
        theta = theta - alpha * ((beta1 * m_t_hat + (1 - beta1) * gradient / (1 - beta1 ** t)) / (np.sqrt(v_t_hat) + epsilon))
    return theta


#%%>>>>>>>>>>>>>> L-BFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）

from scipy.optimize import fmin_l_bfgs_b

def lbfgs(X, y, theta):
    def loss_and_grad(theta):
        loss = np.sum((X @ theta - y) ** 2) / (2 * len(y))
        gradient = X.T @ (X @ theta - y) / len(y)
        return loss, gradient

    theta, _, _ = fmin_l_bfgs_b(loss_and_grad, theta)
    return theta

#%%>>>>>>>>>>>>>> CMA-ES（Covariance Matrix Adaptation Evolution Strategy）
import cma
def cma_es(X, y, theta):
    def loss(theta):
        return np.sum((X @ theta - y) ** 2) / (2 * len(y))

    es = cma.CMAEvolutionStrategy(theta, 0.5)
    es.optimize(loss)
    return es.result.xbest





























