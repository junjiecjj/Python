#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:43:25 2023

@author: jack

https://github.com/Rainwind1995/FGSM

https://blog.csdn.net/wyf2017/article/details/119676908
"""

# 这句话的作用:即使是在Python2.7版本的环境下，print功能的使用格式也遵循Python3.x版本中的加括号的形式
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator


sys.path.append("../")

# 本项目自己编写的库
from  trainers.common import FGSM_draw_image, plotXY


filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


# https://pytorch123.com/FourSection/AdversarialExampleGene/


def data_tf_cnn_mnist(x):
    # ## 1
    # x = transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1, 28, 28))

    # 2
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))  # (-1, 28*28)
    x = x.reshape((1, 28, 28))  # ( 1, 28, 28)
    x = torch.from_numpy(x)
    return x


# 定义LeNet模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



# FGSM算法攻击代码
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad

    # print(f"image.shape = {image.shape}, sign_data_grad.shape = {sign_data_grad.shape}, perturbed_image.shape = {perturbed_image.shape}")
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # 精度计数器
    correct = 0
    adv_examples = []

    # 循环遍历测试集中的所有示例
    for data, target in test_loader:

        # 把数据和标签发送到设备
        data, target = data.to(device), target.to(device)

        # 设置张量的requires_grad属性，这对于攻击很关键
        data.requires_grad = True

        # 通过模型前向传递数据
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # 如果初始预测是错误的，不打断攻击，继续
        if init_pred.item() != target.item():
            continue

        # 计算损失
        loss = F.nll_loss(output, target)

        # 将所有现有的渐变归零
        model.zero_grad()

        # 计算后向传递模型的梯度
        loss.backward()

        # 收集datagrad
        data_grad = data.grad.data
        #print(f"data.shape = {data.shape}, data_grad.shape = {data_grad.shape}")

        # 唤醒FGSM进行攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # 重新分类受扰乱的图像
        output = model(perturbed_data)

        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if final_pred.item() == target.item():
            correct += 1
            # 保存0 epsilon示例的特例
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # 稍后保存一些用于可视化的示例
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # 计算这个epsilon的最终准确度
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {}/{} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # 返回准确性和对抗性示例
    return final_acc, adv_examples



# 设置不同扰动大小
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# 是否使用cuda
use_cuda = True
# 定义我们正在使用的设备
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

# 预训练模型
pretrained_model = "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/mnist_LeNet1.pt"
# 初始化网络
model = Net().to(device)

# 加载已经预训练的模型(没gpu没cuda支持的时候加载模型到cpu上计算)
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))



# 声明 MNIST 测试数据集何数据加载
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/公共的/MLData/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])),
    batch_size=1, shuffle=True)


# 在评估模式下设置模型。在这种情况下，这适用于Dropout图层
model.eval()


accuracies = []
examples = []

# 对每个epsilon运行测试
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# accuracies = [0.9635, 0.9456, 0.9259, 0.8944, 0.8535, 0.8056, 0.7427]


plotXY(epsilons, accuracies, xlabel = r"$\mathrm{\epsilon}$", ylabel = "Accuracy", title = "Accuracy vs Epsilon", legend = "Y vs. X", figsize = (5, 5), savepath = "/home/jack/snap/", savename = "hh2")
FGSM_draw_image(len(epsilons), len(examples[0]), epsilons, examples,  savepath = "/home/jack/snap/", savename = "FGSM_samples2")

### 在每个epsilon上绘制几个对抗样本的例子
# cnt = 0
# plt.figure(figsize=(8, 10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons), len(examples[0]), cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig, adv, ex = examples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.show()


# cnt = 0
# # plt.figure(figsize=(8, 10))
# fig, axs = plt.subplots(len(epsilons),len(examples[0]), figsize=(8, 10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         orig, adv, ex = examples[i][j]
#         axs[i, j].set_title("{} -> {}".format(orig, adv))
#         axs[i, j].imshow(ex, cmap="gray")
#         axs[i, j].set_xticks([])  # #不显示x轴刻度值
#         axs[i, j].set_yticks([] ) # #不显示y轴刻度值
#         if j == 0:
#             axs[i, j].set_ylabel("Eps: {}".format(epsilons[i]), fontsize=14)

# plt.tight_layout()
# plt.show()



























