#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:33:21 2023

@author: jack

https://github.com/cleverhans-lab/cleverhans

https://blog.csdn.net/shuweishuwei/article/details/119823947

https://blog.csdn.net/wyf2017/article/details/119676908

https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h

https://github.com/Rainwind1995/FGSM

https://github.com/Rainwind1995/FGSM

https://github.com/poppybrown/Textcnn-adversarial-training

https://github.com/geyingli/unif/blob/master/uf/task/adversarial.py#L254

https://zhuanlan.zhihu.com/p/166364358

https://zhuanlan.zhihu.com/p/103593948

https://github.com/topics/fgsm

https://github.com/poppybrown/Textcnn-adversarial-training

pip install cleverhans

"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import cv2
from torch.autograd import Variable
#导入cleverhans中的FGSM函数
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method


#获取计算设备 默认是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#图像加载以及预处理
image_path="./baby.png"
orig = cv2.imread(image_path)[..., ::-1]
orig1 = cv2.resize(orig, (224, 224))
img = orig1.copy().astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)

img=np.expand_dims(img, axis=0)
 
img = Variable(torch.from_numpy(img).to(device).float())
print(f"img.shape = {img.shape}")
 
#使用预测模式 主要影响droupout和BN层的行为，用的是Alexnet模型，现成的
model = models.alexnet(pretrained=True, ).to(device).eval()
#取真实标签
label=np.argmax(model(img).data.cpu().numpy())#这里为什么要加cup（）？因为np无法直接转为cuda使用，要先转cpu
print("label = {}".format(label))
 
epoch = 1#训练轮次
target = 796#原始图片的标签
target = Variable(torch.Tensor([float(target)]).to(device).long())#转换数据类型
print(f"target = {target}")



def FGSM(model):
    for i in range(epoch):
        # #（模型，图片数据，扰动值，范数：np.inf、0或1）范数的作用占时不知道
        adver_example = fast_gradient_method(model, img.data, 0.0001, np.inf)
        adver_target = torch.max(model(adver_example),1)[1]
        if adver_target != target:
            print("FGSM attack 成功")
        print("epoch={} adver_target={}".format(epoch,adver_target))
    return adver_example,adver_target,'FGSM attack'


FGSM(model)









