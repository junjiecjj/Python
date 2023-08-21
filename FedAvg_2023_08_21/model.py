#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:54:24 2023

@author: jack
"""







import torch
from torchvision import models
import math


from models.Mnist_2NN  import Mnist_2NN
from models.Mnist_CNN  import Mnist_CNN
from models.WideResNet import WideResNet



def get_model(name = "mnist_cnn",  ):
    if name == "resnet18":
        model = models.resnet18( )
    elif name == "resnet50":
        model = models.resnet50()
    elif name == "densenet121":
        model = models.densenet121()
    elif name == "alexnet":
        model = models.alexnet()
    elif name == "vgg16":
        model = models.vgg16()
    elif name == "vgg19":
        model = models.vgg19()
    elif name == "inception_v3":
        model = models.inception_v3()
    elif name == "googlenet":
        model = models.googlenet()
    elif name == 'mnist_2nn':
        model = Mnist_2NN()
    elif name == 'mnist_cnn':
        model = Mnist_CNN()
    elif name == 'wideResNet':
        model = WideResNet(depth=28, num_classes=10)

    return model



def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
    #    print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)
