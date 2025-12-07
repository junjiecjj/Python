#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:14:50 2025

@author: jack
"""


#  系统库
import numpy as np
import os, sys
import torch, torchvision

def data_tf_mlp_mnist(x):
    ## 1
    # x = transforms.ToTensor()(x)
    # x = (x - 0.5) / 0.5
    # x = x.reshape((-1,))

    ## 2
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

def data_tf_cnn_mnist(x):
    ## 1
    x = torchvision.transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))

    # 2
    # x = np.array(x, dtype='float32') / 255
    # x = (x - 0.5) / 0.5
    # x = x.reshape((1, 28, 28))  # ( 1, 28, 28)
    # x = torch.from_numpy(x)
    return x


class DataGenerator(object):
    def __init__(self, args, datasetname):
        print( f"\n#================================ DataLoader {datasetname} 开始准备 =======================================\n")
        # if args.wanttrain:
        if datasetname == "MNIST":
            # data_tf = torchvision.transforms.Compose([ # torchvision.transforms.Resize(28),
            #                                            torchvision.transforms.ToTensor(),
            #                                            # torchvision.transforms.Normalize([0.5], [0.5])
            #                                           ])
            root = args.dir_minst
            trainset = torchvision.datasets.MNIST(root = root, train = True, download = True, transform = data_tf_cnn_mnist)

        elif datasetname == "CIFAR10":
            data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(64), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5], [0.5]) ])
            root = args.dir_cifar10
            print(f"dataset = {datasetname}")
            trainset = torchvision.datasets.CIFAR10(root = root, train = True, download = True, transform = data_tf)
        else:
            print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")
            raise ValueError('数据集不存在.')


        self.loader_train = torch.utils.data.DataLoader(
            trainset,
            batch_size = args.batch_size,
            shuffle = True,
            pin_memory = 0,
            num_workers=  6,
        )

        # ==================================================== 测试数据集 =======================================================
        self.loader_test = []
        # if  args.wanttest:
        if datasetname == "MNIST":
            data_tf = torchvision.transforms.Compose([ torchvision.transforms.Resize(28),
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize([0.5], [0.5])
                                                      ])
            root = args.dir_minst
            testset = torchvision.datasets.MNIST(root = root, train = False, download = True, transform = data_tf_cnn_mnist)

        elif datasetname == "CIFAR10":
            data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(64),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize([0.5], [0.5])])
            root = args.dir_cifar10
            print(f"dataset = {datasetname}")
            testset = torchvision.datasets.CIFAR10(root = root, train = False, download = True, transform = data_tf)
        else:
            pass

        test_iter = torch.utils.data.DataLoader(
            testset,
            batch_size = args.test_bs,
            shuffle = False,
            num_workers = 6,)

        self.loader_test = test_iter

        print( f"\n#================================ DataLoader {datasetname} 准备完毕 =======================================\n")
        return


#load_data_fashion_mnist()方法返回训练集和测试集。
def load_data_mnist(batch_size, resize = 28, root = '~/SemanticNoise_AdversarialAttack/Data/MNIST'):
    """Use torchvision.datasets module to download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    trans.append(torchvision.transforms.Normalize([0.5], [0.5]))

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def load_data_cifar10(batch_size, resize=32, root='~/SemanticNoise_AdversarialAttack/Data/CIFAR10'):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True,  transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True,  transform=transform_test)

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
























