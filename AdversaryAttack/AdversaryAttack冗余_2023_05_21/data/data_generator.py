# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen

"""

#  系统库
from importlib import import_module
# from torch.utils.data import dataloader

import sys,os

import torch, torchvision
from torch import nn, optim


# 本项目自己编写的库
sys.path.append("..")
from data import srdata
from  ColorPrint import ColoPrint
color =  ColoPrint()


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        print(color.higyellowfg_whitebg( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}, idx_scale = {idx_scale} \n"))
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)



class DataGenerator(object):
    def __init__(self, args, datasetname):
        print(color.fuchsia(f"\n#================================ DataLoader {datasetname} 开始准备 =======================================\n"))

        if args.wanttrain:
            if datasetname == "MNIST":
                data_tf = torchvision.transforms.Compose([#torchvision.transforms.Resize(28),
                                                          torchvision.transforms.ToTensor(),
                                                          #torchvision.transforms.Normalize([0.5], [0.5])
                                                          ])
                root = args.dir_minst
                trainset = torchvision.datasets.MNIST(root = root,          # 表示 MNIST 数据的加载的目录
                                                    train = True,           # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,        # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)    # 表示是否需要对数据进行预处理，none为不进行预处理
            elif datasetname == "FashionMNIST":
                data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(28),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize([0.5], [0.5])
                                                          ])
                root = args.dir_fashionminst
                trainset = torchvision.datasets.FashionMNIST(root = root,   # 表示 FashionMNIST 数据的加载的目录
                                                    train = True,           # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,        # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)    # 表示是否需要对数据进行预处理，none为不进行预处理
            elif datasetname == "CIFAR10":
                data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(64),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize([0.5], [0.5])
                                                            ])
                root = args.dir_cifar10
                print(f"dataset = {datasetname}")
                trainset = torchvision.datasets.CIFAR10(root = root,            # 表示 MNIST 数据的加载的目录
                                                    train = True,               # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,            # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)        # 表示是否需要对数据进行预处理，none为不进行预处理
            else:
                pass

            self.loader_train = torch.utils.data.DataLoader(
                # MyConcatDataset(datasets),
                trainset,
                batch_size = args.batch_size,  # 16
                shuffle = True,
                pin_memory = args.cpu,
                num_workers=  args.n_threads,
            )

        #==================================================== 测试数据集 =======================================================
        self.loader_test = []
        if  args.wanttest:
            if datasetname == "MNIST":
                data_tf = torchvision.transforms.Compose([ # torchvision.transforms.Resize(28),
                                                          torchvision.transforms.ToTensor(),
                                                          # torchvision.transforms.Normalize([0.5], [0.5])
                                                          ])
                root = args.dir_minst
                testset = torchvision.datasets.MNIST(root = root,          # 表示 MNIST 数据的加载的目录
                                                    train = False,           # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,        # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)    # 表示是否需要对数据进行预处理，none为不进行预处理
            elif datasetname == "FashionMNIST":
                data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(28),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize([0.5], [0.5])])
                root = args.dir_fashionminst
                testset = torchvision.datasets.FashionMNIST(root = root,   # 表示 FashionMNIST 数据的加载的目录
                                                    train = False,           # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,        # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)    # 表示是否需要对数据进行预处理，none为不进行预处理
            elif datasetname == "CIFAR10":
                data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize(64),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize([0.5], [0.5])])
                root = args.dir_cifar10
                print(f"dataset = {datasetname}")
                testset = torchvision.datasets.CIFAR10(root = root,            # 表示 MNIST 数据的加载的目录
                                                    train = False,               # 表示是否加载数据库的训练集，false的时候加载测试集
                                                    download = True,            # 表示是否自动下载 MNIST 数据集
                                                    transform = data_tf)        # 表示是否需要对数据进行预处理，none为不进行预处理
            else:
                pass

            test_iter = torch.utils.data.DataLoader(
                testset,
                batch_size = args.test_batch_size,
                shuffle = True,
                num_workers = args.n_threads,)

            self.loader_test.append(
                test_iter,
            )
        print(color.fuchsia(f"\n#================================ DataLoader {datasetname} 准备完毕 =======================================\n"))
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

#load_data_fashion_mnist()方法返回训练集和测试集。
def load_data_fashion_mnist(batch_size, resize = 28, root = '~/SemanticNoise_AdversarialAttack/Data/FashionMNIST'):
    """Use torchvision.datasets module to download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    trans.append(torchvision.transforms.Normalize([0.5], [0.5]))

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
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
























