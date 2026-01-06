#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
# from models.randaug import RandAugment

def get_model(model_name, dataset, img_size, nclass):
    if model_name == 'vggnet':
        from models import vgg
        model = vgg.VGG('VGG11', num_classes=nclass)
        
    elif model_name == 'resnet':
        from models import resnet
        model = resnet.ResNet18(num_classes=nclass)
    elif model_name == 'resnet-s':
        from resnet_s import resnet20
        model = resnet20()

    elif model_name == 'wideresnet':
        from models import wideresnet
        model = wideresnet.WResNet_cifar10(num_classes=nclass, depth=16, multiplier=4)
        
    elif model_name == 'cnnlarge':
        from models import simple
        model = simple.CNNLarge()
        
    elif model_name == 'convmixer':
        from models import convmixer
        model = convmixer.ConvMixer(n_classes=nclass)
    
    elif model_name == 'cnn':
        from models import simple
        
        if dataset == 'mnist':
            model = simple.CNNMnist(num_classes=nclass, num_channels=1)
        elif dataset == 'fmnist':
            model = simple.CNNFashion_Mnist(num_classes=nclass)
        elif dataset == 'cifar':
            model = simple.CNNCifar(num_classes=nclass)
    elif model_name == 'ae':
        from models import simple
        
        if dataset == 'mnist' or dataset == 'fmnist':
            model = simple.Autoencoder()
         
    elif model_name == 'mlp':
        from models import simple

        len_in = 1
        for x in img_size:
            len_in *= x
            model = simple.MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=nclass)
    else:
        exit('Error: unrecognized model')

    return model


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar10' or 'cifar100':
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
#             transforms.RandAugment(num_ops=2, magnitude=14),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
#         transform_train.transforms.insert(0, RandAugment(2, 14))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if args.dataset == 'cifar10':
            data_dir = '../data/cifar/'

            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=transform_train)

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=transform_test)
            
            num_classes = 10
        elif args.dataset == 'cifar100':    
            data_dir = '../data/cifar100/'

            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                           transform=transform_train)

            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                          transform=transform_test)
        
            num_classes = 100
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args)
    
        

    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        num_classes = 10

        
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
                
        

    return train_dataset, test_dataset, num_classes, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_parameter_delta(ws, w0):
    w_avg = copy.deepcopy(ws[0])
    for key in range(len(w_avg)):
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(0, len(ws)):
            # w_avg[key] += ws[i][key] - w0[key]
            w_avg[key] += ws[i][key]
        w_avg[key] = torch.div(w_avg[key], len(ws))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z


def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z


def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z


def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z
