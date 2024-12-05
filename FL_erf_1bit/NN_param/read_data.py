
# -*- coding: utf-8 -*-
"""
Created on 2024/08/26

@author: Junjie Chen

"""


import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch

def data_tf_mnist(x):
    ## 1
    x = transforms.ToTensor()(x)
    x = (x - 0.5) / 0.5
    x = x.reshape((-1, 28, 28))

    return x

class DatasetSplit(TensorDataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

##>>>>>>>>>>>>>>>>>>>>>>>> MNIST ##################################
def mnist_iid(train_data_size, num_users):
    shard_size = train_data_size // num_users   # 300
    random_order = np.random.permutation(train_data_size)
    dict_users = {}
    for user_id in range(num_users):
        dict_users[user_id] = random_order[user_id * shard_size : (user_id + 1) * shard_size]
    return dict_users

def mnist_noniid(train_data_size, labels, num_clients):
    shard_per_user = 2
    num_shards = num_clients * shard_per_user    # 200
    shard_size = train_data_size // num_shards   # 300

    sorted_idx = labels.argsort()
    shards_idx = np.random.permutation(train_data_size // shard_size)
    dict_users = {i: np.array([], dtype = int) for i in range(num_clients)}
    for i in range (num_clients):
        idx_rand = shards_idx[i * shard_per_user : (i + 1)*shard_per_user]
        for r in idx_rand:
            dict_users[i] = np.hstack((dict_users[i], sorted_idx[r*shard_size:(r+1)*shard_size]))
    return dict_users

def get_MNIST(args):
    train_set = datasets.MNIST(root = args.dir_data, train = True, download = True, transform = data_tf_mnist)
    test_set = datasets.MNIST(root = args.dir_data, train = False, download = True, transform = data_tf_mnist)
    testloader = DataLoader(test_set, batch_size = args.test_bs, shuffle = True, )
    labels = np.array(train_set.targets)
    ## data size
    train_data_size = len(train_set.data)  # 60000
    # test_data_size  = len(test_set.data)   # 10000

    if args.IID:
        print(">>> [The Data Partition is IID......]")
        dict_users = mnist_iid(train_data_size, args.num_of_clients)
    else:
        print(">>> [The Data Partition is non-IID......]")
        dict_users = mnist_noniid(train_data_size, labels, args.num_of_clients)

    ## Train data
    train_data = ((train_set.data/255.0 - 0.5)/0.5).reshape(-1,1,28,28)
    local_dt_dict = {}
    for user_id in range (args.num_of_clients):
        local_dt_dict[user_id] = TensorDataset(train_data[dict_users[user_id]], torch.tensor(labels[dict_users[user_id]]))
    ## or
    # local_dt_dict = {}
    # for user_id in range (args.num_of_clients):
    #     local_dt_dict[user_id] = DatasetSplit(train_set, dict_users[user_id])
    return local_dt_dict, testloader

##>>>>>>>>>>>>>>>>>>>>>>>> cifar10 ##################################

def cifar10_iid(train_data_size, num_users):
    shard_size = train_data_size // num_users   # 300
    random_order = np.random.permutation(train_data_size)
    dict_users = {}
    for user_id in range(num_users):
        dict_users[user_id] = random_order[user_id * shard_size : (user_id + 1) * shard_size]
    return dict_users

def cifar10_noniid(train_data_size, labels, num_clients):
    shard_per_user = 2
    num_shards = num_clients * shard_per_user    # 200
    shard_size = train_data_size // num_shards   # 300

    sorted_idx = labels.argsort()
    shards_idx = np.random.permutation(train_data_size // shard_size)
    dict_users = {i: np.array([], dtype = int) for i in range(num_clients)}
    for i in range (num_clients):
        idx_rand = shards_idx[i * shard_per_user : (i + 1)*shard_per_user]
        for r in idx_rand:
            dict_users[i] = np.hstack((dict_users[i], sorted_idx[r*shard_size:(r+1)*shard_size]))
    return dict_users

def get_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root = args.dir_data, train = True, download = True, transform = transform_train)
    test_set = datasets.CIFAR10(root = args.dir_data, train = False, download = True, transform = transform_test)
    testloader = DataLoader(test_set, batch_size = args.test_bs, shuffle = True, )
    labels = np.array(train_set.targets)
    ## data size
    train_data_size = len(train_set.data)  # 60000
    # test_data_size  = len(test_set.data)   # 10000

    if args.IID:
        print(">>> [The Data Partition is IID......]")
        dict_users = mnist_iid(train_data_size, args.num_of_clients)
    else:
        print(">>> [The Data Partition is non-IID......]")
        dict_users = mnist_noniid(train_data_size, labels, args.num_of_clients)

    trainloader, testloader = {}, {}
    for user_id in range(args.num_of_clients):
        trainloader[user_id] = DataLoader(DatasetSplit(train_set, dict_users[user_id]), batch_size = args.local_bs, shuffle=True)
        testloader = DataLoader(test_set, batch_size = args.test_bs, shuffle = False)
    return trainloader, testloader


##>>>>>>>>>>>>>>>>>>>>>>>> get data ##################################
def GetDataSet(args):
    if args.dataset.lower() == 'mnist':
        print(f">>> [{args.dataset} Dataset Is Used for FL......]")
        local_dt_dict, testloader = get_MNIST(args)
    elif args.dataset.lower() == 'cifar10':
        print(f">>> [{args.dataset} Dataset Is Used for FL......]")
        local_dt_dict, testloader = get_cifar10(args)
    return local_dt_dict, testloader











































