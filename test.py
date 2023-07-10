

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params
import copy
import torch
import torchvision
from torchvision import transforms as transforms

def mnist_noniid(train_label, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users*3)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    train_label = train_label.numpy()

    ## sort train_label
    idxs_labels = np.vstack((idxs, train_label))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    ## divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


train = torchvision.datasets.MNIST(root="/home/jack/公共的/MLData/", train=True, download=True, transform=transforms.ToTensor())
train_data = train.data.float().unsqueeze(1)
train_label = train.targets

mean = train_data.mean()
std = train_data.std()
train_data = (train_data - mean) / std

test = torchvision.datasets.MNIST(root="/home/jack/公共的/MLData/", train=False, download=True, transform=transforms.ToTensor())
test_data = test.data.float().unsqueeze(1)
test_label = test.targets
test_data = (test_data - mean) / std

# split MNIST (training set) into non-iid data sets
non_iid = []
num_users = 10
user_dict = mnist_noniid(train_label, num_users)

for i in range(num_users):
    idx = user_dict[i]
    d = train_data[idx]
    targets = train_label[idx].float()
    non_iid.append((d, targets))
non_iid.append((test_data.float(), test_label.float()))



































