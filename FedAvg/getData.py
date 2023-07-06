



import numpy as np
import gzip
import os
import torchvision
from torchvision import transforms as transforms
import torch

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels # dense_to_one_hot(labels)

class GetDataSet(object):
    def __init__(self, dataSetName, isIID = False):
        self.name            = dataSetName
        self.train_data      = None  # 训练集
        self.train_label     = None  # 标签
        self.train_data_size = None  # 训练数据的大小

        self.test_data       = None  # 测试数据集
        self.test_label      = None  # 测试的标签
        self.test_data_size  = None   # 测试集数据Size

        # 如何数据集是mnist
        if self.name == 'mnist':
            # self.mnistDataSetConstruct(isIID)
            self.load_MNIST_torch(isIID)
        elif self.name == 'cifar10':
            self.load_cifar10(isIID)
        else:
            pass

    # mnistDataSetConstruct 数据重构
    def mnistDataSetConstruct(self, isIID = False):
        # 加载数据集
        data_dir = '/home/jack/公共的/MLData/MNIST/raw'
        # data_dir = r'./data/MNIST'
        # python路径拼接os.path.join() 路径变为.\data\MNIST\train-images-idx3-ubyte.gz
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path  = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path  = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]
        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        # 训练数据Size
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        ## 将图片每一张图片变成 28*28 = 784, reshape(60000, 28*28)
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        ## 归一化处理-
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)

        ## 将图片每一张图片变成 28*28 = 784
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        ## 归一化处理-
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        ## 是独立同分布
        ## 一工有60000个样本,100个客户端
        ## IID： 我们首先将数据集打乱，然后为每个Client分配600个样本。
        if isIID:
            print("is IID")
            order = np.arange(self.train_data_size)
            # numpy 中的随机打乱数据方法np.random.shuffle
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else: ## Non-IID：我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
            print("not IID")
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        #print(self.train_label[0:10])
        self.test_data = test_images
        self.test_label = test_labels
        return

    def load_MNIST_torch(self, isIID = False):
        train_tf = torchvision.transforms.Compose([transforms.ToTensor()])
        test_tf  = torchvision.transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root = "/home/jack/公共的/MLData/", train = True,  download = True, transform = train_tf)
        test_set  = torchvision.datasets.MNIST(root = "/home/jack/公共的/MLData/", train = False, download = True, transform = test_tf)

        ## 训练数据Size
        self.train_data_size = train_set.data.shape[0]  # 60000
        self.test_data_size  = test_set.data.shape[0]   # 10000

        ## 训练集
        train_data   =  train_set.data                # 训练数据 torch.Size([60000, 28, 28]), 0-255
        train_labels =  train_set.targets             # (60000)
        # 将训练集转化为（60000，28*28）矩阵
        train_images = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

        ## 测试集
        test_data    =  test_set.data         # 测试数据 torch.Size([10000, 28, 28]), 0-255
        test_labels  =  test_set.targets    # 10000
        # 将测试集转化为（10000，28*28）矩阵
        test_images  = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        ## ---------------------------归一化处理------------------------------
        train_images = train_images.type(torch.float32)
        train_images = torch.mul(train_images, 1.0 / 255.0)

        test_images = test_images.type(torch.float32)
        test_images = torch.mul(test_images, 1.0 / 255.0)
        ## -------------------------------------------------------------------------

        if isIID:
            ## 这里将 60000 个训练数据随机打乱
            order = np.random.permutation(self.train_data_size)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            ## 对数据标签进行排序
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels
        return

    def load_MNIST_np(self, isIID = False):
        train_tf = torchvision.transforms.Compose([transforms.ToTensor()])
        test_tf  = torchvision.transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root = "/home/jack/公共的/MLData/", train = True,  download = True, transform = train_tf)
        test_set  = torchvision.datasets.MNIST(root = "/home/jack/公共的/MLData/", train = False, download = True, transform = test_tf)

        ## 训练数据Size
        self.train_data_size = train_set.data.shape[0]  # 60000
        self.test_data_size  = test_set.data.shape[0]   # 10000

        ## 训练集
        train_data   = np.array(train_set.data)      # 训练数据 torch.Size([60000, 28, 28]), 0-255
        train_labels = np.array(train_set.targets, dtype=np.uint8)   # (60000)
        # 将训练集转化为（60000，28*28）矩阵
        train_images = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

        ## 测试集
        test_data = np.array(test_set.data)        # 测试数据 torch.Size([10000, 28, 28]), 0-255
        test_labels = np.array(test_set.targets)   # 10000
        # 将测试集转化为（10000，28*28）矩阵
        test_images = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

        ## ---------------------------归一化处理------------------------------
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)

        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        ## -------------------------------------------------------------------------

        self.test_data = test_images
        self.test_label = test_labels

        if isIID:
            ## 这里将50000 个训练集随机打乱
            order = np.random.permutation(self.train_data_size)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            ## 对数据标签进行排序
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        return

    # 加载cifar10 的数据
    def load_cifar10(self, isIID = False):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root = "/home/jack/公共的/MLData/CIFAR10", train = True, download = True, transform = train_transform)
        test_set = torchvision.datasets.CIFAR10(root = "/home/jack/公共的/MLData/CIFAR10", train = False, download = True, transform = test_transform)

        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # 将训练集转化为（50000，32*32*3）矩阵
        train_images = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        # 将测试集转化为（10000，32*32*3）矩阵
        test_images = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        ## ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        ## -------------------------------------------------------------------------#

        if isIID:
            # 这里将50000 个训练集随机打乱
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照标签的
            # labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序
            order = np.argsort(train_labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels
        return



# if __name__=="__main__":

# mnistDataSet = GetDataSet('mnist', 0) # test NON-IID
# print(f"{type(mnistDataSet.train_data)} | {type(mnistDataSet.test_data)} | {type(mnistDataSet.train_label)} | {type(mnistDataSet.test_label)}")
# print(f"{mnistDataSet.train_data.shape}, {mnistDataSet.train_label.shape}, {mnistDataSet.test_data.shape}, {mnistDataSet.test_label.shape}")

# if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and  type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
#     print('the type of data is numpy ndarray')
# else:
#     print('the type of data is not numpy ndarray')

# print(mnistDataSet.train_label, mnistDataSet.test_label )














