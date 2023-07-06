import torch
import torchvision
from torch import nn
import sys
import time
import numpy as np

def load_data_fashion_mnist(batch_size, resize=None, root='~/公共的/MLData/FashionMNIST'):
    """Use torchvision.datasets module to download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

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


def try_gpu(i=0):
    """Return gpu device if exists, otherwise return cpu device."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def train(net, train_iter, test_iter, batch_size, optimizer, loss, num_epochs, device= try_gpu()):
    net = net.to(device)
    timer = Timer_lu()
    print("training on", device)
    print(f"len(train_iter) = {len(train_iter)}")

    # in one epoch, it will iterate all training samples
    for epoch in range(num_epochs):
        # Accumulator has 3 parameters: (loss, train_acc, number_of_images_processed)
        metric =  Accumulator(3)
        # all training samples will be splited into batch_size
        print(f"Epoch = {epoch}\n")
        for batch, (X, y) in enumerate(train_iter):
            timer.start()
            # set the network in training mode
            net.train()
            # move data to device (gpu)
            X = X.to(device)
            y = y.to(device)
            #print(f"y = {y}")
            y_hat = net(X)
            l = loss(y_hat, y)
            #print(f"X.shape = {X.shape}, y.shape = {y.shape}.y_hat.shape = {y_hat.shape}, l = {l}")
            # X.shape = torch.Size([128, 1, 96, 96]), y.shape = torch.Size([128]).y_hat.shape = torch.Size([128, 10]), l = 2.303614854812622
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                acc =  accuracy(y_hat, y)
                metric.add(l * X.shape[0], acc, X.shape[0])

            ttmp = timer.stop()
            # metric[0] = l * X.shape[0], metric[2] = X.shape[0]
            train_l = metric[0]/metric[2]
            # metric[1] = number of correct predictions, metric[2] = X.shape[0]
            train_acc = metric[1]/metric[2]
            if (batch+1)%20 == 0:
                print("    Epoch:%d/%d, batch:%d/%d, loss:%.3f, acc:%.3f, time:%.3f(min)"% (epoch+1, num_epochs, batch+1, len(train_iter), l, acc, ttmp/60.0))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        if epoch % 1 == 0:
            print(f'epoch {epoch + 1} : loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    # after training, calculate images/sec
    # variable `metric` is defined in for loop, but in Python it can be referenced after for loop
    print(f"total training time {timer.sum()/60.0:.2f}(min), {metric[2] * num_epochs / timer.sum():.2f} images/sec on {str(device)}")




def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    # Set the model to evaluation mode
    net.eval()  
    if not device:
        device = next(iter(net.parameters())).device
    # Accumulator has 2 parameters: (number of correct predictions, number of predictions)
    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


class Timer_lu:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()



class Accumulator:
    """For accumulating sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)