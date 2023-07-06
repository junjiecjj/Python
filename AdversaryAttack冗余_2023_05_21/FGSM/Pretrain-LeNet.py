#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:43:54 2023

@author: jack
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse
import socket, getpass , os

# 获取当前系统主机名
host_name = socket.gethostname()
# 获取当前系统用户名
user_name = getpass.getuser()
# 获取当前系统用户目录
user_home = os.path.expanduser('~')
home = os.path.expanduser('~')




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



def train(args, model, device, train_loader, optimizer, Loss, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('    Train Epoch: {} [{:*>5d}/{} ({:0>6.2%})]\t Loss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    return



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(f"pred.shape = {pred.shape}, target.shape = {target.shape}, {target.view_as(pred).shape}")
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--dir_minst', type = str, default = home+'/SemanticNoise_AdversarialAttack/Data/Minst', help = 'dataset directory')  # cjj
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
    tr_train = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
    tr_test = transforms.Compose([ transforms.ToTensor(),  ])
    train_data = datasets.MNIST(args.dir_minst, train = True, download = True, transform = tr_train)
    test_data = datasets.MNIST(args.dir_minst, train = False, download = True, transform = tr_test)

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    print(f"len(train_loader) = {len(train_loader)}, len(test_loader) = {len(test_loader)}, {len(train_loader.dataset)} {len(test_loader.dataset)}")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    Loss = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch : {epoch+1}/{args.epochs}({100.0*(epoch+1)/args.epochs:0>5.2f}%)")
        train(args, model, device, train_loader, optimizer, Loss, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # 保存网络中的参数, 速度快，占空间少
    if args.save_model:
        torch.save(model.state_dict(), "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/mnist_LeNet.pt")   # 训练和测试都归一化
        torch.save(model.state_dict(), "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/mnist_LeNet1.pt")  # 只有训练归一化
        torch.save(model.state_dict(), "/home/jack/SemanticNoise_AdversarialAttack/LeNet_AlexNet/mnist_LeNet2.pt")  # 都不归一化


if __name__ == '__main__':
    main()




































