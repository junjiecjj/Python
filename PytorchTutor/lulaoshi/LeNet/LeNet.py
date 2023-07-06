#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:49:23 2022

@author: jack

https://lulaoshi.info/machine-learning/convolutional/lenet


"""



import pandas as pd
import numpy as np
import torch, torchvision, sys
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import os , sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lrs
import collections
import matplotlib.pyplot as plt
import imageio
# 自己的库
from option import args
from data_generator import DataGenerator
import common


#  nn.Conv2d的输入必须为4维的(N,C,H,W)


#网络模型结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 输入 1 * 28 * 28
        self.conv = nn.Sequential(
            # 卷积层1
            # 在输入基础上增加了padding，28 * 28 -> 32 * 32
            # 1 * 32 * 32 -> 6 * 28 * 28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
            # 6 * 28 * 28 -> 6 * 14 * 14
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            # 卷积层2
            # 6 * 14 * 14 -> 16 * 10 * 10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),
            # 16 * 10 * 10 -> 16 * 5 * 5
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # 全连接层1
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),
            # 全连接层2
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        # print(f"img.shape = {img.shape}\nfeature.shape = {feature.shape}\noutput.shape = {output.shape}\nfeature.view(img.shape[0], -1).shape = {feature.view(img.shape[0], -1).shape}")
        """
        img.shape = torch.Size([16, 3, 48, 48])
        feature.shape = torch.Size([256, 16, 5, 5])
        output.shape = torch.Size([256, 10])

        feature.view(img.shape[0], -1).shape = torch.Size([256, 400])
        """
        return output

#网络模型结构
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        # 输入 1 * 28 * 28
        # 卷积层1
        # 在输入基础上增加了padding，28 * 28 -> 32 * 32
        # 1 * 32 * 32 -> 6 * 28 * 28
        self.conv1 = common.conv2d_prelu(in_channels=3, out_channels=16, kernel_size=5,stride=2, pad=4)


        # 6 * 28 * 28 -> 6 * 14 * 14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # kernel_size, stride
        # 卷积层2
        # 6 * 14 * 14 -> 16 * 10 * 10
        self.conv2 = common.conv2d_prelu(in_channels=16, out_channels=32, kernel_size=5, stride=2, pad=2)

        self.conv3 = common.conv2d_prelu(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad=2)

        self.conv4 = common.conv2d_prelu(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad=1)

        self.conv5 = common.conv2d_prelu(in_channels=32, out_channels=3, kernel_size=5, stride=1, pad=1)

        # 16 * 10 * 10 -> 16 * 5 * 5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1
        self.l1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # 全连接层2
        self.l2 = nn.Linear(in_features=120, out_features=84)


        self.l3 = nn.Linear(in_features=84, out_features=10)

        self.deconv1 = common.convTrans2d_prelu(3, 32, 5, 1, 1)
        self.deconv2 = common.convTrans2d_prelu(32, 32, 5, 1, 1)
        self.deconv3 = common.convTrans2d_prelu(32, 32, 5, 1, 2)
        self.deconv4 = common.convTrans2d_prelu(32, 16, 5, 2, 2)
        self.deconv5 = common.convTrans2d_prelu(16, 3, 6, 2, 3)



    def forward(self, img):
        #print(f"107 img.shape = {img.shape}")
        # img.shape = torch.Size([16, 3, 48, 48])

        e1 = self.conv1(img)
        #print(f"e1.shape = {e1.shape}")
        # e1.shape = torch.Size([16, 16, 26, 26])

        #e2 = self.pool1(e1)
        #print(f"e2.shape = {e2.shape}")

        e3 = self.conv2(e1)
        #print(f"e3.shape = {e3.shape}")
        # e3.shape = torch.Size([16, 32, 13, 13])

        #e4 = self.pool2(e3)
        #print(f"e4.shape = {e4.shape}")

        e5 = self.conv3(e3)
        #print(f"e5.shape = {e5.shape}")
        # e5.shape = torch.Size([16, 32, 13, 13])

        e6 = self.conv4(e5)
        #print(f"e6.shape = {e6.shape}")
        # e6.shape = torch.Size([16, 32, 11, 11])

        e7 = self.conv5(e6)
        #print(f"e7.shape = {e7.shape}")
        # e7.shape = torch.Size([16, 3, 9, 9])

        d1 = self.deconv1(e7)
        #print(f"d1.shape = {d1.shape}")
        # d1.shape = torch.Size([16, 32, 11, 11])

        d2 = self.deconv2(d1)
        #print(f"d2.shape = {d2.shape}")
        # d2.shape = torch.Size([16, 32, 13, 13])

        d3 = self.deconv3(d2)
        #print(f"d3.shape = {d3.shape}")
        # d3.shape = torch.Size([16, 32, 13, 13])

        d4 = self.deconv4(d3)
        #print(f"d4.shape = {d4.shape}")
        # d4.shape = torch.Size([16, 16, 25, 25])

        d5 = self.deconv5(d4)
        #print(f"d5.shape = {d5.shape}")
        # d5.shape = torch.Size([16, 3, 48, 48])

        return d5



#训练模型

#load_data_fashion_mnist()方法返回训练集和测试集。
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


#在训练过程中，我们希望看到每一轮迭代的准确度，构造一个evaluate_accuracy方法，计算当前一轮迭代的准确度（模型预测值与真实值之间的误差大小）：
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        #print(f"len(data_iter) = {len(data_iter)}")
        # X.shape = torch.Size([256, 1, 28, 28]), y.shape = torch.Size([256]),
        for X, y in data_iter:
            #print(f"X.shape = {X.shape}, y.shape = {y.shape}, ")
            # len(data_iter) = 40
            if isinstance(net, torch.nn.Module):
                # set the model to evaluation mode (disable dropout)
                net.eval()
                # get the acc of this batch
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # change back to train mode
                net.train()

            n += y.shape[0]
    return acc_sum / n



def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 原本的训练函数
def train_origin(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device=try_gpu()):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        if epoch % 10 == 0:
            # 我认为这里不应该除以batch_count
            print(f'epoch {epoch + 1} : loss {train_l_sum / batch_count:.3f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')

def make_optimizer( net):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    #  filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    # trainable = filter(lambda x: x.requires_grad, net.parameters())
    trainable =  net.parameters()
    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': 0.0003, 'weight_decay': 0 }

    # optimizer = ADAM

    optimizer_class = optim.Adam
    kwargs_optimizer['betas'] = (0.9, 0.999)
    kwargs_optimizer['eps'] =  1e-8

    # scheduler, milestones = 0,   gamma = 0.5
    milestones =  [ 20,  80,   120]  # [200]
    kwargs_scheduler = {'milestones': milestones, 'gamma': 0.6}  # args.gamma =0.5
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

        def reset_state(self):
            self.state = collections.defaultdict(dict)
            self.scheduler.last_epoch = 0
            for param_group in self.param_groups:
                param_group["lr"] = 0.1

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer



class LOSS(nn.modules.loss._Loss):
    def __init__(self, ):
        super(LOSS, self).__init__()
        print('Preparing loss function:')


        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in '1*MSE'.split('+'):  #  ['1*MSE']
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'CrossEntropy':
                    loss_function = nn.CrossEntropyLoss(reduction='mean')

            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function} )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.losslog = torch.Tensor()

        device = torch.device('cpu')
        self.loss_module.to(device)
        self.loss_module.half()

        # TODO
        #if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        # print(f"我正在计算loss\n")
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.losslog[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.losslog[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        #  losslog.shape = [1,len(loss)],[2,len(loss)],[2,len(loss)]...,[epoch,len(loss)]
        self.losslog = torch.cat((self.losslog, torch.zeros(1, len(self.loss))))

    def mean_log(self, n_batches):
        self.losslog[-1].div_(n_batches)
        return self.losslog[-1]

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.losslog[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c/n_samples))

        return ''.join(log)

    # 在同一个画布中画出所有Loss的结果
    def plot_AllLoss(self, apath):
        fig, axs = plt.subplots(len(self.loss),1, figsize=(12,8))

        for i, l in enumerate(self.loss):
            epoch = len(self.losslog[:, i])
            X = np.linspace(1, epoch, epoch)
            label = '{} Loss'.format(l['type'])
            if len(self.loss) == 1:
                axs.set_title(label)
                axs.plot(X, self.losslog[:, i].numpy(), label=label)
                axs.set_xlabel('Epochs')
                axs.set_ylabel(label)
                axs.grid(True)
                axs.legend()
                axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            else:
                axs[i].set_title(label)
                axs[i].plot(X, self.losslog[:, i].numpy(), label=label)
                axs[i].set_xlabel('Epochs')
                axs[i].set_ylabel(label)
                axs[i].grid(True)
                axs[i].legend()
                axs[i].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
        fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
        plt.tight_layout()#  使得图像的四周边缘空白最小化
        out_fig = plt.gcf()
        out_fig.savefig(os.path.join(apath, 'AllTrainLossPlot.pdf'))
        plt.show()
        plt.close(fig)

    # 在不同的画布中画各个损失函数的结果.
    def plot_loss(self, apath):
        epoch = len(self.losslog[:, 0])
        X = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(X, self.losslog[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'TrainLossPlot_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    # 在每个压缩率和信噪比下，所有的epoch训练完再调用保存
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'TrainLossState.pt'))
        torch.save(self.losslog, os.path.join(apath, 'TrainLossLog.pt'))


    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if  os.path.isfile(os.path.join(apath,'TrainLossState.pt')):
            self.load_state_dict(torch.load(os.path.join(apath, 'TrainLossState.pt'), **kwargs))

        if  os.path.isfile(os.path.join(apath,'TrainLossLog.pt')):
            self.losslog = torch.load(os.path.join(apath, 'TrainLossLog.pt'))

        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.losslog)): l.scheduler.step()


# # test LOSS module
# from torch.autograd import Variable
# los = LOSS(args,ckp)

# CompressRate = [1,2,3]
# SNR = [-10,-6,-2,0,2,6,10]

# for cp_idx, CP in enumerate(CompressRate):
#     for snr_idx, snr in enumerate(SNR):
#         for epoch_idx in range(20):
#             los.start_log()
#             for batch in range(20):
#                 sr = torch.randn(1,3,4,4)
#                 hr = torch.randn(1,3,4,4)
#                 lss = los(sr, hr)
#                 lss = Variable(lss, requires_grad = True)
#                 lss.backward()
#             #los.end_log(10)



# los.plot_loss(ckp.dir,)
# los.plot_AllLoss(ckp.dir,)


# los.save(ckp.dir)


# 我认为的训练函数，
def train_cjj(net, dataloder,   optimizer, loss, num_epochs, device=try_gpu()):
    #wr = SummaryWriter("/home/jack/公共的/Python/PytorchTutor/lulaoshi/checkpoint")
    torch.set_grad_enabled(True)
    net = net.to(device)
    net.train()

    print("training on", device)
    # loss = torch.nn.CrossEntropyLoss(reduction='mean')
    #loss = LOSS()


    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        #print(f"len(dataloder.loader_train) = {len(dataloder.loader_train)}\n")
        # len(train_iter) = 235

        loss.start_log()
        for batch_idx, (lr, hr, filename)  in enumerate(dataloder.loader_train):
            #print(f"{batch_idx}, lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n")
            hr = hr.to(device)
            lr = lr.to(device)
            #print(f"hr.shape = {hr.shape}, lr.shape = {lr.shape},  ")
            sr = net(hr)
            #print(f"hr.shape = {hr.shape}, lr.shape = {lr.shape}, sr.shape = {sr.shape}")
            # X.shape = torch.Size([256, 1, 28, 28]), y.shape = torch.Size([256]), sr.shape = torch.Size([256, 10])

            hr = hr/(255.0)
            sr = sr/(255.0)

            l = loss(sr, hr)  # l = 2.3009374141693115
            #l = Variable(l, requires_grad = True)
            # with torch.no_grad():
            #     for a, b, name in zip(hr, y_hat,filename):
            #         filename1 = '/home/jack/公共的/Python/PytorchTutor/lulaoshi/LeNet/image/origin/{}_hr.png'.format(name)
            #         data1 = a.permute(1, 2, 0).cpu().numpy()
            #         imageio.imwrite(filename1, data1)
            #         filename2 = '/home/jack/公共的/Python/PytorchTutor/lulaoshi/LeNet/image/net/{}_lr.png'.format(filename[0])
            #         data2 = b.permute(1, 2, 0).cpu().numpy()
            #         imageio.imwrite(filename2, data2)


            # print(f"sr.shape = {sr.shape}, l.shape = {l.shape}")
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item() #  l.cpu().item() = 2.3009374141693115
            #print(f"train_l_sum = {train_l_sum}")

        epochLos =  loss.mean_log(len(dataloder.loader_train))
        optimizer.schedule()
        lr_list2.append(optimizer.get_lr())
        wr.add_scalar('train/SumLoss',train_l_sum, epoch)
        wr.add_scalar('train/0.9*L1+0.1*MSE',epochLos[-1], epoch)
        #wr.add_scalar('train/0.9*L1',epochLos[-3], epoch)
        #wr.add_scalar('train/0.1*MSE',epochLos[-2], epoch)
        wr.add_scalar('LearningRate',optimizer.get_lr(), epoch)

        if epoch % 10 == 0:
            print(f'epoch {epoch + 1} : loss {train_l_sum :.3f}, ')
    loss1.plot_AllLoss('/home/jack/图片/')





#在整个程序的主逻辑中，设置必要的参数，读入训练和测试数据并开始训练：
#def main():
num_epochs = 200
wr = SummaryWriter("./checkpoint")


net = LeNet1()
images = torch.randn(256,3,28,28)


#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_list2 = []
optimizer = make_optimizer(net)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr)


loss1 = LOSS()

#wr.add_graph(net, images)

Dataloader = DataGenerator(args)

# for param_group in optimizer.param_groups:#在每次更新参数前迭代更改学习率
#     print(f" param_group = {type(param_group)} \n")
#     print(f" param_group['lr'] = {param_group['lr']} \n ")
#     param_group["lr"] = lr


# load data
#train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)




# train
train_cjj(net, Dataloader, optimizer, loss1, num_epochs)

wr.close()



