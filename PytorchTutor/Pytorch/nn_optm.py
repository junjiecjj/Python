#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:18:40 2022

@author: jack

https://pytorchbook.cn/chapter2/2.1.3-pytorch-basics-nerual-network/
本文件主要是测试优化器相关的内容;

本次探索了在不同实验场景下model.zero_grad和potimizer.zero_grad的区别和关系。这两个方法都是对网络进行梯度置零，但是在不同应用场景下有比较优的选择，总结如下：

当仅有一个model，同时optimizer只包含这一个model的参数，那么model.zero_grad和optimizer.zero_grad没有区别，可以任意使用。
当有多个model，同时optimizer包含多个model的参数时，如果这多个model都需要训练，那么使用optimizer.zero_grad是比较好的方式，耗时和防止出错上比对每个model都进行zero_grad要更好。
当有多个model，对于每个model或者部分model有对应的optimizer，同时还有一个total_optimizer包含多个model的参数时。如果是是只想训练某一个model或者一部分model，可以选择对需要训练的那个model进行model.zero_grad，然后使用他对应的optimizer进行优化。如果是想对所有的model进行训练，那么使用total_optimizer.zero_grad是更优的方式。
"""

# 首先要引入相关的包
import torch
# 引入torch.nn并指定别名
import torch.nn as nn
#打印一下版本
torch.__version__


import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        #线性层，输入1350个特征，输出10个特征
        self.fc1   = nn.Linear(1350, 10)  #这里的1350是如何计算的呢？这就要看后面的forward函数
    #正向传播
    def forward(self, x):
        print(x.size()) # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x) #根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        print(x.size()) # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2)) #我们使用池化层，计算结果是15
        x = F.relu(x)
        print(x.size()) # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        #这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        print(x.size()) # 这里就是fc1层的的输入1350
        x = self.fc1(x)
        return x
print("1-"*30)
net = Net()
print(net)


print("2-"*30)
for parameters in net.parameters():
    print(parameters)



#net.named_parameters可同时返回可学习的参数及名称。
print("3-"*30)
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())


#forward函数的输入和输出都是Tensor
print("4-"*30)
input = torch.randn(1, 1, 32, 32) # 这里的对应前面fforward的输入是32
out = net(input)
print(f"out.size() = {out.size()}")

#在反向传播前，先要将所有参数的梯度清零
net.zero_grad()
out.backward(torch.ones(1,10)) # 反向传播的实现是PyTorch自动实现的，我们只要调用这个函数即可


#在nn中PyTorch还预制了常用的损失函数，下面我们用MSELoss用来计算均方误差
y = torch.arange(0,10).view(1,10).float()
criterion = nn.MSELoss()
loss = criterion(out, y)
#loss是个scalar，我们可以直接用item获取到他的python类型的数值
print(loss.item())



#优化器
import torch.optim
out = net(input) # 这里调用的时候会打印出我们在forword函数中打印的x的大小
criterion = nn.MSELoss()
loss = criterion(out, y)
#新建一个优化器，SGD只需要要调整的参数和学习率
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
# 先梯度清零(与net.zero_grad()效果一样)
optimizer.zero_grad()
print("5-"*30)
loss.backward()
print("6-"*30)
#更新参数
optimizer.step()




#========================================================================================
# https://www.jianshu.com/p/9643cba47655?u_atoken=568ba1dd-322d-46a0-8097-8d68d1d0a859&u_asession=01yziPZpcGlZUx6sL3zb5NALh-SWMUpgwwLcUSIri9OLP8qfBnnOPMHavxg4crv3uPX0KNBwm7Lovlpxjd_P_q4JsKWYrT3W_NKPr8w6oU7K8xYXK4ZiJjOBj_JrChRp8xPpcarp92QKzyJKyYjREPlmBkFo3NEHBv0PZUm6pbxQU&u_asig=05INXw5uDoPWF7rcvX5GDmMczVXXpo52EOlLKB0UEwEx0dcIKp8AU3DEKfZZovOfEnXiTSMkbXALvFiuOwI5l3EZi8fD0cLn9JzrBPgdInFDd_EkWGfGaUoFt0zgwnLs3nRQXiXhfn5CWFimiB5FTf4pMiTkQlzWNHltxC3QkRjnD9JS7q8ZD7Xtz2Ly-b0kmuyAKRFSVJkkdwVUnyHAIJzW133U0bA1D264r16WyKRQin7ZDHhIq6RLjYTsjrhXtIWPRPQyB_SKrj-61LB_f61u3h9VXwMyh6PgyDIVSG1W8P1jhq9WAI7deW5maUWQuPMCiaxWNGs0SKRybKdVaVocS8_KYqQM2dyFvA6tiWgc6ldpZUFuHg1IvYPl7tY5TgmWspDxyAEEo4kbsryBKb9Q&u_aref=mWEbfXS7ymBmu6jwEFNvjM5k3dY%3D
#========================================================================================


import torch
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler, Adam
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import matplotlib
matplotlib.get_backend()
matplotlib.use('TkAgg')
import collections


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)


def make_optimizer( net):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    #  filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    trainable = filter(lambda x: x.requires_grad, net.parameters())

    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': 0.1, 'weight_decay': 0 }

    # optimizer = ADAM


    optimizer_class = optim.Adam
    kwargs_optimizer['betas'] = (0.9, 0.999)
    kwargs_optimizer['eps'] =  1e-8


    # scheduler, milestones = 0,   gamma = 0.5
    milestones =  [20, 40, 60, 100, 120]  # [200]
    kwargs_scheduler = {'milestones': milestones, 'gamma': 0.5}  # args.gamma =0.5
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


def train(optimizer):
    optimizer.step()

model = net()
LR = 0.01
opt = make_optimizer(model)
loss = torch.nn.CrossEntropyLoss()

lr_list1 = []
lr_list2 = []
for epoch in range(200):
     for i in range(20):
         y = torch.randint(0, 9, (10,10))*1.0
         opt.zero_grad()
         out = model(torch.randn(10,1))
         lss = loss(out, y)
         lss.backward()
         opt.step()
     opt.schedule()
     lr_list2.append(opt.get_lr())
     lr_list1.append(opt.state_dict()['param_groups'][0]['lr'])
plt.plot(range(200),lr_list1,color = 'r')
#plt.plot(range(100),lr_list2,color = 'b')
out_fig = plt.gcf()
plt.show()



#========================================================================================
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lr_list = []
for epoch in range(100):
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')
plt.show()



#========================================================================================
import numpy as np
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lambda1 = lambda epoch:np.sin(epoch) / epoch
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')
plt.show()


#========================================================================================
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')
plt.show()

#========================================================================================
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')
plt.show()





lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')





#========================================================================================
# https://zhuanlan.zhihu.com/p/352212135
#========================================================================================

# 手动设置学习率衰减1
import torch
import matplotlib.pyplot as plt

from torch.optim import Adam, lr_scheduler
import torch.nn as nn
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lr_list = []
for epoch in range(100):
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
fig = plt.figure()
plt.plot(range(100),lr_list,color = 'r')
plt.show()

# 手动设置学习率衰减2
import torch
import matplotlib.pyplot as plt

from torch.optim import *
import torch.nn as nn
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)

def adjust_learning_rate(optimiz, epoch, base_lr):
     lr = base_lr * (0.1 ** (epoch // 30))
     for param_group in optimizer.param_groups:
          param_group["lr"] = lr

     return lr


model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)

lr_list1 = []
lr_list2 = []
for epoch in range(100):
     lr = adjust_learning_rate(optimizer, epoch, LR)
     lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
     lr_list2.append(lr)
     optimizer.step()
plt.plot(range(100),lr_list1,color = 'r')
plt.plot(range(100),lr_list2,color = 'b')
plt.show()


# 手动设置学习率衰减3
import torch
import matplotlib.pyplot as plt

from torch.optim import *
import torch.nn as nn
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)

def adjust_learning_rate(epoch, base_lr):
     lr = base_lr * (0.1 ** (epoch // 30))

     return lr


model = net()
LR = 0.01


lr_list1 = []
lr_list2 = []
for epoch in range(100):
     lr = adjust_learning_rate(epoch, LR)
     optimizer = Adam(model.parameters(),lr = lr)
     lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
     lr_list2.append(lr)
     optimizer.step()
plt.plot(range(100),lr_list1,color = 'r')
plt.plot(range(100),lr_list2,color = 'b')
plt.show()


#========================================================================================

# lr_scheduler.LambdaLR
initial_lr =0.1
optimizer_1 = torch.optim.Adam(model.parameters(), lr = initial_lr)
scheduler_1 = lr_scheduler.LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch+1))
# train
print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
scheduler_1.step()
# -----------------使用示例2------------------
import numpy as np
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lambda1 = lambda epoch:np.sin(epoch) / epoch
scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')


#========================================================================================
# torch.optim.lr_scheduler.StepLR


lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')



#========================================================================================

# torch.optim.lr_scheduler.MultiStepLR
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')


#========================================================================================
# torch.optim.lr_scheduler.ExponentialLR

lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')



#========================================================================================
# torch.optim.lr_scheduler.CosineAnnealingLR


lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)
for epoch in range(50):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')



#========================================================================================



import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_1(nn.Module):
    def __init__(self, ):
        super(LeNet_1, self).__init__()
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
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


root='/home/jack/公共的/MLData/'
tmpout = "/home/jack/SemanticNoise_AdversarialAttack/tmpout/"


batch_size = 32
trans = []

trans.append( transforms.ToTensor() )
# trans.append( transforms.Normalize([0.5], [0.5]) )
transform =  transforms.Compose(trans)


trainset =  datasets.MNIST(root = root, train = True, download = True, transform = transform) # 表示是否需要对数据进行预处理，none为不进行预处理
testset =  datasets.MNIST(root = root, train = False, download = True, transform = transform) # 表示是否需要对数据进行预处理，none为不进行预处理

train_iter = DataLoader(trainset, batch_size=batch_size, shuffle = False,  )
test_iter = DataLoader(testset, batch_size=batch_size, shuffle = False,  )


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU.")

print(f"len(trainset) = {len(trainset)}, len(train_iter) = {len(train_iter)}, ")
# batch_size = 25, len(trainset) = 60000, len(testset) = 10000, len(train_iter) = 2400, len(test_iter) = 400


model = LeNet_1()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
lossfn = torch.nn.CrossEntropyLoss()


# for epoch in range(100):
optimizer.zero_grad()
clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
print(f"origin: {model.state_dict()['fc2.bias'].is_leaf}, {model.state_dict()['fc2.bias'].shape}, {model.state_dict()['fc2.bias'].device}, {model.state_dict()['fc2.bias'].requires_grad}, {model.state_dict()['fc2.bias'].type()}, {model.state_dict()['fc2.bias'].grad} \n")


for batch, (X, y) in enumerate(train_iter):
    X, y  = X.to(device), y.to(device)
    #print(f"X.requires_grad = {X.requires_grad}, y.requires_grad = {y.requires_grad}")
    y_hat = model(X)
    loss = lossfn(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    print(f"{batch}, 0: {model.state_dict()['fc2.bias'].is_leaf}, {model.state_dict()['fc2.bias'].shape}, {model.state_dict()['fc2.bias'].device}, {model.state_dict()['fc2.bias'].requires_grad}, {model.state_dict()['fc2.bias'].type()}, {model.state_dict()['fc2.bias'].grad} \n {model.state_dict()['fc2.bias']}")
    optimizer.step()
    print(f"{batch}, 1: {model.state_dict()['fc2.bias'].is_leaf}, {model.state_dict()['fc2.bias'].shape}, {model.state_dict()['fc2.bias'].device}, {model.state_dict()['fc2.bias'].requires_grad}, {model.state_dict()['fc2.bias'].type()}, {model.state_dict()['fc2.bias'].grad} \n {model.state_dict()['fc2.bias']}")
#========================================================================================

model = LeNet_1()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
lossfn = torch.nn.CrossEntropyLoss(reduction='none')



# for epoch in range(100):
optimizer.zero_grad()
clipped_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
print(f"model.state_dict() = {model.state_dict()['fc2.bias']} \n\n")




for batch, (X, y) in enumerate(train_iter):
    X, y  = X.to(device), y.to(device)
    #print(f"X.requires_grad = {X.requires_grad}, y.requires_grad = {y.requires_grad}")
    y_hat = model(X)
    loss = lossfn(y_hat, y)  # l = 2.3009374141693115
    for i in range(loss.size(0)):
        loss[i].backward(retain_graph=True)
        print(f"{batch}: {i}: {model.state_dict()['fc2.bias'].is_leaf}, {model.state_dict()['fc2.bias'].shape}, {model.state_dict()['fc2.bias'].device}, {model.state_dict()['fc2.bias'].requires_grad}, {model.state_dict()['fc2.bias'].type()}, {model.state_dict()['fc2.bias'].grad} \n")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        for name, param in model.named_parameters():
            clipped_grads[name] += param.grad
        model.zero_grad()

    # optimizer.zero_grad()
    # loss.backward()
optimizer.step()

#========================================================================================




params = {}
for key, var in model.state_dict().items():
    params[key] = var # .detach().cpu().numpy()
    # print("key:"+str(key)+",var:"+str(var))
    print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()}, {var.grad} \n  {var}" )
    # print(f"张量{key}的Size : "+str(var.size()))

































































































































































































































































































































































