#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:18:40 2022

@author: jack

https://pytorchbook.cn/chapter2/2.1.3-pytorch-basics-nerual-network/
本文件主要是测试优化器相关的内容;
(1) model.zero_grad和opotimizer.zero_grad的区别和关系
(2) 优化器的成员包括哪些，怎么查看学习率、最后的epoch、怎么设置lr
(3) 优化器学习率调整器的种类
    PyTorch学习率调整策略通过torch.optim.lr_scheduler接口实现。PyTorch提供的学习率调整策略分为三大类，分别是
    有序调整：等间隔调整(Step)，按需调整学习率(MultiStep)，指数衰减调整(Exponential)和余弦退火CosineAnnealing。
    自适应调整：自适应调整学习率 ReduceLROnPlateau。
    自定义调整：自定义调整学习率 LambdaLR。

"""


"""

本次探索了在不同实验场景下model.zero_grad和opotimizer.zero_grad的区别和关系。这两个方法都是对网络进行梯度置零，但是在不同应用场景下有比较优的选择，总结如下：

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
input1 = torch.randn(1, 1, 32, 32) # 这里的对应前面fforward的输入是32
out = net(input1)
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
out = net(input1) # 这里调用的时候会打印出我们在forword函数中打印的x的大小
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
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
import collections

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1, 10)
    def forward(self,x):
        return self.fc(x)

###集成了优化器和学习率控制器
def make_optimizer( net, epoch = 100, gamma = 0.9, lr = 0.01):
    trainable = filter(lambda x: x.requires_grad, net.parameters())

    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': lr, 'weight_decay': 0 }

    # optimizer = ADAM
    optimizer_class = optim.Adam
    kwargs_optimizer['betas'] = (0.9, 0.999)
    kwargs_optimizer['eps'] =  1e-8

    milestones =  [i for i in range(epoch)][1:]  # [200]
    kwargs_scheduler = {'milestones': milestones, 'gamma': gamma}  # args.gamma =0.5
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        # def save(self, save_dir):
        #     torch.save(self.state_dict(), self.get_dir(save_dir))
        # def load(self, load_dir, epoch=1):
        #     self.load_state_dict(torch.load(self.get_dir(load_dir)))
        #     if epoch > 1:
        #         for _ in range(epoch): self.scheduler.step()
        # def get_dir(self, dir_path):
        #     return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_last_lr(self):
            return self.scheduler.get_last_lr()[0] ## 返回最近一次 scheduler.step()后的学习率
            # return self.param_groups[0]['lr']

        def get_lr(self):
            ## return optimizer.state_dict()['param_groups'][0]['lr']
            return self.param_groups[0]['lr']

        def set_lr(self, lr):
            # self.scheduler.get_last_lr()[0] = lr
            for param_group in self.param_groups:
                param_group["lr"] = lr

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

#======================= 集成了优化器和学习率控制器的用法0 =======================
model = net()
params = {}
# for key, var in model.state_dict().items():
#     params[key] = var.clone().cpu() #.detach()
#     print(f"{key}, {var.is_leaf}, {var.shape}, {var.device}, {var.requires_grad}, {var.type()} \n  {var}" )

lr_list1 = []
lr_list2 = []
lr_list3 = []
optimizer = make_optimizer(model, gamma=0.97)
# lr_list2.append(optimizer.get_lr())
# lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
# lr_list3.append(optimizer.param_groups[0]['lr'])
loss = torch.nn.CrossEntropyLoss()


print(optimizer.state_dict()['param_groups'][0]['lr'])
print(optimizer.get_lr())
print(optimizer.param_groups[0]['lr'])

for epoch in range(200):
    lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
    lr_list2.append(optimizer.get_lr())
    lr_list3.append(optimizer.param_groups[0]['lr'])
    for i in range(20):
        y = torch.randint(0, 9, (10,10))*1.0
        out = model(torch.randn(10,1))
        lss = loss(out, y)
        optimizer.zero_grad()
        lss.backward()
        optimizer.step()
    optimizer.schedule()
    # lr_list2.append(optimizer.get_lr())
    # lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # lr_list3.append(optimizer.param_groups[0]['lr'])

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)), lr_list1, color = 'r', linestyle = '-', linewidth = 5)
plt.plot(range(len(lr_list2)), lr_list2, color = 'b', linestyle = '--', linewidth = 3)
plt.plot(range(len(lr_list3)), lr_list3, color = 'y', linestyle = '-.', linewidth = 1)

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
# out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

print(lr_list1[:6])
print(lr_list2[:6])


print(optimizer.state_dict()['param_groups'][0]['lr'])
print(optimizer.get_lr())
print(optimizer.param_groups[0]['lr'])

##========== 优化器的成员有哪些 ==========================

print(f"optimizer.defaults = \n{optimizer.defaults}")
print(f"optimizer.param_groups = \n{optimizer.param_groups}")
print(f"optimizer.state_dict() = \n{optimizer.state_dict()}")


##===========  学习率调整器的成员有哪些？ ================
print(f"optimizer.scheduler.base_lrs = {optimizer.scheduler.base_lrs}")
print(f"optimizer.scheduler.gamma  = {optimizer.scheduler.gamma}")
print(f"optimizer.scheduler.get_last_lr()  = {optimizer.scheduler.get_last_lr()}")
print(f"optimizer.scheduler.last_epoch  = {optimizer.scheduler.last_epoch}")
print(f"optimizer.scheduler.gamma  = {optimizer.scheduler.gamma}")
print(f"optimizer.scheduler.last_epoch  = {optimizer.scheduler.last_epoch}")
print(f"optimizer.scheduler.milestones  = {optimizer.scheduler.milestones}")
print(f"optimizer.scheduler.state_dict() = {optimizer.scheduler.state_dict()}")
# optimizer.scheduler.base_lrs = [0.1]
# optimizer.scheduler.gamma  = 0.5
# optimizer.scheduler.get_last_lr()  = [0.003125]
# optimizer.scheduler.last_epoch  = 200
# optimizer.scheduler.gamma  = 0.5
# optimizer.scheduler.last_epoch  = 200
# optimizer.scheduler.milestones  = Counter({20: 1, 40: 1, 60: 1, 100: 1, 120: 1})
# optimizer.scheduler.state_dict() =
    # {'milestones': Counter({20: 1, 40: 1, 60: 1, 100: 1, 120: 1}),
    #  'gamma': 0.5,
    #  'base_lrs': [0.1],
    #  'last_epoch': 200,
    #  '_step_count': 201,
    #  'verbose': False,
    #  '_get_lr_called_within_step': False,
    #  '_last_lr': [0.003125]}


##================== 手动设置上述集成器的学习率1 ==================================

model = net()

optimizer = make_optimizer(model, gamma=0.99, lr = 0.1)
loss = torch.nn.CrossEntropyLoss()

lr_list1 = []
lr_list2 = []
lr_list3 = []
optimizer.set_lr(2)

print(optimizer.get_last_lr())
print(optimizer.get_lr())



# print(lr_list1[:6])
# print(lr_list2[:6])

for epoch in range(200):
    # if epoch == 0:
    lr_list1.append(optimizer.get_lr())
    lr_list2.append(optimizer.get_last_lr())
    ## lr_list3.append(optimizer.get_lr())
    for i in range(20):
        y = torch.randint(0, 9, (10,10))*1.0
        optimizer.zero_grad()
        out = model(torch.randn(10,1))
        lss = loss(out, y)
        optimizer.zero_grad()
        lss.backward()
        optimizer.step()
    optimizer.schedule()
    # if epoch != 0:
    # lr_list1.append(optimizer.get_lr())
    # lr_list2.append(optimizer.get_last_lr())
    # ## lr_list3.append(optimizer.get_lr())

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)), lr_list1, color = 'r', linestyle = '-', linewidth = 5)
plt.plot(range(len(lr_list2)), lr_list2, color = 'b', linestyle = '--', linewidth = 3)
## plt.plot(range(len(lr_list3)), lr_list3, color = 'cyan', linestyle = '-.', linewidth = 1)
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

print(lr_list1[:6])
print(lr_list2[:6])

# optimizer.set_lr(2)
# optimizer.zero_grad()
# optimizer.step()
# # optimizer.schedule()
# print(f"optimizer.get_lr() = {optimizer.get_lr()}")


##================== 手动设置上述集成器的学习率2 ==================================

model = net()

LR = 0.01
optimizer = make_optimizer(model)
loss = torch.nn.CrossEntropyLoss()

lr_list1 = []
lr_list2 = []

for epoch in range(200):
    optimizer.set_lr(epoch/100)
    lr_list1.append(optimizer.get_lr())
    lr_list2.append(optimizer.get_last_lr())

    for i in range(20):
        y = torch.randint(0, 9, (10,10))*1.0
        optimizer.zero_grad()
        out = model(torch.randn(10,1))
        lss = loss(out, y)
        optimizer.zero_grad()
        lss.backward()
        optimizer.step()
    optimizer.schedule()
    # lr_list1.append(optimizer.get_lr())
    # lr_list2.append(optimizer.get_last_lr())


fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)), lr_list1, color = 'r', linestyle = '-', linewidth = 5)
plt.plot(range(len(lr_list2)), lr_list2, color = 'b', linestyle = '--', linewidth = 3)

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

print(lr_list1[:6])
print(lr_list2[:6])

##================== 手动设置上述集成器的学习率3 ==================================
import math
import numpy as np
model = net()


optimizer = make_optimizer(model, gamma = 0.2, lr = 0.1)
loss = torch.nn.CrossEntropyLoss()

lr_list1 = []
lr_list2 = []

for epoch in range(200):
    optimizer.set_lr(np.sin(2* math.pi * epoch/200))
    lr_list1.append(optimizer.get_lr())
    lr_list2.append(optimizer.get_last_lr())

    for i in range(20):
        y = torch.randint(0, 9, (10,10))*1.0
        optimizer.zero_grad()
        out = model(torch.randn(10,1))
        lss = loss(out, y)
        optimizer.zero_grad()
        lss.backward()
        optimizer.step()
    optimizer.schedule()   ## 对比这行使用前后的lr曲线

    # lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # lr_list2.append(optimizer.get_last_lr())


fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)), lr_list1, color = 'r', linestyle = '-', linewidth = 5)
plt.plot(range(len(lr_list2)), lr_list2, color = 'b', linestyle = '--', linewidth = 3)

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()
print(lr_list1[:6])
print(lr_list2[:6])

## 以上三个示例展示了学习率控制和学习率递减的精妙关系：
# 1. 如果没有使用set_lr，或者在epoch循环之前使用了set_lr，则此时的学习率的变化完全是符合预期且是想要的，如例子0和1所示;
# 2. 如果在紧接着epoch后使用 set_lr(epoch、100) 则每epoch的学习率都是 epoch/1000
# 4. optimizer.schedule()会在当时的学习率上 x gamma, 而不管当前的学习率是自然变化的还是被手动设置的


#========================================= 手动控制学习率 ===============================================
model = net()
LR = 0.01
optimizer = Adam(model.parameters(), lr = LR)
lr_list = []
for epoch in range(100):
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

##================ 优化器的成员包括哪些, 怎么查看优化器的参数 ===================
print(f"{optimizer.__dict__}")
# {'defaults': {'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}, '_zero_grad_profile_name': 'Optimizer.zero_grad#Adam.zero_grad', 'state': defaultdict(<class 'dict'>, {}), 'param_groups': [{'params': [Parameter containing:
# tensor([[ 0.6787],
#         [ 0.4035],
#         [ 0.5292],
#         [-0.0933],
#         [ 0.6139],
#         [ 0.3144],
#         [ 0.2878],
#         [-0.6279],
#         [-0.9790],
#         [-0.0769]], requires_grad=True), Parameter containing:
# tensor([-0.8369, -0.7730, -0.9616, -0.1452, -0.2963,  0.6936, -0.0665,  0.9163,
#          0.1320,  0.6001], requires_grad=True)], 'lr': 0.001215766545905694, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}], '_warned_capturable_if_run_uncaptured': True}



print(f"optimizer.defaults = \n{optimizer.defaults}")
# optimizer.defaults =
# {'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}

print(f"optimizer.param_groups = \n{optimizer.param_groups}")
# optimizer.param_groups =
# [{'params': [Parameter containing:
#    tensor([[ 0.6787],
#            [ 0.4035],
#            [ 0.5292],
#            [-0.0933],
#            [ 0.6139],
#            [ 0.3144],
#            [ 0.2878],
#            [-0.6279],
#            [-0.9790],
#            [-0.0769]], requires_grad=True),
#    Parameter containing:
#    tensor([-0.8369, -0.7730, -0.9616, -0.1452, -0.2963,  0.6936, -0.0665,  0.9163,
#             0.1320,  0.6001], requires_grad=True)],
#   'lr': 0.001215766545905694,
#   'betas': (0.9, 0.999),
#   'eps': 1e-08,
#   'weight_decay': 0,
#   'amsgrad': False,
#   'maximize': False,
#   'foreach': None,
#   'capturable': False}]

print(f"optimizer.state_dict() = \n{optimizer.state_dict()}")
# optimizer.state_dict() =
# {'state': {},
#  'param_groups': [{'lr': 0.001215766545905694,
#    'betas': (0.9, 0.999),
#    'eps': 1e-08,
#    'weight_decay': 0,
#    'amsgrad': False,
#    'maximize': False,
#    'foreach': None,
#    'capturable': False,
#    'params': [0, 1]}]}

#======================================= lr_scheduler.ReduceLROnPlateau =================================================
# class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# 功能: 当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。例如，当验证集的loss不再下降时，进行学习率调整；或者监测验证集的accuracy，当accuracy不再上升时，则调整学习率。

# 参数：
    # mode(str)- 模式选择，有 min和max两种模式，min表示当指标不再降低(如监测loss)，max表示当指标不再升高(如监测accuracy)。
    # factor(float)- 学习率调整倍数(等同于其它方法的gamma)，即学习率更新为 lr = lr * factor patience(int)- 直译——"耐心"，即忍受该指标多少个step不变化，当忍无可忍时，调整学习率。注，可以不是连续5次。
    # verbose(bool)- 是否打印学习率信息， print('Epoch {:5d}: reducing learning rate' ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
    # threshold(float)- Threshold for measuring the new optimum，配合threshold_mode使用，默认值1e-4。作用是用来控制当前指标与best指标的差异。
    # threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式，rel和abs。 当threshold_mode = rel，并且mode = max时，dynamic_threshold = best * ( 1 + threshold )； 当threshold_mode = rel，并且mode = min时，dynamic_threshold = best * ( 1 - threshold )； 当threshold_mode = abs，并且mode = max时，dynamic_threshold = best + threshold ； 当threshold_mode = rel，并且mode = max时，dynamic_threshold = best - threshold
    # cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
    # min_lr(float or list)- 学习率下限，可为float，或者list，当有多个参数组时，可用list进行设置。
    # eps(float)- 学习率衰减的最小值，当学习率变化小于eps时，则不调整学习率。



import numpy as np
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


loss = torch.nn.CrossEntropyLoss()

lr_list1 = []
lr_list2 = []
for epoch in range(200):
     for i in range(20):
         y = torch.randint(0, 9, (10,10))*1.0
         optimizer.zero_grad()
         out = model(torch.randn(10,1))
         lss = loss(out, y)
         lss.backward()
         optimizer.step()
     scheduler.step(lss.item())
     # lr_list2.append(scheduler.get_lr())
     lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])


fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)),lr_list1, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

#======================================= lr_scheduler.LambdaLR =================================================
# torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
# 能够根据自己的定义调节LR

# 设置学习率为初始学习率乘以给定lr_lambda函数的值
# new_lr=lr_lambda(last_epoch) * base_lr
# 当 last_epoch=-1时, base_lr为optimizer优化器中的lr
# 每次执行 scheduler.step(),  last_epoch=last_epoch +1
# optimizer：优化器
# lr_lambda：函数或者函数列表
# last_epoch：默认为-1，学习率更新次数计数；注意断点训练时last_epoch不为-1

import numpy as np
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
lambda1 = lambda epoch:np.sin(epoch) / epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
for epoch in range(100):
    # scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    scheduler.step()

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()


#=================================== CosineAnnealingLR ======================================================
# 以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax
#  为周期，在一个周期内先下降，后上升。

import numpy as np
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = 10, eta_min=0,)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#====================================== lr_scheduler.StepLR ==================================================
# class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
# 功能： 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。间隔单位是step。需要注意的是，step通常是指epoch，不要弄成iteration了。

# 等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了。

"""
参数：
optimizer:优化器
step_size(int): 学习率下降间隔数，若为30，则会在30、60、90…个step时，将学习率调整为lr*gamm
gamma(float): 学习率调整倍数，默认为0.1倍，即下降10倍。
last epoch(int): 是从last_start开始后已经记录了多少个epoch， Default: -1.

"""

lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])


fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

#======================================= MultiStepLR =================================================
# torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
# 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。

'''
参数：
milestones(list): 一个 list，每一个元素代表何时调整学习率， list 元素必须是递增的。如 milestones=[30,80,120]
gamma(float): 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。

# StepLR的区别是，调节的epoch是自己定义，无须一定是【30， 60， 90】 这种等差数列；
# 请注意，这种衰减是由外部的设置来更改的。 当last_epoch=-1时，将初始LR设置为LR。
'''



lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [20,80], gamma = 0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()




#======================================= ExponentialLR =================================================
# 按指数衰减调整学习率，调整公式: lr=l∗gamma∗∗epoch

lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()





#========================================================================================
# https://zhuanlan.zhihu.com/p/352212135
#========================================================================================

#===================================== 手动设置学习率衰减1 ====================================
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


fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list)),lr_list, color = 'r')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#=====================================  手动设置学习率衰减2 =====================================
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

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)),lr_list1, color = 'r', )
plt.plot(range(len(lr_list2)),lr_list2, color = 'b', linestyle = '--')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#=====================================  手动设置学习率衰减3 =====================================
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

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
plt.plot(range(len(lr_list1)),lr_list1, color = 'r', )
plt.plot(range(len(lr_list2)),lr_list2, color = 'b', linestyle = '--')

filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()


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

































































































































































































































































































































































