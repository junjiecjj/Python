# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import os
import sys
import numpy as np
import torch.nn as nn
import torch
import collections


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 本项目自己编写的库
# from  ColorPrint import ColoPrint
# color =  ColoPrint()
# sys.path.append("..")
# from  Option import args


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"





def make_optimizer(args, net,  compr = '', snr = ''):
    ### make optimizer and scheduler together

    # optimizer
    #  filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    trainable = filter(lambda x: x.requires_grad, net.parameters())

    #  trainable = net.parameters()
    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}

    # optimizer = ADAM
    if args.optimizer == 'SGD':
        optimizer_class = torch.optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = torch.optim.Adam
        kwargs_optimizer['betas'] = args.betas  # (0.9, 0.999)
        kwargs_optimizer['eps'] = args.epsilon  # 1e-8
    elif args.optimizer == 'RMSprop':
        optimizer_class = torch.optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler, milestones = 0,   gamma = 0.5
    milestones = list(map(lambda x: int(x), args.decay.split('-')))  #  [20, 40, 60, 80, 100, 120]
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.9
    scheduler_class = torch.optim.lr_scheduler.MultiStepLR

    # warmup_class = optimization.get_polynomial_decay_schedule_with_warmup
    # kwargs_warmup = {"num_warmup_steps":args.warm_up_ratio*total_steps, "num_training_steps":total_steps,"power":args.power,"lr_end":args.lr_end}

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
            self.lr = []
            self.cn = self.__class__.__name__
            return

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

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

        def updatelr(self):
            lr = self.get_lr()
            self.lr.append(lr)
            return lr

        def reset_state(self):
            self.state = collections.defaultdict(dict)
            self.scheduler.last_epoch = 0
            #self.scheduler._last_lr = 0
            for param_group in self.param_groups:
                param_group["lr"] = args.lr

            milestones = list(map(lambda x: int(x), args.decay.split('-')))  #  [20, 40, 60, 80, 100, 120]
            kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.5
            self.scheduler = scheduler_class(self, **kwargs_scheduler)

            # kwargs_warmup = {"num_warmup_steps":args.warm_up_ratio*total_steps, "num_training_steps":total_steps,"power":args.power,"lr_end":args.lr_end}
            # self.scheduler = warmup_class(self, **kwargs_warmup)
            return

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    #optimizer._register_scheduler(warmup_class, **kwargs_warmup)

    return optimizer

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)

# model = net()
# LR = 0.01
# opt = make_optimizer(args, model,  )
# loss = torch.nn.CrossEntropyLoss()

# lr_list1 = []
# lr_list2 = []
# for epoch in range(200):
#       for i in range(20):
#           y = torch.randint(0, 9, (10,10))*1.0
#           opt.zero_grad()
#           out = model(torch.randn(10,1))
#           lss = loss(out, y)
#           # lss = Variable(lss, requires_grad = True)
#           lss.backward()
#           opt.step()
#       opt.schedule()
#       lr_list2.append(opt.get_lr())
#       lr_list1.append(opt.state_dict()['param_groups'][0]['lr'])

# fig, axs = plt.subplots(1,1, figsize=(6,6))
# axs.plot(range(len(lr_list1)),lr_list1,color = 'r')
# #plt.plot(range(100),lr_list2,color = 'b')
# out_fig = plt.gcf()
# out_fig.savefig("/home/jack/snap/11.pdf")
# plt.show()
# plt.close(fig)


# from option import args
# ckp = checkpoint(args)
# ckp.InittestDir('aaaa')
# for idx_data, ds in enumerate(args.data_test):
#     for comprate_idx, compressrate in enumerate(args.CompressRateTrain):
#         ckp.InitTestMetric(compressrate, ds)
#         for snr_idx, snr in enumerate( args.SNRtest):
#             ckp.AddTestMetric(compressrate, snr, ds)
#             for i in range(20):
#                 metric = torch.tensor([comprate_idx,comprate_idx+snr_idx])
#                 ckp.UpdateTestMetric(compressrate, ds,metric)
#                 ckp.MeanTestMetric(compressrate, ds,2)

# ckp.PlotTestMetric()

#  使用时：
"""

model = net()
LR = 0.01
optimizer = make_optimizer( args, model, )


lr_list1 = []

for epoch in range(200):
    for X,y in dataloder:
        optimizer.step()
    optimizer.schedule()
    lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(200),lr_list1,color = 'r')

plt.show()

"""
