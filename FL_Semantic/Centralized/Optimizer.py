# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import os
# import sys
import numpy as np
# import torch.nn as nn
import torch
# import collections
# from transformers import optimization

#内存分析工具
# from memory_profiler import profile
# import objgraph
# import gc

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
from matplotlib.pyplot import MultipleLocator


# 本项目自己编写的库
# from  ColorPrint import ColoPrint
# color =  ColoPrint()
# sys.path.append("..")
# from  Option import args


fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"




def make_optimizer(args, net,  compr = '', snr = ''):

    '''
    make optimizer and scheduler together
    '''
    # optimizer
    #  filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    trainable = filter(lambda x: x.requires_grad, net.parameters())

    #  trainable = net.parameters()
    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

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
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.5
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

        # def get_lr(self):
        #     # return self.scheduler.get_lr()[0]
        #     return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

        def updatelr(self):
            lr = self.get_lr()
            self.lr.append(lr)
            return lr

        def save_lr(self, path,  compr = '', tra_snr = 'random'):

            if compr != '' :
                basename = f"_{self.cn}_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
            else:
                basename = f"_{self.cn}"
            torch.save(self.lr, os.path.join(path, f"lr{basename}.pt"))
            self.plot_lr(self.lr, path, compr = compr, snr = tra_snr)
            return

            if compr != '' :
                basename = f"_{self.cn}_compr={compr:.1f}_trainSnr={snr}(dB)"
                title = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}} = {}\mathrm{{(dB)}}$'.format(compr, snr)
            else:
                basename = f"_{self.cn}"
                title = ''
            epoch = len(Lr)
            X = np.linspace(1, epoch, epoch)

            fig = plt.figure(figsize=(6, 5), constrained_layout=True)
            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.plot(X, Lr, )
            plt.xlabel('Epoch',fontproperties=font)
            plt.ylabel('Learning rate',fontproperties=font)
            #plt.title(label,fontproperties=font)
            #plt.grid(True)

            # font1 = {'family':'Times New Roman','style':'normal','size':16}
            # legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            # frame1 = legend1.get_frame()
            # frame1.set_alpha(1)
            # frame1.set_facecolor('none')  # 设置图例legend背景透明

            ax=plt.gca()
            ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(16) for label in labels] #刻度值字号

            fontt  = {'family':'Times New Roman','style':'normal','size':22}
            if title != '':
                plt.suptitle(title, fontproperties=fontt, )

            #plt.tight_layout(pad=2, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            # out_fig.savefig(os.path.join(savepath, f"lr{basename}.pdf"))
            out_fig.savefig(os.path.join(savepath, f"lr{basename}.eps"))
            #plt.show()
            plt.close()
            return

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    #optimizer._register_scheduler(warmup_class, **kwargs_warmup)

    return optimizer


# class net(nn.Module):
#     def __init__(self):
#         super(net,self).__init__()
#         self.fc = nn.Linear(1,10)
#     def forward(self,x):
#         return self.fc(x)

# model = net()
# LR = 0.01
# args.decay = '1-2-3-4-5-6-7-8'
# opt = make_optimizer(args, model, "test" )
# loss = torch.nn.CrossEntropyLoss()

# lr_list = []
# lr_list1 = []
# lr_list2 = []
# for epoch in range(10):
#       lr_list2.append(opt.get_last_lr())
#       lr_list1.append(opt.state_dict()['param_groups'][0]['lr'])
#       for i in range(20):
#           y = torch.randint(0, 9, (10,10))*1.0
#           opt.zero_grad()
#           out = model(torch.randn(10,1))
#           lss = loss(out, y)
#           # lss = Variable(lss, requires_grad = True)
#           lss.backward()
#           opt.step()
#       opt.schedule()
#       lr_list.append(opt.state_dict()['param_groups'][0]['lr'])


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
