
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
# 系统库
import sys,os
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#内存分析工具
from memory_profiler import profile
import objgraph


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator


# 本项目自己编写的库
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True


fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)



class JoinLoss(nn.modules.loss._Loss):
    def __init__(self,  modelname = ''):
        super(JoinLoss, self).__init__()
        # self.len     = Len
        self.cn       = self.__class__.__name__
        self.batchs   = 0

        self.losslog  = torch.Tensor()

        # if args.precision == 'half':
            # self.loss_module.half()

        #print(color.fuchsia(f"\n#============================ LOSS module for {self.modelname} 准备完毕 ==============================\n"))
        return

    def addlog(self):
        #  losslog.shape = [1,len(loss)],[2,len(loss)],[2,len(loss)]...,[epoch,len(loss)]
        self.losslog = torch.cat((self.losslog, torch.zeros(1, 1)))
        self.batchs = 0
        return

    def forward(self, raw_img, real_labs, transmitted_img, classify_outs, comprate):
        crosseny = nn.CrossEntropyLoss(reduction = 'mean')
        mse      = nn.MSELoss(reduction = 'mean')

        self.batchs += 1
        mse_loss = mse(raw_img, transmitted_img)
        cross_los = crosseny(classify_outs, real_labs)
        loss_sum = mse_loss * comprate * comprate + cross_los * (1.0 - comprate)

        self.losslog[-1, -1] += loss_sum.item()
        return loss_sum

    def avg(self, ):
        self.losslog[-1].div_(self.batchs)
        return self.losslog[-1, -1].item()




class myLoss(nn.modules.loss._Loss):
    def __init__(self, args, modelname = ''):
        super(myLoss, self).__init__()
        # print('Preparing loss function:')
        self.cn = self.__class__.__name__
        self.samples = 0
        self.modelname = modelname

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):  #  ['1*MSE']
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = torch.nn.MSELoss(reduction='sum')
            elif loss_type == 'L1':
                loss_function = torch.nn.L1Loss(reduction='sum')
            elif loss_type == 'BCE':
                loss_function = torch.nn.BCELoss(reduction='sum')  # reduction='sum'
            elif loss_type == 'CrossEntropy':
                    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function} )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                # print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.losslog = torch.Tensor()

        self.loss_module.to(args.device)

        if args.precision == 'half':
            self.loss_module.half()

        #print(color.fuchsia(f"\n#============================ LOSS module for {self.modelname} 准备完毕 ==============================\n"))
        return

    #@profile
    def forward(self, sr, hr ):
        # print(f"{sr.shape}   {hr.shape}")
        self.samples += sr.size(0)
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()  #  *sr.shape[0]  # tensor.item()  获取tensor的数值
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.losslog[-1, -1] += loss_sum.item()
        return loss_sum

    # def __getitem__(self, idx):
    #     return self.losslog[-1, idx]

    def reset(self):
        self.losslog = torch.Tensor()
        self.samples = 0
        return

    def addlog(self):
        #  losslog.shape = [1,len(loss)],[2,len(loss)],[2,len(loss)]...,[epoch,len(loss)]
        self.losslog = torch.cat((self.losslog, torch.zeros(1, len(self.loss))))
        self.samples = 0
        return

    def mean_log(self,  ):
        self.losslog[-1].div_( self.samples)
        return self.losslog[-1]

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.losslog[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c/n_samples))

        return ' '.join(log) # '[MSE: 1.0000] [L1: 2.0000] [BCE: 4.0000] [Total: 6.0000]'


    # 在每个压缩率和信噪比下，所有的epoch训练完再调用保存
    def save(self, apath, compr = '', tra_snr = 'random'):
        if compr != '' :
            basename = f"{self.cn}_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
        else:
            basename = f"{self.cn}"
        torch.save(self.losslog, os.path.join(apath, f"{basename}.pt") )
        self.plot_AllLoss(apath, compr = compr, tra_snr = tra_snr)
        return

    # 在不同的画布中画各个损失函数的结果.
    def plot_loss(self, apath, compr = '', snr = ''):
        if compr != '' :
            basename = f"_compr={compr:.1f}_trainSnr={snr}(dB)"
        else:
            basename = ""
        epoch = len(self.losslog[:, 0])
        X = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure(constrained_layout=True)
            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.plot(X, self.losslog[:, i].numpy(), label=label)
            plt.xlabel('Epoch',fontproperties=font)
            plt.ylabel('Training loss',fontproperties=font)
            #plt.title(label,fontproperties=font)
            #plt.grid(True)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
            font1 = {'family':'Times New Roman','style':'normal','size':16}
            legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            ax=plt.gca()
            ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(16) for label in labels] #刻度值字号

            # plt.savefig(os.path.join(apath, f"{self.cn}_{l['type']}_Loss_Plot{basename}.pdf"))
            plt.savefig(os.path.join(apath, f"{self.cn}_{l['type']}_Loss_Plot{basename}.eps"))
            plt.close(fig)
        return

    # 在同一个画布中画出所有Loss的结果
    def plot_AllLoss(self, apath, compr = '', tra_snr = ''):
        if compr != '' :
            title = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}}$'.format(compr, tra_snr)
            basename = f"_compr={compr:.1f}_trainSnr={tra_snr}(dB)"
        else:
            title = ""
            basename = ""
        # fig, axs = plt.subplots(len(self.loss), 1, constrained_layout=True)
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#FF0000','#1E90FF', 'red','cyan','blue','green','#808000','#C000C0', '#FF8C00','#00FF00', '#FFA500']
        if len(self.loss) == 1:
            fig, axs = plt.subplots( constrained_layout=True)
            epoch = len(self.losslog)
            X = np.linspace(1, epoch, epoch)
            label = '{} Loss'.format(self.loss[0]['type'])

            axs.plot(X, self.losslog[:, 0].numpy(), linewidth=2, label=label)
            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            axs.set_xlabel('Epoch', fontproperties=font)
            axs.set_ylabel("Training loss", fontproperties=font)
            #axs.set_title(label, fontproperties=font)
            #axs.grid(True)

            font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 16)
            font1 = {'family':'Times New Roman','style':'normal','size':16}
            #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
            #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete Mono.otf", size=20)
            #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete.otf", size=20)
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs.tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
            labels = axs.get_xticklabels() + axs.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(20) for label in labels]  # 刻度值字号

            #plt.subplots_adjust(top=0.96, bottom=0.1, left=0.03, right=0.97, wspace=0.5, hspace=0.2)#调节两个子图间的距离
            # plt.tight_layout()#  使得图像的四周边缘空白最小化
        else:
            alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
            fig, axs = plt.subplots(len(self.loss), 1, constrained_layout=True)
            for i, l in enumerate(self.loss):
                epoch = len(self.losslog[:,i])
                X = np.linspace(1, epoch, epoch)
                label = '{} Loss'.format(l['type'])

                axs[i].plot(X, self.losslog[:, i].numpy(), linewidth=2, color=color[i], label=label)
                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i].set_xlabel('Epoch', fontproperties=font)
                axs[i].set_ylabel(label, fontproperties=font)
                axs[i].set_title(alabo[i]+f" {l['type']} loss", loc='left', fontproperties=font)
                axs[i].grid(True)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
                font1 = {'family':'Times New Roman','style':'normal','size':16}
                legend1 = axs[i].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                axs[i].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
                labels = axs[i].get_xticklabels() + axs[i].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels]  # 刻度值字号

            plt.subplots_adjust(top=0.96, bottom=0.1, left=0.03, right=0.97, wspace=0.5, hspace=0.2)#调节两个子图间的距离
            # plt.tight_layout()#  使得图像的四周边缘空白最小化
        fontt  = {'family':'Times New Roman','style':'normal','size':22}
        if title != '':
            plt.suptitle(title, fontproperties=fontt, )
        out_fig = plt.gcf()
        # out_fig.savefig(os.path.join(apath, f"{self.cn}_Plot1Fig{basename}.pdf"))
        out_fig.savefig(os.path.join(apath, f"{self.cn}_Plot1Fig{basename}.eps"))
        # plt.show()
        plt.close(fig)
        return



