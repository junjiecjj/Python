
# -*- coding: utf-8 -*-
"""
Created on 2022/07/31

@author: Junjie Chen

"""

# 系统库
from torch.utils.tensorboard import SummaryWriter
import os, sys
import torch
import torch.nn as nn
import numpy as np


# 本项目自己编写的库
# sys.path.append("/home/jack/公共的/Pretrained-IPT-cjj/")
sys.path.append("..")
from Option import args
from  ColorPrint import ColoPrint
color =  ColoPrint()





class SummWriter(SummaryWriter):
    def __init__(self, args, ckp):
        # if args.modelUse == 'DeepSC':
        sdir = os.path.join(args.save, f"{ckp.now}_TensorBoard_{args.modelUse}")

        os.makedirs(sdir, exist_ok=True)
        kwargs_summwrit = {'comment':"", 'purge_step': None, 'max_queue': 10, 'flush_secs':120,}
        super(SummWriter, self).__init__(sdir, **kwargs_summwrit)

        self.loss = []
        for loss in args.loss.split('+'):  #  ['1*MSE']
            weight, loss_type = loss.split('*')
            self.loss.append({'type': loss_type , 'weight': float(weight)} )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0,})

# <<< 训练结果可视化
    # 将不同压缩率和信噪比下的loss画成连续的loss
    def WrTLoss(self, trainloss, epoch):
        for idx, los in enumerate(self.loss):
            self.add_scalar(f"train/Loss/AllLoss/{los['weight']}*{los['type']}", trainloss[idx], epoch)
        return

    # 在一张图画出loss
    def WrTrLossOne(self, compratio, snr,  trainloss, epoch):
        for idx, los in enumerate(self.loss):
            self.add_scalars(f"train/Loss/{los['weight']}*{los['type']}", {f"CompreRatio={compratio},SNR={snr}": trainloss[idx]}, epoch)
        return
    # 在不同图画出loss
    def WrTrainLoss(self,  compratio, snr, trainloss, epoch):
        for idx, los in enumerate(self.loss):
            self.add_scalar(f"train/loss/{los['weight']}*{los['type']}/CompreRatio={compratio},SNR={snr}" , trainloss[idx], epoch)
        return
    # 在一张图画出Psnr,MSE等
    def WrTrMetricOne(self, compratio, snr,  metrics, epoch):
        for idx, met in enumerate(args.metrics):
            self.add_scalars(f"train/Metric/{met}", {f"CompreRatio={compratio},SNR={snr}": metrics[idx]}, epoch)
        return
    # 在不同图画出Psnr,MSE等
    def WrTrainMetric(self, compratio, snr, metrics, epoch):
        for idx, met in enumerate(args.metrics):
            self.add_scalar(f"train/metric/{met}/CompreRatio={compratio},SNR={snr}" , metrics[idx], epoch)
        return
    def WrLr(self, compratio, snr, lr, epoch):
        self.add_scalar(f"LearningRate/CompreRatio={compratio},SNR={snr}" , lr, epoch)
        return
# 训练结果可视化>>>



# <<< 测试结果可视化
    # 不同图可视化
    def WrTestMetric(self, dasename, compratio, snr, metrics):
        for idx, met in enumerate(args.metrics):
            self.add_scalar(f"Test/{dasename}/Y({met})X(snr)/CompreRatio={compratio}" , metrics[idx], snr)
        return
    # 一张图可视化
    def WrTestOne(self, dasename, compratio, snr, metrics):
        for idx, met in enumerate(args.metrics):
            #print(f"{dasename} {met} {compratio} {metrics} {metrics[idx]}")
            self.add_scalars(f"Test/{dasename}/Y{met}X(snr)", {f"CompreRatio={compratio}" : metrics[idx]}, snr)

        return

# 测试结果可视化>>>


    def WrModel(self, model, images ):
        self.add_graph(model, images)
        return
    def WrClose(self):
        print(color.fuchsia(f"\n#================================ 关闭Tensorboard可视化  =======================================\n"))
        # print(f"====================== 关闭Tensorboard可视化 ===================================")
        self.close()
        return




# wr = SummWriter(args)

# x = range(300,400)

# AllEpoch = 0
# for comprate_idx, compressrate in enumerate(args.CompressRateTrain):  #[0.17, 0.33, 0.4]
#     for snr_idx, snr in enumerate(args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
#         for epch_idx in range(200):
#             AllEpoch += 1
#             for batch_idx  in range(10):
#                 pass
#             wr.WrTLoss( torch.tensor(np.random.randint(100,size=(3,))+epch_idx), AllEpoch )

#             #
#             wr.WrTrLossOne(compressrate, snr, torch.tensor(np.random.randint(100,size=(3,))+epch_idx), epch_idx)
#             wr.WrTrPsnrOne(compressrate, snr, torch.tensor(np.random.randint(100)+epch_idx), epch_idx)

#             #
#             wr.WrTrainLoss(compressrate, snr, torch.tensor(np.random.randint(100,size=(3,))+epch_idx), epch_idx)
#             wr.WrTrainPsnr(compressrate, snr, torch.tensor(2.11+epch_idx), epch_idx)

# wr.close()


# wr = SummWriter(args)

# for idx_data, ds in enumerate(args.data_test):
#     for comprate_idx, compressrate in enumerate(args.CompressRateTrain):
#         for snr_idx, snr in enumerate( args.SNRtest):
#             metrics = torch.tensor([snr_idx,snr_idx+idx_data])
#             wr.WrTestMetric(ds, compressrate, snr, metrics)
#             wr.WrTestOne(ds, compressrate, snr, metrics)

#================================================================================================================

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# writer = SummaryWriter('/home/jack/IPT-Pretrain/results/summary')

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train/1', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test/1', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train/2', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test/2', np.random.random(), n_iter)
# r = 5
# for i in range(100):
#     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
#                                     'xcosx':i*np.cos(i/r),
#                                     'tanx': np.tan(i/r)}, i)


# writer.close()


#会发现刚刚的log文件夹里面有文件了。在命令行输入如下，载入刚刚做图的文件（那个./log要写完整的路径）
#  tensorboard --logdir=/home/jack/IPT-Pretrain/results/TensorBoard_IPT/
# 在浏览器输入以下任意一个网址，即可查看结果：（训练过程中可以实时更新显示）
#  http://0.0.0.0:6006/
#  http://localhost:6006/
#  http://127.0.0.1:6006/
