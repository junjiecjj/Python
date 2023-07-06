# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import sys,os
import utility
import torch
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import torch.nn as nn
import imageio

#内存分析工具
from memory_profiler import profile
import objgraph

# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()
# print(color.fuchsia("Color Print Test Pass"))


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, writer):
        self.args = args
        self.scale = args.scale
        #print(f"trainer  self.scale = {self.scale} \n")
        self.wr = writer
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.device = torch.device(args.device if torch.cuda.is_available() and not args.cpu else "cpu")

        len_dataset = len(self.loader_train)
        batch_size = args.batch_size
        epoch = args.epochs
        total_steps = (len_dataset // batch_size) * epoch
        if len_dataset % batch_size == 0:
            total_steps = (len_dataset // batch_size) * epoch
        else:
            total_steps = (len_dataset // batch_size + 1) * epoch
        # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch

        self.optimizer = utility.make_optimizer(args, self.model, args.epochs)

        #self.wr.WrModel(self.model.model, torch.randn(16, 3, 48, 48))
        if self.args.load != '':
            if self.ckp.mark == True:
                self.optimizer.load(self.ckp.loaddir)

        self.error_last = 1e8


    def prepare(self, *args):
        #device = torch.device('cpu' if self.args.cpu else 'cuda')
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device(self.args.device if torch.cuda.is_available() and not self.args.cpu else "cpu")
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    #@profile
    def train1(self):
        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.fuchsia(f"\n#================================ 开始训练, 时刻:{now1} =======================================\n"))

        #lossFn = nn.MSELoss()
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.train()
        torch.set_grad_enabled(True)
        # ind1_scale = self.args.scale.index(1)
        if self.args.freezeIPT:
            for name, param in self.model.named_parameters():
                if "head" in name:
                    param.requires_grad = False
                if "body" in name:
                    param.requires_grad = False
                if "tail" in name:
                    param.requires_grad = False
                else:
                    pass
        else:
            pass
        self.model.print_parameters(self.ckp)
        tm = utility.timer()

        #self.loader_train.dataset.set_scale(ind1_scale)
        #print(f"scale in train = {self.loader_train.dataset.scale[self.loader_train.dataset.idx_scale]}\n")

        self.ckp.write_log(f"#======================================== 开始训练, 开始时刻{now1} =============================================\n", train=True)

        accumEpoch = 0
        # 依次遍历压缩率
        for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  # [0.17, 0.33, 0.4]
            # 依次遍历信噪比
            for snr_idx, snr in enumerate(self.args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
                print(color.fuchsia( f" 开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}, 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练\n"))
                self.ckp.write_log(f"开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}, 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练", train=True)
                # 初始化 特定信噪比和压缩率下 的Psnr日志
                self.ckp.InitMetricLog(compressrate, snr)

                # 遍历epoch
                for epoch_idx in  range(self.ckp.startEpoch, self.ckp.startEpoch+self.args.epochs):

                    accumEpoch += 1
                    self.ckp.UpdateEpoch()
                    #print(f"ckp.SumEpoch = {self.ckp.SumEpoch.requires_grad}\n")

                    #初始化loss日志
                    self.loss.start_log()

                    # 动态增加特定信噪比和压缩率下的Psnr等评价指标日志
                    self.ckp.AddMetricLog(compressrate, snr)

                    loss = 0
                    # 遍历训练数据集
                    for batch_idx, (lr, hr, filename)  in enumerate(self.loader_train):
                        #print(f"{batch_idx}, lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n")
                        # lr.shape = torch.Size([32, 3, 48, 48]), hr.shape = torch.Size([32, 3, 48, 48]), filename = ('0052', '0031',

                        self.optimizer.zero_grad() # 必须在反向传播前先清零。
                        lr, hr = self.prepare(lr, hr)
                        #print(f"lr.dtype = {lr.dtype}, hr.dtype = {hr.dtype}") #lr.dtype = torch.float32, hr.dtype = torch.float32
                        #hr = hr.to(device)
                        #lr = lr.to(device)
                        #print(f"lr.requires_grad = {lr.requires_grad}, hr.requires_grad = {hr.requires_grad} \n")
                        sr = self.model(hr, idx_scale=0, snr=snr, compr_idx=comprate_idx)
                        #hr = hr.div_(self.args.rgb_range)
                        #sr = sr.div_(self.args.rgb_range)

                        # 计算batch内的loss
                        lss = self.loss(sr, hr)
                        #print(f"lss = {lss}")
                        # lss.requires_grad_(True)
                        # lss = Variable(lss, requires_grad = True)
                        #print(f"lss.grad_fn = {lss.grad_fn}\n")
                        #print(f"lss.requires_grad = {lss.requires_grad}\n") #lss.requires_grad = True

                        lss.backward()
                        self.optimizer.step()

                        # 计算bach内的psnr和MSE
                        # with torch.no_grad():
                        metric = utility.calc_metric(sr=sr, hr=hr, scale=1, rgb_range=self.args.rgb_range, metrics=self.args.metrics)
                        # print(f"metric.requires_grad = {metric.requires_grad} {metric.dtype}")

                        # 更新 bach内的psnr
                        self.ckp.UpdateMetricLog(compressrate, snr, metric)

                        # tmp = tm.toc()
                        # self.ckp.write_log(f"\t\tEpoch {epoch_idx+1}/{self.ckp.startEpoch+self.args.epochs} | Batch {batch_idx+1}/{len(self.loader_train)}, 训练完一个 batch: loss = {lss:.3f}, metric = {metric} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)
                        # print(f"\t\tEpoch {epoch_idx+1}/{self.ckp.startEpoch+self.args.epochs} | Batch {batch_idx+1}/{len(self.loader_train)}, 训练完一个 batch: loss = {lss:.3f}, metric = {metric} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")

                        os.makedirs(os.path.join(self.args.TrainImageSave, f'{self.ckp.now}_trainImage','origin'), exist_ok=True)
                        os.makedirs(os.path.join(self.args.TrainImageSave, f'{self.ckp.now}_trainImage', 'net'), exist_ok=True)
                        if accumEpoch == int(len(self.args.CompressRateTrain)*len(self.args.SNRtrain)*self.args.epochs):
                            with torch.no_grad():
                                for a, b, name in zip(hr, sr,filename):
                                    filename1 = os.path.join(self.args.TrainImageSave, f'{self.ckp.now}_trainImage','origin')+'/{}_hr.png'.format(name)
                                    data1 = a.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                                    imageio.imwrite(filename1, data1)
                                    filename2 = os.path.join(self.args.TrainImageSave, f'{self.ckp.now}_trainImage', 'net')+'/{}_lr.png'.format(name)
                                    data2 = b.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                                    imageio.imwrite(filename2, data2)

                    # 学习率递减
                    self.optimizer.schedule()

                    # 计算并更新epoch的PSNR和MSE等metric
                    epochMetric = self.ckp.MeanMetricLog(compressrate, snr, len(self.loader_train))
                    # 计算并更新epoch的loss
                    epochLos = self.loss.mean_log(len(self.loader_train))

                    tmp = tm.toc()
                    self.ckp.write_log(f"\t\t压缩率:{compressrate} ({comprate_idx+1}/{len(self.args.CompressRateTrain)}) |信噪比:{snr}dB ({snr_idx+1}/{len(self.args.SNRtrain)}) | Epoch {epoch_idx+1}/{self.ckp.startEpoch+self.args.epochs} | 训练完一个 Epoch: loss = {epochLos.item():.3f}, metric = {epochMetric} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟) \n", train=True)
                    print(f"\t\t 压缩率:{compressrate} ({comprate_idx+1}/{len(self.args.CompressRateTrain)}) | 信噪比:{snr}dB ({snr_idx+1}/{len(self.args.SNRtrain)}) | Epoch {epoch_idx+1}/{self.ckp.startEpoch+self.args.epochs}, loss = {epochLos.item():.3f}, metric = {epochMetric} | Time {tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)\n")


                    # 断点可视化，在各个压缩率和信噪比下的Loss和PSNR，以及合并的loss
                    self.wr.WrTLoss(epochLos, int(self.ckp.LastSumEpoch+accumEpoch))
                    self.wr.WrTrainLoss(compressrate, snr, epochLos, epoch_idx)

                    # 学习率可视化
                    self.wr.WrLr(compressrate, snr, self.optimizer.get_lr(), epoch_idx)

                    self.wr.WrTrMetricOne(compressrate, snr, epochMetric, epoch_idx)
                    self.wr.WrTrainMetric(compressrate, snr, epochMetric, epoch_idx)
                    tm.reset()

                # 在每个 压缩率+信噪比 组合下都重置一次优化器
                self.optimizer.reset_state()

                # 在训练完每个压缩率和信噪比下的所有Epoch后,保存一次模型
                # nself.ckp.saveModel(self, compressrate, snr, epoch=int(self.ckp.startEpoch+self.args.epochs))
                #exit(0)
        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)
        # 在训练完所有压缩率和信噪比后，保存优化器
        self.ckp.saveOptim(self)
        # 在训练完所有压缩率和信噪比后，保存PSNR等指标日志

        self.ckp.save()
        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.ckp.write_log(f"#========================= 本次训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ================================",train=True)
        # 关闭日志
        self.ckp.done()
        print(f"====================== 关闭训练日志 {self.ckp.log_file.name} ===================================")

        print(color.fuchsia(f"\n#====================== 训练完毕,开始时刻:{now1},结束时刻:{now2},用时:{tm.hold()/60.0:.3f}分钟 ==============================\n"))
        return


    def test1(self):
        
        now1 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.fuchsia(f"\n#================================ 开始测试,时刻{now1} =======================================\n"))
        # 设置随机数种子
        #torch.manual_seed(self.args.seed)
        self.ckp.InittestDir(now=self.ckp.now)
        tm = utility.timer()
        # if self.args.save_results:
        # self.ckp.begin_queue()

        torch.set_grad_enabled(False)

        self.model.eval()
        #self.model.model.eval()

        print(f"共有{len(self.loader_test)}个数据集\n")
        self.ckp.write_log(f"共有{len(self.loader_test)}个数据集")

        # 依次遍历测试数据集
        for idx_data, ds in enumerate(self.loader_test):
            # 得到测试数据集名字
            DtSetName = ds.dataset.name
            print(f"数据集={DtSetName}, 长度={len(ds)}\n")
            self.ckp.write_log(f"开始在数据集{DtSetName}上测试, 长度={len(ds)}\n")

            # 依次遍历压缩率
            for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  #[0.17, 0.33]

                print(color.fuchsia(f" 开始在数据集为:{DtSetName}, 压缩率为:{compressrate} 下测试\n"))
                # 写日志
                self.ckp.write_log(f" 开始在数据集为:{DtSetName}, 压缩率为:{compressrate} 下测试")

                # 初始化测试指标日志
                self.ckp.InitTestMetric(compressrate, DtSetName)

                # 依次遍历信噪比
                for snr_idx, snr in enumerate(self.args.SNRtest):   # [-6, -4, -2, 0, 2, 6, 10, 14, 18]

                    print(f"   数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr}\n")
                    # 写日志
                    self.ckp.write_log(f"   数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr}")

                    # 测试指标日志申请空间
                    self.ckp.AddTestMetric(compressrate, snr, DtSetName)

                    for batch_idx, (lr, hr, filename) in  enumerate(ds):
                        hr = hr.to(self.device)
                        sr = self.model(hr, idx_scale=0, snr=snr, compr_idx=comprate_idx)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        # 保存图片
                        self.ckp.SaveTestFig(DtSetName, compressrate, snr, self.args.SNRtrain[0], filename[0], sr)

                        # 计算batch内(测试时一个batch只有一张图片)的psnr和MSE
                        metric = utility.calc_metric(sr=sr, hr=hr, scale=1, rgb_range=self.args.rgb_range, metrics=self.args.metrics)

                        # 更新具体SNR下一张图片的PSNR和MSE等
                        self.ckp.UpdateTestMetric(compressrate, DtSetName,metric)
                        #print(f"数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr},图片:{filename},指标:{}")

                        tmp = tm.toc()
                        print(f"     数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),图片:{filename}({batch_idx+1}/{len(ds)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}({snr_idx+1}/{len(self.args.SNRtest)}), 指标:{metric},时间:{tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")

                        self.ckp.write_log(f"     数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),图片:{filename}({batch_idx+1}/{len(ds)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}({snr_idx+1}/{len(self.args.SNRtest)}), 指标:{metric},时间:{tmp/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")

                    # 计算某个数据集下的平均指标
                    metrics = self.ckp.MeanTestMetric(compressrate, DtSetName,  len(ds))

                    print(color.fuchsia(f"   数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}dB ({snr_idx+1}/{len(self.args.SNRtest)}), 指标:{metric},时间:{tm.timer/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)"))

                    self.ckp.write_log(f"   数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}dB ({snr_idx+1}/{len(self.args.SNRtest)}), 整个数据集上的平均指标:{metrics}, 此SNR下整个数据集的测试时间:{tm.reset()/60.0:.3f}/{tm.hold()/60.0:.3f}(分钟)")

                    self.wr.WrTestMetric(DtSetName, compressrate, snr, metrics)
                    self.wr.WrTestOne(DtSetName, compressrate, snr, metrics)

        self.ckp.SaveTestLog()
        now2 = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.ckp.write_log(f"===================================  测试结束, 开始时刻:{now1}/结束时刻:{now2}, 用时:{tm.hold()/60.0:.3f}分钟 =======================================================")
        print(color.fuchsia(f"====================== 关闭测试日志  {self.ckp.log_file.name} ==================================="))
        self.ckp.done()
        print(color.fuchsia(f"\n#================================ 完成测试, 开始时刻:{now1}/结束时刻:{now2}, 用时:{tm.hold()/60.0:.3f}分钟 =======================================\n"))
        return

