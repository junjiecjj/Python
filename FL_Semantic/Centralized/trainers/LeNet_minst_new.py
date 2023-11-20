# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""

import sys, os

import numpy as np

import torch

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

sys.path.append("..")
# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()

from . import common, MetricsLog
from attack.fgsm        import  FGSM_Attack
from attack.sem_attack  import  Sem_Attack
from model import AutoEncoder


fontpath  = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

class LeNetMinst_Sem_Trainer():
    def __init__(self, args, loader, model, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test[0]
        self.classify = model
        self.device = args.device

        # 设置不同扰动大小
        self.epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        print(f"len(self.loader_train) = {len(self.loader_train)}, len(self.loader_train.dataset) = {len(self.loader_train.dataset)}")
        print(f"len(self.loader_test ) = {len(self.loader_test )}, len(self.loader_test.dataset) = {len(self.loader_test.dataset)}")

        self.ckp.print_parameters('AttackLog.txt', net = self.classify, name = "LeNet for Minst")
        return

    def Sem_R_SNR(self, SNRtestlist = np.arange(-2, 21, 1)):
        tm = common.myTimer()
        self.classify.eval()
        raw_dim       = 28 * 28

        originRecoder = MetricsLog.TesRecorder(Len = 4)
        attackRecoder = MetricsLog.AttackRecorder(Len = 4)

        print(color.higred(f"\n#==================== 开始对抗测试:{tm.start_str} =======================\n"))
        self.ckp.write_attacklog(f"#==================== 开始对抗测试: {tm.start_str} =======================\n" )
        self.ckp.write_attacklog(f" 压缩率:{self.args.CompRate} \n 训练信噪比: {self.args.SNRtrain}\n 测试信噪比: {self.args.SNRtest}\n 对抗强度:{self.epsilons}\n\n" )

        for idx_c, comrate in enumerate(self.args.CompRate):
            encoded_dim = int(raw_dim * comrate)
            print(f"压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)})"); self.ckp.write_attacklog(f"压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)})" )
            for idx_s, snrtrain in enumerate(self.args.SNRtrain):
                originRecoder.add_item( comrate, snrtrain,)
                print(f"  信噪比:{snrtrain} dB ({idx_s+1}/{len(self.args.SNRtrain)})"); self.ckp.write_attacklog(f"  信噪比: {snrtrain} dB ({idx_s+1}/{len(self.args.SNRtrain)})\n\n" )
                self.communicator = AutoEncoder.AED_cnn_mnist(encoded_space_dim = encoded_dim, snr = snrtrain, quantize = self.args.quantize ).to(self.device)
                attacker = Sem_Attack(self.classify, self.communicator.encoder, self.communicator.decoder)
                if self.args.pretrain == True:
                    d1 = "NoQuan_JoinLoss"
                    d2 = "R_SNR"
                    # predir = f"/home/{self.args.user_name}/SemanticNoise_AdversarialAttack/ModelSave/{d1}/{d2}/AE_Minst_noQuant_joinLoss_R={comrate:.1f}_trainSnr=noiseless.pt"
                    predir = f"{self.args.user_home}/SemanticNoise_AdversarialAttack/ModelSave/{d1}/{d2}/AE_Minst_noQuant_joinLoss_R={comrate:.1f}_trainSnr={snrtrain}.pt"
                    self.communicator.load_state_dict(torch.load(predir, map_location = self.device ))
                    self.communicator.train()
                for snr_ti, snrtest in enumerate(SNRtestlist):
                    self.ckp.write_attacklog(f"    #=====压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)}),训练信噪比:{snrtrain} dB ({idx_s+1}/{len(self.args.SNRtrain)}),测试信噪比为{snrtest}(dB)({snr_ti+1}/{len(SNRtestlist)})下测试 =======")
                    print(color.red(f"    #=======压缩率:{comrate:.2f} ({idx_c+1}/{len(self.args.CompRate)}),训练信噪比:{snrtrain} dB ({idx_s+1}/{len(self.args.SNRtrain)}),测试信噪比为{snrtest}(dB)({snr_ti+1}/{len(SNRtestlist)})下测试 ========="))
                    self.ckp.write_attacklog("        #===================== 原图传输 ======================")
                    print(color.green("        #==================== 原图传输 ==================="))
                    self.communicator.set_snr(snrtest)
                    attackRecoder.add_item( comrate, snrtrain, snrtest )
                    originRecoder.add_snr( comrate, snrtrain, snrtest )

                    metric = MetricsLog.Accumulator(5)
                    with torch.no_grad():
                        for batch, (imgs, labels) in enumerate(self.loader_test):
                            imgs, labels            = common.prepare(self.device, self.args.precision, imgs, labels)
                            # 原始样本语义传输
                            recv_origin_imgs        = self.communicator(imgs)
                            # 传输后分类
                            pred_origin_labs        = self.classify(recv_origin_imgs).detach().cpu()
                            # 计算准确率
                            labels                  = labels.detach().cpu()
                            acc                     = common.accuracy(pred_origin_labs, labels )
                            imgs, recv_origin_imgs  = imgs.detach().cpu(), recv_origin_imgs.detach().cpu()

                            batch_01_psnr           = common.PSNR_torch(imgs, recv_origin_imgs, )
                            imgs                    = common.data_inv_tf_cnn_mnist_batch_3D(imgs)
                            recv_origin_imgs        = common.data_inv_tf_cnn_mnist_batch_3D(recv_origin_imgs)
                            batch_avg_psnr          = common.PSNR_torch_Batch(imgs, recv_origin_imgs, )
                            metric.add(acc, batch_01_psnr, batch_avg_psnr, 1, imgs.size(0))
                        accuracy     = metric[0]/metric[4]
                        avg_batch_01 = metric[1]/metric[3]
                        avg_batch    = metric[2]/metric[3]
                        originRecoder.assign( comrate,  snrtrain,  torch.tensor([accuracy, avg_batch_01, avg_batch ]))
                        self.ckp.write_attacklog(f"        {snrtest:>10}, {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} ")
                        print(color.green(f"        {snrtest:>10}(dB), {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} "))

                    self.ckp.write_attacklog(f"        #=============== 带对抗噪声, 时刻: {tm.now()} ================")
                    print(color.green(f"        #=============== 带对抗噪声, 时刻: {tm.now()} ================"))
                    for eps in self.epsilons:
                        attackRecoder.add_eps( comrate, snrtrain,  snrtest,  eps)
                        metric = MetricsLog.Accumulator(5)
                        # with torch.no_grad():
                        for batch, (imgs, labels) in enumerate(self.loader_test):
                            imgs, labels     = common.prepare(self.device, self.args.precision, imgs, labels)        # 选择精度且to(device)
                            # 对原始样本进行攻击
                            grad_sign = attacker.perturb(imgs, labels, eps = eps)
                            # 攻击样本语义传输
                            self.communicator.eval()
                            recv_imgs        = self.communicator(imgs, attack_vector = grad_sign)
                            # 传输后分类
                            pred_labs        = self.classify(recv_imgs).detach().cpu()
                            # 计算准确率
                            labels           = labels.detach().cpu()
                            acc              = common.accuracy(pred_labs, labels )
                            imgs, recv_imgs  = imgs.detach().cpu(), recv_imgs.detach().cpu()

                            batch_01_psnr    = common.PSNR_torch(imgs, recv_imgs, )
                            imgs         = common.data_inv_tf_cnn_mnist_batch_3D(imgs)
                            recv_imgs        = common.data_inv_tf_cnn_mnist_batch_3D(recv_imgs)
                            batch_avg_psnr   = common.PSNR_torch_Batch(imgs, recv_imgs, )
                            metric.add(acc, batch_01_psnr, batch_avg_psnr,  1, imgs.size(0))
                        accuracy     = metric[0]/metric[4]
                        avg_batch_01 = metric[1]/metric[3]
                        avg_batch    = metric[2]/metric[3]
                        attackRecoder.assign(comrate, snrtrain,  snrtest, torch.tensor([accuracy, avg_batch_01, avg_batch ]))
                        self.ckp.write_attacklog(f"        epsilon:{eps:>10}, {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} ")
                        print(color.green(f"        epsilon:{eps:>10}, {accuracy:>12.3f} {avg_batch_01:>12.3f}, {avg_batch:>12.3f} "))
                    originRecoder.save(self.ckp.savedir)
                    attackRecoder.save(self.ckp.savedir)
        originRecoder.save(self.ckp.savedir)
        attackRecoder.save(self.ckp.savedir)
        print(color.higred(f"\n#================================ 完成测试, 开始时刻:{tm.start_str}/结束时刻:{tm.now()}  =======================================\n"))
        return











































