

import torch
import torch.nn as nn
import os, sys
import numpy as np

sys.path.append("..")
#  工具
from trainers import common as tcommon



class FGSM_Attack(object):
    """Reproduce Fast Gradients Sign Method (FGSM)
    in the paper 'Explaining and harnessing adversarial examples'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- Magnitude of perturbation
    """
    def __init__(self, target_cls,    pixel_min = -1, pixel_max = 1):
        self.min = pixel_min
        self.max = pixel_max
        self.attack_name = "FGSM"
        self.criterion = nn.CrossEntropyLoss()

        self.target_cls = target_cls
        self.device = next(target_cls.parameters()).device

        return


    def perturb(self, imgs, labels, fgsm_eps = 0.1,):
        self.eps = fgsm_eps
        self.target_cls.eval()

        # print(f"1   imgs.shape = {imgs.shape}, labels.shape = {labels.shape}")
        # imgs = imgs.to(self.device)
        # labels = labels.to(self.device)

        imgs.requires_grad = True
        # print(f"0   imgs.requires_grad = {imgs.requires_grad}")
        outputs = self.target_cls(imgs)
        loss = self.criterion(outputs, labels)

        # ###  1
        # loss.backward()
        # grad_sign = imgs.grad.data.sign()
        # adversarial_examples = imgs +  self.eps * grad_sign
        # adversarial_examples = torch.clamp(adversarial_examples, min = -1, max = 1)

        ###  2
        gradient_sign = torch.autograd.grad(loss, imgs)[0].sign()
        adversarial_examples = imgs +  self.eps * gradient_sign
        adversarial_examples = torch.clamp(adversarial_examples, min = self.min, max = self.max)
        # print(f"2   adversarial_examples.shape = {adversarial_examples.shape}, labels.shape = {labels.shape}")
        return adversarial_examples, labels

    def inference(self, data_loader, save_path = '', file_name = '',  epsilon = 0.1):
        """[summary]
        Arguments:
            save_path {[type]} -- [description]
            data_loader {[type]} -- [description]
        """
        self.target_cls.eval()

        correct = 0
        accumulated_num = 0
        batch_01_psnr   = 0.0
        batch_psnr      = 0.0
        batch_sum       = 0

        for batch, (imgs, labels) in enumerate(data_loader):
            batch_sum        += 1
            imgs, labels     = tcommon.prepare(self.device, 'single', imgs, labels)
            adv_imgs, labels = self.perturb(imgs, labels, fgsm_eps = epsilon)
            outputs          = self.target_cls(adv_imgs)
            correct          += (outputs.argmax(axis = 1) == labels).sum().item()
            accumulated_num  += imgs.size(0)

            imgs, adv_imgs   =  imgs.detach().cpu(), adv_imgs.detach().cpu()
            batch_01_psnr    += tcommon.PSNR_torch(imgs, adv_imgs, )
            imgs             =  tcommon.data_inv_tf_cnn_mnist_batch_3D(imgs)
            adv_imgs         =  tcommon.data_inv_tf_cnn_mnist_batch_3D(adv_imgs)
            batch_psnr       += tcommon.PSNR_torch_Batch(imgs, adv_imgs, )
        acc =  correct / accumulated_num
        batch_01_psnr  /= batch_sum
        batch_psnr     /= batch_sum

        cols = 5
        adv_exps = []
        ## 固定的选前几张图片
        idx         = np.arange(0, cols, 1)
        ## 原图
        labels      = data_loader.dataset.targets[idx].to(self.device)
        real_image  = data_loader.dataset.data[idx]
        ## 原图的预处理
        real_image  = tcommon.data_tf_cnn_mnist_batch(real_image).to(self.device)
        ## 对原始样本进行攻击
        adv_imgs    = self.perturb(real_image, labels, fgsm_eps = epsilon)[0]
        # adv_imgs,   = tcommon.prepare(self.device, "single", adv_imgs)
        ## 攻击样本分类
        adv_labs    = self.target_cls(adv_imgs).detach().cpu().argmax(axis = 1)
        adv_imgs    = adv_imgs.detach().cpu()
        adv_imgs    = tcommon.data_inv_tf_cnn_mnist_batch_2D(adv_imgs)
        for i in range(cols):
            adv_exps.append((labels[i].item(), adv_labs[i].item(), adv_imgs[i] ))

        return acc, batch_01_psnr, batch_psnr, adv_exps



















