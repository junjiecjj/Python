"""
FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
import os, sys
import torch


sys.path.append("..")
#  工具
from trainers import common as tcommon



class Sem_Attack(object):
    """Reproduce Fast Gradients Sign Method (FGSM)
    in the paper 'Explaining and harnessing adversarial examples'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- Magnitude of perturbation
    """
    def __init__(self, target_cls, encoder, decoder, pixel_min = -1, pixel_max = 1):
        self.min = pixel_min
        self.max = pixel_max
        self.attack_name = "Before Channel attack"
        self.criterion = nn.CrossEntropyLoss()

        self.target_cls = target_cls
        self.encoder = encoder
        self.decoder = decoder
        self.device = next(target_cls.parameters()).device
        return

    def perturb(self, imgs, labels, eps = 0.1,):
        self.eps = eps
        self.target_cls.eval()
        self.encoder.train()
        self.decoder.train()

        # print(f"1   imgs.shape = {imgs.shape}, labels.shape = {labels.shape}")
        # imgs = imgs.to(self.device)
        # labels = labels.to(self.device)

        imgs.requires_grad = True
        # print(f"0   imgs.requires_grad = {imgs.requires_grad}")

        encoded = self.encoder(imgs)
        # encoded.requires_grad = True
        # print(f"1   encoded.requires_grad = {encoded.requires_grad}")

        decoded = self.decoder(encoded)
        pred_labs = self.target_cls(decoded)
        loss = self.criterion(pred_labs, labels)

        # ###  1
        # loss.backward()
        # gradient_sign = encoded.grad.data # .sign()
        # adversarial_examples = imgs +  self.eps * grad_sign
        # adversarial_examples = torch.clamp(adversarial_examples, min = -1, max = 1)

        ###  2
        gradient_sign = torch.autograd.grad(loss, encoded)[0] #.sign()
        adversarial_encoded = encoded +  self.eps * gradient_sign
        ## adversarial_encoded = torch.clamp(adversarial_examples, min = self.min, max = self.max)
        ## print(f"2   adversarial_examples.shape = {adversarial_examples.shape}, labels.shape = {labels.shape}")
        print(f"2 {gradient_sign.min()}, {gradient_sign.max()}")
        # print(f"0  {gradient_sign}")
        return gradient_sign

    def inference(self, data_loader, save_path = '', file_name = '',  epsilon = 0.1):
        """[summary]
        Arguments:
            save_path {[type]} -- [description]
            data_loader {[type]} -- [description]
        """
        self.target_cls.eval()

        correct = 0
        accumulated_num = 0
        batch_01_psnr = 0.0
        batch_psnr = 0.0
        batch_sum = 0
        for batch, (imgs, labels) in enumerate(data_loader):
            batch_sum        += 1
            imgs, labels     = tcommon.prepare(self.device, 'single', imgs, labels)
            adv_imgs, labels = self.perturb(imgs, labels, eps = epsilon)
            outputs          = self.target_cls(adv_imgs)
            correct          += (outputs.argmax(axis = 1) == labels).sum().item()
            accumulated_num  += imgs.size(0)

            imgs, adv_imgs   =  imgs.detach().cpu(), adv_imgs.detach().cpu()
            batch_01_psnr    += tcommon.PSNR_torch(imgs, adv_imgs, )
            imgs             =  tcommon.data_inv_tf_cnn_mnist_batch_3D(imgs)
            adv_imgs         =  tcommon.data_inv_tf_cnn_mnist_batch_3D(adv_imgs)
            batch_psnr       += tcommon.PSNR_torch_Batch(imgs, adv_imgs, )
        acc =  correct / accumulated_num
        batch_01_psnr /= batch_sum
        batch_psnr    /= batch_sum
        return acc, batch_01_psnr, batch_psnr



















