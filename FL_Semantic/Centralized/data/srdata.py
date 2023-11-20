
# -*- coding: utf-8 -*-
"""
Created on 2023/04/25

@author: Junjie Chen


此文件的作用：
实现DataSet类, 继承自data.Dataset, 继承必须将__getitem__()和__len__()实现。
"""

# 系统库
import glob
import random
import pickle
import io

import PIL.Image as pil_image
import sys,os

import numpy as np
import imageio
import torch
import torch.utils.data as data
import torchvision.transforms as tfs
#内存分析工具
from memory_profiler import profile
import objgraph


#  本项目自己编写的库
# sys.path.append("/home/jack/公共的/Pretrained-IPT-cjj/")
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()
from data import common



def search(root, target="JPEG"):
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            #print('[-]', path)
            item_list.extend(search(path, target))
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
        elif path.split('/')[-2] == target or path.split('/')[-3] == target or path.split('/')[-4] == target:
            item_list.append(path)
        else:
            ttt = 1
            #print('[!]', path)
    return item_list

class SRData(data.Dataset):
    #  test:  args, name = 'Set5',    train=False, benchmark=True
    #  test:  args, name = 'DIV2K',   train=False, benchmark=False
    #  train: args, name = 'DIV2K',   train=True,  benchmark=False
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        # self.split = 'train' if train else 'test'
        # self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.modelUse == 'VDSR')
        self.scale = args.scale  #  [1]
        self.idx_scale = 0
        # print(f" {name}  {train}  {benchmark}\n")

        if self.name in ['Set1','Set2','Set3','Set5', 'Set14', 'B100', 'Urban100']:
            self._set_filesystem_benchmark(args.dir_data)
            self.images_hr_png, self.images_lr_png = self._scan_benchmark()
            if self.args.useBIN == True:
                self.images_hr_bin, self.images_lr_bin = self._make_bin_img_magnify()

        if self.name in ['DIV2K','DIV2K_16','DIV2K_64',]:
            #print(f"srdata.py  69   {self.name}\n")
            self._set_filesystem_div2k(args.dir_data)
            self.images_hr_png, self.images_lr_png = self._scan_div2k()
            if self.args.useBIN == True:
                self.images_hr_bin, self.images_lr_bin = self._make_bin_img_magnify()


        # 去雨任务，且数据集是去雨任务的数据集'Rain100L'
        if self.name in ['Rain100L',] and self.args.derain:
            self.apath = os.path.join(args.dir_data, "Rain100L")
            self.dir_lr = os.path.join(self.apath, 'rainy')
            self.ext = ('.png', '.png')
            self.images_lr_png = sorted(search(self.dir_lr, "rain"))
            self.images_hr_png = [path.replace("rainy/","no") for path in self.images_lr_png]
            if self.args.useBIN == True:
                self.images_hr_bin, self.images_lr_bin = self._make_bin_img_rain100l()


        if self.name in ['CBSD68', ] and self.args.denoise:
            self._set_filesystem_CBSD68(args.dir_data)
            self.images_hr_png, self.images_lr_png = self._scan_CBSD68()
            if self.args.useBIN == True:
                self.images_hr_bin, self.images_lr_bin = self._make_bin_img_cbsd68()


        if train:
            print(f"train ={train} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            n_patches = args.batch_size * args.test_every # 16*1000
            n_images = len(args.data_train) * len(self.images_hr_png)

            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
            # print(f"n_patches = {n_patches}, n_images = {n_images} repead = {self.repeat}\n")
            # n_patches = 16000, n_images = 800  repead = 20


    def _scan_benchmark(self):
        names_hr = sorted( glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) )

        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}x{}{}'.format(s, filename, s, self.ext[1] )))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si]=names_hr

        return names_hr, names_lr


    def _scan_div2k(self):
        #  列表，列表的每个元素为“每张图片的路径+文件名”  '/home/jack/IPT-Pretrain/Data/DIV2K/DIV2K_train_HR/0801.png'
        names_hr = sorted( glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) )

        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}x{}{}'.format(s, filename, s, self.ext[1] )))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si]=names_hr

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr


    def _scan_CBSD68(self):
        names_hr = sorted( glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) )
        names_lr = sorted( glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0])) )

        return names_hr, names_lr


    def _set_filesystem_CBSD68(self, dir_data):
        # dir_data = '/home/jack/IPT-Pretrain/Data/'
        self.apath = os.path.join(dir_data, self.name)       # /home/jack/IPT-Pretrain/Data/DIV2K
        self.dir_lr = os.path.join(self.apath, 'noisy{sigma}'.format(sigma = str(self.args.sigma)) )         # /home/jack/IPT-Pretrain/Data/DIV2K/HR
        self.dir_hr = os.path.join(self.apath, 'original_png')  #  /home/jack/IPT-Pretrain/Data/DIV2K/LR_bicubic
        # if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')
        return

    def _set_filesystem_benchmark(self, dir_data):
        # dir_data = '/home/jack/IPT-Pretrain/Data/'
        self.apath = os.path.join(dir_data, 'benchmark', self.name)  # /home/jack/IPT-Pretrain/Data/benchmark/Set5
        self.dir_hr = os.path.join(self.apath, 'HR')                 # /home/jack/IPT-Pretrain/Data/benchmark/Set5/HR
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')     # /home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic

        self.ext = ('', '.png')
        return


    def _set_filesystem_div2k(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.ext = ('.png', '.png')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')

        data_range = [r.split('-') for r in self.args.data_range.split('/')]
        if self.train:
            data_range = data_range[0]   # data_range = ['1', '800']
        else:
            if  len(data_range) == 1:
                data_range = data_range[0]
            else:  # 进入这里
                data_range = data_range[1]
                #print(f"data_range = {data_range}\n")  # data_range = ['801', '810']
        self.begin, self.end = list(map(lambda x: int(x), data_range))  # [801, 810]
        return

    def _make_bin_img_magnify(self):
        path_bin =  os.path.join(self.apath, 'bin')
        os.makedirs(path_bin, exist_ok=True)

        os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)

        for s in self.scale:
            if s == 1:
                os.makedirs(self.dir_hr, exist_ok=True)
            else:
                os.makedirs(os.path.join(self.dir_lr.replace(self.apath, path_bin), 'X{}'.format(s)), exist_ok=True)

        images_hr_bin, images_lr_bin = [], [[] for _ in self.scale]
        for img in self.images_hr_png:
            b = img.replace(self.apath, path_bin)
            b = b.replace(self.ext[1], '.pt')
            images_hr_bin.append(b)
            self._check_and_load(self.args.ext, img, b, verbose=True)

        for idx_s, ll in enumerate(self.images_lr_png):
            for img in ll:
                b = img.replace(self.apath, path_bin)
                b = b.replace(self.ext[1], '.pt')
                images_lr_bin[idx_s].append(b)
                self._check_and_load(self.args.ext, img, b, verbose=True)
        return images_hr_bin, images_lr_bin




    def _make_bin_img_cbsd68(self):
        path_bin =  os.path.join(self.apath, 'bin')
        os.makedirs(path_bin, exist_ok=True)

        os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
        os.makedirs(self.dir_lr.replace(self.apath, path_bin), exist_ok=True)

        images_hr_bin, images_lr_bin = [], []
        for img in self.images_hr_png:
            b = img.replace(self.apath, path_bin)
            b = b.replace(self.ext[1], '.pt')
            images_hr_bin.append(b)
            self._check_and_load(self.args.ext, img, b, verbose=True)

        for img  in  self.images_lr_png:
            b = img.replace(self.apath, path_bin)
            b = b.replace(self.ext[1], '.pt')
            images_lr_bin.append(b)
            self._check_and_load(self.args.ext, img, b, verbose=True)
        return images_hr_bin, images_lr_bin



    """
    打开文件并读取 /cache/data/DIV2K/LR_bicubic/baboonx2.png
    并保存为二进制：/cache/data/DIV2K/bin/LR_bicubic/baboonx2.pt
    """
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)
        return

    #@profile
    def __getitem__(self, idx):
        if self.train == False and self.name in ['Rain100L'] and self.args.derain:  # 不进入此处
            norain, rain, filename = self._load_rain_test(idx)
            pair = common.set_channel(*[norain, rain], n_channels = self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range = self.args.rgb_range)
            # 先返回rain，再返回norain
            return pair_t[1], pair_t[0], filename
        if self.train == False and self.name in ['CBSD68'] and self.args.denoise: # 不进入此处
            hr,lr, filename = self._load_cbsd68_test(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0],pair_t[0], filename
        if self.name in ['Set1','Set2','Set3','Set5', 'Set14', 'B100', 'Urban100','DIV2K','DIV2K_16','DIV2K_64',]:
            # 默认，图像缩放任务
            lr, hr, filename = self._load_file(idx)
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)  # rgb_range=255
            return pair_t[0], pair_t[1], filename


#====================================================================================
#  dataset length and get index
#====================================================================================

    def __len__(self):
        if self.train:
            #return len(self.images_hr_png) * self.repeat
            return len(self.images_hr_png)
        else:
            if self.args.derain:
                return int(len(self.images_hr_png)/self.args.derain_test)
            return len(self.images_hr_png)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr_png)
        else:
            return idx

#====================================================================================
#  load img file
#====================================================================================

    def _load_cbsd68_test(self, idx):
        idx = self._get_index(idx)

        if self.args.useBIN == True:
            f_hr = self.images_hr_bin[idx]
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            #print(f"\n{self.name} 正在使用二进制图像源 {f_hr}\n")

            with open(f_hr, 'rb') as _f:
                origin = pickle.load(_f)
            if self.images_lr_bin != []:
                f_lr = self.images_lr_bin[idx]
                with open(f_lr, 'rb') as _f:
                    noise = pickle.load(_f)
            else:
                noise = []

            return origin, noise, filename

        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        origin = imageio.imread(f_hr)
        noise = imageio.imread(f_lr)
        return origin, noise, filename


    def _load_rain_test(self, idx):
        idx = self._get_index(idx)
        if self.args.useBIN ==  True:
            f_hr = self.images_hr_bin[idx]
            f_lr = self.images_lr_bin[idx]
            filename, _ = os.path.splitext(os.path.basename(f_lr))
            with open(f_hr, 'rb') as _f:
                norain = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                rain = pickle.load(_f)
            #print(f"\n{self.name} 正在使用二进制图像源 {f_lr}\n")
            return norain, rain, filename

        f_hr = self.images_hr_png[idx]
        f_lr = self.images_lr_png[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.args.useBIN == True:
            f_hr = self.images_hr_bin[idx]
            f_lr = self.images_lr_bin[self.idx_scale][idx]
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            #print(f"\n{self.name} 正在使用二进制图像源 {f_hr}\n")
            return lr, hr, filename

        f_hr = self.images_hr_png[idx]
        f_lr = self.images_lr_png[self.idx_scale][idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)
        return lr, hr, filename

#====================================================================================
#  get patch
#====================================================================================

    def get_cbsd68_patch_hr(self, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.args.patch_size,  # 48
                scale=1
            )
        return hr

    def get_cbsd68_patch(self, *arg):
        scale = self.scale[self.idx_scale]
        def  _get_patch_cbsd68(img):
            if self.train:
                img = self.get_patch_img_hr(
                    img,
                    patch_size=self.args.patch_size,  # 48
                    scale=1
                )
            return img
        return [_get_patch_cbsd68(a) for a in arg]

    def get_patch_hr(self, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.args.patch_size,  # 48
                scale=1
            )

        return hr


    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        ih, iw = img.shape[:2]

        tp = patch_size  # 48
        ip = tp // scale  # 48

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret



    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if  self.train:

            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size*scale,
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
            #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno},  idx_scale = {idx_scale} \n"))
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
        return
