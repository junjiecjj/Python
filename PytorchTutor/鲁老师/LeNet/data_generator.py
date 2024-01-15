# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

#  系统库
from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import sys,os

# 本项目自己编写的库
import srdata

sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        print(color.higyellowfg_whitebg( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n\
    idx_scale = {idx_scale} \n"))
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)



class DataGenerator(object):
    def __init__(self, args):
        print(color.fuchsia(f"\n#================================ DataLoader 开始准备 =======================================\n"))
        self.loader_train = None
        if args.wanttrain:
            datasets = []
            for trainname in args.data_train:
                # datasets.append(srdata.SRData(args, name=trainname, train=False, benchmark=True))

                if trainname in ['CBSD68', 'DIV2K','DIV2K_cut','Rain100L']:
                    trainset = srdata.SRData(args, name=trainname, train=True, benchmark=False)
                    print(f"train set is {trainname}\n")
                else:
                    print(f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n训练数据库里没有训练数据集{trainname}\n" )

            self.loader_train = dataloader.DataLoader(
                # MyConcatDataset(datasets),
                trainset,
                batch_size=32,  # 16
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=  args.n_threads,
            )

        self.loader_test = []
        if  args.wanttest:
            for testname in args.data_test:
                #print(color.higyellowfg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n testname = {testname}" ) )
                if testname in ['Set1','Set2','Set3','Set5', 'Set14', 'B100', 'Urban100', 'CBSD68', 'Rain100L']:
                    testset = srdata.SRData(args, name=testname, train=False, benchmark=True)
                    #print(f"testset = {testset}\n")
                elif testname in ['DIV2K',]:
                    testset = srdata.SRData(args, name=testname, train=False, benchmark=False)
                else:
                    print(f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n测试数据库里没有测试数据集{testname}\n" )

                self.loader_test.append(
                    dataloader.DataLoader(
                        testset,
                        batch_size=  args.test_batch_size,  #  1
                        shuffle=False,
                        pin_memory=not args.cpu,
                        num_workers= args.n_threads,
                    )
                )
        print(color.fuchsia(f"\n#================================ DataLoader 准备完毕 =======================================\n"))
























