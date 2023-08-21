

import numpy  as np
import os,time
import random
# from multiprocessing import Process, Pool
# import multiprocessing


class LDPC(object):
    def __init__(self):
        self.codedim  = 20         # 码的维数，编码前长度
        self.codelen  = 30         # 码的长度，编码后长度，码字长度
        self.codechk  = 10         # 校验位的个数
        # self.coderate = 0.0      # 码率
        # self.num_row  = 3
        # self.num_col  = 7
        # self.encH = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        # self.encH     = np.array([[1,0,0,1,1,0,0],[0,1,0,0,1,1,1],[0,0,1,0,0,1,1]])
        self.encH     = np.random.randint(low = 0, high = 2, size = (self.codechk, self.codelen), dtype = np.int8 )
        self.a        = np.array([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6],[0.7, 0.8]])
        ##
        print("LDPC初始化完成...")

    def chan(self,i):
        time.sleep(random.random())
        self.a += i
        time.sleep(random.random())
        c=[1,2,3]
        time.sleep(random.random())
        return self.a ,i**2, c

    def achan(self,i):
        self.a[i, :] += i

    def encoder(self, uu):
        cc = np.zeros(self.codelen, dtype = np.int8)
        cc[self.codechk:] = uu
        for i in range(self.codechk):
            cc[i] = np.logical_xor.reduce(np.logical_and(uu[:], self.encH[i, self.codechk:]))
        return cc

    def decoder(self, yy):
        uu_hat = np.zeros(self.codedim, dtype = np.int8)
        cc_hat = np.zeros(self.codelen, dtype = np.int8 )

        return












































































