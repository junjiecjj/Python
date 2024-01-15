#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:34:17 2024
@author: jack
"""

import numpy as np

class CRC(object):
    def __init__(self, K = 20, N = 31, chklen = 11, Generator = None):
        # if Generator == None:
        #     # self.G = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype = np.int8)
        # else:
        self.G = Generator
        assert len(self.G) == (chklen + 1)

        self.chklen = chklen
        self.n = self.chklen + 1
        self.K = K  ## 信息位长度
        self.N = N  ## 编码后总长度
        assert self.N == self.K + chklen
        return

    def encoder(self, uu):
        assert uu.shape[0] == self.K
        cc = np.zeros(self.N, dtype = np.int8)
        cc[:self.K] = uu.copy()
        q = []
        for i in range(self.K):
            if cc[i] == 1:
                q.append(1)
                # for j in range(self.n):
                cc[i:i + self.n] ^= self.G
            else:
                q.append(0)
        check_code = cc[-self.chklen:]
        cc[:self.K] = uu.copy()
        cc[self.K:] = check_code.copy()

        return cc, q, check_code

    def check(self, cc):
        # print(cc)
        uu = cc[:self.K].copy()
        cc_hat, _, _ = self.encoder(uu)
        print(cc)
        print(cc_hat)
        return  (cc == cc_hat).all()





uu = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0, 1])
Generator =  np.array([1, 1, 0, 1, 0, 1])

chklen = 5
K = 10
N = 15

crc = CRC(K,N,chklen, Generator)

cc, q, check_code = crc.encoder(uu)

print(f'信息oinfo：{len(uu)}\t\n {uu}' )
print(f'生成多项式p：{len(crc.G)}\t\n {crc.G}')
print(f'商q：{len(q)}\t\n {q}')
print(f'余数check_code：{len(check_code)}\t\n {check_code}')
print(f'编码cc：{len(cc)}\t\n {cc}')


cc[0] = 0
print(crc.check(cc))











