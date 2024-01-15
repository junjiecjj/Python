#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
https://github.com/RQC-QApp/polar-codes/tree/master
https://github.com/mcba1n/polar-codes
"""



import math
import numpy as np
import copy
import sys
import os
from functools import reduce
import utility



class CodeWord(object):
    def __init__(self, length):
        self.p = None
        self.cfrom  = None
        self.cc = np.zeros(length, dtype = np.int8)
        self.p0 = np.zeros(length)
        return

class Polar(object):
    def __init__(self, args):
        ## code parameters
        self.args = args
        self.codedim = 64                            # 码的维数，编码前长度
        self.codelen = 128                           # 码的长度，编码后长度，码字长度
        self.codechk = self.codelen - self.codedim      # 校验位的个数
        self.coderate = self.codedim/self.codelen       # 码率
        self.list_size = 8
        self.frozenbook = np.loadtxt("Polar_Without_CRC_64_128.txt", delimiter=' ', dtype=np.int8)
        return

    def reverse(self, cc):
        N = int(math.log2(self.codelen))
        for level in range(N):
            interval = 2**level
            for merge_start in range(0, self.codelen, 2*interval):
                for stride in range(interval):
                    cc[merge_start + stride] ^= cc[interval + merge_start + stride]
        return cc

    ##=============================  SC, withoutCRC =============================================
    def encoder_withoutCRC(self, uu):
        cc = np.zeros(self.codelen, dtype = np.int8)
        cc[np.where(self.frozenbook == 0)] = uu

        N = int(math.log2(self.codelen))
        for level in range(N):
            interval = 2**level
            for merge_start in range(0, self.codelen, 2*interval):
                for stride in range(interval):
                    cc[merge_start + stride] ^= cc[interval + merge_start + stride]
        return cc

    def decoderSC_withoutCRC(self, p0, cc, frozenbook, length):
        ## 递归终止条件，判决。
        if length == 1:
            cc[0] = p0[0] < 0.5 and frozenbook[0] == 0
        else:
            ## 左子树处理
            length = int(length/2)
            lp0 = np.zeros(length)
            for i in range(length):
                lp0[i] = p0[i]*p0[i + length] + (1 - p0[i])*(1 - p0[i + length])
            ## 递归处理
            self.decoderSC_withoutCRC(lp0, cc, frozenbook, length)
            ## 右子树处理
            rp0 = np.zeros(length)
            for i in range(length):
                if cc[i]:
                    rp0[i] = (1 - p0[i])*p0[i + length]/(1 - lp0[i])
                else:
                    rp0[i] = p0[i] * p0[i + length] / lp0[i]
            ## 递归处理
            self.decoderSC_withoutCRC(rp0, cc[length:], frozenbook[length:], length)
            ## 根节点左边合并处理
            for i in range(length):
                cc[i] ^= cc[i + length]
        return

    ##=============================  SCL, withoutCRC =============================================
    def decoder_SCL(self, nodes, pos, length):
        ## 递归终止条件
        if length == 1:
            if self.frozenbook[pos]:
                for i in range(len(nodes)):
                    nodes[i].cc[0] = 0
                    nodes[i].p *= nodes[i].p0[0]
            else:
                cws_num =  len(nodes)
                for i in range(cws_num):
                    nodes.append(copy.deepcopy(nodes[i]))
                for i in range(2):
                    for j in range(cws_num):
                        nodes[i*cws_num + j].cc[0] = i
                        if i == 0:
                            nodes[i*cws_num + j].p *= nodes[i*cws_num + j].p0[0]
                        else:
                            nodes[i*cws_num + j].p *= (1 - nodes[i*cws_num + j].p0[0])
                nodes.sort(key = lambda x : x.p, reverse=True)
                if len(nodes) > self.list_size:
                    nodes[:] = nodes[:self.list_size]
                norm  = nodes[0].p
                if norm > 0:
                    while nodes[-1].p == 0:
                        nodes.pop()
                    for i in range(len(nodes)):
                        nodes[i].p /= norm
                else:
                    for i in range(len(nodes)):
                        nodes[i].p = 1
        else:
            ## 左节点处理
            length =  length//2
            # print(length)
            left = []
            for cfrom, cw in enumerate(nodes):
                lcw = CodeWord(length)
                lcw.p = cw.p
                lcw.cfrom = cfrom
                for i in range(length):
                    lcw.p0[i] = cw.p0[i] * cw.p0[i + length] + (1 - cw.p0[i]) * (1 - cw.p0[i + length])
                left.append(lcw)
            ## 递归处理
            self.decoder_SCL(left, pos, length)
            ## 右节点处理
            right = []
            for cfrom, lcw in enumerate(left):
                rcw = CodeWord(length)
                rcw.p = lcw.p
                rcw.cfrom = cfrom
                cw = nodes[lcw.cfrom]
                for i in range(length):
                    lcw_p0_i = cw.p0[i]*cw.p0[i+length] + (1 - cw.p0[i])*(1 - cw.p0[i+length])
                    if lcw.cc[i]:
                        rcw.p0[i] = (1-cw.p0[i])*cw.p0[i + length]/(1-lcw_p0_i) if (1 - lcw_p0_i) else 0.5
                    else:
                        rcw.p0[i] = cw.p0[i]*cw.p0[i + length]/lcw_p0_i if lcw_p0_i else 0.5
                right.append(rcw)
            ## 向右递归
            self.decoder_SCL(right, pos + length, length)
            ## 合并，替换
            new_node = []
            for i, rcw in enumerate(right):
                ncw = CodeWord( length * 2 )
                lcw = left[rcw.cfrom]
                for j in range(length):
                    ncw.cc[j] = lcw.cc[j] ^ rcw.cc[j]
                    ncw.cc[j + length] = rcw.cc[j]
                ncw.p = rcw.p
                ncw.cfrom = lcw.cfrom
                new_node.append(ncw)
            nodes.clear()
            for cw in new_node:
                nodes.append(copy.deepcopy(cw))
        return

    def decoderSCL_withoutCRC(self, yy_p0 ):
        nodes = [CodeWord(self.codelen)]
        nodes[0].p = 1
        nodes[0].cfrom = 0
        nodes[0].p0 = copy.deepcopy(yy_p0)
        self.decoder_SCL(nodes, 0, self.codelen)
        # print(len(nodes))

        cc_hat = nodes[0].cc
        cc = self.reverse(nodes[0].cc.copy())

        uu_hat = np.zeros(self.codedim, dtype = np.int8)
        j = 0
        for i in range(self.codelen):
            if self.frozenbook[i]:
                assert cc[i] == 0
            else:
                uu_hat[j] = cc[i]
                j += 1
        return uu_hat, cc_hat


# class CodeWord1(object):
#     def __init__(self, length):
#         self.p = 1
#         self.cfrom  = 0
#         self.cc = np.zeros(length, dtype = np.int8)
#         self.p0 = np.zeros(length)
#         return

# def cha(node):
#     node.append(CodeWord1(128))
#     node[-1].p = -2.189
#     node[-1].cfrom = 3.3

#     new_node = [CodeWord1(12)]
#     new_node[-1].p = 1.23
#     new_node.append(CodeWord1(12))
#     new_node[-1].p = 2.345
#     new_node.append(CodeWord1(12))
#     new_node[-1].p = 4.232

#     node.clear()
#     for cw in new_node:
#         node.append(cw)

#     # tmp = copy.deepcopy(node[:])
#     # node[:] = copy.deepcopy(new_node)
#     # new_node[:] = copy.deepcopy(tmp[:])
#     return


# nodes = [CodeWord1(12)]

# nodes.append(CodeWord1(12))
# nodes[-1].p = -2.189
# nodes[-1].cfrom = 3.3

# nodes.append(CodeWord1(12))
# nodes[-1].p = 1.29
# nodes[-1].cfrom = 66.3

# nodes.append(CodeWord1(12))
# nodes[-1].p = 4.59
# nodes[-1].cfrom = 72.3


# cha(nodes)























































































































































































































































































































































































































































































































































































































































































































































































