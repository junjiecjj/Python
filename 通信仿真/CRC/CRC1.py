#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:19:17 2022

https://www.jianshu.com/p/8185e125afaf

@author: jack
"""

class CRC:
    """循环冗余检验

    parameters
    ----------

    info : list
        需要被编码的信息

    crc_n : int, default: 32
        生成多项式的阶数

    p : list
        生成多项式

    q : list
        crc后得到的商

    check_code : list
        crc后得到的余数，即计算得到的校验码

    code : list
        最终的编码

    ----------

    """

    def __init__(self, info, crc_n=32):
        self.info = info

        # 初始化生成多项式p
        loc = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 2, 1, 0]
        if crc_n == 8:
            loc = [8, 2, 1, 0]
        elif crc_n == 16:
            loc = [16, 15, 2, 0]
        p = [0 for i in range(crc_n + 1)]
        for i in loc:
            p[i] = 1

        info = self.info.copy()
        times = len(info)
        n = crc_n + 1

        # 左移补零
        for i in range(crc_n):
            info.append(0)
        # 除
        q = []
        for i in range(times):
            if info[i] == 1:
                q.append(1)
                for j in range(n):
                    info[j + i] = info[j + i] ^ p[j]
            else:
                q.append(0)

        # 余数
        check_code = info[-crc_n::]

        # 生成编码
        code = self.info.copy()
        for i in check_code:
            code.append(i)

        self.crc_n = crc_n
        self.p = p
        self.q = q
        self.check_code = check_code
        self.code = code

    def print_format(self):
        """格式化输出结果"""

        print('{:10}\t{}'.format('信息：', self.info))
        print('{:10}\t{}'.format('生成多项式：', self.p))
        print('{:10}\t{}'.format('商：', self.q))
        print('{:10}\t{}'.format('余数：', self.check_code))
        print('{:10}\t{}'.format('编码：', self.code))


import numpy as np
# 随机信息
m = np.random.randint(0, 2, 10)
m = list(m)
crc = CRC(m, 32)
crc.print_format()