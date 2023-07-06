#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:29:41 2022

@author: jack
"""

import multiprocessing

def f(n, a):

    n.value = 3.14

    a[0] = 5

if __name__ == '__main__':

    num = multiprocessing.Value('d', 0.0)

    arr = multiprocessing.Array('i', range(10))

    p = multiprocessing.Process(target=f, args=(num, arr))

    p.start()

    p.join()

    print(num.value)

    print(arr[:])