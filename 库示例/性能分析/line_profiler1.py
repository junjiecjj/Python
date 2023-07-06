#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:26:22 2022

@author: jack

https://zhuanlan.zhihu.com/p/88193562
"""

from line_profiler import LineProfiler
import random

def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]
        

if __name__=='__main__':
    numbers = [random.randint(1,100) for i in range(1000)]
    lp = LineProfiler()
    lp_wrapper = lp(do_stuff)
    lp_wrapper(numbers)
    lp.print_stats()