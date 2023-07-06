#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:29:00 2022

@author: jack

https://zhuanlan.zhihu.com/p/88193562
"""

from line_profiler import LineProfiler
import random


def do_one_stuff(numbers):
    l = [numbers[i]/43 for i in range(len(numbers))]
def do_other_stuff(numbers):
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]
def do_stuff(numbers):
    for i in range(3):
        print(i)
        s = sum(numbers)
        do_one_stuff(numbers)
        do_other_stuff(numbers)
if __name__=='__main__':
    numbers = [random.randint(1,100) for i in range(1000)]
    lp = LineProfiler()
    lp.add_function(do_one_stuff)
    lp.add_function(do_other_stuff)
    lp_wrapper = lp(do_stuff)
    lp_wrapper(numbers)
    lp.print_stats()
    
    
"""



❯ kernprof -l  -v line_profiler2.py
0
1
2
Timer unit: 1e-06 s

Total time: 0.000515 s
File: line_profiler2.py
Function: do_one_stuff at line 13

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    13                                           def do_one_stuff(numbers):
    14         3        515.0    171.7    100.0      l = [numbers[i]/43 for i in range(len(numbers))]

Total time: 0.000921 s
File: line_profiler2.py
Function: do_other_stuff at line 15

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    15                                           def do_other_stuff(numbers):
    16         3        921.0    307.0    100.0      m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

Total time: 0.001631 s
File: line_profiler2.py
Function: do_stuff at line 17

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    17                                           def do_stuff(numbers):
    18         4          4.0      1.0      0.2      for i in range(3):
    19         3         65.0     21.7      4.0          print(i)
    20         3         55.0     18.3      3.4          s = sum(numbers)
    21         3        549.0    183.0     33.7          do_one_stuff(numbers)
    22         3        958.0    319.3     58.7          do_other_stuff(numbers)

Wrote profile results to line_profiler2.py.lprof
Timer unit: 1e-06 s


"""