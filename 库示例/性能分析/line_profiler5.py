#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:34:25 2022

@author: jack

https://blog.csdn.net/guofangxiandaihua/article/details/77825524
"""

from line_profiler import LineProfiler
import random
 
def do_other_stuff(numbers):
    s = sum(numbers)
 
def do_stuff(numbers):
    do_other_stuff(numbers)
    l = [numbers[i]/43 for i in range(len(numbers))]
    m = ['hello'+str(numbers[i]) for i in range(len(numbers))]
 
numbers = [random.randint(1,100) for i in range(1000)]
lp = LineProfiler()
lp.add_function(do_other_stuff)   # add additional function to profile
lp_wrapper = lp(do_stuff)
lp_wrapper(numbers)
lp.print_stats()

"""

❯ kernprof -l  -v line_profiler5.py
Timer unit: 1e-06 s

Total time: 7e-06 s
File: line_profiler5.py
Function: do_other_stuff at line 14

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    14                                           def do_other_stuff(numbers):
    15         1          7.0      7.0    100.0      s = sum(numbers)

Total time: 0.000478 s
File: line_profiler5.py
Function: do_stuff at line 17

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    17                                           def do_stuff(numbers):
    18         1         11.0     11.0      2.3      do_other_stuff(numbers)
    19         1        158.0    158.0     33.1      l = [numbers[i]/43 for i in range(len(numbers))]
    20         1        309.0    309.0     64.6      m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

Wrote profile results to line_profiler5.py.lprof
Timer unit: 1e-06 s


"""