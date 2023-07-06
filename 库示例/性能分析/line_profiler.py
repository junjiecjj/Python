#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:09:30 2022

@author: jack
"""

# line_profiler_test.py
from line_profiler import LineProfiler
import numpy as np

@profile
def test_profiler():
    for i in range(100):
        a = np.random.randn(100)
        b = np.random.randn(1000)
        c = np.random.randn(10000)
    return None

if __name__ == '__main__':
    test_profiler()
    
    
    
    
"""
run with:
kernprof -l  -v line_profiler.py

这里我们就直接得到了逐行的性能分析结论。简单介绍一下每一列的含义：
代码在代码文件中对应的行号、被调用的次数、该行的总共执行时间、单次执行所消耗的时间、执行时间在该函数下的占比，最后一列是具体的代码内容。

❯ kernprof -l  -v line_profiler.py
Wrote profile results to line_profiler.py.lprof
Timer unit: 1e-06 s

Total time: 0.040359 s
File: line_profiler.py
Function: test_profiler at line 13

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    13                                           @profile
    14                                           def test_profiler():
    15       101        274.0      2.7      0.7      for i in range(100):
    16       100       1254.0     12.5      3.1          a = np.random.randn(100)
    17       100       3757.0     37.6      9.3          b = np.random.randn(1000)
    18       100      35073.0    350.7     86.9          c = np.random.randn(10000)
    19         1          1.0      1.0      0.0      return None


"""
