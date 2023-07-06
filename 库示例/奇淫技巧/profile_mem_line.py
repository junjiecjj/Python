#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:49:37 2022

@author: jack

ipython -m memory_profiler memory_profiler_test.py 

"""

from memory_profiler import profile
import pretty_errors

# 【重点】进行配置
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
)

@profile(precision=4, stream=open('memory_profiler.log','w+'))

@profile
def test1():
    c = []
    a = [1, 2, 3] * (2 ** 20)
    b = [1] * (2 ** 20)
    c.extend(a)
    c.extend(b)
    del b
    del c

@profile
def test2():
    c=0
    for item in range(1000):
        c+=1
    print (c)





#第 3 式：按调用函数分析代码运行时间,profile
#平凡方法
def relu(x):
	return (x if x>0 else 0)

def main2():
	result = [relu(x) for x in range(-100000,100000,1)]
	return result
import profile as Profile

def test3():
    Profile.run('main2()')
    
#第 3 式：按调用函数分析代码运行时间,cprofile
import cProfile


# 直接把分析结果打印到控制台
def test4():
    cProfile.run("main2()")
    
 # 把分析结果保存到文件中
def test5():
    cProfile.run("main2()", filename="result.out")

# 增加排序方式
def test6():
    cProfile.run("main2()", filename="result.out", sort="cumulative")


#第 4 式：按行分析代码运行时间
#平凡方法
from line_profiler import LineProfiler
def test7():
    lprofile = LineProfiler(main2,relu)
    lprofile.run('main2()')
    lprofile.print_stats()


#耗时测试

#$ python -m cProfile main.py
#测试某代码块耗时

# 代码块耗时定义
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timeblock(label):
    tic = perf_counter()
    try:
        yield
    finally:
        toc = perf_counter()
        print('%s : %s' % (label, toc - tic))

# 代码块耗时测试
def test8():
    with timeblock('counting'):
        pass
#代码耗时优化的一些原则

#专注于优化产生性能瓶颈的地方，而不是全部代码。
#避免使用全局变量。局部变量的查找比全局变量更快，将全局变量的代码定义在函数中运行通常会快 15%-30%。
#避免使用.访问属性。使用 from module import name 会更快，将频繁访问的类的成员变量 self.member 放入到一个局部变量中。
#尽量使用内置数据结构。str, list, set, dict 等使用 C 实现，运行起来很快。
#避免创建没有必要的中间变量，和 copy.deepcopy()。
#字符串拼接，例如 a + ':' + b + ':' + c 会创造大量无用的中间变量，':',join([a, b, c]) 效率会高不少。另外需要考虑字符串拼接是否必要，例如 print(':'.join([a, b, c])) 效率比 print(a, b, c, sep=':') 低。



if __name__ == "__main__":
    print("\033[4;32m  test1().......memory_profiler...........  \033[0m")
    test1()
    
    print("\033[4;32m  test2()........memory_profiler..........  \033[0m")
    test2()
    
    print("\033[4;32m  test3().......profiler...........  \033[0m")
    test3()

    print("\033[4;32m  test4().......cprofiler...........  \033[0m")
    test4()
        
    print("\033[4;32m  test5().......cprofiler...........  \033[0m")
    test5()
        
    print("\033[4;32m  test6().......cprofiler...........  \033[0m")
    test6()
        
        
        
    print("\033[4;32m  test7()........LineProfiler............  \033[0m")
    test7()
    
    print("\033[4;32m  test8()........perf............  \033[0m")
    test8()    
    
    
