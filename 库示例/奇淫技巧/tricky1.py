#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 21:04:05 2022
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453843967&idx=1&sn=0227e9a6e7333ca16cd2ec8cbba929be&chksm=87eabe36b09d37205a9d8b16e32605d3c194f098edf166e7deb383d26297f24c4635576e8265&mpshare=1&scene=1&srcid=1220C7WFsXvgNjsDdbU36CjL&sharer_sharetime=1647688526465&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AVhqzZ85COwPr2EjnyFryi4%3D&acctmode=0&pass_ticket=HtUZpbsBpQ0mIk%2BoPK4K47uZExu7fNHfVfHlbsTOmMv74UV9HL0sgd84sBqhifTg&wx_header=0#rd
@author: jack
"""

# 1. 易混淆操作
# 本节对一些 Python 易混淆的操作进行对比。

# 1.1 有放回随机采样和无放回随机采样

import random
random.choices(seq, k=1)  # 长度为k的list，有放回采样
random.sample(seq, k)     # 长度为k的list，无放回采样


#1.2 lambda 函数的参数

func = lambda y: x + y          # x的值在函数运行时被绑定
func = lambda y, x=x: x + y     # x的值在函数定义时被绑定


#1.3 copy 和 deepcopy

import copy
y = copy.copy(x)      # 只复制最顶层
y = copy.deepcopy(x)  # 复制所有嵌套部分
#复制和变量别名结合在一起时，容易混淆：

a = [1, 2, [3, 4]]

# Alias.
b_alias = a  
assert b_alias == a and b_alias is a

# Shallow copy.
b_shallow_copy = a[:]  
assert b_shallow_copy == a and b_shallow_copy is not a and b_shallow_copy[2] is a[2]

# Deep copy.
import copy
b_deep_copy = copy.deepcopy(a)  
assert b_deep_copy == a and b_deep_copy is not a and b_deep_copy[2] is not a[2]
#

#对别名的修改会影响原变量，（浅）复制中的元素是原列表中元素的别名，而深层复制是递归的进行复制，对深层复制的修改不影响原变量。

#1.4 == 和 is

x == y  # 两引用对象是否有相同值
x is y  # 两引用是否指向同一对象

#1.5 判断类型

type(a) == int      # 忽略面向对象设计中的多态特征
isinstance(a, int)  # 考虑了面向对象设计中的多态特征


#1.6 字符串搜索

str.find(sub, start=None, end=None); str.rfind(...)     # 如果找不到返回-1
str.index(sub, start=None, end=None); str.rindex(...)   # 如果找不到抛出ValueError异常


#1.7 List 后向索引
#这个只是习惯问题，前向索引时下标从0开始，如果反向索引也想从0开始可以使用~。

print(a[-1], a[-2], a[-3])
print(a[~0], a[~1], a[~2])

#2. C/C++ 用户使用指南
#不少 Python 的用户是从以前 C/C++ 迁移过来的，这两种语言在语法、代码风格等方面有些不同，本节简要进行介绍。

#2.1 很大的数和很小的数

#C/C++ 的习惯是定义一个很大的数字，Python 中有 inf 和 -inf：

a = float('inf')
b = float('-inf')

#2.2 布尔值

#C/C++ 的习惯是使用 0 和非 0 值表示 True 和 False， Python 建议直接使用 True 和 False 表示布尔值。

a = Trueb = False


#2.3 判断为空

#C/C++ 对空指针判断的习惯是 if (a) 和 if (!a)。Python 对于 None 的判断是：

if x is None:    pass
#如果使用 if not x，则会将其他的对象（比如长度为 0 的字符串、列表、元组、字典等）都会被当做 False。

#2.4 交换值

#C/C++ 的习惯是定义一个临时变量，用来交换值。利用 Python 的 Tuple 操作，可以一步到位。

a, b = b, a
#2.5 比较

#C/C++ 的习惯是用两个条件。利用 Python 可以一步到位。

if 0 < a < 5:    pass
#2.6 类成员的 Set 和 Get

#C/C++ 的习惯是把类成员设为 private，通过一系列的 Set 和 Get 函数存取其中的值。在 Python 中虽然也可以通过 @property、@setter、@deleter 设置对应的 Set 和 Get 函数，我们应避免不必要的抽象，这会比直接访问慢 4 - 5 倍。

#2.7 函数的输入输出参数

#C/C++ 的习惯是把输入输出参数都列为函数的参数，通过指针改变输出参数的值，函数的返回值是执行状态，函数调用方对返回值进行检查，判断是否成功执行。在 Python 中，不需要函数调用方进行返回值检查，函数中遇到特殊情况，直接抛出一个异常。

#2.8 读文件

#相比 C/C++，Python 读文件要简单很多，打开后的文件是一个可迭代对象，每次返回一行内容。

with open(file_path, 'rt', encoding='utf-8') as f:   
   for line in f:      
       print(line)       # 末尾的\n会保留
#2.9 文件路径拼接

#C/C++ 的习惯通常直接用 + 将路径拼接，这很容易出错，Python 中的 os.path.join 会自动根据操作系统不同补充路径之间的 / 或 \ 分隔符：

import os
os.path.join('usr', 'lib', 'local')
#2.10 解析命令行选项

#虽然 Python 中也可以像 C/C++ 一样使用 sys.argv 直接解析命令行选择，但是使用 argparse 下的 ArgumentParser 工具更加方便，功能更加强大。

#2.11 调用外部命令

#虽然 Python 中也可以像 C/C++ 一样使用 os.system 直接调用外部命令，但是使用 subprocess.check_output 可以自由选择是否执行 Shell，也可以获得外部命令执行结果。

import subprocess
# 如果外部命令返回值非0，则抛出subprocess.CalledProcessError异常
result = subprocess.check_output(['cmd', 'arg1', 'arg2']).decode('utf-8')  
# 同时收集标准输出和标准错误
result = subprocess.check_output(['cmd', 'arg1', 'arg2'], stderr=subprocess.STDOUT).decode('utf-8')  
# 执行shell命令（管道、重定向等），可以使用shlex.quote()将参数双引号引起来
result = subprocess.check_output('grep python | wc > out', shell=True).decode('utf-8')
#2.12 不重复造轮子

#不要重复造轮子，Python称为batteries included即是指Python提供了许多常见问题的解决方案。

#3. 常用工具
#3.1 读写 CSV 文件

import csv
# 无header的读写
with open(name, 'rt', encoding='utf-8', newline='') as f:  # newline=''让Python不将换行统一处理    
    for row in csv.reader(f):       
        print(row[0], row[1])  # CSV读到的数据都是str类型
with open(name, mode='wt') as f:    
    f_csv = csv.writer(f)    
    f_csv.writerow(['symbol', 'change'])
    
# 有header的读写
with open(name, mode='rt', newline='') as f:   
    for row in csv.DictReader(f):      
        print(row['symbol'], row['change'])
with open(name, mode='wt') as f:   
    header = ['symbol', 'change']    
    f_csv = csv.DictWriter(f, header)    
    f_csv.writeheader()    
    f_csv.writerow({'symbol': xx, 'change': xx})
#注意，当 CSV 文件过大时会报错：_csv.Error: field larger than field limit (131072)，通过修改上限解决

import sys
csv.field_size_limit(sys.maxsize)
#csv 还可以读以 \t 分割的数据

f = csv.reader(f, delimiter='\t')
#3.2 迭代器工具

#itertools 中定义了很多迭代器工具，例如子序列工具：

import itertools
# itertools.islice(iterable, start=None, stop, step=None)
itertools.islice('ABCDEF', 2, None) #-> C, D, E, F

#itertools.filterfalse(predicate, iterable)         # 过滤掉predicate为False的元素
itertools.filterfalse(lambda x: x < 5, [1, 4, 6, 4, 1]) #-> 6

#itertools.takewhile(predicate, iterable)           # 当predicate为False时停止迭代
itertools.takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]) #-> 1, 4

#itertools.dropwhile(predicate, iterable)           # 当predicate为False时开始迭代
itertools.dropwhile(lambda x: x < 5, [1, 4, 6, 4, 1]) #-> 6, 4, 1

#itertools.compress(iterable, selectors)            # 根据selectors每个元素是True或False进行选择
itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1]) #-> A, C, E, F


#序列排序：
#itertools.sorted(iterable, key=None, reverse=False)

#itertools.groupby(iterable, key=None)              # 按值分组，iterable需要先被排序
itertools.groupby(sorted([1, 4, 6, 4, 1])) #-> (1, iter1), (4, iter4), (6, iter6)

#itertools.permutations(iterable, r=None)           # 排列，返回值是Tuple
itertools.permutations('ABCD', 2) #-> AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB, DC

#itertools.combinations(iterable, r=None)           # 组合，返回值是Tuple
#itertools.combinations_with_replacement(...)
itertools.combinations('ABCD', 2) #-> AB, AC, AD, BC, BD, CD


#多个序列合并：
#itertools.chain(*iterables)                        # 多个序列直接拼接
itertools.chain('ABC', 'DEF') # -> A, B, C, D, E, F

import heapq
# heapq.merge(*iterables, key=None, reverse=False)   # 多个序列按顺序拼接
itertools.merge('ABF', 'CDE') # -> A, B, C, D, E, F

# zip(*iterables)                                    # 当最短的序列耗尽时停止，结果只能被消耗一次
# itertools.zip_longest(*iterables, fillvalue=None)  # 当最长的序列耗尽时停止，结果只能被消耗一次


#3.3 计数器
#计数器可以统计一个可迭代对象中每个元素出现的次数。

import collections
# 创建
# collections.Counter(iterable)

# 频次
#collections.Counter[key]                 # key出现频次
# 返回n个出现频次最高的元素和其对应出现频次，如果n为None，返回所有元素
#collections.Counter.most_common(n=None)

# 插入/更新
#collections.Counter.update(iterable)
#counter1 + counter2; counter1 - counter2  # counter加减

# 检查两个字符串的组成元素是否相同
# collections.Counter(list1) == collections.Counter(list2)
#3.4 带默认值的 Dict
#当访问不存在的 Key 时，defaultdict 会将其设置为某个默认值。

import collections
collections.defaultdict(type)  # 当第一次访问dict[key]时，会无参数调用type，给dict[key]提供一个初始值
#3.5 有序 Dict

import collections
collections.OrderedDict(items=None)  # 迭代时保留原始插入顺序
#4. 高性能编程和调试
#4.1 输出错误和警告信息

#向标准错误输出信息

import sys
sys.stderr.write('')
#输出警告信息

import warnings
# warnings.warn(message, category=UserWarning)  
# category的取值有DeprecationWarning, SyntaxWarning, RuntimeWarning, ResourceWarning, FutureWarning
#控制警告消息的输出

#$ python -W all     # 输出所有警告，等同于设置warnings.simplefilter('always')
#$ python -W ignore  # 忽略所有警告，等同于设置warnings.simplefilter('ignore')
#$ python -W error   # 将所有警告转换为异常，等同于设置warnings.simplefilter('error')
#4.2 代码中测试

#有时为了调试，我们想在代码中加一些代码，通常是一些 print 语句，可以写为：

# 在代码中的debug部分
if __debug__:
    pass
#一旦调试结束，通过在命令行执行 -O 选项，会忽略这部分代码：

#$ python -0 main.py
#4.3 代码风格检查

#使用 pylint 可以进行不少的代码风格和语法检查，能在运行之前发现一些错误

# pylint main.py
#4.4 代码耗时

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
with timeblock('counting'):
    pass
#代码耗时优化的一些原则

#专注于优化产生性能瓶颈的地方，而不是全部代码。
#避免使用全局变量。局部变量的查找比全局变量更快，将全局变量的代码定义在函数中运行通常会快 15%-30%。
#避免使用.访问属性。使用 from module import name 会更快，将频繁访问的类的成员变量 self.member 放入到一个局部变量中。
#尽量使用内置数据结构。str, list, set, dict 等使用 C 实现，运行起来很快。
#避免创建没有必要的中间变量，和 copy.deepcopy()。
#字符串拼接，例如 a + ':' + b + ':' + c 会创造大量无用的中间变量，':',join([a, b, c]) 效率会高不少。另外需要考虑字符串拼接是否必要，例如 print(':'.join([a, b, c])) 效率比 print(a, b, c, sep=':') 低。
#5. Python 其他技巧
#5.1 argmin 和 argmax

items = [2, 1, 3, 4]
argmin = min(range(len(items)), key=items.__getitem__)
#argmax同理。

#5.2 转置二维列表

A = [['a11', 'a12'], ['a21', 'a22'], ['a31', 'a32']]
A_transpose = list(zip(*A))  # list of tuple
A_transpose = list(list(col) for col in zip(*A))  # list of list
#5.3 一维列表展开为二维列表

A = [1, 2, 3, 4, 5, 6]

# Preferred.
list(zip(*[iter(A)] * 2))