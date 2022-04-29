#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:12:55 2022

@author: jack
"""
#============================================= 迭代======================================================

#那么，如何判断一个对象是可迭代对象呢？方法是通过collections.abc模块的Iterable类型判断：
from collections.abc import Iterable

isinstance('abc', Iterable) # str是否可迭代
# True
isinstance([1,2,3], Iterable) # list是否可迭代
# True
isinstance(123, Iterable) # 整数是否可迭代
# False


#=============================================  列表生成式 ======================================================
import os # 导入os模块，模块的概念后面讲到

[d for d in os.listdir('.')] # os.listdir可以列出文件和目录


#for循环其实可以同时使用两个甚至多个变量，比如dict的items()可以同时迭代key和value：
d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
     print(k, '=', v)


#因此，列表生成式也可以使用两个变量来生成list：
d = {'x': 'A', 'y': 'B', 'z': 'C' }
[k + '=' + v for k, v in d.items()]
# ['y=B', 'x=A', 'z=C']


#最后把一个list中所有的字符串变成小写：
L = ['Hello', 'World', 'IBM', 'Apple']
[s.lower() for s in L]
# ['hello', 'world', 'ibm', 'apple']


#if ... else
#使用列表生成式的时候，有些童鞋经常搞不清楚if...else的用法。
#例如，以下代码正常输出偶数：
[x for x in range(1, 11) if x % 2 == 0]
# [2, 4, 6, 8, 10]
# 但是，我们不能在最后的if加上else：
[x for x in range(1, 11) if x % 2 == 0 else 0]
#   File "<stdin>", line 1
#     [x for x in range(1, 11) if x % 2 == 0 else 0]
#                                               ^
# SyntaxError: invalid syntax
#这是因为跟在for后面的if是一个筛选条件，不能带else，否则如何筛选？


#另一些童鞋发现把if写在for前面必须加else，否则报错：

[x if x % 2 == 0 for x in range(1, 11)]
#   File "<stdin>", line 1
#     [x if x % 2 == 0 for x in range(1, 11)]
#                        ^
# SyntaxError: invalid syntax
#这是因为for前面的部分是一个表达式，它必须根据x计算出一个结果。因此，考察表达式：x if x % 2 == 0，它无法根据x计算出结果，因为缺少else，必须加上else：

[x if x % 2 == 0 else -x for x in range(1, 11)]
#[-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]
#上述for前面的表达式x if x % 2 == 0 else -x才能根据x计算出确定的结果。
#可见，在一个列表生成式中，for前面的if ... else是表达式，而for后面的if是过滤条件，不能带else。




#=============================================  生成器 ======================================================
"""
通过列表生成式，我们可以直接创建一个列表。但是，受到内存限制，列表容量肯定是有限的。而且，创建一个包含100万个元素的列表，不仅占用很大的存储空间，如果我们仅仅需要访问前面几个元素，那后面绝大多数元素占用的空间都白白浪费了。

所以，如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，从而节省大量的空间。在Python中，这种一边循环一边计算的机制，称为生成器：generator。

要创建一个generator，有很多种方法。第一种方法很简单，只要把一个列表生成式的[]改成()，就创建了一个generator：
"""

L = [x * x for x in range(10)]
L
#[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
g = (x * x for x in range(10))
g
#<generator object <genexpr> at 0x1022ef630>
#创建L和g的区别仅在于最外层的[]和()，L是一个list，而g是一个generator。

#我们可以直接打印出list的每一个元素，但我们怎么打印出generator的每一个元素呢？

#如果要一个一个打印出来，可以通过next()函数获得generator的下一个返回值：

next(g)
#0
next(g)
#1
next(g)
#4
next(g)
#9
next(g)
#16
next(g)
#25
next(g)
#36
next(g)
#49
next(g)
#64
next(g)
#81
next(g)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# StopIteration
#我们讲过，generator保存的是算法，每次调用next(g)，就计算出g的下一个元素的值，直到计算到最后一个元素，没有更多的元素时，抛出StopIteration的错误。

#当然，上面这种不断调用next(g)实在是太变态了，正确的方法是使用for循环，因为generator也是可迭代对象：

g = (x * x for x in range(10))
for n in g:
     print(n)



#比如，著名的斐波拉契数列（Fibonacci），除第一个和第二个数外，任意一个数都可由前两个数相加得到：
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
    return 'done'
fib(6)


def fib1(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'
f = fib1(6)
for n in f:
     print(n)


#这里，最难理解的就是generator函数和普通函数的执行流程不一样。普通函数是顺序执行，遇到return语句或者最后一行函数语句就返回。
#而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
#举个简单的例子，定义一个generator函数，依次返回数字1，3，5：

def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield(3)
    print('step 3')
    yield(5)

o = odd()
next(o)
next(o)
next(o)


#请务必注意：调用generator函数会创建一个generator对象，多次调用generator函数会创建多个相互独立的generator。
#原因在于odd()会创建一个新的generator对象，上述代码实际上创建了3个完全独立的generator，对3个generator分别调用next()当然每个都会返回第一个值。
#正确的写法是创建一个generator对象，然后不断对这一个generator对象调用next()：

g = odd()
next(g)






def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment
for n in frange(0, 4, 0.5):
     print(n)





#=============================================  迭代器 ======================================================

"""
我们已经知道，可以直接作用于for循环的数据类型有以下几种：

一类是集合数据类型，如list、tuple、dict、set、str等；

一类是generator，包括生成器和带yield的generator function。

这些可以直接作用于for循环的对象统称为可迭代对象：Iterable。

可以使用isinstance()判断一个对象是否是Iterable对象：






"""

from collections.abc import Iterable


isinstance([], Iterable)
# True
isinstance({}, Iterable)
# True
isinstance('abc', Iterable)
# True
isinstance((x for x in range(10)), Iterable)
# True
isinstance(100, Iterable)
# false



"""
而生成器不但可以作用于for循环，还可以被next()函数不断调用并返回下一个值，直到最后抛出StopIteration错误表示无法继续返回下一个值了。

可以被next()函数调用并不断返回下一个值的对象称为迭代器：Iterator。

可以使用isinstance()判断一个对象是否是Iterator对象：


"""


from collections.abc import Iterator
isinstance((x for x in range(10)), Iterator)
#True
isinstance([], Iterator)
#False
isinstance({}, Iterator)
#False
isinstance('abc', Iterator)
#False


#生成器都是Iterator对象，但list、dict、str虽然是Iterable，却不是Iterator。

#把list、dict、str等Iterable变成Iterator可以使用iter()函数：

isinstance(iter([]), Iterator)
#True
isinstance(iter('abc'), Iterator)
#True
#你可能会问，为什么list、dict、str等数据类型不是Iterator？
#这是因为Python的Iterator对象表示的是一个数据流，Iterator对象可以被next()函数调用并不断返回下一个数据，直到没有数据时抛出StopIteration错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下一个数据，所以Iterator的计算是惰性的，只有在需要返回下一个数据时它才会计算。
#Iterator甚至可以表示一个无限大的数据流，例如全体自然数。而使用list是永远不可能存储全体自然数的。



'''
凡是可作用于for循环的对象都是Iterable类型；

凡是可作用于next()函数的对象都是Iterator类型，它们表示一个惰性计算的序列；

集合数据类型如list、dict、str等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。

Python的for循环本质上就是通过不断调用next()函数实现的，例如：
'''
for x in [1, 2, 3, 4, 5]:
    pass
#实际上完全等价于：

# 首先获得Iterator对象:
it = iter([1, 2, 3, 4, 5])
# 循环:
while True:
    try:
        # 获得下一个值:
        x = next(it)
    except StopIteration:
        # 遇到StopIteration就退出循环
        break




































































































































































































































































































































































































