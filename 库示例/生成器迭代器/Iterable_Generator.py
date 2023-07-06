#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:12:55 2022

@author: jack

这里讲: 列表生成器, 生成器, 迭代器, 列表生成器, 列表迭代器等 的区别, 联系及用法

在使用Python的过程中, 经常会和列表/元组/字典(list/tuple/dict)、容器(container)、可迭代对象(iterable)、迭代器(iterator)、生成器(generator)等这些名词打交道, 众多的概念掺杂到一起难免会让人一头雾水,

我们先来看下容器(container)和可迭代对象(iterable).

那么什么是容器呢？
容器是一种把多个元素组织在一起的数据结构,容器中的元素可以逐个迭代获取,可以用 in,not in 关键字判断元素是否包含在容器中. 通常这类数据结构把所有的元素存储在内存中(也有一些特例,并不是所有的元素都放在内存,比如迭代器和生成器对象),我们常用的 string、set、list、tuple、dict 都属于容器对象.

"""
#使用dir()
#如果要获得一个对象的所有属性和方法, 可以使用dir()函数, 它返回一个包含字符串的list, 比如, 获得一个str对象的所有属性和方法:

print(dir('ABC'))


#==========================================================================================================================================
#                                                              (一) 容器
#==========================================================================================================================================

print('h' in 'hello')
print('z' not in 'hello')
print(1 in [1, 2, 3])
print(5 not in [1, 2, 3])



#==========================================================================================================================================
#                                                              (二) 可迭代对象(iterable)
#==========================================================================================================================================

"""
(2.1)  那什么是可迭代对象(iterable) ?
定义一:
    1. 实现__iter__成为迭代对象: 有了__iter__方法了,因此,此时已经是迭代对象了. 在python中, 只有实现__iter__方法的才是叫迭代对象.
        通过实现__next__方法成为迭代器: 上述仅仅只是一个迭代对象,并不是迭代器；因此,要想成为迭代器,还需要实现迭代器协议:即实现next方法,要么返回下一个元素,要么引起终止迭代.
定义二:
    2. 可迭代对象 (iterable) : 可直接作为for循环的对象, 统称为可迭代对象. python从可迭代对象中获取迭代器, 但凡是可以返回一个迭代器的对象都可称之为可迭代对象.
    我们已经知道,可以直接作用于for循环的数据类型有以下几种:
         (1) 集合数据类型 : list, tuple, dict, set, str, bytes
         (2) generator (数据结构)  : 生成器、带 yield 的generator function
    这些可以直接作用于for循环的对象统称为可迭代对象:Iterable.

可以使用isinstance()判断一个对象是否是Iterable对象:


在循环遍历自定义容器对象时,会使用 python 内置函数 iter() 调用遍历对象的 __iter__() 获得一个迭代器, 之后再循环对这个迭代器使用 next() 调用迭代器对象的__next__(). __iter__()只会被调用一次, 而__next__()会被调用 n 次.

通过索引的方式进行迭代取值, 实现简单, 但仅适用于序列类型 : 字符串, 列表, 元组.
对于没有索引的字典、集合等非序列类型, 必须找到一种不依赖索引来进行迭代取值的方式, 这就用到了迭代器.

可以返回一个迭代器的对象都可以称之为可迭代对象,我们来看一个例子:
"""
#================================================================



# 那么, 如何判断一个对象是可迭代对象呢？ 方法是通过collections.abc模块的Iterable类型判断:
from collections.abc import Iterable, Iterator

x = [1, 2, 3]
print(f"{isinstance(x, Iterable)}")  # True
print(f"{isinstance(x, Iterator)}")  # False
print(f"{type(x)}")                  # <class 'list'>

a = iter(x)
print(f"{isinstance(a, Iterable)}")  # True
print(f"{isinstance(a, Iterator)}")  # False
print(f"{type(a)}")                  # <class 'list_iterator'>


print(issubclass(list,   Iterable)) # True
print(issubclass(list,   Iterator)) # False

print(issubclass(dict,   Iterable)) # True
print(issubclass(dict,   Iterator)) # False

print(issubclass(str,    Iterable)) # True
print(issubclass(str,   Iterator)) # False

print(issubclass(tuple,  Iterable)) # True
print(issubclass(tuple,   Iterator)) # False

print(issubclass(set,    Iterable)) # True
print(issubclass(set,   Iterator)) # False

print(issubclass(int,    Iterable)) # False
print(issubclass(int,   Iterator)) # False

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

isinstance('abc', Iterable) # str是否可迭代
# True
isinstance([1,2,3], Iterable) # list是否可迭代
# True
isinstance(123, Iterable) # 整数是否可迭代
# False


#================================================================
from collections.abc import Iterable, Iterator


# 首先创建一个类, 实例化一个普通的对象.
class Dogs(object):
    def __init__(self,nums):  # 我家有nums条狗
        self.nums = nums

dogs = Dogs(10)
print(dir(dogs))          # 通过dir查看下对象dogs拥有的方法以及属性,发现没有 __iter__ 方法
print(f"{isinstance(dogs, Iterable)}")  # False
print(f"{isinstance(dogs, Iterator)}")  # False
print(f"{type(dogs)}")
# 通过dir()方法可以看出,dogs类并没有__iter__方法,因此并不是一个迭代对象. 在python中,只有实现__iter__方法的才是叫迭代对象.

#================================================================
# 首先创建一个类,实例化一个普通的对象.
class Dogs(object):
    def __init__(self,nums):  # 我家有nums条狗
        self.nums = nums
    def __iter__(self):       # 通过实现__iter__方法,则对象就成了迭代对象
        return self


dogs = Dogs(10)
print(dir(dogs))          # 通过dir查看下对象dogs拥有的方法以及属性,发现没有 __iter__ 方法
print(f"{isinstance(dogs, Iterable)}")  # True
print(f"{isinstance(dogs, Iterator)}")  # False
print(f"{type(dogs)}")
# 此时,通过dir()方法就可以看见dogs有了__iter__方法了,因此,此时已经是迭代对象了.

# 如果没有给这个类中加入 __next__ 方法,是不可以对其进行 iter() 而变成迭代器对象的.
# iter(dogs)
# TypeError: iter() returned non-iterator of type 'Dogs'
#


#============================================= 迭代 ======================================================

"""
如果给定一个 list 或 tuple,我们可以通过 for 循环来遍历这个 list 或 tuple, 这种遍历我们称为迭代(Iteration) .
Python 的 for 循环不仅可以用在 list 或 tuple 上, 还可以作用在其他可迭代对象上.


list 这种数据类型虽然有下标, 但很多其他数据类型是没有下标的, 但是, 只要是可迭代对象, 无论有无下标, 都可以迭代, 比如 dict 就可以迭代:
因为 dict 的存储不是按照 list 的方式顺序排列, 所以, 迭代出的结果顺序很可能不一样.
默认情况下, dict 迭代的是 key. 如果要迭代 value, 可以用 for value in d.values(), 如果要同时迭代 key 和 value, 可以用f or k, v in d.items().
由于字符串也是可迭代对象, 因此,也可以作用于 for 循环:
"""


d = {'a': 1, 'b': 2, 'c': 3}

print(f"{isinstance(d, Iterable)}")  # True
print(f"{isinstance(d, Iterator)}")  # False
print(f"{type(d)}")                  # <class 'dict'>


for key in d:
    print(key)
for k, v in d.items():
    print(k, v)
for ch in 'ABC':
    print(ch)

# 可以看出, for in 只要求后面的是可迭代对象而不要求是迭代器


#================================================================


#============================================== for 循环原理 =============================================
# (1) 当我们使用 for 循环时, 只要作用于一个可迭代对象, for循环就可以正常运行, 而我们不太关心该对象究竟是 list 还是其他数据类型, for...in...这个结构后面跟 一定是一个可迭代的对象.
# (2) 当执行了 for 循环的语句之后, 首先得保证作用于一个可迭代对象, 然后系统会调用 iter() 方法, 将可迭代对象 (in 后面的内容)  转化(如果 in 后面的对象已经是一个迭代器, 则不需要, 这一过程是系统完成的)成一个迭代器对象, 然后调用 迭代器对象中的 next() 方法, 将迭代器中的对象一个个的顺序输出.
#     如下图代码中所示的对比效果.
# (3) for … in… 这个语句其实做了两件事. 第一件事是获得一个可迭代器, 即调用了__iter__()函数, 第二件事是循环的过程,循环调用 __next__() 函数.
# (4) for...in...的迭代实际是将可迭代对象转换成迭代器, 再重复调用next()方法实现的。

# for 循环在工作时:  只要作用于一个可迭代对象, for循环就可以正常运行, 而我们不太关心该对象究竟是list还是其他数据类型.
# 首先会调用可迭代对象 good 内置的 iter 方法拿到一个迭代器对象,
# 然后再调用该迭代器对象的 next 方法将取到的值赋给 item, 执行循环体完成一次循环,
# 周而复始, 直到捕捉StopIteration(停止迭代) ,结束迭代.


# 有了迭代器后, 便可以不依赖索引迭代取值了, 使用while循环的实现方式如下 :
goods = ['mac','lenovo','acer','dell']
i = iter(goods) #每次都需要重新获取一个迭代器对象
while True:
    try:
        print(next(i))
    except StopIteration: #捕捉异常终止循环
        break

goods = ['mac','lenovo','acer','dell']
for item in iter(goods):
    print(item)
# 迭代器不但可以作用于for循环, 还可以被 next() 函数不断调用并返回下一个值, 直到最后抛出 StopIteration 错误表示无法继续返回下一个值了.

#for循环又称为迭代循环, i 后可以跟任意可迭代对象, 上述while循环可以简写为:
goods = ['mac','lenovo','acer','dell']
print(f"{isinstance(goods, Iterable)}")  # True
print(f"{isinstance(goods, Iterator)}")  # False
print(f"{type(goods)}")                  # <class 'list'>


for item in goods:
    print(item)

# 可以看出, for in 只要求后面的是可迭代对象而不要求是迭代器




#==========================================================================================================================================
#                                                              (三) 迭代器(iterator),  for的原理, 迭代过程
#==========================================================================================================================================

"""
那什么是 迭代器(iterator) ?
定义:
    上述仅仅只是一个迭代对象,并不是迭代器；因此,要想成为迭代器,还需要实现迭代器协议:即实现next方法,要么返回下一个元素,要么引起终止迭代.
    迭代器（Iterator）是实现了__iter__()方法和__next()__方法的对象。
    任何实现了__iter__()和__next__()(Python2.x中实现next())方法的对象都是迭代器, __iter__()返回迭代器自身, __next__()返回容器中的下一个值, 如果容器中没有更多元素了, 则抛出StopIteration异常.

迭代器与列表的区别在于,构建迭代器的时候,不像列表把所有元素一次性加载到内存,而是以一种延迟计算(lazy evaluation) 方式返回元素,这正是它的优点. 比如列表中含有一千万个整数,需要占超过100M的内存,
而迭代器只需要几十个字节的空间. 因为它并没有把所有元素装载到内存中,而是等到调用next()方法的时候才返回该元素(按需调用 call by need 的方式,本质上 for 循环就是不断地调用迭代器的next()方法) .


迭代器是一个可以记住遍历的位置的对象.

迭代器对象从集合的第一个元素开始访问,直到所有的元素被访问完结束. 迭代器只能往前不会后退.

字符串, 列表或元组对象都可用于创建迭代器:

创建迭代器的方式:
(1) 方法一:
可以自己通过实现__iter__和__next__方法来定义迭代器,

迭代环境是通过调用内置函数iter去尝试寻找__iter__方法来实现的, 而这种方法应该返回一个迭代器, 如果没有找到__iter__方法, python会改用__getitem__机制
    把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__() .
    __iter__() 方法返回一个特殊的迭代器对象,  这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成.
    __next__() 方法 (Python 2 里是 next()) 会返回下一个迭代器对象.

(2) 方法二:
    通过 iter() 把可迭代对象转为迭代器: 如果是list等直接 iter(x), 而这些内置的数据结构往往没有实现__next__, 如果是自定义的类, 则这个类必须实现 __next__方法.
    迭代器是可迭代对象的一个子集, 它是一个可以记住遍历的位置的对象, 它与列表、元组、集合、字符串这些可迭代对象的区别就在于next方法的实现, 其他列表、元组、集合、字符串这些可迭代对象可以很简单的转化成迭代器, 通过Python内置的iter函数能够轻松把可迭代对象转化为迭代器, 下面来看一个例子,

迭代器的优缺点
优点
1、为序列和非序列类型提供了一种统一的迭代取值方式.
2、惰性计算 : 迭代器对象表示的是一个数据流, 可以只在需要时才去调用next来计算出一个值.
迭代器同一时刻在内存中只有一个值, 因而可以存放无限大的数据流.
而对于其他容器类型, 如列表, 需要把所有的元素都存放于内存中, 受内存大小的限制, 可以存放的值的个数是有限的.

缺点
1、除非取尽, 否则无法获取迭代器的长度.
2、只能取下一个值, 不能回到开始.
迭代器产生后的唯一目标就是重复执行next方法直到值取尽, 否则就会停留在某个位置, 等待下一次调用next.
若是要再次迭代同个对象, 只能重新调用iter方法创建一个新的迭代器对象.
如果有两个或者多个循环使用同一个迭代器, 必然只会有一个循环能取到值.

你可能会问,为什么list、dict、str等数据类型不是Iterator？
这是因为Python的Iterator对象表示的是一个数据流,Iterator对象可以被next()函数调用并不断返回下一个数据,直到没有数据时抛出StopIteration错误. 可以把这个数据流看做是一个有序序列,但我们却不能提前知道序列的长度,只能不断通过next()函数实现按需计算下一个数据,所以Iterator的计算是惰性的,只有在需要返回下一个数据时它才会计算.

"""

x = [1,2,3,4,5]
print(f"{isinstance(x, Iterable)}")  # True
print(f"{isinstance(x, Iterator)}")  # False
print(f"{type(x)}")                  # <class 'list'>

x = iter(x)
print(f"{isinstance(x, Iterable)}")  # True
print(f"{isinstance(x, Iterator)}")  # True
print(f"{type(x)}")                  # <class 'list_iterator'>


#===========================================================

class Fib(object):
    def __init__(self, Max = 0):
        super(Fib, self).__init__()
        self.prev = 0
        self.curr = 1
        self.max = Max

    def __iter__(self):
        return self

    def __next__(self):
        if self.max > 0:
            self.max -= 1
            # 当前要返回的元素的值
            value = self.curr
            # 下一个要返回的元素的值
            self.curr += self.prev
            # 设置下一个元素的上一个元素的值
            self.prev = value
            return value
        else:
            raise StopIteration

    # 兼容Python2.x
    def next(self):
        return self.__next__()


fib = Fib(12)
print(f"{isinstance(fib, Iterable)}")  # True
print(f"{isinstance(fib, Iterator)}")  # False
print(f"{type(fib)}")                  # <class '__main__.Fib'>
print(f"{type(iter(fib))}")            # <class '__main__.Fib'>

# 调用next()的过程
for n in fib:
    print(n)
# raise StopIteration
print(next(fib))


#===========================================================
class Dogs(object):
    def __init__(self,nums):  # 我家有nums条狗
        self.nums = nums
        self.start = -1
    def __iter__(self):       # 通过实现__iter__方法,则对象就成了迭代对象
        return self
    def __next__(self):       # 实现next方法,即迭代器协议;每一次for循环都调用该方法
        self.start +=1
        if self.start >= self.nums:# 若超出,则停止迭代
            raise StopIteration()
        return self.start


dogs = Dogs(10)
print(f"{isinstance(dogs, Iterable)}")  # True
print(f"{isinstance(dogs, Iterator)}")  # False
print(f"{type(dogs)}")                  # <class '__main__.Dogs'>
print(f"{type(iter(dogs))}")            # <class '__main__.Dogs'>
for dog in dogs:
    print(f"{type(dogs)}")
    print(dog)

# 此时,我们就成功封装出一个迭代器,那么,for就可以根据迭代器协议遍历类中元素.
# 而dogs就被称之为 实现了迭代器协议的可迭代对象.

#===========================================================
class testclass():
    def __init__(self,data=1):
        self.data = data
    def __iter__(self):
        return self
    def __next__(self):
        if self.data > 7:
            raise StopIteration
        else:
            self.data += 1
            return self.data


test = testclass(3)
print(f"{isinstance(test, Iterable)}")  # True
print(f"{isinstance(test, Iterator)}")  # True
print(f"{type(test)}")                  # <class '__main__.testclass'>

for item in test:
    print(item)




#  含有__next__()函数的对象都是一个迭代器, 所以test也可以说是一个迭代器. 如果去掉__iter__()函数, 就不能用于for  ... in .....循环中:

class test():
    def __init__(self,data=1):
        self.data = data

    #def __iter__(self):
    #    return self
# 唯一需要注意下的就是__next__中必须控制iterator的结束条件, 不然就死循环了
    def __next__(self):
        if self.data > 7:
            raise StopIteration
        else:
            self.data+=1
            return self.data

# for item in test(2):
#     print(item)
#     # TypeError: 'test' object is not iterable


# 但是可以通过下面的方式调用 :
class testclass():
    def __init__(self,data=1):
        self.data = data

    def __next__(self):
        if self.data > 7:
            raise StopIteration
        else:
            self.data += 1
            return self.data

t = testclass(3)
print(f"{isinstance(test, Iterable)}")  # True
print(f"{isinstance(test, Iterator)}")  # True
print(f"{type(test)}")                  # <class '__main__.testclass'>

for i in range(3):
    print(t.__next__())

# iter()函数与next()函数 : iter是将一个对象 (列表) 变成迭代器对象,使用next函数式取迭代器中的下一个数据
it = iter([1, 2, 3, 4, 5])
# 循环:
while True:
    try:
        # 获得下一个值:
        x = next(it)
        print(x)
    except StopIteration:
        # 遇到StopIteration就退出循环
        break


#===========================================================
# 现在回头看一下经常用到的list:
for i in [1,2,3]:
    print(i)
# 之所以能够利用for进行遍历,是因为list本身就是一个可迭代对象,内部实质上已经实现了__iter__方法. 但是此时[1, 2, 3]并不是迭代器, 只是可迭代对象:
a = [1, 2, 3]
# print(f"{next(a)}")    # TypeError: 'list' object is not an iterator
# 只有在 把a 变为迭代器后才能使用next方法:
b = iter(a)
print(f"{next(b)}")
print(f"{next(b)}")
print(f"{next(b)}")
print(f"{next(b)}")
# 1
# 2
# 3
# StopIteration



#============================== 列表和迭代器区别 ==================================
"""
(1) 列表不论遍历多少次, 表头位置始终是第一个元素；
(2) 迭代器遍历结束后, 不再指向原来的表头位置, 而是为最后元素的下一个位置；
(3) 需要注意, 我们无法通过调用 len 获得迭代器的长度, 只能迭代到最后一个末尾元素时, 才知道其长度.

遍历列表, 表头位置始终不变；
遍历迭代器, 表头位置相应改变；
next 函数执行一次, 迭代对象指向就前进一次；
StopIteration 触发时, 意味着已到迭代器尾部；


"""
# 通过 type 关键字可以看到列表和迭代器的类型是不同的.
a = [1,2,3]
b = iter(a)
print(f"type(a) = {type(a)}")
# list
print(f"type(b) = {type(b)}")
#  list_iterator

for i in a:
    print(f"{i}")
# 1
# 2
# 3

for i in b:
    print(f"{i}")
# 1
# 2
# 3

for i in a:
    print(f"{i}")
# 1
# 2
# 3

for i in b:
    print(f"{i}")
#

# 列表每次迭代都是从第一个元素开始；
# 而迭代器在迭代结束后再次迭代就不会有任何值. 因为一旦迭代结束, 就指向迭代器最后一个元素的下一个位置；
# 只有迭代器对象才能与内置函数 next 结合使用,  next 一次, 迭代器就前进一次, 指向一个新的元素. 所以, 要想迭代器 b 重新指向 a 的表头, 需要重新创建一个新的迭代器.

b = iter(a)

print(f"next(b)")
# 1

print(f"next(b)")
# 2

print(f"next(b)")
# 3

print(f"next(b)")
#  StopIteration:

#==========================================================================================================================================
#                                                              (四) 列表推导式/ 表达式
#==========================================================================================================================================
import os # 导入os模块,模块的概念后面讲到

# 列出当前目录下的所有目录, 只需要一行代码:
a = [x for x in os.listdir('.') if os.path.isdir(x)]
print(f"{a}")


# 要列出所有的.py文件, 也只需一行代码:
a = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.py']
print(f"{a}")


#
Dir = [dirs for dirs in os.listdir('.')] # os.listdir可以列出文件和目录

for d in  Dir:
    print(f"d = {d}")


# 使用两层循环, 可以生成全排列:
a = [m + n for m in 'ABC' for n in 'XYZ']
print(f"{a}")

# 列表生成式也可以使用两个变量来生成list:
d = {'x': 'A', 'y': 'B', 'z': 'C' }
a = [k + '=' + v for k, v in d.items()]
print(f"{a}")


#for循环其实可以同时使用两个甚至多个变量,比如dict的items()可以同时迭代key和value:
d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
    print(k, '=', v)



#最后把一个list中所有的字符串变成小写:
L = ['Hello', 'World', 'IBM', 'Apple']
[s.lower() for s in L]
# ['hello', 'world', 'ibm', 'apple']


#if ... else
#使用列表生成式的时候,有些童鞋经常搞不清楚if...else的用法.
#例如,以下代码正常输出偶数:
a = [x for x in range(1, 11) if x % 2 == 0]
print(f"{a}")
# [2, 4, 6, 8, 10]
# 但是,我们不能在最后的if加上else:
[x for x in range(1, 11) if x % 2 == 0 else 0]
#   File "<stdin>", line 1
#     [x for x in range(1, 11) if x % 2 == 0 else 0]
#                                               ^
# SyntaxError: invalid syntax
#这是因为跟在for后面的if是一个筛选条件,不能带else,否则如何筛选？


#另一些童鞋发现把if写在for前面必须加else,否则报错:
[x if x % 2 == 0 for x in range(1, 11)]
#   File "<stdin>", line 1
#     [x if x % 2 == 0 for x in range(1, 11)]
#                        ^
# SyntaxError: invalid syntax
#这是因为for前面的部分是一个表达式,它必须根据x计算出一个结果. 因此,考察表达式:x if x % 2 == 0,它无法根据x计算出结果,因为缺少else,必须加上else:

a = [x if x % 2 == 0 else -x for x in range(1, 11)]
print(f"{a}")
#[-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]
#上述for前面的表达式x if x % 2 == 0 else -x才能根据x计算出确定的结果.
#可见,在一个列表生成式中,for前面的if ... else是表达式,而for后面的if是过滤条件,不能带else.



#==========================================================================================================================================
#                                                              (五) 生成器:  生成器的两种用法: yeild 和 生成器表达式。
#==========================================================================================================================================

#==========================================================================================================================================
#                      (5.1) 生成器:   生成器表达式 / 生成器推导式:
#==========================================================================================================================================

"""
普通函数用return返回一个值,还有一种函数用yield返回值, 这种函数叫生成器函数。函数被调用时会返回一个生成器对象。生成器其实是一种特殊的迭代器, 不过这种迭代器更加优雅, 它不需要像普通迭代器一样实现__iter__()和__next__()方法了, 只需要一个yield关键字。生成器一定是迭代器（反之不成立）, 因此任何生成器也是一种懒加载的模式生成值。

(1) yield 能够临时挂起当前函数, 记下其上下文（包括局部变量、待决的 try catch 等）, 将控制权返回给函数调用者。

(2) 当下一次再调用其所在生成器时, 会恢复保存的上下文, 继续执行剩下的语句, 直到再遇到 yield 或者退出为止。

(3) 在调用生成器运行的过程中, 每次遇到 yield 时函数会暂停并保存当前所有的运行信息, 返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。
(4) yield 和 return 区别
    yield可以用于返回值, 但不同于return。
    函数一旦遇到return就结束了, 销毁上下文（弹出栈帧）, 将控制权返回给调用者。
    而yield可以保存函数的运行状态, 挂起函数, 用来返回多次值。
    因此, 以 yield 进行执行流控制的函数称为生成器函数, 以 return 进行执行流控制的函数就是普通函数。
    由于可以临时挂起函数的执行, yield 可以充当其调用者和被挂起函数间交互的桥梁。

通过列表生成式,我们可以直接创建一个列表. 但是,受到内存限制,列表容量肯定是有限的. 而且,创建一个包含100万个元素的列表,不仅占用很大的存储空间,如果我们仅仅需要访问前面几个元素,那后面绝大多数元素占用的空间都白白浪费了.
所以,如果列表元素可以按照某种算法推算出来,那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list,从而节省大量的空间. 在Python中,这种一边循环一边计算的机制,称为生成器:generator.
要创建一个generator,有很多种方法. 第一种方法很简单,只要把一个列表生成式的[]改成(),就创建了一个generator:
"""

L = [x * x for x in range(10)]
print(f"{L}")
#[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
g = (x * x for x in range(10))
print(f"{g}")
#<generator object <genexpr> at 0x1022ef630>
#创建L和g的区别仅在于最外层的[]和(), L是一个list,而g是一个generator.

#我们可以直接打印出list的每一个元素,但我们怎么打印出generator的每一个元素呢？

#如果要一个一个打印出来,可以通过next()函数获得generator的下一个返回值:

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
#我们讲过,generator保存的是算法,每次调用next(g),就计算出g的下一个元素的值,直到计算到最后一个元素,没有更多的元素时,抛出StopIteration的错误.


#当然,上面这种不断调用next(g)实在是太变态了,正确的方法是使用for循环,因为generator也是可迭代对象:

g = (x * x for x in range(10))
for n in g:
    print(n)


#==========================================================================================================================================
#                                   (5.2) 生成器:  生成器函数 yeild: 针对表达式形式的yield, 生成器对象必须事先被初始化一次,
#==========================================================================================================================================

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return "I'm done with you"

# 针对表达式形式的yield, 生成器对象必须事先被初始化一次, 首先, 定义的 g = generator() 并不是函数调用, 而是产生生成器对象。
f = fib(4)

print(f"{next(f)}")
print(f"{next(f)}")
print(f"{next(f)}")
print(f"{next(f)}")
# 1
# 1
# 2
# 3
print(f"{next(f)}")
# StopIteration: done


f = fib(6)
# 把函数改成generator函数后, 我们基本上从来不会用next()来获取下一个返回值, 而是直接使用for循环来迭代:
for n in fib(6):
    print(n)
# 1
# 1
# 2
# 3
# 5
# 8


g = fib(6)
while True:
    try:
        x = next(g)
        print('g:', x)
    except StopIteration as e:
        print('Generator return value:', e.value)
        break
# g: 1
# g: 1
# g: 2
# g: 3
# g: 5
# g: 8
# Generator return value: I'm done with you





#==============================================================================
import numpy as np


# 信源发生器, 结合 numpy 产生无穷的数据
def generate_infty_data(batchsz = 2, dim = 3):
    while True:
        data = np.random.randn(batchsz, dim)
        # data = data + 3
        yield data
    return "I'm done with you"

# 1
infty = generate_infty_data()

for data in infty:
    print(f"data = \n{data}")
    time.sleep(1)

# 2   信源发生器, 结合 numpy 产生无穷的数据
data_iter = real_data_generator(batchsz = 2, dim = 3)
cnt = 0
for i in range(14222):
    cnt += 1
    x = next(data_iter)
    print(x)


#==============================================================================

def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield(3)
    print('step 3')
    yield(5)


o = odd()
print(f"{next(o)}\n")
print(f"{next(o)}\n")
print(f"{next(o)}\n")
print(f"{next(o)}\n")
# step 1
# 1
# step 2
# 3
# step 3
# 5
# Traceback (most recent call last):
#   File "/tmp/ipykernel_2613665/2308099519.py", line 14, in <module>
#     print(f"{next(o)}")
# StopIteration

# 可以看到, odd不是普通函数, 而是generator函数, 在执行过程中, 遇到yield就中断, 下次又继续执行。执行3次yield后, 已经没有yield可以执行了, 所以, 第4次调用next(o)就报错。



#==============================================================================
# 可以编写装饰器来完成为所有表达式形式 yield 对应生成器的初始化操作, 如下

def init(func):
    def wrapper(*args,**kwargs):
        g = func(*args,**kwargs)
        next(g)
        return g
    return wrapper

@init
def eater():
    print('Ready to eat.')
    while True:
        food = yield
        print('get the food: %s, and start to eat.' %food)


#==============================================================================
# 表达式形式的yield也可以用于返回多次值, 即 变量名=yield 值 的形式, 如下
def eater():
    print('Ready to eat')
    food_list = []
    while True:
        food = yield food_list
        food_list.append(food)

e=eater()
print(f"{next(e)}")
# Ready to eat
# []
e.send('蒸羊羔')
# ['蒸羊羔']
e.send('蒸熊掌')
# ['蒸羊羔', '蒸熊掌']
e.send('蒸鹿尾儿')
# ['蒸羊羔', '蒸熊掌', '蒸鹿尾儿']

#==============================================================================

def read_by_chunks(file, chunk_size=1024):
    while True:
        data = file.read(chunk_size)
        if not data:
            break
        yield data

f = open('your_big_file.dat')
for chunk in read_by_chunks(f):
    process_chunk(chunk)

#================================== send ============================================

# 在函数内可以采用表达式形式的yield
def eater():
    print('Ready to eat.')
    while True:
        food = yield
        print('get the food: %s, and start to eat.' % food)


g=eater() # 得到生成器对象
print(f"{g}")
# <generator object eater at 0x7f93d5de39e0>

print(f"{next(g)}")
# Ready to eat.
# None

print(f"{next(g)}")
# get the food: None, and start to eat.
# None


print(f"{next(g)}")
# get the food: None, and start to eat.
# None

# 可以拿到函数的生成器对象持续为函数体send值, 如下
print(f"{g.send('包子')}")
# get the food: 包子, and start to eat.
# None

print(f"{g.send('鸡腿')}")
# get the food: 鸡腿, and start to eat.
# None

print(f"{g.send(None)}")
## 让函数挂起在food=yield的位置, 等待调用g.send()方法为函数体传值。
# # g.send(None)等同于next(g)。
# get the food: None, and start to eat.
# None


#==================================== send ==========================================
import time

# yield 就是 return 返回一个值, 并且记住这个返回的位置, 下次迭代就从这个位置后（下一行）开始。next方法 和 send方法都可以返回下一个元素,
# 区别在于send可以传递参数给yield表达式, 这时传递的参数会作为yield表达式的值, 而yield的参数是返回给调用者的值。

# 1
def func(n):
    print(' Ready to eat.')
    for i in range(0, n):
        print(f'    eat. {i}')
        # yield相当于return, 下一次循环从yield的下一行开始
        arg = yield i
        print('    func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f"j = {j}")
    print(f'  main-next: {next(f)}', )
    if j%2 == 0:
        print(f'  main-send: {f.send(100+j) }\n' )
    time.sleep(1)


# 2
def func(n):
    print(' Ready to eat.')
    for i in range(0, n):
        print(f'    eat. {i}')
        # yield相当于return, 下一次循环从yield的下一行开始
        arg = yield i
        print('    func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f"j = {j}")
    if j == 0:
        print(f'  main-next: {next(f)}', )
    print(f'  main-send: {f.send(100+j) }\n' )
    time.sleep(1)

# 3
def func(n):
    print(' Ready to eat.')
    for i in range(0, n):
        print(f'    eat. {i}')
        # yield相当于return, 下一次循环从yield的下一行开始
        arg = yield i
        print('    func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f"j = {j}")
    if j %2 == 0:
        print(f'  main-next: {next(f)}', )
    print(f'  main-send: {f.send(100+j) }\n' )
    time.sleep(1)


# 4
def func(n):
    print(' Ready to eat.')
    for i in range(0, n):
        #  print(f'    arg. {arg}') error
        print(f'    eat. {i}')
        # yield相当于return, 下一次循环从yield的下一行开始
        arg = yield i
        print('    func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f"j = {j}")
    print(f'  main-next: {next(f)}', )
    print(f'  main-send: {f.send(100 + j) }\n' )
    time.sleep(1)


# 5
def func(n):
    print(' Ready to eat.')
    for i in range(0, n):
        #  print(f'    arg. {arg}') error
        print(f'    eat. {i}')
        # yield相当于return, 下一次循环从yield的下一行开始
        arg = yield i
        print('    func', arg)

# if __name__ == '__main__':
f = func(10)
for j in range(12):
    print(f"j = {j}")
    if j == 0:
        print(f'  main-next: {next(f)}', )
    else:
        print(f'  main-send: {f.send(100 + j) } ' )
        print(f'  main-next: {next(f)}\n', )
    time.sleep(1)

"""
从以上4个实验可以看出:
(1) 生成器的 send 和 next 都可以触发生成器
(2) send 和 next 都可以接收返回值
(3) 第一次调用时, 请使用next()语句或是send(None), 不能使用send发送一个非None的值, 否则会出错的, 可以看到, can't send non-None value to a just-started generator因为生成器just-started generator, 是没有Python yield语句来接收这个值的。
(4) 区别在于send可以传递参数给yield表达式, 这时传递的参数会作为 yield 表达式的, 且类似于提前为表达式占坑的行为, 等send进来后再赋值给 yield表达式,  而 yield 的参数是返回给 send 调用者的值。
(5) f.send(None))  和next(f)的效果一模一样
(6) recv = yeild 3: 在执行顺序上, 就算是这次是 f.send("haha") 进来, 这次只会把 3 返回给 f.send的调用者, 只有在下次 f.send('didi') 时才会把 'didi' 赋值给 recv


send()方法:
先理解个概念【挂起】: 意思就是暂时保留先不进行, 等待需要时再进行。

我们执行的起点是: n = yield i, 并且这个n值并不是i值, 而是通过send()传递过来的值,
即: n = send(), 但是我们没有调用send()方法, 所以, n自然而然为None,

注意: nl = yield r 这里是有两步完成的也就是开关的两种情况, 完成终止时（关时）, 返回函数yield r的数, 即产生r的值, 开始时再次调用next()时, 这个过程中完成nl的赋值操作和send参数

recv = yeild 3: 在执行顺序上, 是先 yield 返回 3, 当再次next的时候才会赋值给recv。
"""



def _generator():
    for i in range(4):
        n = yield i
        if n == 'hello':
            print('  world')
        else:
            print(f"  {str(n)}")

g = _generator()     # 生成器对象
print(f"g.send(None) = {g.send(None)} ")         # 相当于g.__next__()
time.sleep(1)
print(f"g.send('100') = {g.send('100')} ")
time.sleep(1)
print(f"g.send('Python') = {g.send('Python')} ")
time.sleep(1)
print(f"g.send('hello') = {g.send('hello')} ")
time.sleep(1)




def mygen():
    yield 1
    yield 2
    # yield 返回3, 然后将send过来的msg保存到recv
    recv = yield 3
    yield f"{recv} 这是mygen收到的msg"


a = mygen()

ret = next(a) # 等同于 a.send(None)
print(ret)

ret = a.send('hello 1')
print(ret)

ret = a.send('hello 2')
print(ret)

ret = a.send('hello 3')
print(ret)

# send会引发一次generator的next, 让generator继续执行, 传过来的value会赋值给yield关键字左边的变量, 而yield关键字右边的值就会返还给send。在执行顺序上, 是先yield返回3, 当再次next的时候才会赋值给recv。



#!/usr/bin/python3
def MyGenerator():
        value=yield 1
        yield value
        return done

gen=MyGenerator()
print(next(gen))
print(next(gen))
print(next(gen))
print(gen.send("I am Value"))

# 运行过程，
# 用next启动了生成器gen，知道到yield 1时返回1。

# 然后我们再用gen的内部方法send进入gen，而且还带回来一个值“I am Value”。这时候，继续执行yield 1后的代码“value=”，把带回来的值“I am Value”赋给value。直到遇到yield value，把value返回。







#=======================================================================================
#3.2 迭代器工具

#itertools 中定义了很多迭代器工具, 例如子序列工具:

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


#序列排序:
#itertools.sorted(iterable, key=None, reverse=False)

#itertools.groupby(iterable, key=None)              # 按值分组, iterable需要先被排序
itertools.groupby(sorted([1, 4, 6, 4, 1])) #-> (1, iter1), (4, iter4), (6, iter6)

#itertools.permutations(iterable, r=None)           # 排列, 返回值是Tuple
itertools.permutations('ABCD', 2) #-> AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB, DC

#itertools.combinations(iterable, r=None)           # 组合, 返回值是Tuple
#itertools.combinations_with_replacement(...)
itertools.combinations('ABCD', 2) #-> AB, AC, AD, BC, BD, CD


#多个序列合并:
#itertools.chain(*iterables)                        # 多个序列直接拼接
itertools.chain('ABC', 'DEF') # -> A, B, C, D, E, F

import heapq
# heapq.merge(*iterables, key=None, reverse=False)   # 多个序列按顺序拼接
itertools.merge('ABF', 'CDE') # -> A, B, C, D, E, F

# zip(*iterables)                                    # 当最短的序列耗尽时停止, 结果只能被消耗一次
# itertools.zip_longest(*iterables, fillvalue=None)  # 当最长的序列耗尽时停止, 结果只能被消耗一次






#=============================================  生成器 ======================================================




#比如,著名的斐波拉契数列(Fibonacci),除第一个和第二个数外,任意一个数都可由前两个数相加得到:
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


#这里,最难理解的就是generator函数和普通函数的执行流程不一样. 普通函数是顺序执行,遇到return语句或者最后一行函数语句就返回.
#而变成generator的函数,在每次调用next()的时候执行,遇到yield语句返回,再次执行时从上次返回的yield语句处继续执行.
#举个简单的例子,定义一个generator函数,依次返回数字1,3,5:

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


#请务必注意:调用generator函数会创建一个generator对象,多次调用generator函数会创建多个相互独立的generator.
#原因在于odd()会创建一个新的generator对象,上述代码实际上创建了3个完全独立的generator,对3个generator分别调用next()当然每个都会返回第一个值.
#正确的写法是创建一个generator对象,然后不断对这一个generator对象调用next():

g = odd()
next(g)






def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment
for n in frange(0, 4, 0.5):
     print(n)


































































































































































































































































































































































































