#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:28:18 2022

@author: jack

本程序主要演示了list、numpy、tuple、dict、string、number等数据类型的深浅拷贝问题
"""

"""
#print("a = \n{},\nb = \n{}, ".format(a,b, ))
#print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c, ))
#print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
#print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne=\n{}".format(a,b,c,d,e,))
#print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne=\n{},\nf=\n{}".format(a,b,c,d,e,f))

#print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))
#print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
#print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
#print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},\nid(e)=\n{}, ".format(id(a),id(b),id(c),id(d),id(e), ))
#print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},\nid(e)=\n{},\nid(f)=\n{}".format(id(a),id(b),id(c),id(d),id(e),id(f)))

"""


import numpy as np
from IPython import get_ipython
import gc


a = np.arange(12).reshape(2,6)

b = a.T
c = a.reshape(3,4)


print("\n\na = \n{},\na.view() = \n{},\nb=\n{},\nc =\n{}".format(a,a.view(),b,c))

b[0,:]=123

print("\n\na = \n{},\na.view() = \n{},\nb=\n{},\nc =\n{}".format(a,a.view(),b,c))



a1 = a
b1 = a1.T
c1 = a1.reshape(3,4)

print("\n\na1 = \n{},\nb1=\n{},\nc1 =\n{}".format(a1, b1,c1))

a1[:,0]=[0,6]

print("\n\na = \n{},\nb=\n{},\nc =\n{}".format(a,b,c))

print("\n\na1 = \n{},\nb1=\n{},\nc1 =\n{}".format(a1, b1,c1))

print(a1 == a, '\n',id(a1) == id(a))





# 删除全部用户自定义变量
#import re
#for x in dir():
#    if not re.match('^__',x) and x!="re":
#        exec(" ".join(("del",x)))


#清楚所有变量
# get_ipython().magic('reset -sf')


# https://blog.csdn.net/u014465934/article/details/79488819
"""
引用：

在 python 中，对象赋值实际上是对象的引用。
当创建一个对象，然后把它赋给另一个变量的时候，python 并没有拷贝这个对象，而只是拷贝了这个对象的引用。
直接赋值相当于视图（numpy），没有 copy 到额外的空间中。
"""

print("============================================== 1 =========================================================")
a = tuple('furzoom')
b = a

print("a = \n{},\nb = \n{} ".format(a,b, ))

print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))



"""
浅拷贝：
如果 list 中没有引用其它对象，那么浅拷贝【copy () 和切片两种方法】就和深拷贝一样一样了。
"""
print("============================================== 2 =========================================================")
# 1. copy()在没有其他对象时，和深拷贝一样
a = [1, 2, 3]
b = a.copy()
b[0] = 333

print("a = \n{},\nb = \n{} ".format(a,b, ))

#2.  切片方法在没有其他对象情况下，和深拷贝一样
print("============================================== 3 =========================================================")
a = [1,2,3] 
b = a[:]
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 0

print("a = \n{},\nb = \n{} ".format(a,b, ))

#浅拷贝有两种方式，一个是工厂方法 copy ()，另一个是通过切片操作。
print("============================================== 4 =========================================================")
#1.   copy()
a = [1,3,[2,3]]
b = a.copy()
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[2][0] = 999
b[1] = 2
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[2][1] = 324
a[1] = 76
print("a = \n{},\nb = \n{} ".format(a,b, ))

#2.  切片操作
print("============================================== 5 =========================================================")
a = [1,3,[2,3]]
b = a[:]
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[2][0] =1999
b[1] = 874
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[2][1] = 976
a[1] = 44
print("a = \n{},\nb = \n{} ".format(a,b, ))

print("============================================== 6 =========================================================")
#下面就是赋值操作了，也就是引用，一个变另一个也变
a = [1,3,[2,3]]
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[2][0] = 78
b[1] = 55
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[2][1] = 564
a[1] = 96
print("a = \n{},\nb = \n{} ".format(a,b, ))
#深拷贝
#深复制，不仅复制对象本身，同时也复制该对象所引用的对象。
print("============================================== 7 =========================================================")
import copy
a = [1,2,3]
b =copy.deepcopy(a)
print("a = \n{},\nb = \n{} ".format(a,b, ))
b[0] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))

c = [1,2,[3,4]]
d = copy.deepcopy(c)
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0] = 0
d[2][1] = 2
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1] = 6
c[2][1] = 7
print("c = \n{},\nd = \n{} ".format(c,d ))

"""
总结：
浅拷贝 只拷贝父对象，不会拷贝对象的内部的子对象。
深拷贝 拷贝对象及其子对象
"""



"""
2.Numpy 中的视图、副本
数组切片返回的对象是原始数组的视图（视图 = 引用）。
Python 的列表 List 切片得到的是副本（副本 = 浅拷贝）。
Numpy 中的赋值 (引用)、数组切片、copy ()

"""
print("============================================== 8 =========================================================")
#1.赋值/引用
a = np.array([1,2,3,4])
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[2] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))


#2.数组切片（也就是视图/引用）
print("============================================== 9 =========================================================")
c = np.array([1,2,3,4])
d = c[:]
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0] = 0
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1] = 1
print("c = \n{},\nd = \n{} ".format(c,d ))


c = np.arange(6).reshape(2,3)
d = c[:]  #c,d一起变
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0,1] = 32
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1,1] = 54
print("c = \n{},\nd = \n{} ".format(c,d ))


#3.copy()【类似列表中的deepcopy()】
print("============================================== 10 =========================================================")
import numpy as np
import copy
a = np.array([1,2,3,4])
b = a.copy()
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 0
print("c = \n{},\nd = \n{} ".format(c,d ))

c = np.arange(6).reshape(2,3)
d = c.copy() #c,d无关联
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0,1] = 32
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1,1] = 54
print("c = \n{},\nd = \n{} ".format(c,d ))

"""
3.Python 函数参数传递（值传递还是引用传递）
Python 值传递和引用传递区别，哪些类型值传递，哪些是引用传递？
值传递：方法调用时，实际参数把它的值传递给对应的形式参数，方法执行中形式参数值的改变不影响实际参数的值。

"""

print("============================================== 11 =========================================================")
a = 520
b=a
print("a = \n{},\nb = \n{} ".format(a,b, ))
b=a+1
print("a = \n{},\nb = \n{} ".format(a,b, ))

#引用传递：也称地址传递，在方法调用时，实际上是把参数的引用 (传的是地址，而不是参数的值) 传递给方法中对应的形式参数，在方法执行中，对形式参数的操作实际上就是对实际参数的操作，方法执行中形式参数值的改变将会影响实际参数的值。
print("============================================== 12 =========================================================")
a = [1,2]
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))

b.append(3)
print("a = \n{},\nb = \n{} ".format(a,b, ))

#在 Python 中，数字、字符或者元组等不可变对象类型都属于值传递，而字典 dict 或者列表 list 等可变对象类型属于引用传递。
#如果要想修改新赋值后原对象不变，则需要用到 python 的 copy 模块，即对象拷贝。对象拷贝又包含浅拷贝和深拷贝。下面用例子来说明:
print("============================================== 13 =========================================================")
import copy
l1 = [[1, 2], 3]
l2 = copy.copy(l1)
l3 = copy.deepcopy(l1)
print("l1 = \n{},\nl2 = \n{},\nl3 = \n{}, ".format(l1,l2,l3, ))

l2.append(4)
l2[0].append(5)
l3[0].append(6)
print("l1 = \n{},\nl2 = \n{},\nl3 = \n{}, ".format(l1,l2,l3, ))

#l1 = [[1, 2, 5], 3]
#l2 = [[1, 2, 5], 3, 4]
#l3 = [[1, 2, 6], 3]

#从上例可以看出，copy.copy 属于浅拷贝，拷贝的是第一层 list，而 copy.deepcopy 属于深拷贝，对 list 所有子元素都进行深拷贝。




#=====================================================================================
# https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html
#=====================================================================================
#字典浅拷贝实例
#实例
a = {1: [1,2,3]}
b = a.copy()  
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[1].append(4)
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[1].append(5)
print("a = \n{},\nb = \n{} ".format(a,b, ))


#深度拷贝需要引入 copy 模块：
#实例
import copy
a = {1: [1,2,3]}
c = copy.deepcopy(a)
print("a = \n{},\nc = \n{} ".format(a,c, ))

a[1].append(5)
print("a = \n{},\nc = \n{} ".format(a,c, ))

c[1].append(6)
print("a = \n{},\nc = \n{} ".format(a,c, ))


"""
浅复制要分两种情况进行讨论：
1）当浅复制的值是不可变对象（字符串、元组、数值类型）时和“赋值”的情况一样，对象的id值（id()函数用于获取对象的内存地址）与浅复制原来的值相同。
2）当浅复制的值是可变对象（列表、字典、集合）时会产生一个“不是那么独立的对象”存在。有两种情况：
第一种情况：复制的对象中无复杂子对象，原来值的改变并不会影响浅复制的值，同时浅复制的值改变也并不会影响原来的值。原来值的id值与浅复制原来的值不同。
第二种情况：复制的对象中有复杂子对象（例如列表中的一个子元素是一个列表），如果改变其中复杂子对象，浅复制的值改变会影响原来的值。 改变原来的值中的复杂子对象的值会影响浅复制的值。
"""
print("============================ 1 ====================================================\n")
import copy
a = [1, 2, 3, 4, ['a', 'b']] #原始对象
b = a                       #赋值，传对象的引用
c = copy.copy(a)            #对象拷贝，浅拷贝
d = copy.deepcopy(a)        #对象拷贝，深拷贝
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

a.append(5)                 #修改对象a
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

a[4].append('c')            #修改对象a中的['a', 'b']数组对象
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

print("============================ 2 ====================================================\n")
import copy
a = [1, 2, 3, 4, ['a', 'b']] #原始对象
b = a                       #赋值，传对象的引用
c = copy.copy(a)            #对象拷贝，浅拷贝
d = copy.deepcopy(a)        #对象拷贝，深拷贝
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

b.append(6)                 #修改对象a
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

b[4].append('d')            #修改对象a中的['a', 'b']数组对象
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))


print("============================ 3 ====================================================\n")

import copy
a = [1, 2, 3, 4, ['a', 'b']] #原始对象
b = a                       #赋值，传对象的引用
c = copy.copy(a)            #对象拷贝，浅拷贝
d = copy.deepcopy(a)        #对象拷贝，深拷贝
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

c.append(7)                 #修改对象a
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

c[4].append('e')            #修改对象a中的['a', 'b']数组对象
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

print("============================ 4 ====================================================\n")

import copy
a = [1, 2, 3, 4, ['a', 'b']] #原始对象
b = a                       #赋值，传对象的引用
c = copy.copy(a)            #对象拷贝，浅拷贝
d = copy.deepcopy(a)        #对象拷贝，深拷贝
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

d.append(8)                 #修改对象a
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))

d[4].append('f')            #修改对象a中的['a', 'b']数组对象
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))



#=====================================================================================
# https://zhuanlan.zhihu.com/p/25221086
#=====================================================================================



#=====================================================================================
# https://zhuanlan.zhihu.com/p/54011712
#=====================================================================================


#对于不可变对象的深浅拷贝

import copy
a=(1,2,3)

print("=====赋值=====")
b=a
print(a)
print(b)
print(id(a))
print(id(b))



print("=====浅拷贝=====")
b=copy.copy(a)
print(a)
print(b)
print(id(a))
print(id(b))

print("=====深拷贝=====")
b=copy.deepcopy(a)
print(a)
print(b)
print(id(a))
print(id(b))









#=====================================================================================
# https://songlee24.github.io/2014/08/15/python-FAQ-02/
#=====================================================================================

a = [1, 2, 3]
b = a
print(id(a), id(b), sep='\n')

"""
二、浅拷贝（shallow copy）

注意：浅拷贝和深拷贝的不同仅仅是对组合对象来说，所谓的组合对象就是包含了其它对象的对象，如列表，类实例。而对于数字、字符串以及其它“原子”类型，没有拷贝一说，产生的都是原对象的引用。

所谓“浅拷贝”，是指创建一个新的对象，其内容是原对象中元素的引用。（拷贝组合对象，不拷贝子对象）

常见的浅拷贝有：切片操作、工厂函数、对象的copy()方法、copy模块中的copy函数。
"""

a = [1, 2, 3]
b = list(a)
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

a[0]= 12
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

import copy
a = [1, 2, 3]
b = copy.copy(a)
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

a[0]= 12
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))


import copy
a = [1, 2, 3]
b = a[:]
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

a[0]= 12
print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

"""
三、深拷贝（deep copy）
所谓“深拷贝”，是指创建一个新的对象，然后递归的拷贝原对象所包含的子对象。深拷贝出来的对象与原对象没有任何关联。

深拷贝只有一种方式：copy模块中的deepcopy函数。
"""

import copy
a = [1, 2, 3]
b = copy.deepcopy(a)

print("a = \n{},\nb = \n{} ".format(a,b, ))
print(id(a), id(b))          # a和b身份不同

for x, y in zip(a, b):       # 但它们包含的子对象身份相同
    print(id(x), id(y))

"""
看了上面的例子，有人可能会疑惑：

为什么使用了深拷贝，a和b中元素的id还是一样呢？

答：这是因为a的每个元素为不可变对象，当需要一个新的对象时，python可能会返回已经存在的某个类型和值都一致的对象的引用。而且这种机制并不会影响 a 和 b 的相互独立性，因为当两个元素指向同一个不可变对象时，对其中一个赋值不会影响另外一个。

我们可以a的每个元素变为可变对象，即列表，来确切地展示“浅拷贝”与“深拷贝”的区别：
"""

import copy
a = [[1, 2],[5, 6], [8, 9]]
b = copy.copy(a)              # 浅拷贝得到b
c = copy.deepcopy(a)          # 深拷贝得到c
print(id(a), id(b))           # a 和 b 不同
for x, y in zip(a, b):        # a 和 b 的子对象相同
    print(id(x), id(y))


print(id(a), id(c))           # a 和 c 不同

for x, y in zip(a, c):        # a 和 c 的子对象也不同
    print(id(x), id(y))



#=====================================================================================
# https://www.cnblogs.com/yoyoketang/p/14449962.html
#=====================================================================================


# 1.str
a = "hello"
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a = "world"
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

# 2.list
a = [1, 2, 3]
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a.append(4)
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

"""
这是个很有趣的事情，字符串重新赋值给b后，改变原来a的值，b不会跟着变。
但是list重新赋值给bb后，改变aa的值，bb的值也跟着变了。
这里有个知识点：在python中，都是将“对象的引用(内存地址)”赋值给变量的。其次，在python中有6个标准数据类型，他们分为可变和不可变两类。

"""

"""
在python中有6个标准数据类型，他们分为可变和不可变两类。

不可变类型：Number（数字）String（字符串）Tuple（元组）
可变类型：List（列表）Dictionary（字典）Set（集合）
可变对象和不可变对象的内存地址可以通过id函数获取

可变对象：可变对象可以在其 id() 保持固定的情况下改变其取值；
不可变对象：具有固定值的对象。不可变对象包括数字、字符串和元组。这样的对象不能被改变。如果必须存储一个不同的值，则必须创建新的对象。
id(object)： 函数用于获取对象的内存地址，函数返回对象的唯一标识符，标识符是一个整数。


"""

#字符串和数字都是不可变类型，不同变量赋值一样，通过id获取的内存地址是一样的
a = "abc"
b = "abc"

print(id(a))
print(id(b))  
print(a is b)  # True

c = 100
d = 100
print(id(c))
print(id(d))  
print(c is d)    # True

#list、dict 和 set集合是可变类型，虽然值一样，但是id获取的内存地址不一样
a = {"key": "123"}
b = {"key": "123"}

print(id(a))
print(id(b))
print(a is b)  # False
print(a == b)  # True

c = [1, 2, 3]
d = [1, 2, 3]
print(id(c))
print(id(d))
print(c is d)  # False
print(c == d)  # True


a = {"key": "123"}
b = a

print(id(a))
print(id(b))
print(a is b)  # True
print(a == b) # True

c = [1, 2, 3]
d = c
print(id(c))
print(id(d))
print(c is d)  # True
print(c == d)  # True

"""
 python 中的深拷贝和浅拷贝使用 copy 模块

1.浅拷贝会创建一个新的容器对象(compound object)
2.对于对象中的元素，浅拷贝就只会使用原始元素的引用（内存地址）

常见的浅拷贝操作有：
使用切片操作[:]
使用工厂函数（如list/dict/set）
copy模块的copy()方法
对象的copy()方法

深拷贝 A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
上面这段话是官方文档上的描述，也是有2个含义：
1.深拷贝和浅拷贝一样，都会创建一个新的容器对象(compound object)
2.和浅拷贝的不同点在于，深拷贝对于对象中的元素，深拷贝都会重新生成一个新的对象
 
 
"""


# 浅拷贝使用 copy 模块的 copy 方法
import copy


a = [1, "hello", [2, 3], {"key": "123"}]
b = copy.copy(a)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

# a和b容器里面的元素对象id
print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# del删除的是变量，而不是数据。
del a,b


a = [1, "hello", [2, 3], {"key": "123"}]
b = copy.copy(a)  # or b = a.copy()
# 如果改变a里面的不可变对象数字和字符串，此时a和b的值就不一样了，但是b的后面没改变的元素还是指向a
# 改变a的 数字和字符串对象
a[0] = 2

# a 和b 的值不一样了
print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

# 但是后面的元素还是指的a
print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# 改变a的 数字和字符串对象
b[0] = 32

# a 和b 的值不一样了
print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

# 但是后面的元素还是指的a
print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# 改变a的 数字和字符串对象
a[1] = 'world'

# a 和b 的值不一样了
print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

# 但是后面的元素还是指的a
print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# 改变a的 数字和字符串对象
b[1] = 'hhhhhh'

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

# a 和b 的值不一样了
print(a)
print(b)

# 但是后面的元素还是指的a
print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

#如果改变a里面的可变对象， 把[2, 3]里面的3改成 [2, 4]
# 改变a的 可变对象 [2, 4]
a[2][1] = 4

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))


# 改变a的 可变对象 [2, 4]
b[2][1] = 657

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# 改变a的 可变对象 [2, 4]
a[2].append(7)

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))


# 改变a的 可变对象 [2, 4]
b[2].append(99)

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

#浅拷贝使用 copy 模块的 deepcopy 方法
import copy
a = [1, "hello", [2, 3], {"key": "123"}]

b = copy.deepcopy(a)

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

# 改变a的 可变对象 [2, 4]
a[2][1] = 5454

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))


# 改变a的 可变对象 [2, 4]
b[2][1] = 33

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))


#赋值跟浅拷贝 深拷贝是有区别的

a = [1, "hello", [2, 3], {"key": "123"}]
b = a


print(id(a))
print(id(b))

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))

a[0] = 2

print(a)
print(b)

print(id(a))    # 外面容器拷贝了，所以a和b的id不一样
print(id(b))

print(id(a[0]))
print(id(b[0]))

print(id(a[1]))
print(id(b[1]))

print(id(a[2]))
print(id(b[2]))

print(id(a[3]))
print(id(b[3]))


#=====================================================================================
# https://cloud.tencent.com/developer/article/1809082
#=====================================================================================

"""
拷贝对可变类型和不可变类型的区别
copy.copy() 对于可变类型，会进行浅拷贝。
copy.copy() 对于不可变类型，不会拷贝，仅仅是指向。
copy.deepcopy() 深拷贝对可变、不可变类型都一样递归拷贝所有，对象完全独立

所谓的不可变指的是所指向的内存中的内容不可变。
同一份内存地址，其内容发生了改变，但地址依旧不变。说明是可变数据类型例如 list, set, dict。


常见的浅拷贝有：切片操作、工厂函数、对象的copy()方法、copy模块中的copy函数。
深拷贝只有一种方式：copy模块中的deepcopy函数。

数据类型                         是否可变

数字 Number                     不可变  
字符串 str                      不可变  
元组 tuple                      不可变  
列表 list                       可变  
集合 set                        可变 
字典 dict                       可变   


三者对比
d = c 赋值引用，c 和 d 都指向同一个对象
e = c.copy() 浅拷贝，c  和 e 是一个 独立的对象，但他们的 子对象还是指向统一对象即引用。
f = copy.deepcopy(c) 深度拷贝，c 和  f  完全拷贝了父对象及其子对象，两者是完全独立的。

"""




#直接赋值

a = [11, 22, 33]
b = a

print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a.append(44)
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a = {"name": "hui"}
b = a

print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a["age"] = 21
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))


# 利用内置模块 copy 实现浅拷贝
import copy
a = [1, 2]
b = [3, 4]
c = [a, b]
d = c
e = c.copy()


print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))



a.append(5)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))


b.append(6)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))





c.append(23)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))


d.append(54)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))


e.append(76)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},\ne = \n{}, ".format(a,b,c,d, e))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, \nid(e) = \n{}".format(id(a),id(b),id(c),id(d),id(e) ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, \nid(e[0]) = \n{},\nid(e[1]) = \n{}".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]),id(e[0]),id(e[1]) ))

"""
可以看出直接赋值 c 和 d 是同一对象，而浅拷贝 copy 的 c 和 e 是一个分别独立的对象，但他们的子对象 a , b 还是 指向统一对象即引用。

因此当 c.append(7) 后，只有 c 对象改变了，而浅拷贝的 e 还是没有变化。

当 a.append(5), b.append(6) 后，c, d, e 对象依然内容一致。

"""


#浅拷贝测试
# 可变类型list
a = [1, 2, 3]
b = copy.copy(a)

print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a.append(4)
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))


# 不可变类型 tuple
import copy
a = (1, 2, 3, 4)
b = copy.copy(a)
c = copy.deepcopy(a)

#a.append(4)  error
print("a = \n{},\nb = \n{}  ,\nc = \n{} ".format(a,b, c))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) =\n{}".format(id(a),id(b), id(c)))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

# 不可变类型 tuple
import copy
a = (1,12.12,'jack',[1,2])
b = copy.copy(a)
c = copy.deepcopy(a)

#a.append(4)  error
print("a = \n{},\nb = \n{}  ,\nc = \n{} ".format(a,b, c))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) =\n{}".format(id(a),id(b), id(c)))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))


# a[0] = 121  error

# a[2] = 'kkk'  error

a[3].append(3)
print("a = \n{},\nb = \n{}  ,\nc = \n{} ".format(a,b, c))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) =\n{}".format(id(a),id(b), id(c)))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))



import copy
a =[1,'jack',[1,2,3], (4, 5, 6)]
b = copy.copy(a)
c = copy.deepcopy(a)


print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))


a[0] = 12 
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

b[0] = 32
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

c[0] = 65
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

a[1] = 'wang'
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

b[1] = 'chen'
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))


c[1] = 'hhh'
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

a[2].append(43)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))

b[2].append(77)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))


c[2].append(89)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{}, ".format(id(a),id(b),id(c), ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]), ))


a=['hello',[1,2,3]]
b=a[:]
[print(id(x)) for x in a]
[print(id(x)) for x in b]
a[0]='world'
a[1].append(4)

print("a = \n{},\nb = \n{}, ".format(a,b,  ))






# 通过 copy.deepcopy() 来实现深拷贝
a = [1, 2]
b = [3, 4]
c = [a, b]
d = copy.deepcopy(c)


print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},  ".format(a,b,c,d,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},  ".format(id(a),id(b),id(c),id(d), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, ".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]), ))



c.append(5)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},  ".format(a,b,c,d,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},  ".format(id(a),id(b),id(c),id(d), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, ".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]), ))


a.append(33)
b.append(53)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},  ".format(a,b,c,d,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},  ".format(id(a),id(b),id(c),id(d), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, ".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]), ))


d[0].append(88)
d[1].append(44)

print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{},  ".format(a,b,c,d,  ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{},  ".format(id(a),id(b),id(c),id(d), ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid([0]) = \n{},\nid(d[1]) = \n{}, ".format(id(c[0]),id(c[1]),id(d[0]),id(d[1]), ))



"""
深拷贝测试    深拷贝则是完全拷贝，互不影响 

深拷贝是在另一块地址中创建一个新的变量或容器，同时容器内的元素的地址也是新开辟的，仅仅是值相同而已，是完全的副本。也就是说（ 新瓶装新酒 ）。

这里可以看出，深拷贝后，a和b的地址以及a和b中的元素地址均不同，这是完全拷贝的一个副本，修改a后，发现b没有发生任何改变，因为b是一个完全的副本，元素地址与a均不同，a修改不影响b。
"""
import copy
a = ([1, 2], [3, 4])
b = copy.copy(a)
c = copy.deepcopy(a)

# 浅拷贝不可变类型id()一致, 深拷贝不可变类型id()不一致
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},  ".format(id(a),id(b),id(c),  ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(b[0]) = \n{},\nid(b[1]) = \n{}, \nid(c[0]) = \n{},\nid(c[1]) = \n{}, ".format(id(a[0]),id(a[1]),id(b[0]),id(b[1]), id(c[0]),id(c[1]),))

a[0].append(12)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},  ".format(id(a),id(b),id(c),  ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(b[0]) = \n{},\nid(b[1]) = \n{}, \nid(c[0]) = \n{},\nid(c[1]) = \n{}, ".format(id(a[0]),id(a[1]),id(b[0]),id(b[1]), id(c[0]),id(c[1]),))

b[0].append(32)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},  ".format(id(a),id(b),id(c),  ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(b[0]) = \n{},\nid(b[1]) = \n{}, \nid(c[0]) = \n{},\nid(c[1]) = \n{}, ".format(id(a[0]),id(a[1]),id(b[0]),id(b[1]), id(c[0]),id(c[1]),))


c[0].append(78)
print("a = \n{},\nb = \n{},\nc = \n{}, ".format(a,b,c, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},  ".format(id(a),id(b),id(c),  ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(b[0]) = \n{},\nid(b[1]) = \n{}, \nid(c[0]) = \n{},\nid(c[1]) = \n{}, ".format(id(a[0]),id(a[1]),id(b[0]),id(b[1]), id(c[0]),id(c[1]),))


a=['hello',[1,2,3]]

b=copy.deepcopy(a)


[print(id(x)) for x in a]
[print(id(x)) for x in b]

a[0]='world'
a[0]='world'
print("a = \n{},\nb = \n{}, ".format(a,b,  ))






#============================================================================================
#  下面根据是list,set, dict,  tuple,string,number数据来展示
# 不可变数据（3 个）：Number（数字）、String（字符串）、Tuple（元组）；
# 可变数据（3 个）：List（列表）、Dictionary（字典）、Set（集合）。
#============================================================================================


#============================================================================================
#  set
#============================================================================================
import copy
a = {9,  2.32, 'jack', (1,2,3)}
b=a
c = a.copy()
d = copy.deepcopy(a)

#

print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c, d ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c), id(d)))
#id(set[0]) 会出错，set[0]无法id()
#print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},  ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]), ))

a.add(12)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c, d ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c), id(d)))


b.add(54)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c, d ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c), id(d)))



c.add(76)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c, d ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c), id(d)))



d.add(98)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c, d ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c), id(d)))



#因为set中的元素不能为list、dict、set，只能为 tuple,string,number，因此set的拷贝很简单


#============================================================================================
#  tuple  # 不可变类型 tuple
#============================================================================================

import copy
a = (1,12.12,'jack',[1,2],{'name':'jack'})
b = a
c = copy.copy(a)
d = copy.deepcopy(a)


#a.append(4)  error
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

# a[0] = 121  error

# a[2] = 'kkk'  error

a[3].append(3)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

b[3].append(4)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

c[3].append(5)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

d[3].append(6)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))


a[4]['age']=28
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

b[4]['home']='jiaotan'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

c[4]['hig']=172
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))


d[4]['wei']=70
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))


#============================================================================================
#  list
#============================================================================================
import copy
a =[1,'jack',[1,2,3], (4, 5, 6),{'name':'jack'}]
b=a
c = copy.copy(a)
d = copy.deepcopy(a)


print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))


a[0] = 12 
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

b[0] = 32
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

c[0] = 65
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

a[1] = 'wang'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

b[1] = 'chen'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

c[1] = 'hhh'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

a[2].append(43)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

b[2].append(77)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))


c[2].append(89)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))
print("id(a[0]) = \n{},\nid(a[1]) = \n{},\nid(a[2]) = \n{},\nid(a[3]) = \n{},\nid(a[4]) = \n{}, ".format(id(a[0]),id(a[1]),id(a[2]),id(a[3]),id(a[4]) ))
print("id(b[0]) = \n{},\nid(b[1]) = \n{},\nid(b[2]) = \n{},\nid(b[3]) = \n{},\nid(b[4]) = \n{},  ".format(id(b[0]),id(b[1]),id(b[2]),id(b[3]),id(b[4])  ))
print("id(c[0]) = \n{},\nid(c[1]) = \n{},\nid(c[2]) = \n{},\nid(c[3]) = \n{},\nid(c[4]) = \n{},  ".format(id(c[0]),id(c[1]),id(c[2]),id(c[3]),id(c[4])  ))
print("id(d[0]) = \n{},\nid(d[1]) = \n{},\nid(d[2]) = \n{},\nid(d[3]) = \n{},\nid(d[4]) = \n{},  ".format(id(d[0]),id(d[1]),id(d[2]),id(d[3]), id(d[4]) ))

#============================================================================================
#  number
#============================================================================================


a = 12.13
b=a
c=copy.copy(a)
d = copy.deepcopy(a)



print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))


a = 323
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

b =21
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

c = 75
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

d = 8
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



#============================================================================================
#   string
#============================================================================================


a = 'jack'
b=a
c=copy.copy(a)
d = copy.deepcopy(a)



print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))


a = 'chrss'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

b =  'marry'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

c = 'lili'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

d = 'nana'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



#============================================================================================
#   dict
#============================================================================================
a = {(20, 30):'good','name':'Allen', 'age':14, 'gender':'male',12:'kkk','haha':[1,2,3],'fff':(4,5),'set':{4,5,6}}
b=a
c=copy.copy(a)
d = copy.deepcopy(a)

print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))


a['age']=21
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))


b['name']='wang'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

c['gender']='female'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

d[12]='....'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



a['haha'].append(4)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

b['haha'].append(5)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

c['haha'].append(6)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



d['haha'].append(7)
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



a['home']='jingdez'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

b['street']='fuhonglu'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))

c['school'] = 'fu liang yi zhong'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))



d[(3,4)] = 'tuple'
print("a = \n{},\nb = \n{},\nc = \n{},\nd = \n{}, ".format(a,b,c,d, ))
print("id(a) = \n{},\nid(b) = \n{},\nid(c) = \n{},\nid(d) = \n{}, ".format(id(a),id(b),id(c),id(d) ))










