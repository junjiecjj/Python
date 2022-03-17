#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:45:47 2022
http://c.biancheng.net/view/2239.html

https://www.runoob.com/python/python-func-sorted.html
@author: jack

sort 与 sorted 区别：

sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。

list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。


sorted 语法：
sorted() 函数的基本语法格式如下：
list = sorted(iterable, key=None, reverse=False)  

其中，iterable 表示指定的序列，key 参数可以自定义排序规则；reverse 参数指定以升序（False，默认）还是降序（True）进行排序。sorted() 函数会返回一个排好序的列表。
注意，key 参数和 reverse 参数是可选参数，即可以使用，也可以忽略。


返回值
返回重新排序的列表。

"""

a = [5,7,6,3,4,1,2]
b = sorted(a)       # 保留原列表
print("a  = {}".format(a))
# [5, 7, 6, 3, 4, 1, 2]
print("b  = {}".format(b))
#[1, 2, 3, 4, 5, 6, 7]
 
L=[('b',2),('a',1),('c',3),('d',4)]
#sorted(L, cmp=lambda x,y:cmp(x[1],y[1]))   # 利用cmp函数
# [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
print("sorted(L, key=lambda x:x[1]) = {}".format(sorted(L, key=lambda x:x[1]) ))              # 利用key
# [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
 
 
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
print("sorted(students, key=lambda s: s[2]) = {}".format(sorted(students, key=lambda s: s[2])))             # 按年龄排序
# [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
 
print("sorted(students, key=lambda s: s[2], reverse=True)  = {}".format(sorted(students, key=lambda s: s[2], reverse=True) ))      # 按降序
# [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]




#对列表进行排序
a = [5,3,4,2,1]
print("sorted(a)1 = {}".format(sorted(a)))
#对元组进行排序
a = (5,4,3,1,2)
print("sorted(a)2 = {}".format(sorted(a)))
#字典默认按照key进行排序
a = {4:1,\
     5:2,\
     3:3,\
     2:6,\
     1:8}
print("sorted(a)3 = {}".format(sorted(a.items())))
#对集合进行排序
a = {1,5,3,2,4}
print("sorted(a)4 = {}".format(sorted(a)))
#对字符串进行排序
a = "51423"
print("sorted(a)5 = {}".format(sorted(a)))

#再次强调，使用 sorted() 函数对序列进行排序， 并不会在原序列的基础进行修改，而是会重新生成一个排好序的列表。例如：

#对列表进行排序
a = [5,3,4,2,1]
print("sorted(a)6 = {}".format(sorted(a)))
#再次输出原来的列表 a
print("a   = {}".format(a))

#除此之外，sorted(）函数默认对序列中元素进行升序排序，通过手动将其 reverse 参数值改为 True，可实现降序排序。例如：
 
#对列表进行排序
a = [5,3,4,2,1]
print(sorted(a,reverse=True))


#另外在调用 sorted() 函数时，还可传入一个 key 参数，它可以接受一个函数，该函数的功能是指定 sorted() 函数按照什么标准进行排序。例如：
chars=['http://c.biancheng.net',\
       'http://c.biancheng.net/python/',\
       'http://c.biancheng.net/shell/',\
       'http://c.biancheng.net/java/',\
       'http://c.biancheng.net/golang/']
#默认排序
print("sorted(chars) = {}".format(sorted(chars)))
#自定义按照字符串长度排序
print("sorted(chars,key=lambda x:len(x)) = {}".format(sorted(chars,key=lambda x:len(x))))