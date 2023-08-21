#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:40:31 2022

@author: jack

set：无序，不重复，可修改
（把set理解key的集合，更合适，因为set中存在的就是不可变对象）
list：有序，可重复，可修改
tuple：有序，可重复，不可修改


Python中list，tuple，dict和set的主要区别：tuple是一个不可改变的list，set是一个没有Value的dict，list，dict和set的数据是可变的，tuple数据是不可变的！

列表list是最自由的，可以使用索引、切片，可以进行计算和修改；

元组tuple是不自由的，数据不能更改，但是和list一样具有序列，可以用索引和切片；

字典dict是由无序的键值对构成，可以通过key去索引value的值，修改数据可以通过key来修改对应的value；

set集合是无序的，不重复的，和字典类似也是使用中括号{}表示，区别在于字典是用键值对，而set由数据或者元素或者列表构成；
"""



#=====================================================================================
#           Python 元组（Tuple）
#=====================================================================================
#Tuple可以看做是一种“不变”的List，访问也是通过下标，用小括号（）表示：
a = (3.14, 'China', 'Jason')
b = ('jack','iii',123,90.12)
#但是不能重新赋值替换：
# a[1] = 'America'   error
print("a = \n{},\nb = \n{},\na[1] = \n{} ".format(a,b, a[1]))
# a =
# (3.14, 'China', 'Jason'),
# b =
# ('jack', 'iii', 123, 90.12),
# a[1] =
# China
for i in a:
    print(i)
# 3.14
# China
# Jason



l1,l2,l3 = a
print("l1 = {},l2={},l3 = {}".format(l1,l2,l3))

# 元组运算符
print("len(a) = %d"%(len(a)))  #

c = a+b
print(c)
# (3.14, 'China', 'Jason', 'jack', 'iii', 123, 90.12)


a = (3.14, 'China','China','China', 'Jason')
if 3.14 in a:
    print("true")
# true

for i in a: #
    print(i )
# 3.14
# China
# China
# China
# Jason

#  内置函数
num = (1,2,3,12,32,4,6,7,123.12)
print(max(num)) #最大值
print(min(num))  #最小值
print("len(a) = %d"%(len(a)))  #长度
# 123.12
# 1
# len(a) = 5



#元组中的元素不允许进行删除，但是元组的整体是可以删除的，使用del 删除整个元组
#del a[1]
del a

t = (1, 2, 3, 1, 2)
# index方法：根据元素找到其位置
print(t.index(1, 2))  # 寻找第2个元素1的位置


#也没有pop和insert、append方法。
#可以创建空元素的tuple:
t = ()
#或者单元素tuple (比如加一个逗号防止和声明一个整形歧义):
t = (3.14,)

"""
要知道如果你希望一个函数返回多个返回值，其实只要返回一个tuple就可以了，因为tuple里面的含有多个值，而且是不可变的（就像是java里面的final）。
当然，tuple也是可变的，比如:

这是因为Tuple所谓的不可变指的是指向的位置不可变，因为本例子中第四个元素并不是基本类型，而是一个List类型，所以t指向的该List的位置是不变的，但是List本身的内容是可以变化的，因为List本身在内存中的分配并不是连续的。

"""
t = (3.14, 'China', 'Jason', ['A', 'B'])
print(t)
# (3.14, 'China', 'Jason', ['A', 'B'])
L = t[3]
L[0] = 122
L[1] = 233
print(t)
# (3.14, 'China', 'Jason', [122, 233])






#=====================================================================================
#           Python set集合
#=====================================================================================
"""
python的set中元素必须是unhashable(不可修改的)。

因此，可以修改的list就不能作为元素放入set中。
set里面不可以存放list

元组是不可变对象，list属于可变对象不能被hash。
#因为set中的元素不能为list、dict、set，只能为 tuple,string,number，
"""

# 可以使用大括号 { } 或者 set() 函数创建集合，但是注意如果创建一个空集合必须用 set() 而不是 { }，因为{}是用来表示空字典类型的


#1.用{}创建set集合
person ={"student","teacher","babe",123,321,123} #同样各种类型嵌套,可以赋值重复数据，但是存储会去重
print(len(person))  #存放了6个数据，长度显示是5，存储是自动去重.
print(person) #但是显示出来则是去重的

#空set集合用set()函数表示
person1 = set() #表示空set，不能用person1={}
print(len(person1))
print(person1)

#1.set对字符串也会去重，因为字符串属于序列。
str1 = set("abcdefgabcdefghi")
str2 = set("abcdefgabcdefgh")
print(str1,str2)
print(str1 - str2) #-号可以求差集
print(str2-str1)  #空值
#print(str1+str2)  #set里不能使用+号




s = {'s', 'e', 't'}
print(s)


s = set(['a, b, c, d, e'])
print(s)


#集合也可以用表达式（推导）的方式创建
s = {x * 2 for x in 'abc'}  #{'aa', 'bb', 'cc}
print(s)
s = {x **2 for x in range(1,5)}   #{1, 4, 9, 16}
print(s)


#set就像是把Dict中的key抽出来了一样，类似于一个List，但是内容又不能重复，通过调用set()方法创建：
s = set(['A', 'B', 'C'])

print( 'A' in s)

print ('D' in s)


s = set([('Adam', 95), ('Lisa', 85), ('Bart', 59)])

#tuple
for x in s:
    print( x[0],':',x[1])






#Python 中可以用于集合的函数主要有 add( )、clear( )、copy( )、discard( )、remove( )、pop( )、difference( )、intersection( )、union( ) 等。

# add( ) 方法用于为集合添加一个元素，例如：
a={'a', 'b', 'c'}
a.add('d')
print(a)
a.add('d')
print(a)
# 使用update()可以增加多个元素，update可以使用元组、列表、字符串或其他集合作为参数
s = {"P","y"}
s.add('t')
print(s)
s.update(['a','e','o'])
print(s)

s.update(['H','e'],{'l','m','n','o'})
print(s)

#clear( ) 方法用于清空一个集合，例如：
a={'a', 'b', 'c'}
a.clear()
print(a)

#copy( ) 方法用于复制一个集合，例如：
a={'a', 'b', 'c'}
b = a.copy()
print(b)

# discard( ) 方法用于删除集合中一个指定元素，例如：
a={'a', 'b', 'c'}
a.discard('b')
print(a)


#remove( ) 方法与 discard( ) 方法作用相同，区别在于 remove() 方法在移除集合中一个不存在的元素时会发生错误，而 discard( ) 方法不会。
a={'a', 'b', 'c'}
a.remove('b')
print(a)

a.remove('b')
print(a)


# pop() 方法用于从集合中随机移除一个元素，例如：
a={'a', 'b', 'c', 'd', 'e', 'f', 'g'}
print(a)
a.pop()
print(a)

s = set("abcd")
print(s.pop())
print(s)

s.clear() #清空
print(s)


# 两个不同的集合可以执行交、并、补、差等运算，例如：

drawer = {'pen', 'pencil', 'ruler', 'eraser'}
desk = {'pen', 'book ', 'cup'}

s = drawer | desk   #两个集合的并集
print(s)
s = drawer.union(desk)   #两个集合的并集
print(s)

s = drawer & desk   #两个集合的交集
print(s)
s = drawer.intersection(desk)   #两个集合的交集
print(s)

s = drawer ^ desk    #两个集合的交集的补集
print(s)
s = drawer.symmetric_difference(desk)  #返回一个新集合，包含所有只在其中一个集合中出现的元素。
print(s)

s = drawer - desk    #两个集合的差集
print(s)
s = drawer.difference(desk)    #两个集合的差集
print(s)



# 子集
A= set('abcd')
B = set('cdef')
C = set('ab')

print(C<A)# 子集

print(C.issubset(A))# 子集


































































































































































































































































































