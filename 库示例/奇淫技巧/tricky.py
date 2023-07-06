#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 19:33:34 2022
https://mp.weixin.qq.com/s?__biz=MzIxNjM4NDE2MA==&mid=2247515244&idx=3&sn=b187e07a6972b4b6fb6e67ae94be769b&chksm=978b2ba3a0fca2b5088f498e2808240a21e06e18a9b4388b7a5b96e23eed41879ebf4700f9ee&mpshare=1&scene=1&srcid=0117t9sSfJ7VJ0bLKzKiSyVJ&sharer_sharetime=1647688437348&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AVyJ9vEftE2N9nSd%2F9aPe08%3D&acctmode=0&pass_ticket=HtUZpbsBpQ0mIk%2BoPK4K47uZExu7fNHfVfHlbsTOmMv74UV9HL0sgd84sBqhifTg&wx_header=0#rd
@author: jack
"""
#下面的方法可以检查给定列表中是否有重复的元素。它使用了 set() 属性，该属性将会从列表中删除重复的元素。
def all_unique(lst):
    return len(lst) == len(set(lst))

x = [1,1,2,2,3,2,3,4,5,6]
y = [1,2,3,4,5]
all_unique(x) # False
all_unique(y) # True

#检测两个字符串是否互为变位词（即互相颠倒字符顺序）
from collections import Counter

def anagram(first, second):
    return Counter(first) == Counter(second)
anagram("abcd3", "3acdb") # True


#以下代码段可用来检查对象的内存使用情况。
import sys
variable = 30
print(sys.getsizeof(variable)) # 24



#以下方法将以字节为单位返回字符串长度。

def byte_size(string):
    return(len(string.encode(UTF-8)))
byte_size( 'sasas' ) # 4    byte_size( Hello World ) # 11




#以下代码不需要使用循环即可打印某个字符串 n 次
n = 2;
s ="Programming"; print(s * n);
# ProgrammingProgramming



#以下代码段使用 title() 方法将字符串内的每个词进行首字母大写。
s = "programming is awesome"
print(s.title()) # Programming Is Awesome

#以下方法使用 range() 将列表分块为指定大小的较小列表。

from math import ceil
def chunk(lst, size):
    return list( map(lambda x: lst[x * size:x * size + size], list(range(0, ceil(len(lst) / size)))))    chunk([1,2,3,4,5],2) # [[1,2],[3,4],5]



#以下方法使用 fliter() 删除列表中的错误值（如：False, None, 0 和“”）

def compact(lst):
    return list(filter(bool, lst))
compact([0, 1, False, 2,   , 3,  a ,  s , 34]) # [ 1, 2, 3,  a ,  s , 34 ]


#以下代码段可以用来转换一个二维数组。

array = [[ a ,  b ], [ c ,  d ], [ e ,  f ]]
transposed = zip(*array)
print(transposed) # [( a ,  c ,  e ), ( b ,  d ,  f )]



#以下代码可以在一行中用各种操作符进行多次比较。
a = 3
print( 2 < a < 8) # True
print(1 == a < 2) # False




#以下代码段可将字符串列表转换为单个字符串，列表中的每个元素用逗号分隔。


hobbies = ["basketball", "football", "swimming"]
print("My hobbies are: " + ", ".join(hobbies)) # My hobbies are: basketball, football, swimming




#以下方法可计算字符串中元音字母（‘a’, ‘e’, ‘i’, ‘o’, ‘u’）的数目。

import re
def count_vowels(str):
    return len(len(re.findall(r [aeiou] , str, re.IGNORECASE)))
count_vowels( foobar ) # 3
count_vowels( gym ) # 0




#以下方法可用于将给定字符串的第一个字母转换为小写。

def decapitalize(string):
    return str[:1].lower() + str[1:]
decapitalize( FooBar ) #  fooBar
decapitalize( FooBar ) #  fooBar



#以下方法使用递归来展开潜在的深度列表。


def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
            return ret
def deep_flatten(lst):
    result = []
    result.extend(spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, lst))))
    return resultdeep_flatten([1, [2], [[3], 4], 5]) # [1,2,3,4,5]




#该方法只保留第一个迭代器中的值，从而发现两个迭代器之间的差异。

def difference(a, b):
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return list(comparison)
difference([1,2,3], [1,2,4]) # [3]

#下面的方法在将给定的函数应用于两个列表的每个元素后，返回两个列表之间的差值。
def difference_by(a, b, fn):
    b = set(map(fn, b))
    return [item for item in a if fn(item) not in b]
from math import floor
difference_by([2.1, 1.2], [2.3, 3.4],floor) # [1.2]
difference_by([{  x : 2 }, {  x : 1 }], [{  x : 1 }], lambda v : v[ x ]) # [ { x: 2 } ]

#以下方法可在一行中调用多个函数。

def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
a, b = 4, 5
print((subtract if a > b else add)(a, b)) # 9



#以下方法使用 set() 方法仅包含唯一元素的事实来检查列表是否具有重复值。

def has_duplicates(lst):
    return len(lst) != len(set(lst))

x = [1,2,3,4,5,5]
y = [1,2,3,4,5]
has_duplicates(x) # True
has_duplicates(y) # False



#以下方法可用于合并两个词典。
def merge_two_dicts(a, b):
    c = a.copy()   # make a copy of a
    c.update(b)    # modify keys and values of a with the ones from b
    return c
a = {  x : 1,  y : 2}
b = {  y : 3,  z : 4}
print(merge_two_dicts(a, b)) # { y : 3,  x : 1,  z : 4}

#在Python 3.5及更高版本中，你还可以执行以下操作：


def merge_dictionaries(a, b):
    return {**a, **b}
a = {  x : 1,  y : 2}
b = {  y : 3,  z : 4}
print(merge_dictionaries(a, b)) # { y : 3,  x : 1,  z : 4}




#以下方法可将两个列表转换成一个词典。

def to_dictionary(keys, values):
    return dict(zip(keys, values))

keys = ["a", "b", "c"]
values = [2, 3, 4]
print(to_dictionary(keys, values)) # { a : 2,  c : 4,  b : 3}




#以下方法将字典作为输入，然后仅返回该字典中的键。


list = ["a", "b", "c", "d"]
for index, element in enumerate(list):
    print("Value", element, "Index ", index, )
# ( Value ,  a ,  Index  , 0)
# ( Value ,  b ,  Index  , 1)
#( Value ,  c ,  Index  , 2)
# ( Value ,  d ,  Index  , 3)






#以下代码段可用于计算执行特定代码所需的时间。
import time
start_time = time.time()
a = 1
b = 2
c = a + b
print(c) #3
end_time = time.time()
total_time = end_time - start_time
print("Time: ", total_time)
# ( Time:  , 1.1205673217773438e-05)



#你可以将 else 子句作为 try/except 块的一部分，如果没有抛出异常，则执行该子句。

try:
    2*3
except TypeError:
    print("An exception was raised")
else:
    print("Thank God, no exceptions were raised.")
#Thank God, no exceptions were raised.



#以下方法返回列表中出现的最常见元素。

def most_frequent(list):
    return max(set(list), key = list.count)

list = [1,2,1,2,3,2,1,4,2]
most_frequent(list)

#以下方法可检查给定的字符串是否为回文结构。该方法首先将字符串转换为小写，然后从中删除非字母数字字符。最后，它会将新的字符串与反转版本进行比较。

def palindrome(string):
    from re import sub
    s = sub( [W_] , string.lower())
    return s == s[::-1]
palindrome(' taco cat ') # True




#以下代码段将展示如何编写一个不使用 if-else 条件的简单计算器。
import operator
action = {
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
    "*": operator.mul,
    "**": pow
}
print(action[-](50, 25)) # 25


#以下算法通过实现 Fisher-Yates算法 在新列表中进行排序来将列表中的元素顺序随机打乱。

from copy import deepcopy
from random import randint
def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst

foo = [1,2,3]
shuffle(foo) # [2,3,1] , foo = [1,2,3]

#以下方法可使列表扁平化，类似于JavaScript中的[].concat(…arr)。


def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret
spread([1,2,3,[4,5,6],[7],8,9]) # [1,2,3,4,5,6,7,8,9]

#以下是交换两个变量的快速方法，而且无需使用额外的变量。

def swap(a, b):
  return b, a
a, b = -1, 14
swap(a, b) # (14, -1)


#以下代码段显示了如何在字典中没有包含要查找的键的情况下获得默认值。





d = { a : 1,  b : 2}
print(d.get( c , 3)) # 3






































