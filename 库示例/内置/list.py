#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:55:51 2022

@author: jack


Python列表函数&方法
Python包含以下函数:

序号	函数
1	cmp(list1, list2)
比较两个列表的元素
2	len(list)
列表元素个数
3	max(list)
返回列表元素最大值
4	min(list)
返回列表元素最小值
5	list(seq)
将元组转换为列表
Python包含以下方法:

序号	方法
1	list.append(obj)
在列表末尾添加新的对象
2	list.count(obj)
统计某个元素在列表中出现的次数
3	list.extend(seq)
在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
4	list.index(obj)
从列表中找出某个值第一个匹配项的索引位置
5	list.insert(index, obj)
将对象插入列表
6	list.pop([index=-1])
移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
7	list.remove(obj)
移除列表中某个值的第一个匹配项
8	list.reverse()
反向列表中元素
9	list.sort(cmp=None, key=None, reverse=False)
对原列表进行排序

"""

"""
# Python中List的复制（直接复制、浅拷贝、深拷贝）
"""
"""
直接赋值：
如果用 = 直接赋值，是非拷贝方法。
这两个列表是等价的，修改其中任何一个列表都会影响到另一个列表。
"""
old = [1,[1,2,3],3]
new = []
for i in range(len(old)):
    new.append(old[i])

new[0] = 3
new[1][0] = 3

'''
-----------------------
Before:
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After:
[3, [3, 2, 3], 3]
[3, [3, 2, 3], 3]
-----------------------
'''


'''
浅拷贝：
1.copy()方法
对于List来说，其第一层，是实现了深拷贝，但对于其内嵌套的List，仍然是浅拷贝。

因为嵌套的List保存的是地址，复制过去的时候是把地址复制过去了，嵌套的List在内存中指向的还是同一个。
'''
old = [1,[1,2,3],3]
new = old.copy()

new[0] = 3
new[1][0] =3

'''
---------------------
Before:
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After:
[1, [3, 2, 3], 3]
[3, [3, 2, 3], 3]
---------------------
'''


'''
2.使用列表生成式
使用列表生成式产生新列表也是一个浅拷贝方法，只对第一层实现深拷贝。
'''
old = [1,[1,2,3],3]
new = [i for i in old]

new[0] = 3
new[1][0] = 3

'''
----------------------
Before
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After
[1, [3, 2, 3], 3]
[3, [3, 2, 3], 3]
----------------------
'''


"""
3.for循环遍历
通过for循环遍历，将元素一个个添加到新列表中。这也是一个浅拷贝方法，只对第一层实现深拷贝。
"""
old = [1,[1,2,3],3]
new = []
for i in range(len(old)):
    new.append(old[i])

new[0] = 3
new[1][0] = 3

'''
-----------------------
Before:
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After:
[1, [3, 2, 3], 3]
[3, [3, 2, 3], 3]
-----------------------
'''

'''
4.使用切片
通过使用 [ : ] 切片，可以浅拷贝整个列表，同样的，只对第一层实现深拷贝。
'''
old = [1,[1,2,3],3]
new = old[:]

new[0] = 3
new[1][0] = 3

'''
------------------
Before:
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After:
[1, [3, 2, 3], 3]
[3, [3, 2, 3], 3]
------------------
'''

"""
深拷贝：
如果用deepcopy()方法，则无论多少层，无论怎样的形式，得到的新列表都是和原来无关的，这是最安全最清爽最有效的方法。

需要import copy
"""

import copy
old = [1,[1,2,3],3]
new = copy.deepcopy(old)

new[0] = 3
new[1][0] = 3

'''
-----------------------
Before:
[1, [1, 2, 3], 3]
[1, [1, 2, 3], 3]
After:
[1, [1, 2, 3], 3]
[3, [3, 2, 3], 3]
-----------------------
'''





#========================================================================
#   https://www.jianshu.com/p/50da60d54a14
#========================================================================
list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]

#访问列表中的值
list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5, 6, 7 ]

print("list1[0]: ", list1[0])
print("list2[1:5]: ", list2[1:5])



#更新列表
list = []          ## 空列表
list.append('Google')   ## 使用 append() 添加元素
list.append('Runoob')
print (list)


#删除列表元素
list1 = ['physics', 'chemistry', 1997, 2000]

print (list1)
del list1[2]
print ("After deleting value at index 2 : ")
print (list1)


L = ['Google', 'Runoob', 'Taobao']
print("L[2] = {}".format(L[2]))
print(" L[-2] = {}".format( L[-2]))
print("L[1:] = {}".format(L[1:]))

#列表元素的个数
L = [1,2,3,4]
print(len(L))



#max(list)
#返回列表的最大值
L = [1,2,3,4]
print(max(L))




#min(list)
#返回列表的最小值
li = [1, 2, 3, 4, 5]
print(len(li))
# 5
print(max(li))
# 5
print(min(li))
# 1




#list(seq)
#将元组转化为列表
aTuple = (1, 'd', 7)
#print(list(aTuple))
# [1, 'd', 7]

a = [5,7,6,3,4,1,2]
b = sorted(a)       # 保留原列表


L=[('b',2),('a',1),('c',3),('d',4)]

sorted(L, key=lambda x:x[1])               # 利用key



students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=lambda s: s[2])            # 按年龄排序


sorted(students, key=lambda s: s[2], reverse=True)       # 按降序


#返回迭代器 reversed()
L = [3,5,2,1,4]
L.reverse()
print(L)


# 追加对象到list, list.append(obj)
L = [1,2,3]
L.append(4)
print(L)
# [1, 2, 3, 4]


L.append([5, 6])
print(L)
# [1, 2, 3, 4, [5, 6]]

# list.count(obj)
# 统计某个元素在列表中出现的个数
L = [1,2,3,4,1,2,1]
print(L.count(1))
# 3
print(L.count(2))
# 2


#list.extend(list)
#追加一个可迭代对象到list
L = [1,2,3]
A = [4,5,6]
L.extend(A)
print(L)
# [1,2,3,4,5,6]
# !!!! 该方法返回值为None，修改的是原列表


#list.index(obj)
#计算List中某个对象第一次出现的位置
L= [1,2,3,3]
print(L.index(3))
# 2


#list.insert(index,obj)
#在指定位置增加一个元素
L= [1,2,3]
L.insert(0,10)
print(L)
# [10,1,2,3]


#list.pop(obj=list[-1])
#移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
L = [1, 2, 3, 4, 5]
print(L.pop())
# 5
print(L)
# [1,2,3,4]
L.pop(1)
print(L)
# [1,3,4]

#list.remove(obj)
#移除第一个匹配的指定对象
L= [1,2,3,3]
L.remove(3)
print(L)
# [1,2,3]


#list.clear(obj)
#清空列表，等同于del list[:]
L = [1,2,3,4]
L.clear()
print(L)
# []


#list.copy(obj)
#用于复制列表，等同于list[:]
L1 = [1,2,3,4]
L2 = L1.copy()
print(L2)
# [1,2,3,4]


#'str'.join(list)
#将列表变成字符串

li = ['my','name','is','bob']
print(' '.join(li))
# 'my name is bob'

print('_'.join(li))
# 'my_name_is_bob'

s = ['my','name','is','bob']
print(' '.join(s))
# 'my name is bob'

print('..'.join(s))
# 'my..name..is..bob'


#split(seq,maxsplit=-1)

b = 'my..name..is..bob'
print(b.split())
# ['my..name..is..bob']

print(b.split(".."))
# ['my', 'name', 'is', 'bob']

print(b.split("..",0))
# ['my..name..is..bob']

print(b.split("..",1))
# ['my', 'name..is..bob']

print(b.split("..",2))
# ['my', 'name', 'is..bob']

print(b.split("..",-1))
# ['my', 'name', 'is', 'bob']

#可以看出 b.split("..",-1)等价于b.split("..")


list1, list2 = [123, 'xyz', 'zara'], [456, 'abc']

print ("First list length : ", len(list1));
print ("Second list length : ", len(list2));


#========================================================================
#  https://www.runoob.com/python/python-lists.html
#========================================================================

#append() 方法用于在列表末尾添加新的对象。
aList = [123, 'xyz', 'zara', 'abc'];
aList.append( 2009 );
print( "Updated List : ", aList);




#count() 方法用于统计某个元素在列表中出现的次数。
aList = [123, 'xyz', 'zara', 'abc', 123];

print ("Count for 123 : ", aList.count(123))
print( "Count for zara : ", aList.count('zara'))




#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
aList = [123, 'xyz', 'zara', 'abc', 123];
bList = [2009, 'manni'];
aList.extend(bList)

print ("Extended List : ", aList )


#index() 函数用于从列表中找出某个值第一个匹配项的索引位置。

#语法
#index()方法语法：
#list.index(x[, start[, end]])
aList = [123, 'xyz', 'runoob', 'abc']

print ("xyz 索引位置: ", aList.index( 'xyz' ))
print ("runoob 索引位置 : ", aList.index( 'runoob', 1, 3 ))



#insert() 函数用于将指定对象插入列表的指定位置。
#语法
#insert()方法语法：
#list.insert(index, obj)
aList = [123, 'xyz', 'zara', 'abc']

aList.insert( 3, 2009)

print ("Final List : ", aList)



#pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。

# pop()方法语法：
#list.pop([index=-1])

list1 = ['Google', 'Runoob', 'Taobao']
list_pop=list1.pop(1)
print ("删除的项为 :", list_pop)
print ("列表现在为 : ", list1)

#remove() 函数用于移除列表中某个值的第一个匹配项。
#语法
#remove()方法语法：
#list.remove(obj)
aList = [123, 'xyz', 'zara', 'abc', 'xyz'];

aList.remove('xyz');
print ("List : ", aList);
aList.remove('abc');
print ("List : ", aList);



#remove() 函数用于移除列表中某个值的第一个匹配项。
# 语法
# remove()方法语法：
# list.remove(obj)

aList = [123, 'xyz', 'zara', 'abc', 'xyz'];

aList.remove('xyz');
print ("List : ", aList);
aList.remove('abc');
print ("List : ", aList);


# reverse() 函数用于反向列表中元素。
# 语法
# reverse()方法语法：
 #list.reverse()
aList = [123, 'xyz', 'zara', 'abc', 'xyz']

aList.reverse()
print ("List : ", aList)


#sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。

#语法
#sort()方法语法：

# list.sort(cmp=None, key=None, reverse=False)
# 参数
# cmp -- 可选参数, 如果指定了该参数会使用该参数的方法进行排序。
# key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
#reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。
aList = ['123', 'Google', 'Runoob', 'Taobao', 'Facebook'];

aList.sort();
print("List : ")
print(aList)


# 列表
vowels = ['e', 'a', 'u', 'o', 'i']

# 降序
vowels.sort(reverse=True)

# 输出结果
print('降序输出:')
print( vowels )


# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

# 列表
random = [(2, 2), (3, 4), (4, 1), (1, 3)]

# 指定第二个元素排序
random.sort(key=takeSecond)

# 输出类别
print('排序列表：')
print(random)
