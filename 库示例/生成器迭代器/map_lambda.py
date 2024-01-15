#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:30:45 2022

@author: jack


匿名函数: 关键字 lambda 表示匿名函数, 冒号前面的x表示函数参数.

匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果。

用匿名函数有个好处，因为函数没有名字，不必担心函数名冲突。此外，匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数：


"""

f = lambda x: x * x
print(f"f(5) = {f(5)}")


#同样，也可以把匿名函数作为返回值返回，比如：
def build(x, y):
    return lambda: x * x + y * y

f1 = build(2, 3)
print(f"f1() = {f1()}")



import numpy as np
def get_generator_input_sampler(mean, std):
    return lambda m, n: np.random.normal(loc = mean, scale = std,  size = (m, n))  # Uniform-dist data into generator, _NOT_ Gaussian

f2 = get_generator_input_sampler(0, 2)
print(f"f2(4, 5).shape = {f2(4, 5).shape}")


# 考虑function为None的情形。error
# list(map(None, [1,2,3])) #[1, 2, 3]

# list(map(None, [1,2,3], [4,5,6])) #[(1, 4), (2, 5), (3, 6)]

# list(map(None, [1,2,3], [4,5])) #[(1, 4), (2, 5), (3, None)]


#考虑function为lambda表达式的情形。此时lambda表达式:的左边的参数的个数与map函数sequence的个数相等, :右边的表达式是左边一个或者多个参数的函数。

print(list(map(lambda x: x+1, [1,2,3])) ) #[2, 3, 4]

print(list(map(lambda x, y:x+y, [1,2,3], [4,5,6]))) #[5, 7, 9]

print(list(map(lambda x, y:x == y, [1,2,3], [4,5,6]))) #[False, False, False]

def f(x):
    return True if x==1 else False
print(list(map(lambda x: f(x), [1,2,3])) ) #[True, False, False]


# 考虑函数不为lambda表达式的情形:
def f1(x):
    return True if x==1 else False
print(list(map(f1, [1,2,3]))) #[True, False, False]
# [True, False, False]

s = [1,2,3]
print(list(map(lambda x:x+1,s)))
# [2, 3, 4]

s = [1,2,3]
print(list(map(lambda x, y, z:x*y*z, s , s, s)))
# [1, 8, 27]


mylist = [3,6,3,2,4,8,23]
sorted(mylist, key=lambda x: x%2==0)

random = [(2, 2), (3, 4), (4, 1), (1, 3)]
#按第二个元素升序排列
s = sorted(random,key=lambda x:x[1])
print(s)
#按第一个元素降序排列
ss = sorted(random,key=lambda x:x[0])
print(ss)
# [(4, 1), (2, 2), (1, 3), (3, 4)]
# [(1, 3), (2, 2), (3, 4), (4, 1)]

a=[('b',3),('a',2),('d',4),('c',1)]
##按照第一个元素排序
sorted(a,key=lambda x:x[0])
# [('a', 2), ('b', 3), ('c', 1), ('d', 4)]

## 按照第二个元素排序
sorted(a,key=lambda x:x[1])
# [('c',1),('a',2),('b',3),('d',4)]

# 求字符串每个单词的长度
sentence = "Welcome To Beijing!"
words = sentence.split()
lengths  = map(lambda x:len(x),words)
print(list(lengths))
# [7,2,8]



name = ['0_7.png',
 '1_2.png',
 '10_0.png',
 '11_6.png',
 '12_9.png',
 '13_0.png',
 '14_1.png',
 '15_5.png',
 '16_9.png',
 '17_7.png',
 '18_3.png',
 '19_4.png',
 '2_1.png',
 '3_0.png',
 '4_4.png',
 '5_1.png',
 '6_4.png',
 '7_9.png',
 '8_5.png',
 '9_9.png',
 'raw_grid_images.png']
name.remove('raw_grid_images.png')

na = sorted(name, key=lambda x: int(x.split('_')[0]))



def increment(n):
    return lambda x:x+n

f=increment(4)
f(2)
# 6

a = np.arange(12).reshape(3,4)
f(a)

Names = ['Anne', 'Amy', 'Bob', 'David', 'Carrie', 'Barbara', 'Zach']
B_Name= filter(lambda x: x.startswith('B'),Names)
print(list(B_Name))
# ['Bob', 'Barbara']

# 求两个列表元素的和
a = [1,2,3,4]
b = [5,6,7,8]
print(list(map(lambda x,y:x+y, a,b)))

# [6,8,10,12]

# 4、按年龄升序
students = {'john':15, 'jane':12,'dave':10}
sorted(students.items(), key=lambda s: s[1])

# 4、按年龄升序
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
sorted(students, key=lambda s: s[2])
# 结果：
# [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
# 5、按年龄降序
sorted(students, key=lambda s: s[2], reverse=True)
# 结果：
# [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]

import reduce
# 2、两数相加，lambda 写法
reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
# 结果：
# 15


# 2、将列表[1, 2, 3]中能够被3整除的元素过滤出来
newlist = filter(lambda x: x % 3 == 0, [1, 2, 3])
print(list(newlist))
# 结果： [3]


# 1、 [x for x in data if condition]
# 此时data数据中只有满足if条件才能保留，最后生成一列表.
res=[i  for  i  in  range(10)  if  i%2==0]
print(res)
# [0, 2, 4, 6, 8]

# 2、[exp1 if condition else exp2 for x in data]
# 此处if…else，当data中的数据满足if条件时，exp1，否则exp2，最后生成一列表.

res=[i   if  i%2==0  else i+1  for  i  in  range(10)]
print(res)
# [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]



a = [x+y for x in range(1,5) if x > 2 for y in range(1,5) if y < 4]
print(a)
# [4, 5, 6, 5, 6, 7]


# 执行顺序
for x in range(1,5):
    if x >2 :
        for y in range(1,5):
            if y < 4:
                print(x+y)
# 4
# 5
# 6
# 5
# 6
# 7

#1-100中3的倍数

p = [x for x in range(1,12) if x%3 == 0]
print(p)
# [3, 6, 9]

#1-100中，不是3的倍数的数去相反数，其余的数保持不变
q = [x if x%3==0 else -x for x in range(1,12)]
print(q)

# [-1, -2, 3, -4, -5, 6, -7, -8, 9, -10, -11]



