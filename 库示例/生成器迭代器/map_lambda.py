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


s = [1,2,3]
print(list(map(lambda x:x+1,s)))


s = [1,2,3]
print(list(map(lambda x, y, z:x*y*z, s , s, s)))




















