#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:30:45 2022

@author: jack
"""
# 考虑function为None的情形。
# list(map(None, [1,2,3])) #[1, 2, 3]

# list(map(None, [1,2,3], [4,5,6])) #[(1, 4), (2, 5), (3, 6)]

# list(map(None, [1,2,3], [4,5])) #[(1, 4), (2, 5), (3, None)]


#考虑function为lambda表达式的情形。此时lambda表达式:的左边的参数的个数与map函数sequence的个数相等, :右边的表达式是左边一个或者多个参数的函数。

print( list((map(lambda x: x+1, [1,2,3])))) #[2, 3, 4]

print(list(map(lambda x, y:x+y, [1,2,3], [4,5,6]))) #[5, 7, 9]

print(list(map(lambda x, y:x == y, [1,2,3], [4,5,6]))) #[False, False, False]

def f(x):
    return True if x==1 else False
print(list(map(lambda x: f(x), [1,2,3])) ) #[True, False, False]


#考虑function为lambda表达式的情形。此时lambda表达式:的左边的参数的个数与map函数sequence的个数相等, :右边的表达式是左边一个或者多个参数的函数。

print(list(map(lambda x: x+1, [1,2,3])) ) #[2, 3, 4]

print(list(map(lambda x, y:x+y, [1,2,3], [4,5,6])) ) #[5, 7, 9]

print(list(map(lambda x, y:x == y, [1,2,3], [4,5,6])) ) #[False, False, False]

def f2(x):
    return True if x==1 else False
print(list(map(lambda x: f2(x), [1,2,3])) ) #[True, False, False]

# 考虑函数不为lambda表达式的情形:
def f1(x):
    return True if x==1 else False
print(list(map(f1, [1,2,3]))) #[True, False, False]


s = [1,2,3]
print(list(map(lambda x:x+1,s)))



s = [1,2,3]
print(list(map(lambda x,y,z:x*y*z ,s,s,s)))