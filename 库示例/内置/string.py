#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:19:03 2022

@author: jack
"""


# 1
name = 'leo'
n=37
s = '{} has {} message.'.format(name,n)
print(s)

# 2
name = 'leo'
n=37
s = '{name} has {n} message.' 
print(s.format_map(vars()))


name = 'jack'
n = 43
print(s.format_map(vars()))





"""
Python3 startswith()方法
Python3 字符串 Python3 字符串

描述
startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查。

语法
string.startswith(value, start, end)
参数值
参数	描述
value	必需。检查字符串是否以其开头的值。
start	可选。整数，规定从哪个位置开始搜索。
end	可选。整数，规定结束搜索的位置。
返回值
如果检测到字符串则返回True，否则返回False。
"""
str = "this is string example....wow!!!"
print (str.startswith( 'this' ))   # 字符串是否以 this 开头
print (str.startswith( 'string', 8 ))  # 从第九个字符开始的字符串是否以 string 开头
print (str.startswith( 'this', 2, 4 )) # 从第2个字符开始到第四个字符结束的字符串是否以 this 开头


txt = "Hello, welcome to my world."
x = txt.startswith("Hello")
print(x)

txt = "Hello, welcome to my world."
x = txt.startswith("wel", 7, 20)
print(x)


"""
语法
string.endswith(value, start, end)
参数值
参数	描述
value	必需。检查字符串是否以之结尾的值。
start	可选。整数。规定从哪个位置开始检索。
end	可选。整数。规定从哪个位置结束检索。

定义和用法
如果字符串以指定值结尾，则 endswith() 方法返回 True，否则返回 False。
"""

txt = "Hello, welcome to my world."
x = txt.endswith("my world.")
print(x)


txt = "Hello, welcome to my world."
x = txt.endswith("my world.", 5, 11)
print(x)



Str='Runoob example....wow!!!'
suffix='!!'
print (Str.endswith(suffix))
print (Str.endswith(suffix,20))
suffix='run'
print (Str.endswith(suffix))
print (Str.endswith(suffix, 0, 19))


str = "this is string example....wow!!!";
suffix = "wow!!!";
print( str.endswith(suffix))
print( str.endswith(suffix,20))
suffix = "is";
print( str.endswith(suffix, 2, 4))
print (str.endswith(suffix, 2, 6))




















































































































































