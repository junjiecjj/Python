#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:40:44 2022

@author: jack

1.Python strip()函数 介绍

函数原型

声明：

s为字符串，rm为要删除的字符序列
s.strip(rm) 删除s字符串中开头、结尾处，位于 rm删除序列 的字符(如果rm中不包含　开头或结尾　的那个字母，则不会删除)
s.lstrip(rm) 删除s字符串中开头处，位于 rm删除序列 的字符(如果rm中不包含开头的那个字母，则不会删除)
s.rstrip(rm) 删除s字符串中结尾处，位于 rm删除序列 的字符(如果rm中不包含结尾的那个字母，则不会删除)
(1)当rm为空时，默认删除空白符（包括'\n', '\r', '\t', ' ')
(2)这里的rm删除序列是只要边（开头或结尾）上的字符在删除序列内，就删除掉。


strip用于字符串头部和尾部的指定字符串，默认为空格或换行符。



二.split

这个函数的用法是拆分字符串，然后把分割之后的字符串放到一个列表里并返回。默认情况下是根据换行符"\n"和空格" ",以及“\t”进行分割，比如我们有代码：





"""

import numpy as np

Str = "123123\n"
print("str.strip() = {}".format(Str.strip()))

Str="I love I"
print("str.strip(\"I\") = {}".format(Str.strip("I")))#删除收尾的I


Str = '     123'
print("str.strip() = {}".format(Str.strip()))



Str='\t\tabc'
print("str.strip() = {}".format(Str.strip()))

Str = 'sdff\r\n'
print("str.strip() = {}".format(Str.strip()))


Str = '123abc'
print("str.strip('21') = {}".format(Str.strip('21')))
print("str.strip('12') = {}".format(Str.strip('12')))


Str = ('www.google.com')
print (Str)


str_split = Str.split('.')
print("str.split('.') = {}".format(Str.split('.')))

#按照某一个字符分割，且分割n次。如按‘.'分割1次
str_split = Str.split('.',1)
print (str_split)

#(3).split()函数后面还可以加正则表达式，例如：
str_split = Str.split('.')[0]
print (str_split)


str_split = Str.split('.')[::-1]
print (str_split)



str_split = Str.split('.')[::]
print (str_split)

#按反序列排列，[::]安正序排列
Str = Str + '.com.cn'
str_split = Str.split('.')[::-1]
print (str_split)
str_split = Str.split('.')[:-1]
print (str_split)


#ip ==> 数字
ip2num = lambda x:sum([256**j*int(i) for j,i in enumerate(x.split('.')[::-1])])
ip2num('192.168.0.1')

# 数字 ==> ip # 数字范围[0, 255^4]
num2ip = lambda x: '.'.join([str(x/(256**i)%256) for i in range(3,-1,-1)])
num2ip(3232235521)




#最后，python怎样将一个整数与IP地址相互转换？

import socket
import struct

int_ip = 123456789
socket.inet_ntoa(struct.pack('I',socket.htonl(int_ip)))#整数转换为ip地址

#‘7.91.205.21'
str(socket.ntohl(struct.unpack("I",socket.inet_aton("255.255.255.255"))[0]))#ip地址转换为整数
#‘4294967295'






























































