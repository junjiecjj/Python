#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: args.py
# Author: 陈俊杰
# Created Time: 2022年04月22日 星期五 23时28分00秒

# mail: 2716705056@qq.com
#此程序的功能是：

# https://www.cnblogs.com/bingabcd/p/6671368.html


#########################################################################



def func_arg(farg, *args):
    print("formal arg:", farg)
    for arg in args:
        print("another arg:", arg)
func_arg(1,"youzan",'dba','hello')
# 输出结果如下：
# formal arg: 1
# another arg: youzan
# another arg: dba
# another arg: hello



def func_arg(farg, earg, *args):
    print("formal arg:", farg)
    for arg in args:
        print("another arg:", arg)
func_arg(1,"youzan",'dba','hello')
# 输出结果如下：
# formal arg: 1
# another arg: dba
# another arg: hello




#利用它转换参数为字典
def kw_dict(**kwargs):
    return kwargs
print(kw_dict(a=1,b=2,c=3))

# 输出结果如下：
# {'a': 1, 'b': 2, 'c': 3}






def test_args(first, *args):
    print('Required argument: ', first)
    print(type(args))
    for v in args:
        print ('Optional argument: ', v)

test_args(1, 2, 3, 4)




def test_kwargs(first, *args, **kwargs):
   print('Required argument: ', first)
   print(type(args))
   for v in args:
      print ('Optional argument (args): ', v)
   print(type(kwargs))
   for k, v in kwargs.items():
      print ('Optional argument %s (kwargs): %s' % (k, v))

test_kwargs(1, 2, 3, 4, k1=5, k2=6)



def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

args = ("two", 3, 5)
test_args_kwargs(*args)




kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)



#================================================================================================
# https://blog.csdn.net/luckytanggu/article/details/51714757
#================================================================================================
# 一、位置参数
# 调用函数时根据函数定义的参数位置来传递参数。
def print_hello(name, sex):
	sex_dict = {1: u'先生', 2: u'女士'}
	print('hello %s %s, welcome to python world!' %(name, sex_dict.get(sex, u'先生')))


# 两个参数的顺序必须一一对应，且少一个参数都不可以
print_hello('tanggu', 1)


# 二、关键字参数
# 用于函数调用，通过“键-值”形式加以指定。可以让函数更加清晰、容易使用，同时也清除了参数的顺序需求。

# 以下是用关键字参数正确调用函数的实例
print_hello('tanggu', sex=1)
print_hello(name='tanggu', sex=1)
print_hello(sex=1, name='tanggu')

# 以下是错误的调用方式
# print_hello(1, name='tanggu')
# print_hello(name='tanggu', 1)
# print_hello(sex=1, 'tanggu')
# 通过上面的代码，我们可以发现：有位置参数时，位置参数必须在关键字参数的前面，但关键字参数之间不存在先后顺序的

# 三、默认参数
# 用于定义函数，为参数提供默认值，调用函数时可传可不传该默认参数的值（注意：所有位置参数必须出现在默认参数前，包括函数定义和调用）

# 正确的默认参数定义方式--> 位置参数在前，默认参数在后
def print_hello(name, sex=1):
	sex_dict = {1: u'先生', 2: u'女士'}
	print('hello %s %s, welcome to python world!' %(name, sex_dict.get(sex, u'先生')))

# 错误的定义方式
# def print_hello1(sex=1, name):
# 	sex_dict = {1: u'先生', 2: u'女士'}
# 	print('hello %s %s, welcome to python world!' %(name, sex_dict.get(sex, u'先生')))

# 调用时不传sex的值，则使用默认值1
print_hello('tanggu')

print_hello('tanggu',2)

#调用时传入sex的值，并指定为2
print_hello('tanggu', sex=2)




# 基本原则是：先位置参数，默认参数，包裹位置，包裹关键字(定义和调用都应遵循)
def func(name, age, sex=1, *args, **kargs):
	print (name, age, sex, args, kargs)


func('tanggu', 25, 2, 'music', 'sport', class1=2)
# tanggu 25 1 ('music', 'sport') {'class1'=2}


#================================================================================================
# https://blog.csdn.net/cadi2011/article/details/86641811
#================================================================================================

#定义一个名为temp的函数，参数列表共4个参数

def temp(first,second="Hello World",*args,**kwargs):
    print(first)
    print(second)
    print(args)
    print(kwargs)

# 1、参数first称为位置参数

# 2、参数second称为默认参数

# 3、参数*args称为可变参数

# 4、参数**kwargs也称为可变参数




#关键字参数
#1、函数调用时，指定参数名称，称为关键字参数（别和默认参数混淆，这里是函数调用）

def temp(a,b,c):
    print(a)
    print(b)
    print(c)


temp(100, 32, c = 1100)


# 命名关键字参数
# 1、英文名：Keyword-only parameter

# 2、特点：必须使用关键字方式传递参数

# 3、语法

def only_kw(a,*,b,c):
    print(a)
    print(b)
    print(c)


only_kw(100,b=1000,c=99) #b和c必须使用参数名传递参数



#命名位置参数
def only_position(a,b,/):
    print(a)
    print(b)
#函数调用时，必须使用位置参数方式传递参数……，不能再使用关键字参数调用该函数


only_position(1,2)


#  only_position(a = 1,b = 2)   # error
















