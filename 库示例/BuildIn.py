#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 20:58:25 2022

@author: jack

68 个 Python 内置函数详解
https://mp.weixin.qq.com/s/OjC4IvHRKh-qLUL0PIj4bg

内置函数就是Python给你提供的，拿来直接用的函数，比如print.，input等。
截止到python版本3.6.2 ，python一共提供了68个内置函数，具体如下👇
abs()           dict()        help()         min()         setattr()
all()           dir()         hex()          next()        slice() 
any()           divmod()      id()           object()      sorted() 
ascii()         enumerate()   input()        oct()         staticmethod() 
bin()           eval()        int()          open()        str() 
bool()          exec()        isinstance()   ord()         sum() 
bytearray()     ﬁlter()       issubclass()   pow()         super() 
bytes()         ﬂoat()        iter()         print()       tuple() 
callable()      format()      len()          property()    type() 
chr()           frozenset()   list()         range()       vars() 
classmethod()   getattr()     locals()       repr()        zip() 
compile()       globals()     map()          reversed()    __import__() 
complex()       hasattr()     max()          round() 
delattr()       hash()        memoryview()   set()

"""


# 和数字相关
# 1. 数据类型
# bool : 布尔型(True,False)
# int : 整型(整数)
# float : 浮点型(小数)
# complex : 复数
# 2. 进制转换
# bin() 将给的参数转换成二进制
# otc() 将给的参数转换成八进制
# hex() 将给的参数转换成十六进制
print(bin(10))  # 二进制:0b1010
print(hex(10))  # 十六进制:0xa
print(oct(10))  # 八进制:0o12



# 3. 数学运算
# abs() 返回绝对值
# divmode() 返回商和余数
# round() 四舍五入
# pow(a, b) 求a的b次幂, 如果有三个参数. 则求完次幂后对第三个数取余
# sum() 求和
# min() 求最小值
# max() 求最大值
print(abs(-2))  # 绝对值:2
print(divmod(20,3)) # 求商和余数:(6,2)
print(round(4.50))   # 五舍六入:4
print(round(4.51))   #5
print(pow(10,2,3))  # 如果给了第三个参数. 表示最后取余:1
print(sum([1,2,3,4,5,6,7,8,9,10]))  # 求和:55
print(min(5,3,9,12,7,2))  #求最小值:2
print(max(7,3,15,9,4,13))  #求最大值:15



# 和数据结构相关
# 1. 序列
# （1）列表和元组

# list() 将一个可迭代对象转换成列表
# tuple() 将一个可迭代对象转换成元组
print(list((1,2,3,4,5,6)))  #[1, 2, 3, 4, 5, 6]
print(tuple([1,2,3,4,5,6]))  #(1, 2, 3, 4, 5, 6)



# （2）相关内置函数

# reversed() 将一个序列翻转, 返回翻转序列的迭代器
# slice() 列表的切片
lst = "你好啊"
it = reversed(lst)   # 不会改变原列表. 返回一个迭代器, 设计上的一个规则
print(list(it))  #['啊', '好', '你']
lst = [1, 2, 3, 4, 5, 6, 7]
print(lst[1:3:1])  #[2,3]
s = slice(1, 3, 1)  #  切片用的
print(lst[s])  #[2,3]


# （3）字符串

# str() 将数据转化成字符串
print(str(123)+'456')  #123456
# format()     与具体数据相关, 用于计算各种小数, 精算等.
s = "hello world!"
print(format(s, "^20"))  #剧中
print(format(s, "<20"))  #左对齐
print(format(s, ">20"))  #右对齐
#     hello world!    
# hello world!        
#         hello world!
print(format(3, 'b' ))    # 二进制:11
print(format(97, 'c' ))   # 转换成unicode字符:a
print(format(11, 'd' ))   # ⼗进制:11
print(format(11, 'o' ))   # 八进制:13 
print(format(11, 'x' ))   # 十六进制(⼩写字母):b
print(format(11, 'X' ))   # 十六进制(大写字母):B
print(format(11, 'n' ))   # 和d⼀样:11
print(format(11))         # 和d⼀样:11

print(format(123456789, 'e' ))      # 科学计数法. 默认保留6位小数:1.234568e+08
print(format(123456789, '0.2e' ))   # 科学计数法. 保留2位小数(小写):1.23e+08
print(format(123456789, '0.2E' ))   # 科学计数法. 保留2位小数(大写):1.23E+08
print(format(1.23456789, 'f' ))     # 小数点计数法. 保留6位小数:1.234568
print(format(1.23456789, '0.2f' ))  # 小数点计数法. 保留2位小数:1.23
print(format(1.23456789, '0.10f'))  # 小数点计数法. 保留10位小数:1.2345678900
print(format(1.23456789e+3, 'F'))   # 小数点计数法. 很大的时候输出INF:1234.567890
# bytes() 把字符串转化成bytes类型
bs = bytes("今天吃饭了吗", encoding="utf-8")
print(bs)  #b'\xe4\xbb\x8a\xe5\xa4\xa9\xe5\x90\x83\xe9\xa5\xad\xe4\xba\x86\xe5\x90\x97'
#   bytearray()    返回一个新字节数组. 这个数字的元素是可变的, 并且每个元素的值得范围是[0,256)

ret = bytearray("alex" ,encoding ='utf-8')
print(ret[0])  #97
print(ret)  #bytearray(b'alex')
ret[0] = 65  #把65的位置A赋值给ret[0]
print(str(ret))  #bytearray(b'Alex')


# ord() 输入字符找带字符编码的位置
# chr() 输入位置数字找出对应的字符
# ascii() 是ascii码中的返回该值 不是就返回u
print(ord('a'))  # 字母a在编码表中的码位:97
print(ord('中'))  # '中'字在编码表中的位置:20013

print(chr(65))  # 已知码位,求字符是什么:A
print(chr(19999))  #丟

for i in range(65536):  #打印出0到65535的字符
    print(chr(i), end=" ")

print(ascii("@"))  #'@'


# repr() 返回一个对象的string形式
s = "今天\n吃了%s顿\t饭" % 3
print(s)#今天# 吃了3顿    饭
print(repr(s))   # 原样输出,过滤掉转义字符 \n \t \r 不管百分号%
#'今天\n吃了3顿\t饭'


# 2. 数据集合
# 字典：dict 创建一个字典
# 集合：set 创建一个集合
# frozenset() 创建一个冻结的集合，冻结的集合不能进行添加和删除操作。

# 3. 相关内置函数
# len() 返回一个对象中的元素的个数
# sorted() 对可迭代对象进行排序操作 (lamda)
# 语法：sorted(Iterable, key=函数(排序规则), reverse=False)

# Iterable: 可迭代对象
# key: 排序规则(排序函数), 在sorted内部会将可迭代对象中的每一个元素传递给这个函数的参数. 根据函数运算的结果进行排序
# reverse: 是否是倒叙. True: 倒叙, False: 正序
lst = [5,7,6,12,1,13,9,18,5]
lst.sort()  # sort是list里面的一个方法
print(lst)  #[1, 5, 5, 6, 7, 9, 12, 13, 18]

ll = sorted(lst) # 内置函数. 返回给你一个新列表  新列表是被排序的
print(ll)  #[1, 5, 5, 6, 7, 9, 12, 13, 18]

l2 = sorted(lst,reverse=True)  #倒序
print(l2)  #[18, 13, 12, 9, 7, 6, 5, 5, 1]
#根据字符串长度给列表排序
lst = ['one', 'two', 'three', 'four', 'five', 'six']
def f(s):
    return len(s)
l1 = sorted(lst, key=f, )
print(l1)  #['one', 'two', 'six', 'four', 'five', 'three']


# enumerate() 获取集合的枚举对象
lst = ['one','two','three','four','five']
for index, el in enumerate(lst,1):    # 把索引和元素一起获取,索引默认从0开始. 可以更改
    print(index)
    print(el)
# 1
# one
# 2
# two
# 3
# three
# 4
# four
# 5
# five


# all() 可迭代对象中全部是True, 结果才是True
# any() 可迭代对象中有一个是True, 结果就是True
print(all([1,'hello',True,9]))  #True
print(any([0,0,0,False,1,'good']))  #True


# zip() 函数用于将可迭代的对象作为参数, 将对象中对应的元素打包成一个元组, 然后返回由这些元组组成的列表. 如果各个迭代器的元素个数不一致, 则返回列表长度与最短的对象相同
lst1 = [1, 2, 3, 4, 5, 6]
lst2 = ['醉乡民谣', '驴得水', '放牛班的春天', '美丽人生', '辩护人', '被嫌弃的松子的一生']
lst3 = ['美国', '中国', '法国', '意大利', '韩国', '日本']
print(zip(lst1, lst1, lst3))  #<zip object at 0x00000256CA6C7A88>
for el in zip(lst1, lst2, lst3):
    print(el)
# (1, '醉乡民谣', '美国')
# (2, '驴得水', '中国')
# (3, '放牛班的春天', '法国')
# (4, '美丽人生', '意大利')
# (5, '辩护人', '韩国')
# (6, '被嫌弃的松子的一生', '日本')


# fiter() 过滤 (lamda)
# 语法：fiter(function. Iterable)

# function: 用来筛选的函数. 在ﬁlter中会自动的把iterable中的元素传递给function. 然后根据function返回的True或者False来判断是否保留留此项数据 , Iterable: 可迭代对象


def func(i):    # 判断奇数
    return i % 2 == 1
lst = [1,2,3,4,5,6,7,8,9]
l1 = filter(func, lst)  #l1是迭代器
print(l1)  #<filter object at 0x000001CE3CA98AC8>
print(list(l1))  #[1, 3, 5, 7, 9]


# map() 会根据提供的函数对指定序列列做映射(lamda)
# 语法 : map(function, iterable)

# 可以对可迭代对象中的每一个元素进行映射. 分别去执行 function

def f(i):    return i
lst = [1,2,3,4,5,6,7,]
it = map(f, lst) # 把可迭代对象中的每一个元素传递给前面的函数进行处理. 处理的结果会返回成迭代器print(list(it))  #[1, 2, 3, 4, 5, 6, 7]


# 和作用域相关
# locals() 返回当前作用域中的名字
# globals() 返回全局作用域中的名字
def func():
    a = 10
    print(locals())  # 当前作用域中的内容
    print(globals())  # 全局作用域中的内容
    print("今天内容很多")
func()
# {'a': 10}
# {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': 
# <_frozen_importlib_external.SourceFileLoader object at 0x0000026F8D566080>, 
# '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' 
# (built-in)>, '__file__': 'D:/pycharm/练习/week03/new14.py', '__cached__': None,
#  'func': <function func at 0x0000026F8D6B97B8>}
# 今天内容很多


# 和迭代器生成器相关
# range() 生成数据
# next() 迭代器向下执行一次, 内部实际使⽤用了__ next__()⽅方法返回迭代器的下一个项目
# iter() 获取迭代器, 内部实际使用的是__ iter__()⽅方法来获取迭代器
for i in range(15,-1,-5):
    print(i)
# 15
# 10
# 5
# 0
lst = [1,2,3,4,5]
it = iter(lst)  #  __iter__()获得迭代器
print(it.__next__())  #1
print(next(it)) #2  __next__()  
print(next(it))  #3
print(next(it))  #4


# 字符串类型代码执行
# eval() 执行字符串类型的代码. 并返回最终结果
# exec() 执行字符串类型的代码
# compile() 将字符串类型的代码编码. 代码对象能够通过exec语句来执行或者eval()进行求值
s1 = input("请输入a+b:")  #输入:8+9
print(eval(s1))  # 17 可以动态的执行代码. 代码必须有返回值
s2 = "for i in range(5): print(i)"
a = exec(s2) # exec 执行代码不返回任何内容

# 0
# 1
# 2
# 3
# 4
print(a)  #None

# 动态执行代码
exec("""
def func():
    print(" 我是周杰伦")
""" )
func()  #我是周杰伦
code1 = "for i in range(3): print(i)"
com = compile(code1, "", mode="exec")   # compile并不会执行你的代码.只是编译
exec(com)   # 执行编译的结果
# 0
# 1
# 2

code2 = "5+6+7"
com2 = compile(code2, "", mode="eval")
print(eval(com2))  # 18

code3 = "name = input('请输入你的名字:')"  #输入:hello
com3 = compile(code3, "", mode="single")
exec(com3)
print(name)  #hello


# 输入输出
# print() : 打印输出
# input() : 获取用户输出的内容
print("hello", "world", sep="*", end="@") # sep:打印出的内容用什么连接,end:以什么为结尾
#hello*world@


# 内存相关
# hash() : 获取到对象的哈希值(int, str, bool, tuple). hash算法:(1) 目的是唯一性 (2) dict 查找效率非常高, hash表.用空间换的时间 比较耗费内存

s = print(hash(s))  #-168324845050430382lst = [1, 2, 3, 4, 5]print(hash(lst))  #报错,列表是不可哈希的  id() :  获取到对象的内存地址s = 'alex'print(id(s))  #2278345368944


# 文件操作相关
# open() : 用于打开一个文件, 创建一个文件句柄
f = open('softmax.py',mode='r',encoding='utf-8')
f.read()
f.close()


# 模块相关
# __ import__() : 用于动态加载类和函数
# 让用户输入一个要导入的模块
import os
name = input("请输入你要导入的模块:")
__import__(name)    # 可以动态导入模块


# 帮 助
# help() : 函数用于查看函数或模块用途的详细说明
print(help(str))  #查看字符串的用途


# 调用相关
# callable() : 用于检查一个对象是否是可调用的. 如果返回True, object有可能调用失败, 但如果返回False. 那调用绝对不会成功
a = 10
print(callable(a))  #False  变量a不能被调用
#
def f():
    print("hello")
    print(callable(f))   # True 函数是可以被调用的
    
    
# 查看内置属性
# dir() : 查看对象的内置属性, 访问的是对象中的__dir__()方法
print(dir(tuple))  #



#============================================
# int() 接下来介绍的是相对少见的用法，int 可以将 2 进制到 36 进制的字符串、字节串（bytes）或者字节数组（bytearray）实例转换成对应的 10 进制整数。
# 具体的调用形式为：int(x, base=10)，其中 x 即为字符串、字节串或字节数组的实例。
# 二进制数字可以用 0b 或者 0B 做前缀，八进制数字可以用 0o 或者 0O 做前缀，十六进制数字可以用 0x 或者 0X 做前缀，前缀是可选的。
#============================================


int()               # 不传入参数时，得到结果0

print( " int(3) = {}".format(int(3)) )

print( " int(3.6) = {}".format(int(3.6)) )

#  如果是带参数base的话，12要以字符串的形式进行输入，12 为 16进制
print( " int('12',16) = {}".format(int('12',16)) ) 

print( " int('0xa',16)   = {}".format(int('0xa',16)  ) )

print( " int('10',8)     = {}".format(int('10',8)   ) )

print( " int('0x0010',base=16)   = {}".format(int('0x0010',base=16)   ) )

x = '6'
num1 = int(x)
num2 = int(x, 10)
print(num1)
print(num2)

x = '10'
num1 = int(x, 2)
num2 = int(x, 8)
num3 = int(x, 16)
print(num1)
print(num2)
print(num3)


# 带正号
x = '+a0'
num = int(x, 16)
print(num)


# 带负号
x = '-a0'
num = int(x, 16)
print(num)

# 两端带空白
x = '  \t  +a0\r\n  '
num = int(x, 16)
print(num)



# base=0 时按照代码字面量直接解析
# base 为 0 的时候会按照代码字面量（code literal）解析，即只能把 2、8、10、16 进制数字表示转换为 10 进制。对于 2、8、16 进制必须指明相应进制的前缀，否则会按照 10 进制解析。

x = '10'
num = int(x, 0)
print(num)

x = '0b10'
num = int(x, 0)
print(num)

x = '0o10'
num = int(x, 0)
print(num)

x = '0x10'
num = int(x, 0)
print(num)


# 如果指定了base参数，那么第一个参数必须是字符类型
# 下面的例子就是指定字符为2进制数据，将其转换为十进制
int('11', base=2)

# 如果不指定，则认为字符是十进制数据
int('101')





"""
描述
getattr() 函数用于返回一个对象属性值。

语法
getattr 语法：

getattr(object, name[, default])
参数
object -- 对象。
name -- 字符串，对象属性。
default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。
返回值
返回对象属性值。
"""
class A(object):
     def set(self, a, b):
          x = a        
          a = b        
          b = x        
          print( a, b   )

a = A()       
c = getattr(a, 'set')
c(a='1', b='2')

getattr(A(), 'set')(a=12,b=23)




"""
Python compile() 函数
Python 内置函数 Python 内置函数

描述
compile() 函数将一个字符串编译为字节代码。

语法
以下是 compile() 方法的语法:

compile(source, filename, mode[, flags[, dont_inherit]])
参数
source -- 字符串或者AST（Abstract Syntax Trees）对象。。
filename -- 代码文件名称，如果不是从文件读取代码则传递一些可辨认的值。
mode -- 指定编译代码的种类。可以指定为 exec, eval, single。
flags -- 变量作用域，局部命名空间，如果被提供，可以是任何映射对象。。
flags和dont_inherit是用来控制编译源码时的标志
返回值
返回表达式执行结果。
"""

str = "for i in range(0,10): print(i)" 
c = compile(str,'','exec')   # 编译为字节代码对象 
print(f"c = {c}")

exec(c)




"""
Python locals() 函数
Python 内置函数 Python 内置函数

描述
locals() 函数会以字典类型返回当前位置的全部局部变量。

对于函数, 方法, lambda 函式, 类, 以及实现了 __call__ 方法的类实例, 它都返回 True。

语法
locals() 函数语法：

locals()
参数
无
返回值
返回字典类型的局部变量。
"""


def runoob(arg):    # 两个局部变量：arg、z
     z = 1
     print (locals())

runoob(4)
#{'z': 1, 'arg': 4}      # 返回一个名字/值对的字典





"""
Python xrange() 函数
Python 内置函数 Python 内置函数

描述
xrange() 函数用法与 range 完全相同，所不同的是生成的不是一个数组，而是一个生成器。

语法
xrange 语法：

xrange(stop)
xrange(start, stop[, step])
参数说明：

start: 计数从 start 开始。默认是从 0 开始。例如 xrange(5) 等价于 xrange(0， 5)
stop: 计数到 stop 结束，但不包括 stop。例如：xrange(0， 5) 是 [0, 1, 2, 3, 4] 没有 5
step：步长，默认为1。例如：xrange(0， 5) 等价于 xrange(0, 5, 1)
返回值
返回生成器。
"""


xrange(8)
#xrange(8)
list(xrange(8))
#[0, 1, 2, 3, 4, 5, 6, 7]
range(8)                 # range 使用
#[0, 1, 2, 3, 4, 5, 6, 7]
xrange(3, 5)
xrange(3, 5)
list(xrange(3,5))
#[3, 4]
range(3,5)               # 使用 range
#[3, 4]
xrange(0,6,2)
xrange(0, 6, 2)              # 步长为 2
list(xrange(0,6,2))
#[0, 2, 4]



"""
Python vars() 函数
Python 内置函数 Python 内置函数

描述
vars() 函数返回对象object的属性和属性值的字典对象。

语法
vars() 函数语法：

vars([object])
参数
object -- 对象
返回值
返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。
"""

class Runoob:
     a = 1
print(vars(Runoob))
# {'a': 1, '__module__': '__main__', '__doc__': None}
runoob = Runoob()
print(vars(runoob))





"""
Python bytearray() 函数
Python 内置函数 Python 内置函数

描述
bytearray() 方法返回一个新字节数组。这个数组里的元素是可变的，并且每个元素的值范围: 0 <= x < 256。

语法
bytearray()方法语法：

class bytearray([source[, encoding[, errors]]])
参数
如果 source 为整数，则返回一个长度为 source 的初始化数组；
如果 source 为字符串，则按照指定的 encoding 将字符串转换为字节序列；
如果 source 为可迭代类型，则元素必须为[0 ,255] 中的整数；
如果 source 为与 buffer 接口一致的对象，则此对象也可以被用于初始化 bytearray。
如果没有输入任何参数，默认就是初始化数组为0个元素。
返回值
返回新字节数组。
"""
bytearray()
# bytearray(b'')

bytearray([1,2,3])
#bytearray(b'\x01\x02\x03')


bytearray('runoob', 'utf-8')
#bytearray(b'runoob')



"""
Python hex() 函数
Python 内置函数 Python 内置函数

描述
hex() 函数用于将10进制整数转换成16进制，以字符串形式表示。

语法
hex 语法：

hex(x)
参数说明：

x -- 10进制整数
返回值
返回16进制数，以字符串形式表示。
"""
hex(255)
# '0xff'
hex(-42)
# '-0x2a'
hex(1L)
# '0x1L'
hex(12)
# '0xc'
type(hex(12))
# <class 'str'>      # 字符串



"""
Python frozenset() 函数
Python 内置函数 Python 内置函数

描述
frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。

语法
frozenset() 函数语法：

class frozenset([iterable])
参数
iterable -- 可迭代的对象，比如列表、字典、元组等等。
返回值
返回新的 frozenset 对象，如果不提供任何参数，默认会生成空集合。
"""

a = frozenset(range(10))     # 生成一个新的不可变集合
a
frozenset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = frozenset('runoob') 
b
#frozenset(['b', 'r', 'u', 'o', 'n'])   # 创建不可变集合





"""
描述
hasattr() 函数用于判断对象是否包含对应的属性。

语法
hasattr 语法：

hasattr(object, name)
参数
object -- 对象。
name -- 字符串，属性名。
返回值
如果对象有该属性返回 True，否则返回 False。

实例
以下实例展示了 hasattr 的使用方法：
"""

class Coordinate:
    x = 10
    y = -5
    z = 0
 
point1 = Coordinate() 
print(hasattr(point1, 'x'))
print(hasattr(point1, 'y'))
print(hasattr(point1, 'z'))
print(hasattr(point1, 'no'))  # 没有该属性




"""
Python int() 函数
Python 内置函数 Python 内置函数

描述
int() 函数用于将一个字符串或数字转换为整型。

语法
以下是 int() 方法的语法:

class int(x, base=10)
参数
x -- 字符串或数字。
base -- 进制数，默认十进制。
返回值
返回整型数据。
"""
int()               # 不传入参数时，得到结果0
#0
int(3)
#3
int(3.6)
#3
int('12',16)        # 如果是带参数base的话，12要以字符串的形式进行输入，12 为 16进制
#18
int('0xa',16)  
#10  
int('10',8)  
#8

#x 有两种：str / int

#1、若 x 为纯数字，则不能有 base 参数，否则报错；其作用为对入参 x 取整
int(3.1415926)
#3

int(-11.123)
#-11

int(2.5,10)
#报错
int(2.5)
#2

#2、若 x 为 str，则 base 可略可有。
#base 存在时，视 x 为 base 类型数字，并将其转换为 10 进制数字。
#若 x 不符合 base 规则，则报错。如:

int("9",2)  #报错，因为2进制无9
int("9")
#9 
#默认10进制
int("3.14",8)
int("1.2")
#均报错，str须为整数
int("1001",2)
#9
# "1001"才是2进制格式，并转化为十进制数字9
int("0xa",16)
#10
# ≥16进制才会允许入参为a,b,c...
int("b",8) #报错
int("123",8)
#83
#视123为8进制数字，对应的10进制为83




"""
Python input() 函数
Python 内置函数 Python 内置函数

1、在 Python2.x 中 raw_input( ) 和 input( )，两个函数都存在，其中区别为:

raw_input( ) 将所有输入作为字符串看待，返回字符串类型。
input( ) 只能接收"数字"的输入，在对待纯数字输入时具有自己的特性，它返回所输入的数字的类型（ int, float ）。
2、在 Python3.x 中 raw_input( ) 和 input( ) 进行了整合，去除了 raw_input( )，仅保留了 input( ) 函数，其接收任意任性输入，将所有输入默认为字符串处理，并返回字符串类型。

函数语法
input([prompt])
参数说明：

prompt: 提示信息
"""

a = input("input:") # #input的输出结果都是作为字符串
#input:123                  # 输入整数
type(a)
#<type 'int'>               # 整型
a = input("input:")    
#input:"runoob"           # 正确，字符串表达式
type(a)
#<type 'str'>             # 字符串
a = input("input:")
#input:runoob               # 报错，不是表达式
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File "<string>", line 1, in <module>
#NameError: name 'runoob' is not defined
#<type 'str'>



"""
Python ord() 函数
Python 内置函数 Python 内置函数

描述
ord() 函数是 chr() 函数（对于8位的ASCII字符串）或 unichr() 函数（对于Unicode对象）的配对函数，它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值，如果所给的 Unicode 字符超出了你的 Python 定义范围，则会引发一个 TypeError 的异常。

语法
以下是 ord() 方法的语法:

ord(c)
参数
c -- 字符。
返回值
返回值是对应的十进制整数。
"""


ord('a')
#97
ord('b')
#98
ord('c')
#99



"""
Python hash() 函数
Python 内置函数 Python 内置函数

描述
hash() 用于获取取一个对象（字符串或者数值等）的哈希值。

语法
hash 语法：

hash(object)
参数说明：

object -- 对象；
返回值
返回对象的哈希值。

实例
以下实例展示了 hash 的使用方法："""


hash('test')            # 字符串
#2314058222102390712
hash(1)                 # 数字
#1
hash(str([1,2,3]))      # 集合
#1335416675971793195
hash(str(sorted({'1':1}))) # 字典
#7666464346782421378


"""
Python oct() 函数
Python 内置函数 Python 内置函数

描述
oct() 函数将一个整数转换成 8 进制字符串。

Python2.x 版本的 8 进制以 0 作为前缀表示。
Python3.x 版本的 8 进制以 0o 作为前缀表示。
语法
oct 语法：

oct(x)
参数说明：

x -- 整数。
"""


oct(10)
#'0o12'
oct(20)
#'0o24'
oct(15)
#'0o17'




"""
Python exec 内置语句
Python 内置函数 Python 内置函数

描述
exec 执行储存在字符串或文件中的Python语句，相比于 eval，exec可以执行更复杂的 Python 代码。

需要说明的是在 Python2 中exec不是函数，而是一个内置语句(statement)，但是Python 2中有一个 execfile() 函数。可以理解为 Python 3 把 exec 这个 statement 和 execfile() 函数的功能够整合到一个新的 exec() 函数中去了。

语法
以下是 exec 的语法:

exec obj
参数
obj -- 要执行的表达式。
返回值
exec 返回值永远为 None。
"""


exec ('print ("Hello World")')
Hello World
# 单行语句字符串
exec "print 'runoob.com'"
runoob.com
 
#  多行语句字符串
exec ("""for i in range(5):
     print ("iter time: %d" % i)
     """)



x = 10
expr = """
z = 30
sum = x + y + z
print(sum)
"""
def func():
    y = 20
    exec(expr)
    exec(expr, {'x': 1, 'y': 2})
    exec(expr, {'x': 1, 'y': 2}, {'y': 3, 'z': 4})
    
func()





"""
https://www.runoob.com/python/python-func-super.html
super()

Python super() 函数
Python 内置函数 Python 内置函数

描述
super() 函数是用于调用父类(超类)的一个方法。

super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。

MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。

语法
以下是 super() 方法的语法:

super(type[, object-or-type])
参数
type -- 类。
object-or-type -- 类，一般是 self

"""



class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print ('Parent')
    
    def bar(self,message):
        print ("%s from Parent" % message)
 
class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），并用FooParent的初始化方法初始化子类FooChild
        super(FooChild,self).__init__()    
        print ('Child')
        
    def bar(self,message):
        super(FooChild, self).bar(message)
        print ('Child bar fuction')
        print (self.parent)

if __name__ == '__main__':
    fooChild = FooChild()
    fooChild.bar('HelloWorld')




"""
Python bin() 函数
Python 内置函数 Python 内置函数

描述
bin() 返回一个整数 int 或者长整数 long int 的二进制表示。

语法
以下是 bin() 方法的语法:

bin(x)
参数
x -- int 或者 long int 数字
返回值
字符串。
"""
bin(10)
# '0b1010'
bin(20)
# '0b10100'


"""
Python Tuple(元组) tuple()方法
Python 元组Python 元组

描述
Python 元组 tuple() 函数将列表转换为元组。

语法
tuple()方法语法：

tuple( iterable )
参数
iterable -- 要转换为元组的可迭代序列。
返回值
返回元组。
"""

tuple([1,2,3,4])
tuple({1:2,3:4})    #针对字典 会返回字典的key组成的tuple
tuple((1,2,3,4))    #元组会返回元组自身



aList = [123, 'xyz', 'zara', 'abc'];
aTuple = tuple(aList)
print ("Tuple elements : ", aTuple)


test_list1 = ('a','b','c')
test_list2 = ['x','y','z']
test_tuple = tuple(test_list2)
# test_list2 可以修改，tuple() 函数不是改变值的类型，而是返回改变类型后的值，原值不会被改变
test_list2[2] = '这是修改的'
#下面这行报错，元组不可修改
# test_list1[2]='这是修改的'
print(test_list1)
print(test_list2)
print(test_tuple)





"""
Python open() 函数
Python 内置函数 Python 内置函数

python open() 函数用于打开一个文件，创建一个 file 对象，相关的方法才可以调用它进行读写。

更多文件操作可参考：Python 文件I/O。

函数语法
open(name[, mode[, buffering]])
参数说明：

name : 一个包含了你要访问的文件名称的字符串值。

mode : mode 决定了打开文件的模式：只读，写入，追加等。所有可取值见如下的完全列表。这个参数是非强制的，默认文件访问模式为只读(r)。

buffering : 如果 buffering 的值被设为 0，就不会有寄存。如果 buffering 的值取 1，访问文件时会寄存行。如果将 buffering 的值设为大于 1 的整数，表明了这就是的寄存区的缓冲大小。如果取负值，寄存区的缓冲大小则为系统默认。

不同模式打开文件的完全列表：

模式	描述
t	文本模式 (默认)。
x	写模式，新建一个文件，如果该文件已存在则会报错。
b	二进制模式。
+	打开一个文件进行更新(可读可写)。
U	通用换行模式（不推荐）。
r	以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
rb	以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等。
r+	打开一个文件用于读写。文件指针将会放在文件的开头。
rb+	以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。一般用于非文本文件如图片等。
w	打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb	以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
w+	打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb+	以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
a	打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+	打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。
file 对象方法
file.read([size])：size 未指定则返回整个文件，如果文件大小 >2 倍内存则有问题，f.read()读到文件尾时返回""(空字串)。

file.readline()：返回一行。

file.readlines([size]) ：返回包含size行的列表, size 未指定则返回全部行。

for line in f: print line ：通过迭代器访问。

f.write("hello\n")：如果要写入字符串以外的数据,先将他转换为字符串。

f.tell()：返回一个整数,表示当前文件指针的位置(就是到文件头的字节数)。

f.seek(偏移量,[起始位置])：用来移动文件指针。

偏移量: 单位为字节，可正可负
起始位置: 0 - 文件头, 默认值; 1 - 当前位置; 2 - 文件尾
f.close() 关闭文件


Python File(文件) 方法
open() 方法
Python open() 方法用于打开一个文件，并返回文件对象，在对文件进行处理过程都需要使用到这个函数，如果该文件无法被打开，会抛出 OSError。

注意：使用 open() 方法一定要保证关闭文件对象，即调用 close() 方法。

open() 函数常用形式是接收两个参数：文件名(file)和模式(mode)。

open(file, mode='r')
完整的语法格式为：

open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
参数说明:

file: 必需，文件路径（相对或者绝对路径）。
mode: 可选，文件打开模式
buffering: 设置缓冲
encoding: 一般使用utf8
errors: 报错级别
newline: 区分换行符
closefd: 传入的file参数类型
opener: 设置自定义开启器，开启器的返回值必须是一个打开的文件描述符。
mode 参数有：

模式	描述
t	文本模式 (默认)。
x	写模式，新建一个文件，如果该文件已存在则会报错。
b	二进制模式。
+	打开一个文件进行更新(可读可写)。
U	通用换行模式（不推荐）。
r	以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。
rb	以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。一般用于非文本文件如图片等。
r+	打开一个文件用于读写。文件指针将会放在文件的开头。
rb+	以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。一般用于非文本文件如图片等。
w	打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb	以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
w+	打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
wb+	以二进制格式打开一个文件用于读写。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
a	打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
ab	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
a+	打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。
ab+	以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。
默认为文本模式，如果要以二进制模式打开，加上 b 。

file 对象
file 对象使用 open 函数来创建，下表列出了 file 对象常用的函数：

序号	方法及描述
1	file.close() 关闭文件。关闭后文件不能再进行读写操作。

2	 file.flush() 刷新文件内部缓冲，直接把内部缓冲区的数据立刻写入文件, 而不是被动的等待输出缓冲区写入。

3	 file.fileno() 返回一个整型的文件描述符(file descriptor FD 整型), 可以用在如os模块的read方法等一些底层操作上。

4	file.isatty() 如果文件连接到一个终端设备返回 True，否则返回 False。

5	file.next() 返回文件下一行。

6	file.read([size]) 从文件读取指定的字节数，如果未给定或为负则读取所有。

7	file.readline([size]) 读取整行，包括 "\n" 字符。

8	file.readlines([sizeint]) 读取所有行并返回列表，若给定sizeint>0，则是设置一次读多少字节，这是为了减轻读取压力。

9	file.seek(offset[, whence]) 设置文件当前位置

10	file.tell() 返回文件当前位置。

11	file.truncate([size]) 截取文件，截取的字节通过size指定，默认为当前文件位置。

12	file.write(str) 将字符串写入文件，返回的是写入的字符长度。

13	file.writelines(sequence) 向文件写入一个序列字符串列表，如果需要换行则要自己加入每行的换行符。

"""
f = open('string.py')
f.read()




"""
Python globals() 函数
Python 内置函数 Python 内置函数

描述
globals() 函数会以字典类型返回当前位置的全部全局变量。

语法
globals() 函数语法：

globals()
参数:无
返回值: 返回全局变量的字典。
"""
a='runoob'
print(globals()) # globals 函数返回一个全局变量的字典，包括所有导入的变量。


"""
os.mkdir()
用法
mkdir(path,mode)
1
参数
path ：要创建的目录的路径（绝对路径或者相对路径）
mode ： Linux 目录权限数字表示，windows 请忽略该参数
权限种类分为三种，分别为 读，写，可执行。
身份为 owners，groups，others 三种。
3 × 3 共有 9种 权限
使用 3 个数字表示3个身份的权限
对于任何一个身份，可读 为 4，可写为 2，可执行为 1，用户具有的权限为相应的权重相加的结果，例如具有可读和可写权限，但是没有执行权限，则数字为 1 + 2=3。
默认是 777
注意： mkdir 只能创建一个目录，不能递归创建目录，例如创建 ./two/three 目录的时候，./two 目录必须存在，否则报错，另外需要注意的是，如果已经存在了目录，则创建目录也会失败！



os.makedirs()
用法
makedirs(path, mode=0o777, exist_ok=False):
1
参数
path : 要创建的目录，绝对路径或者相对路径
mode： 和上面一样，windows 用户请忽略
exist_ok：如果已经存在怎么处理，默认是 False ，即：已经存在程序报错。当为 True 时，创建目录的时候如果已经存在就不报错。

"""




# python 3
import os
path1="./one"
path2="./one/two"
path3="./two/three"

try:
    os.mkdir(path1)
    print("创建"+path1+"成功")
except:
    pass

try:
    os.mkdir(path2)
    print("创建" + path2 + "成功")
except:
    pass

try:
    os.mkdir(path3)
except:
    print("无法创建{0}目录".format(path3))

# 输出：
# 创建./one成功
# 创建./one/two成功
# 无法创建./two/three目录



import os
path1="./one"
path2="./one/two"
path3="./two/three"

try:
    os.makedirs(path1,exist_ok=True)
    print("创建"+path1+"成功，或者目录已经存在")
except:
    pass

try:
    os.makedirs(path2,exist_ok=True)
    print("创建" + path2 + "成功，或者目录已经存在")
except:
    pass

try:
    os.makedirs(path3,exist_ok=True)
    print("可以递归创建{0}目录".format(path3))
except:
    pass

# 输出结果
# 创建./one成功，或者目录已经存在
# 创建./one/two成功，或者目录已经存在
# 可以递归创建./two/three目录


# change directory
import os, sys

path = "/home/jack/tmp"

retval = os.getcwd()
print ( f"当前工作目录为 {retval} ")


os.chdir(path)
retval = os.getcwd()
print ( f"当前工作目录为 {retval} ")









































































































