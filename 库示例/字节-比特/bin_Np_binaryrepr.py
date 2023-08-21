#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:20:21 2023
@author: jack

本文档是记录
(一) 2/8/10/16进制互转: int、bin、hex、oct
    其中int主要是将数字转为整数或者将字符串转为整数，如'1001'转为10进制的整数；而 bin hex oct是将整数转为相应的进制数字符串便于人阅读非10进制的整数;

(二)  怎么将 整数 转为相应的 0-1比特字符串， 以及怎么将 0-1比特串字符串 转为 整数;
(三) numpy.binary_repr(num, width=None)
    返回输入数字的二进制字符串。

"""



###   2/8/10/16 进制赋值
a = 0b1000
print(a)
# 8

a = 0o1100
print(a)
# 576

a = 1110
print(a)
# 1110

a = 0x1111
print(a)
# 4369

a = 0x100
print(a)
# 256


"""
2/8/10/16进制互转:

(一) int()
    描述: int() 函数用于将一个字符串或数字转换为整型。

    语法:以下是 int() 方法的语法:

    class int(x, base=10)
      参数: x -- 字符串或数字。
          base -- 进制数，默认十进制。
      返回值: 返回整型数据。


(二) bin()
    bin() 返回一个整数 int 或者长整数 long int 的二进制表示。
      参数  x: int 或者 long int 数字

    Python 中的整型是补码形式存储的
    Python 中 bin 一个负数（十进制表示），输出的是它的原码的二进制表示加上个负号，方便查看（方便个鬼啊）
    Python 中 bin 一个负数（十六进制表示），输出的是对应的二进制表示。（注意此时）
    所以你为了获得负数（十进制表示）的补码，需要手动将其和十六进制数 0xffffffff 进行按位与操作，得到结果也是个十六进制数，再交给 bin() 进行输出，得到的才是你想要的补码表示。


(三)  oct()
  描述
  oct() 函数将一个整数转换成 8 进制字符串。

  Python2.x 版本的 8 进制以 0 作为前缀表示。
  Python3.x 版本的 8 进制以 0o 作为前缀表示。
  语法
  oct 语法：oct(x)
  参数说明：
    x :10 进制整数
    返回值
    返回 8 进制字符串。


(四)：hex()
    Python hex() 函数:Python 内置函数 Python 内置函数
    描述
    hex() 函数用于将10进制整数转换成16进制，以字符串形式表示。
    hex 语法：hex(x)
    参数说明：
        x :10 进制整数
        返回值:返回16进制数，以字符串形式表示。


(五) ord()
     ord() 函数是 chr() 函数（对于 8 位的 ASCII 字符串）的配对函数，它以一个字符串（Unicode 字符）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值。

    语法以下是 ord() 方法的语法: ord(c)
    参数
      c -- 字符。
      返回值
      返回值是对应的十进制整数。

(六) chr()
  Python3 chr() 函数
  Python3 内置函数Python3 内置函数

  描述：chr() 用一个整数作参数，返回一个对应的字符。

  语法以下是 chr() 方法的语法:chr(i)
  参数：i -- 可以是 10 进制也可以是 16 进制的形式的数字，数字范围为 0 到 1,114,111 (16 进制为0x10FFFF)。
  返回值：返回值是当前整数对应的 ASCII 字符。

"""



import numpy as np

##============================================================================
# (一) int()



a = int()               # 不传入参数时，得到结果0
print(a)
# 0
a = int(3)
print(a)
# 3
a = int(3.6)
print(a)
# 3
a = int('12', 16)        # 如果是带参数base的话，12要以字符串的形式进行输入，12 为 16进制
print(a)
# 18
a = int('0xa', 16)
print(a)
# 10
a = int('10', 8)
print(a)
# 8


# 使用 int() 函数的时候要主要注意一点，如果提供的字符串不能转换成指定进制的数字，那么会报异常，就像下面这样，所以在使用这个函数的时候最好放到 try 语句中。
# val = int('128', 2)

# 如果指定了base参数，那么第一个参数必须是字符类型
# 下面的例子就是指定字符为2进制数据，将其转换为十进制
int('11', base = 2)
# 3

# 如果不指定，则认为字符是十进制数据
int('101')
# 101



# base=0 时按照代码字面量直接解析
# base 为 0 的时候会按照代码字面量（code literal）解析，即只能把 2、8、10、16 进制数字表示转换为 10 进制。对于 2、8、16 进制必须指明相应进制的前缀，否则会按照 10 进制解析。
x = '10'
num = int(x, 0)
print(num)
# 10

x = '0b10'
num = int(x, 0)
print(num)
# 2

x = '0o10'
num = int(x, 0)
print(num)
# 8

x = '0x10'
num = int(x, 0)
print(num)
# 16

# 对于 2、8、16 进制必须指明相应进制的前缀，否则会按照 10 进制解析。提供的字符串不能转换成指定进制的数字，那么会报异常
int('0b10')
# ValueError: invalid literal for int() with base 10: '0b10'


val = int('10')
print(type(val), val)

val = int('0xa', 16)
print(type(val), val)
val = int('a', 16)
print(type(val), val)

val = int('0b1010', 2)
print(type(val), val)
val = int('1010', 2)
print(type(val), val)

val = int('101', 3)
print(type(val), val)

val = int('20', 5)
print(type(val), val)

# 结果均为 <class 'int'> 10


#============================================
# int() 接下来介绍的是相对少见的用法，int 可以将 2 进制到 36 进制的字符串、字节串（bytes）或者字节数组（bytearray）实例转换成对应的 10 进制整数。
# 具体的调用形式为：int(x, base=10)，其中 x 即为字符串、字节串或字节数组的实例。
# 二进制数字可以用 0b 或者 0B 做前缀，八进制数字可以用 0o 或者 0O 做前缀，十六进制数字可以用 0x 或者 0X 做前缀，前缀是可选的。
#============================================



int()               # 不传入参数时，得到结果0
# 0

print( " int(3) = {}".format(int(3)) )
 # int(3) = 3


print( " int(3.6) = {}".format(int(3.6)) )
 # int(3.6) = 3


#  如果是带参数base的话，12要以字符串的形式进行输入，12 为 16进制
print( " int('12',16) = {}".format(int('12',16)) )
 # int('12',16) = 18

print( " int('0xa',16)   = {}".format(int('0xa',16)  ) )
 # int('0xa',16)   = 10

print( " int('10',8)     = {}".format(int('10',8)   ) )
 # int('10',8)     = 8


print( " int('0x0010',base=16)   = {}".format(int('0x0010',base=16)   ) )
 # int('0x0010',base=16)   = 16


x = '6'
num1 = int(x)
num2 = int(x, 10)
print(num1)
print(num2)
# 6
# 6


x = '10'
num1 = int(x, 2)
num2 = int(x, 8)
num3 = int(x, 16)
print(num1)
print(num2)
print(num3)
# 2
# 8
# 16

# 带正号
x = '+a0'
num = int(x, 16)
print(num)
# 160

# 带负号
x = '-a0'
num = int(x, 16)
print(num)
# -160


# 两端带空白
x = '  \t  +a0\r\n  '
num = int(x, 16)
print(num)
# 160


##============================================================================
# (二) bin()
print ('13 和 17 的二进制形式：')
a, b = 13, 17
print (bin(a), bin(b))
print ('\n')


a = bin(-3)
print(a)
# -0b11

a = bin(3)
print(a)
# 0b11

print (np.binary_repr(-3, width = 8))
b = bin(-3 & 0xff)
print(b)
# 11111101
# 0b11111101

c = bin(0xfd)
print(c)
# 0b11111101



##============================================================================
# (三) oct

a = oct(10)
print(a)
# 0o12

a = oct(-10)
print(a)
# -0o12

a = oct(20)
print(a)
# 0o24

a = oct(-20)
print(a)
# -0o24

a = oct(15)
print(a)
# 0o17

##============================================================================
# (四) hex()
a = hex(255)
print(a)
# '0xff'

a = hex(-42)
print(a)
# '-0x2a'
a = hex(12)
print(a)
# '0xc'

a = type(hex(12))
print(a)
# <class 'str'>      # 字符串


##============================================================================
# (五) ord()
a = ord('a')
print(a)

a = ord('€')
print(a)

##============================================================================
# (六) chr()

a = chr(0x30)
print(a)
# '0'
a = chr(97)
print(a)
# 'a'
a = chr(8364)
print(a)
# '€'

##============================================================================
# (七) 综合

## 十六进制转换
a = "FF"
# 十六进制转二进制
b = bin(int(a, 16))[2:]
# 因为上文提过bin函数带有两位数的前缀，故加上[2:]舍去前缀
print(b)      # 11111111
# 十六进制转八进制
b = oct(int(a, 16))[2:]
print(b)      # 377:   3*8**2+7*8+7 = 255
# 十六进制转十进制
b = int(a, 16)
print(b)      # 255


## 二进制转换：
a = "1011100"
# 二进制转八进制
b = oct(int(a, 2))[2:]
# [2:]从第二位数开始读起
print(b)      # 134
# 二进制转十进制
b = int(a, 2)
print(b)      # 92
# 二进制转十六进制
b = hex(int(a, 2))[2:]
print(b)      # 5c

## 十进制转换：
a = "92"
# 十进制转二进制
b = bin(int(a, 10))[2:]
# 加上[2:]从第二位数开始读起
print(b)      # 1011100
# 十进制转八进制
b = oct(int(a, 10))[2:]
print(b)      # 134
# 十进制转十六进制
b = hex(int(a, 10))[2:]
print(b)      # 5c


number = 0.0
print(number, "in hex =", float.hex(number))
# 0.0 in hex = 0x0.0p+0
number = 19.5
print(number, "in hex =", float.hex(number))
# 19.5 in hex = 0x1.3800000000000p+4




#==============================================================================
print ('13 和 17 的位与：')
print (np.bitwise_and(13, 17))
print ( 13 & 17)
print (np.binary_repr(13 & 17, width = 8))
print(bin(13 & 17))
# 13 和 17 的位与：
# 1
# 1
# 00000001
# 0b1

print ('13 和 17 的位或：')
print (np.bitwise_or(13, 17))
print ( 13 | 17)
print (np.binary_repr(13 | 17, width = 8))
print(bin(13 | 17))
# 13 和 17 的位或：
# 29
# 29
# 00011101
# 0b11101



"""
numpy.binary_repr(num, width = None)
    返回输入数字的二进制字符串。
    对于负数,如果没有给出宽度,前面会加上一个减号。如果给了宽度,则返回该数字的二进制补码,与该宽度有关。
    在二进制补码系统中，负数由绝对值的二进制补码表示。这是在计算机上表示有符号整数的最常用方法。N 位二进制补码系统可以表示 \(-2^{N-1}\) 到 \(+2^{N-1}-1\) 范围内的每个整数。
"""

# 对于负数,如果没有给出宽度,前面会加上一个减号
print ('-3 的二进制表示：')
print(np.binary_repr(-3))
# -11
# 如果给了宽度,则返回该数字的二进制补码,与该宽度有关。
print ('-3 的二进制表示：')
print(np.binary_repr(-3, width = 8))
# 11111101

# 如果给了宽度,则返回该数字的二进制补码,与该宽度有关。
print ('-3 的二进制表示：')
print(np.binary_repr(-3, width = 6))
# 111101

print ('13 的二进制表示：')
print (np.binary_repr(13, width = 8))
# 00001101

print ('242 的二进制表示：')
print (np.binary_repr(242, width = 8))
# 11110010



print(np.binary_repr(-128))
# -10000000

print(np.binary_repr(-128, width = 8))
# 10000000

print(np.binary_repr(128))
# 10000000

print(np.binary_repr(128, width = 8))
# 10000000

print(np.binary_repr(127))
# 1111111

print(np.binary_repr(127, width = 8))
# 01111111

print(np.binary_repr(-127))
# -1111111

print(np.binary_repr(-127, width = 8))
# 10000001

print(np.binary_repr(1))
# 1

print(np.binary_repr(1, width = 8))
# 00000001

print(np.binary_repr-(1))
# error

print(np.binary_repr(-1, width = 8))
# 111111111





import numpy as np

print ('将 10 左移两位：')
print (np.left_shift(10, 2))
print ('\n')

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
print ('\n')

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。




print ('将 40 右移两位：')
print (np.right_shift(40,2))
print ('\n')

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
print ('\n')

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。






























































































































































































































































































































































































