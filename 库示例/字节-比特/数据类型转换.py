#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:50:04 2023

@author: jack
数据类型转换是个很基础的操作，很多语言中都要做这些转换；

本文档是记录 Python 中常用的数据类型转换，注意，这里的类型转换是指将：
(一)  int <----> bytes 相互转换， 这里的 bytes 是一个新的数据结构：字节序列(字节数据类型)，指计算机中的字节序列/字节串，这与 bin() hex() oct() 不同， bin() hex() oct()是将整数转为相应进制的字符串便于人阅读；而这里的函数都是将整数转化为计算机中存储的字节；
    分析一个网络数据包/二进制文件，基本进行的操作就是将bin十六进制转int、转byte、转str；而相反，构造一个网络数据包/二进制文件，要做的就是将int、将byte、将str转bin十六进制。

(二)  字符串和byte互转



"""
# https://zhuanlan.zhihu.com/p/354060745
# https://blog.csdn.net/albertsh/article/details/114465098

# https://blog.csdn.net/Gordennizaicunzai/article/details/107853981

# https://blog.csdn.net/Clovera/article/details/79293108

# https://www.delftstack.com/zh/howto/python/how-to-convert-int-to-bytes-in-python-2-and-python-3/

# https://www.delftstack.com/zh/howto/python/how-to-convert-bytes-to-integers/

# https://zhuanlan.zhihu.com/p/621519809

import binascii
import struct
def example(express, result=None):
    if result == None:
        result = eval(express)
    print(express, ' ==> ', result)

"""
如何将整型 int 转换为字节 bytes? 有三种方法将 int 转为 bytes：
(一) ：bytes 函数返回一个新的 bytes 对象，该对象是一个 0 <= x < 256 区间内的整数不可变序列。它是 bytearray 的不可变版本。
    使用 bytes 轻松地将整数 0~255 转换为字节数据类型。
    bytes 相比 str 而言更接近底层数据，也更接近存储的形式，它其实是一个字节的数组，类似于 C 语言中的 char []，每个字节是一个范围在 0-255 的数字。
    https://blog.csdn.net/albertsh/article/details/114465098


(二) int.to_bytes函数
    功能：是int.from_bytes的逆过程，把十进制整数，转换为bytes类型的格式。
    x.to_bytes(1,byteorder='little', signed=False)
    第一个参数，表示转换之后字节的个数
    第二个参数，表示大小端;
    第三个参数，表示这个int类型的整数是有符号的还是无符号整数。
    第一个参数是转换后的字节数据长度，第二个参数 byteorder 将字节顺序定义为 little 或 big-endian，可选参数 signed 确定是否使用二进制补码来表示整数。

    int.to_bytes(length, byteorder, *, signed=False)
    length: 用几个字节来表示数据，示例中的1024用“2”字节表示即可，如果数字大小超过字节可表示的范围，会报OverflowError异常
    byteorder: 选择用哪种编码，“big”或“little”
    signed: 编码的数据是否有符号（正/负），默认False表示无符号，如果待编码数据为负数，需要将参数值改为True


(三) struct.pack(fmt, num) 函数把数字转化成bytes
    struct.pack 函数中的第一个参数是格式字符串，它指定了字节格式比如长度，字节顺序(little/big endian)等。

    字符	字节顺序	大小	对齐方式
    @	按原字节	按原字节	按原字节
    =	按原字节	标准	无
    <	小端	标准	无
    >	大端	标准	无
    !	网络（=	大端）	标准

    格式	C类型	Python类型	标准大小
    x	填充字节	无
    c	char	长度为1的字节串	1
    b	signedchar	整数	1
    B	unsignedchar	整数	1
    ?	_Bool	bool	1
    h	short	整数	2
    H	unsignedshort	整数	2
    i	int	整数	4
    I	unsignedint	整数	4
    l	ong	整数	4
    L	unsignedlong	整数	4
    q	longlong	整数	8
    Q	unsignedlonglong	整数	8
    n	ssize_t	整数
    N	size_t	整数
    e	 浮点数	2
    f	float	浮点数4
    d	double	浮点数	8
    s	char[]	字节串
    p	char[]	字节串
    P	void*	整数

"""


import struct

# 四、网络数据包/二进制文件中的各种互转
# 分析一个网络数据包/二进制文件，基本进行的操作就是将bin十六进制转int、转byte、转str；而相反，构造一个网络数据包/二进制文件，要做的就是将int、将byte、将str转bin十六进制。
###  (一) bytes 和 int 相互转换
##================================================================
##  int -> bytes: 整数转 bytes/字节序列/字节串
##================================================================

##======================================
## (1) 使用 bytes() 函数把int数组转成bytes/字节序列/字节串, 将整数 0~255 转换为 字节数据类型(bytes/字节序列/字节串)
##======================================
# 整数需要放在括号 [] 中，否则你得到将是占用该整数个字节位置的空字节，而不是相应的字节本身。
val = bytes([2])
print(type(val), val)
# <class 'bytes'> b'\x02'
val = bytes(2)
print(type(val), val)
# <class 'bytes'> b'\x00\x00'


# 参考上面的生成的数组，可以通过数组生成相同的结果
# https://blog.csdn.net/albertsh/article/details/114465098
num_array = [57, 18, 0, 0]
val = bytes(num_array)
print(type(val), val)
for v in val:
    print(v)
# <class 'bytes'> b'9\x12\x00\x00'
# 57
# 18
# 0
# 0

num_array = [57, 18, 0, 1]
val = bytes(num_array)
print(type(val), val)
#<class 'bytes'> b'9\x12\x00\x01'


num_array = [57, 18, 0, 1]
val = bytes(num_array)
print(type(val), val)
#<class 'bytes'> b'9\x12\x00\x01'

a = bytes([1,2,3,4])
print(type(a), a)
for v in a:
    print(v)
# <class 'bytes'> b'\x01\x02\x03\x04'
# 1
# 2
# 3
# 4

import numpy as np
a = bytes(np.array([1,2,3,4], dtype = np.int32))
print(a)
# b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00'
print(type(a))
# <class 'bytes'>


import numpy as np
a = bytes(np.array([-1,2,3,4], dtype = np.int32))
print(a)
# b'\xff\xff\xff\xff\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00'
print(type(a))
# <class 'bytes'>


a = bytes('hello','ascii')
print(a)
# b'hello'
print(type(a))
# <class 'bytes'>


# bytes 是 Python 3 中的内置数据类型，因此，你可以使用 bytes 关键字直接来定义字节。
a = bytes(18)
print(a)
# b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
type(a)
# <class 'bytes'>

a = b'\x01\x21\31\41'
type(a)
# <class 'bytes'>
# a[0]
# Out[281]: 1

# a[1]
# Out[282]: 33 = 2x16 + 1

# a[2]
# Out[283]: 25 = 3x8 + 1

# a[3]
# Out[284]: 33 = 4x8 + 1

a = bytes((1,2,ord('1'),ord('2'))) # 可迭代类型，元素是数字
print(a)
# b'\x01\x0212'

##======================================
## (2)  使用 to_bytes() 转换成定长 bytes/字节序列/字节串
##======================================

num = 4665
a = num.to_bytes(length = 4, byteorder = 'little', signed = False)
print(type(a), a)
for v in a:
    print(v)
# <class 'bytes'> b'9\x12\x00\x00'
# 57
# 18
# 0
# 0

# 4665 = 18 * 256 + 57，我们发现两个字节就能存储这个数字，一个18，一个57，要想组成4个字节的数组需要补充两个空位，也就是补充两个0，这时就涉及到一个排列顺序，是 [0,0,18,57] 还是 [57, 18, 0, 0] 呢，这就是函数参数中的字节序 byteorder，little 表示小端，big 表示大端，输出的时候都是默认从低地址向高地址进行的.这里选择的小端 [57, 18, 0, 0] 的排列。
# 看到这里可能会迷糊，好像和结果不一样啊，其实这只是一个表示问题，57 的 ASCII 码对应这个字符 ‘9’，18 表示成16进制就是 ‘0x12’，这里写成 b’9\x12\x00\x00’ 只是便于识别而已，实际上内存存储的就是 [57, 18, 0, 0] 这一串数字对应的二进制编码 ‘00111001 00010010 00000000 00000000’。

num = 4665
a = num.to_bytes(length = 4, byteorder = 'big', signed = False)
print(type(a), a)
for v in a:
    print(v)
# <class 'bytes'> b'\x00\x00\x129'
# 0
# 0
# 18
# 57
##===============
Val = 3
a = Val.to_bytes(length=1, byteorder='little', signed = False)
print(a)
a = Val.to_bytes(length=2, byteorder='little', signed = False)
print(a)
a = Val.to_bytes(length=3, byteorder='little', signed = False)
print(a)
a = Val.to_bytes(length=4, byteorder='little', signed = False)
print(a)
for v in a:
    print(v)
# b'\x03'
# b'\x03\x00'
# b'\x03\x00\x00'
# b'\x03\x00\x00\x00'
# 3
# 0
# 0
# 0
# 从上图的输出也可以看出，使用的是小端序，低位优先。输出的时候都是默认从低地址向高地址进行的，所以低位存储在低地址，首先被输出，高位存储在高地址，最后被输出。

a = 3
val = a.to_bytes(length=1, byteorder='big', signed = False)
print(val)
val = a.to_bytes(length=2, byteorder='big', signed = False)
print(val)
val = a.to_bytes(length=3, byteorder='big', signed = False)
print(val)
val = a.to_bytes(length=4, byteorder='big', signed = False)
print(val)
for v in val:
    print(v)
# b'\x03'
# b'\x00\x03'
# b'\x00\x00\x03'
# b'\x00\x00\x00\x03'
# 0
# 0
# 0
# 3
# 从上图可以看出，采用的是大端序。输出的时候都是默认从低地址向高地址进行的，所以高位存储在低地址，首先被输出，低位存储在高地址，最后被输出。


# 第三个参数，表示这个int类型的整数是有符号的还是无符号整数。
# 例如采用1个字节表示的数字，如果是有符号，那么最大的正数是127，超过了就认定是负数，这个时候处理就会报错，如下图：

a = 134
val = a.to_bytes(length=1, byteorder='big', signed = True)
print(val)

# 如果采用无符号数，就不会报错：
a = 134
val = a.to_bytes(length=1, byteorder='big', signed = False)
print(val)
# b'\x86'
#===============

##======================================
### (3)  使用 struct.pack() 函数把数字转化成 bytes/字节序列/字节串
##======================================

num = 4665
val = struct.pack("<I", num)
print(type(val), val)
for v in val:
    print(v)
# <class 'bytes'> b'9\x12\x00\x00'
# 57
# 18
# 0
# 0
# 这里的 "<I" 表示将一个整数转化成小端字节序的4字节数组，其他的类型还有：
# 参数	含义
# >	大端序
# <	小端序
# B	uint8类型
# b	int8类型
# H	uint16类型
# h	int16类型
# I	uint32类型
# i	int32类型
# L	uint64类型
# l	int64类型
# s	ascii码，s前带数字表示个数

val = struct.pack('<HH', 1,2)
print(type(val), val)
# <class 'bytes'> b'\x01\x00\x02\x00'

val = struct.pack('<LL', 1,2)
print(type(val), val)
# <class 'bytes'> b'\x01\x00\x00\x00\x02\x00\x00\x00'

struct.pack('>L', 0x12345678)
# b'\x124Vx

struct.pack('<L', 0x12345678)
# b'xV4\x12'

struct.pack('9si2s',b'HTTP/1.1 ',200,b'OK')
# b'HTTP/1.1 \x00\x00\x00\xc8\x00\x00\x00OK'




import timeit
timeit.timeit('bytes([255])', number=1000000)
# 0.31296982169325455
timeit.timeit('struct.pack("B", 255)', setup='import struct', number=1000000)
# 0.2640925447800839
timeit.timeit('(255).to_bytes(1, byteorder="little")', number=1000000)
# 0.5622947660224895




"""
如何将字节 bytes 转换为整数 int ?
(一) res = int.from_bytes(x)的含义是把bytes类型的变量x，转化为十进制整数，并存入res中。其中bytes类型是python3特有的类型。
     函数参数：int.from_bytes(bytes, byteorder, *, signed=False)。在IDLE或者命令行界面中使用help(int.from_bytes)命令可以查看具体介绍。bytes是输入的变量；byteorder主要有两种：'big'和'little'；signed=True表示需要考虑符号位。

(二) struct.unpack(fmt, string)
    Python 内部模块 struct 可以将二进制数据（字节）转换为整数。它可以双向转换 Python 2.7 中的字节（实际上是字符串）和整数。


"""

import struct

##================================================================
##  bytes -> int: bytes/字节序列/字节串 转整数
##================================================================
## 使用 int.from_bytes() 把 bytes 转化成int. 输出的时候都是默认从低地址向高地址进行的,也就是下面的9是最低地址，\x12其次...，而小端表示:现在bys是以小端的形式呈现的，所以9是低序字节，\x12其次;
bys = b'9\x12\x00\x00'
val = int.from_bytes(bys, byteorder='little', signed=False)
print(type(val), val)
#<class 'int'> 4665

# 输出的时候都是默认从低地址向高地址进行的,也就是下面的9是最低地址，\x12其次...，而大端表示:现在bys是以小端的形式呈现的，所以9是高序字节，\x12其次;
bys = b'9\x12\x00\x00'
val = int.from_bytes(bys, byteorder='big', signed=False)
print(type(val), val)
# <class 'int'> 957480960

testBytes = b'\xF1\x10'
int.from_bytes(testBytes, byteorder='big')
# 61712
# 字节表示将会被转换为一个整数。int.from_bytes() 有第三个选项 signed，它将整数类型赋值为 signed 或 unsigned。
testBytes = b'\xF1\x10'
int.from_bytes(testBytes, byteorder='big', signed=True)
# -3824

# 当字节是 unsigned chart 时使用 []
# 如果数据格式是 unsigned char 且只包含一个字节，则可以直接使用对象索引进行访问来获取该整数。
testBytes = b'\xF1\x10'
testBytes[0]  # 241
testBytes[1]  # 16

## (2) 使用 struct.unpack() 把 bytes 转化成int
bys = b'9\x12\x00\x00'
val = struct.unpack("<I", bys)
print(type(val), val)
#<class 'tuple'> (4665,)

import struct
testBytes = b'\x00\x01\x00\x02'
testResult = struct.unpack('>HH', testBytes)
print(testResult)
# (1, 2)

# 格式字符串 >HH 包含两部分，
# > 表示二进制数据是 big-endian，或换句话说，数据是从高端（最高有效位）排序的。例如，\x00\0x1 中，\x00 是高字节，\x01 是低字节。
# HH 表示字节字符串中有两个 H 类型的对象。H 表示它是 unsigned short 整数，占用了 2 个字节。
# 如果分配的数据格式不同，你可以从相同的字符串中得到不同的结果。
testResult = struct.unpack('<HH', testBytes)
print(testResult)
# (256, 512)

# 这里，< 表示字节顺序是 little-endian。因此\x00\x01 变成 00+1*256 = 256 不是 0*256+1 = 1 了。
testBytes = b'\x00\x01\x00\x02'
testResult = struct.unpack('<BBBB', testBytes)
print(testResult)
# (0, 1, 0, 2)

# 友情提示
# 格式字符串中指定的数据长度应与给定数据相同，否则将会报错。
testResult = struct.unpack('<BBB', b'\x00\x01\x00\x02')



###  (二) bytes 和 str 相互转换/  字符串和byte互转
##================================================================
##  str 和 bytes
##================================================================
## 使用 encode() 函数完成 str -> bytes
s = '大漠孤烟直qaq'
val = s.encode('utf-8')
print(type(val), val)

# <class 'bytes'> b'\xe5\xa4\xa7\xe6\xbc\xa0\xe5\xad\xa4\xe7\x83\x9f\xe7\x9b\xb4qaq'


## 使用 decode() 函数完成 bytes -> str
bys = b'\xe5\xa4\xa7\xe6\xbc\xa0\xe5\xad\xa4\xe7\x83\x9f\xe7\x9b\xb4qaq'
val = bys.decode('utf-8')
print(type(val), val)

# <class 'str'> 大漠孤烟直qaq




# bytes() 把字符串转化成bytes类型
bs = bytes("今天吃饭了吗", encoding="utf-8")
print(bs)  #b'\xe4\xbb\x8a\xe5\xa4\xa9\xe5\x90\x83\xe9\xa5\xad\xe4\xba\x86\xe5\x90\x97'
#   bytearray()    返回一个新字节数组. 这个数字的元素是可变的, 并且每个元素的值得范围是[0,256)


# https://zhuanlan.zhihu.com/p/354060745
##================================================================
## 2.1 bin十六进制与int互转实现
##================================================================
# bin十六进制转int主要在分析二进制文件、数据包头时获取长度等值时使用；相反，int转bin十六进制就是在构造二进制文件、数据包头时写入长度等值时使用。
# 另外注意把bin十六进制当数值时有大端和小端两种模式，大端意思是开头（低地址）权重大，小端为开头（低地址）权重小。文件系统一般用小端模式，网络传输一般用大端模式。

# int转bin十六进制----num_var.to_bytes(lenght,byteorder)，lenght表示转成的多少个字节；byteorder可为big或little分别表示转bin十六进制时使用大端模式还是小端模式。
# bin十六进制转int----int.from_bytes(byte_var,byteorder)，byte_var是要转成数值的变bin十六进制变量，byteorder还是一样可为big或little，分别表示从bin十六进制转为数值时把bin十六进制当大端模式还是小端模式处理。
a = int.from_bytes(b'\x01\x79', 'big')
print(a)
# 377
a = 377
a.to_bytes(2, 'big')
# b'\x01y'
# b’\x01y’其实就是b’\x01\x79’（y的ascii编码是十六进制的79）


##================================================================
## 2.2 bin十六进制和byte互转实现
##================================================================

# bin十六进制转byte主要在分析二进制文件、数据包头时获取mac地址、密钥等平时就以十六进制表示的值时使用；相反，byte转bin十六进制就是在构造二进制文件、数据包头时写入mac地址、密钥等平时就以十六进制表示的值时使用。

# bin十六进制转byte----binascii.b2a_hex(bin_var)，bin_var为byte变量常从二进制文件中读出； 如binascii.b2a_hex(b’\x04\xf9\x38\xad\x13\x26’)结果为b’04f9381326‘
# byte转bin十六进制----binascii.a2b_hex(hex_byte_var)，hex_byte_var为十六进制字节串； 如binascii.a2b_hex(b’04f9381326’)结果为b’\x04\xf98\x13&’（8对应的ascii编码是38，&对应的ascii编码是26）

import binascii

a = binascii.b2a_hex(b'\x04\xf9\x38\xad\x13\x26')
print(a)
# b'04f938ad1326'

a = binascii.a2b_hex(b'04f938ad1326')
print(a)
# b'\x04\xf98\xad\x13&'

a = binascii.hexlify(b'\x04\xf9\x38\xad\x13\x26')
print(a)
# b'04f938ad1326'

a = binascii.unhexlify(b'04f938ad1326')
print(a)
# b'\x04\xf98\xad\x13&'

 # b2a_hex 与 hexlify 功能一致； a2b_hex 与 unhexlify 一致。
 # b2a_hex 与 a2b_hex 相反； hexlify 与 unhexlify 相反。

##================================================================
## 2.3 bin十六进制与str互转
##================================================================

# bin十六进制转主要在分析二进制文件、数据包头时获取其量的字符串时使用；相反，byte转bin十六进制就是在构造二进制文件、数据包头时写入字符串时使用。

# bin十六进制与str互转其实就是上边第三大点中的字符串和byte互转；此处的bin十六进制就是上边第三大点中的byte的本质。（b’\x48\x54\x54\x50’和b’HTTP’在内存中是一模一样的）


var = b'\x48\x54\x54\x50'
var.decode('utf-8')
# 'HTTP'
var = 'HTTP'
var.encode('utf-8')
 # b'HTTP'

##================================================================
## https://zhuanlan.zhihu.com/p/195204018
##================================================================


print('整数之间的进制转换:')
print("10进制转16进制", end=': ');example("hex(16)")
print("16进制转10进制", end=': ');example("int('0x10', 16)")
print("类似的还有oct()， bin()")

print('\n-------------------\n')

print('字符串转整数:')
print("10进制字符串", end=": ");example("int('10')")
print("16进制字符串", end=": ");example("int('10', 16)")
print("16进制字符串", end=": ");example("int('0x10', 16)")

print('\n-------------------\n')

print('字节串转整数:')
print("转义为short型整数", end=": ");example(r"struct.unpack('<hh', bytes(b'\x01\x00\x00\x00'))")
print("转义为long型整数", end=": ");example(r"struct.unpack('<L', bytes(b'\x01\x00\x00\x00'))")

print('\n-------------------\n')

print('整数转字节串:')
print("转为两个字节", end=": ");example("struct.pack('<HH', 1,2)")
print("转为四个字节", end=": ");example("struct.pack('<LL', 1,2)")

print('\n-------------------\n')

print('字符串转字节串:')
print('字符串编码为字节码', end=": ");example(r"'12abc'.encode('ascii')")
print('数字或字符数组', end=": ");example(r"bytes([1,2, ord('1'),ord('2')])")
print('16进制字符串', end=': ');example(r"bytes().fromhex('010210')")
print('16进制字符串', end=': ');example(r"bytes(map(ord, '\x01\x02\x31\x32'))")
print('16进制数组', end =': ');example(r'bytes([0x01,0x02,0x31,0x32])')

print('\n-------------------\n')

print('字节串转字符串:')
print('字节码解码为字符串', end=": ");example(r"bytes(b'\x31\x32\x61\x62').decode('ascii')")
print('字节串转16进制表示,夹带ascii', end=": ");example(r"str(bytes(b'\x01\x0212'))[2:-1]")
print('字节串转16进制表示,固定两个字符表示', end=": ");example(r"str(binascii.b2a_hex(b'\x01\x0212'))[2:-1]")
print('字节串转16进制数组', end=": ");example(r"[hex(x) for x in bytes(b'\x01\x0212')]")







##================================================================
## 2.1 bin十六进制与int互转实现
##================================================================













##================================================================
## 2.1 bin十六进制与int互转实现
##================================================================




























































































































































