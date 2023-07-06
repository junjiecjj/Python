#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:21:48 2023

@author: jack

f = open(sourcefile, mode = 'r', encoding = "utf-8")


(一) 读取的三种方法:
Python 提供了如下 3 种函数, 它们都可以帮我们实现读取文件中数据的操作:
    f.read() 每次读取整个文件, 它通常将读取到底文件内容放到一个字符串变量中, 也就是说 .read() 生成文件内容是一个字符串类型。
    f.readline()每只读取文件的一行, 通常也是读取到的一行内容放到一个字符串变量中, 返回str类型。
    f.readlines()每次按行读取整个文件内容, 将读取到的内容放到一个列表中, 返回list类型。

对于借助 open() 函数, 并以可读模式（包括 r、r+、rb、rb+）打开的文件, 可以调用 read() 函数逐个字节（或者逐个字符）读取文件中的内容。
如果文件是以文本模式（非二进制模式）打开的, 则 read() 函数会逐个字符进行读取; 反之, 如果文件以二进制模式打开, 则 read() 函数会逐个字节进行读取。

read() 函数的基本语法格式如下:
file.read([size])

其中, file 表示已打开的文件对象; size 作为一个可选参数, 用于指定一次最多可读取的字符（字节）个数, 如果省略, 则默认一次性读取所有内容。
注意, 当操作文件结束后, 必须调用 close() 函数手动将打开的文件进行关闭, 这样可以避免程序发生不必要的错误。



(二) 写入的三种方法:
    (1)  file.write(str)将字符串写入文件, 返回的是写入的字符长度。

    (2) file.writelines(sequence)向文件写入一个序列字符串列表, 如果需要换行则要自己加入每行的换行符。
        writelines() 方法是用于将字符串列表写入文件的方法。但是需要注意以下几点:
        writelines() 方法只接受字符串列表作为参数。如果要写入单个字符串, 请使用 write() 方法。
        writelines() 方法不会在字符串之间自动添加换行符, 需要手动将其添加到字符串中。
        writelines() 方法不会在列表的最后添加空行, 如果需要在最后一行添加空行, 请手动添加一个包含换行符的空字符串。
        在使用 writelines() 方法时, 需要保证传递的参数是一个字符串列表。如果参数是一个生成器对象, 需要将其转换为列表再传递。


    (3) print() 函数
    可以使用 print() 函数向文件写入内容, 需要指定 file 参数为打开的文件对象。例如:
        with open('example.txt', 'w') as f:
            print('Hello, world!', file=f)

        以下是 print() 函数的常用参数及其详细介绍:
        print() 函数是 Python 中用于打印输出信息到终端的内置函数。print() 函数可以接受多个参数, 并将它们打印输出到终端。
        以下是 print() 函数的常用参数及其详细介绍:
        print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
        *objects: 一个或多个要打印输出的对象, 可以是字符串、数字、变量等。可以接受任意数量的参数。
        sep: 用于分隔多个参数的字符, 默认是一个空格。在打印输出多个参数时, sep 参数将作为它们之间的分隔符。
        end: 用于表示打印输出结束的字符, 默认是一个换行符。在打印输出最后一个参数之后, end 参数将作为它们之后的字符。
        file: 用于指定输出的文件对象, 默认是标准输出设备 sys.stdout。可以将输出重定向到文件中, 以便将输出保存到文件中而不是终端。
        flush: 用于指定是否立即刷新缓冲区, 默认为 False。如果将 flush 参数设置为 True, 则输出将立即写入文件, 而不是等待缓冲区满了再写入。

"""


sourcefile = '/home/jack/公共的/Python/库示例/文件IO/runoob.txt'
mode = 'r'

#=================================================================================================================
#                                              (1) read
#=================================================================================================================
"""
read()直接读取字节到字符串中, 包括了换行符

特点是: 读取整个文件, 将文件内容放到一个字符串变量中。
劣势是: 如果文件非常大, 尤其是大于内存时, 无法使用read()方法。

"""
#======================================================================================

#以 utf-8 的编码格式打开指定文件
f = open(sourcefile,  encoding = "utf-8")
#输出读取到的数据
con = f.read()
print(con)
# 1:www.runoob.com
# 2:www.runoob.com
# 3:www.runoob.com
# 4:www.runoob.com
# 5:www.runoob.com

# jack
# dick
# hahahaha
# 傻逼
# 傻缺
# ！！！！
#关闭文件
f.close()

#======================================================================================

#以二进制形式打开指定文件
f = open(sourcefile, 'rb+')
#输出读取到的数据
print(f.read())
# b'1:www.runoob.com\n2:www.runoob.com\n3:www.runoob.com\n4:www.runoob.com\n5:www.runoob.com\n\njack\ndick\nhahahaha\n\xe5\x82\xbb\xe9\x80\xbc\n\xe5\x82\xbb\xe7\xbc\xba\n\xef\xbc\x81\xef\xbc\x81\xef\xbc\x81\xef\xbc\x81\n'

#关闭文件
f.close()

#======================================================================================

#以二进制形式打开指定文件, 该文件编码格式为 utf-8
f = open(sourcefile, 'rb+')
byt = f.read()
print(f"\n转换前:  = {byt}")

after  =byt.decode('utf-8')
print(f"\n转换后: {after}")
#关闭文件
f.close()

# 转换前:  = b'1:www.runoob.com\n2:www.runoob.com\n3:www.runoob.com\n4:www.runoob.com\n5:www.runoob.com\n\njack\ndick\nhahahaha\n\xe5\x82\xbb\xe9\x80\xbc\n\xe5\x82\xbb\xe7\xbc\xba\n\xef\xbc\x81\xef\xbc\x81\xef\xbc\x81\xef\xbc\x81\n'

# 转换后: 1:www.runoob.com
# 2:www.runoob.com
# 3:www.runoob.com
# 4:www.runoob.com
# 5:www.runoob.com

# jack
# dick
# hahahaha
# 傻逼
# 傻缺
# ！！！！
#=================================================================================================================
#                                             (2)   readline
#=================================================================================================================
"""
　readline()  读取整行, 包括行结束符, 并作为字符串返回

特点: readline()方法每次读取一行; 返回的是一个字符串对象, 保持当前行的内存
缺点: 比readlines慢得多

"""

sourcefile = '/home/jack/公共的/Python/库示例/文件IO/runoob.txt'
mode = 'r'

#========================================================================================


with open(sourcefile, mode) as f:
    # 读取一行数据,
    byt = f.readline()
    print(byt)
    # 1:www.runoob.com

#========================================================================================

with open(sourcefile, mode) as f:
    # 读取一行数据,不仅如此, 在逐行读取时, 还可以限制最多可以读取的字符（字节）数, 例如:
    byt = f.readline(3)
    print(byt)
    # 1:w


#========================================================================================

f = open(sourcefile, mode)
line = f.readline()
while line:
    print(line, end="")
    line = f.readline()
f.close()

#========================================================================================

try:
    f = open(sourcefile, mode)
    while True:
        text_line = f.readline()
        if text_line:
            print(f"{type(text_line)}, {text_line}")
        else:
            break
finally:
    f.close()

# <class 'str'>, 1:www.runoob.com

# <class 'str'>, 2:www.runoob.com

# <class 'str'>, 3:www.runoob.com

# <class 'str'>, 4:www.runoob.com

# <class 'str'>, 5:www.runoob.com

# <class 'str'>,

# <class 'str'>, jack

# <class 'str'>, dick

# <class 'str'>, hahahaha

# <class 'str'>, 傻逼

# <class 'str'>, 傻缺

# <class 'str'>, ！！！！



#=================================================================================================================
#                                           (3)  readlines
#=================================================================================================================

"""
特点: 一次性读取整个文件; 自动将文件内容分析成一个行的列表。
　readlines()读取所有行然后把它们作为一个字符串列表返回。

注意: 三种方法都是直接读取字节到字符串中, 包括换行符\n。

"""



sourcefile = '/home/jack/公共的/Python/库示例/生成器迭代器/runoob.txt'
mode = 'r'
f = open(sourcefile,  encoding = "utf-8")
# 读取一行数据,不仅如此, 在逐行读取时, 还可以限制最多可以读取的字符（字节）数, 例如:
byt = f.readlines( )
print(byt)
#['1:www.runoob.com\n', '2:www.runoob.com\n', '3:www.runoob.com\n', '4:www.runoob.com\n', '5:www.runoob.com\n', '\n', 'jack\n', 'dick\n', 'hahahaha\n', '傻逼\n', '傻缺\n', '！！！！\n']

f.close()

for line in byt:                          #依次读取每行
    line = line.strip()                             #去掉每行头尾空白
    print(f"读取的数据为: {line}")




#=================================================================================================================
#                                           (1)  write
#=================================================================================================================


with open('data.txt','w') as f:    #设置文件对象
    f.write(str)                 #将字符串写入文件中


#====================================================


f = open('1.txt', 'w')
f.write("abcd")
f.close() # 当文件结束使用后记住需要关闭文件

f = open('1.txt', 'r')
print(f.read())
f.close


#=================================================================================================================
#                                           (1)  writelines
#=================================================================================================================

# 单层列表

data = ['a','b','c']
#单层列表写入文件
with open("data.txt", "w") as f:
    f.writelines(data)

#双层列表写入文件
#第一种方法，每一项用空格隔开，一个列表是一行写入文件
data =[['a','b','c'],['a','b','c'],['a','b','c']]
with open("data.txt", "w") as f:                                                   #设置文件对象
    for i in data:                                                                 #对于双层列表中的数据
        i = str(i).strip('[').strip(']').replace(',','').replace('\'','')+'\n'     #将其中每一个列表规范化成字符串
        f.write(i)                                                                 #写入文件

#第二种方法，直接将每一项都写入文件
data =[ ['a','b','c'],['a','b','c'],['a','b','c']]
with open("data.txt", "w") as f:                                                   #设置文件对象
    for i in data:                                                                 #对于双层列表中的数据
        f.writelines(i)                                                            #写入文件


#====================================================


f = open('1.txt', 'w')
f.writelines("ab\ncd")
f.close() # 当文件结束使用后记住需要关闭文件

f = open('1.txt', 'r')
print(f.read())
f.close


#=================================================================================================================
#                                           (3)  print
#=================================================================================================================

# 将输出重定向到文件
a = 12.32434
with open('output.txt', 'w') as f:
    print(f"Hello World {a:>10.2f}", file=f, flush=True)
    # 立即刷新缓冲区
    # print("Hello World", flush=True)














