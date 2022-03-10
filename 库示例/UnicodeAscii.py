#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:39:03 2022

@author: jack
"""

import unicodedata



s = u"Marek Čech"   #(u表示是unicode而非 ascii码，不加报错！)
line = unicodedata.normalize('NFKD',s).encode('ascii','ignore')
print (line)





"""
Python 模块 unicodedata 提供了一种利用 Unicode 和实用程序功能中的字符数据库的方法，这些功能大大简化了对这些字符的访问、过滤和查找。

unicodedata 具有一个名为 normalize()的函数，该函数接受两个参数，即 Unicode 字符串的规范化形式和给定的字符串。

规范化的 Unicode 格式有 4 种类型：NFC，NFKC，NFD 和 NFKD。要了解更多信息，可以使用官方文档来详细了解每种类型。本教程将全程使用 NFKD 规范化形式。
让我们声明一个包含多个 unicode 字符的字符串。
"""
stringVal = u'Här är ett exempel på en svensk mening att ge dig.'

print(unicodedata.normalize('NFKD', stringVal).encode('ascii', 'ignore'))
#调用 normalize() 方法后，将调用链接到函数 encode()，该函数将完成从 Unicode 到 ASCII 的转换。
#字符串值之前的 u 字符可帮助 Python 识别字符串值包含 unicode 字符；这样做是出于类型安全的目的。
#第一个参数指定转换类型，第二个参数强制执行字符无法转换时应执行的操作。在这种情况下，第二个参数传递 ignore，它将忽略任何无法转换的字符。
#输出：b'Har ar ett exempel pa en svensk mening att ge dig.'


#请注意，原始字符串（ä和å）中的 unicode 字符已被其 ASCII 字符对等体（a）取代。
#字符串开头的 b 符号表示该字符串是字节文字，因为在字符串上使用了 encode() 函数。要删除符号和封装字符串的单引号，请在调用 encode() 之后将其链式调用 decode()，以将其重新转换为字符串文字。
print(unicodedata.normalize('NFKD', stringVal).encode('ascii', 'ignore').decode())

#让我们尝试另一个示例，该示例使用 replace 作为 encode() 函数中的第二个参数。
#对于此示例，让我们尝试一个字符串，该字符串具有不包含 ASCII 对应字符的字符。
stringVal = u'áæãåāœčćęßßßわた'

print(unicodedata.normalize('NFKD', stringVal).encode('ascii', 'replace').decode())

#replace 参数直接将没有 ASCII 对应的字符替换成问号 ? 符号。如果我们在同一字符串上使用 ignore：
print(unicodedata.normalize('NFKD', stringVal).encode('ascii', 'ignore').decode())



str1 = '汗'
print (repr(str1))

str2 = u'汗'
print (repr(str2))
str3 = str2.encode('utf-8')
str4 = str2.encode('gbk')
print (repr(str3))
print (repr(str4))
str5 = str3.decode('utf-8') 
print( repr(str5))
str6 = str4.decode('gbk') 
print( repr(str6))









