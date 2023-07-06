#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 23:49:17 2022

@author: jack

https://geek-docs.com/python/python-tutorial/python-fstring.html#:%7E:text=Python%20f-string%20%E6%98%AF%E6%89%A7%E8%A1%8C%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%A0%BC%E5%BC%8F%E5%8C%96%E7%9A%84%E6%9C%80%E6%96%B0%20Python%20%E8%AF%AD%E6%B3%95%E3%80%82%20%E8%87%AA%20Python%203.6,Python%20%E4%B8%AD%E6%A0%BC%E5%BC%8F%E5%8C%96%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9A%84%E6%96%B9%E5%BC%8F%E3%80%82%20f%20%E5%AD%97%E7%AC%A6%E4%B8%B2%E7%9A%84%E5%89%8D%E7%BC%80%E4%B8%BA%20f%20%EF%BC%8C%E5%B9%B6%E4%BD%BF%E7%94%A8%20%7B%7D%20%E6%8B%AC%E5%8F%B7%E8%AF%84%E4%BC%B0%E5%80%BC%E3%80%82

Python f 字符串教程

Python f 字符串教程显示了如何使用 f 字符串在 Python 中格式化字符串。

文章目录

1 Python f 字符串
2 Python 字符串格式
3 Python f 字符串表达式
4 Python f 字符串字典
5 Python 多行 f 字符串
6 Python f 字符串调用函数
7 Python f 字符串对象
8 Python F 字符串转义字符
9 Python f 字符串格式化日期时间
10 Python f 字符串格式化浮点数
11 Python f 字符串格式化宽度
12 Python f 字符串对齐字符串
13 Python f 字符串数字符号
Python f 字符串
Python f-string 是执行字符串格式化的最新 Python 语法。 自 Python 3.6 起可用。 Python f 字符串提供了一种更快，更易读，更简明且不易出错的在 Python 中格式化字符串的方式。

f 字符串的前缀为f，并使用{}括号评估值。

在冒号后指定用于类型，填充或对齐的格式说明符； 例如：f'{price:.3}'，其中price是变量名。

Python 字符串格式
以下示例总结了 Python 中的字符串格式设置选项。

formatting_strings.py

#!/usr/bin/env python3

name = 'Peter'
age = 23

print('%s is %d years old' % (name, age))
print('{} is {} years old'.format(name, age))
print(f'{name} is {age} years old')
Py
该示例使用两个变量设置字符串格式。

print('%s is %d years old' % (name, age))
Py
这是最旧的选项。 它使用%运算符和经典字符串格式指定，例如%s和%d。

print('{} is {} years old'.format(name, age))
Py
从 Python 3.0 开始，format()函数被引入以提供高级格式化选项。

print(f'{name} is {age} years old')
Py
从 Python 3.6 开始，Python f 字符串可用。 该字符串具有f前缀，并使用{}评估变量。

$ python formatting_string.py
Peter is 23 years old
Peter is 23 years old
Peter is 23 years old
Py
Python f 字符串表达式
我们可以将表达式放在{}括号之间。

expressions.py

#!/usr/bin/env python3

bags = 3
apples_in_bag = 12

print(f'There are total of {bags * apples_in_bag} apples')
Py
该示例对 f 字符串中的表达式求值。

$ python expressions.py
There are total of 36 apples
Py
Python f 字符串字典
我们可以使用 f 字符串中的字典。

dicts.py

#!/usr/bin/env python3

user = {'name': 'John Doe', 'occupation': 'gardener'}

print(f"{user['name']} is a {user['occupation']}")
Py
该示例以 f 字符串形式评估字典。

$ python dicts.py
John Doe is a gardener
Py
Python 多行 f 字符串
我们可以使用多行字符串。

multiline.py

#!/usr/bin/env python3

name = 'John Doe'
age = 32
occupation = 'gardener'

msg = (
    f'Name: {name}\n'
    f'Age: {age}\n'
    f'Occupation: {occupation}'
)

print(msg)
Py
该示例显示了多行 f 字符串。 F 弦放在方括号之间； 每个字符串前面都带有f字符。

$ python multiline.py
Name: John Doe
Age: 32
Occupation: gardener
Py
Python f 字符串调用函数
我们还可以在 f 字符串中调用函数。

call_function.py

#!/usr/bin/env python3

def mymax(x, y):

    return x if x > y else y

a = 3
b = 4

print(f'Max of {a} and {b} is {mymax(a, b)}')
Py
该示例在 f 字符串中调用自定义函数。

$ python call_fun.py
Max of 3 and 4 is 4
Py
Python f 字符串对象
Python f 字符串也接受对象。 对象必须定义了__str__()或__repr__()魔术函数。

objects.py

#!/usr/bin/env python3

class User:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    def __repr__(self):
        return f"{self.name} is a {self.occupation}"

u = User('John Doe', 'gardener')

print(f'{u}')
Py
该示例评估 f 字符串中的对象。

$ python objects.py
John Doe is a gardener
Py
Python F 字符串转义字符
下面的示例显示如何对 f 字符串中的某些字符进行转义。

escaping.py

#!/usr/bin/env python3

print(f'Python uses {{}} to evaludate variables in f-strings')
print(f'This was a \'great\' film')
Py
为了避免花括号，我们将字符加倍。 单引号以反斜杠字符转义。

$ python escaping.py
Python uses {} to evaludate variables in f-strings
This was a 'great' film
Py
Python f 字符串格式化日期时间
以下示例格式化日期时间。

format_datetime.py

#!/usr/bin/env python3

import datetime

now = datetime.datetime.now()

print(f'{now:%Y-%m-%d %H:%M}')
Py
该示例显示格式化的当前日期时间。 日期时间格式说明符位于<colon>：</colon>字符之后。

$ python format_datetime.py
2019-05-11 22:39
Py
Python f 字符串格式化浮点数
浮点值的后缀为f。 我们还可以指定精度：小数位数。 精度是一个点字符后的值。

format_floats.py

#!/usr/bin/env python3

val = 12.3

print(f'{val:.2f}')
print(f'{val:.5f}')
Py
该示例打印格式化的浮点值。

$ python format_floats.py
12.30
12.30000
Py
输出显示具有两位和五个小数位的数字。

Python f 字符串格式化宽度
宽度说明符设置值的宽度。 如果该值短于指定的宽度，则该值可以用空格或其他字符填充。

format_width.py

#!/usr/bin/env python3

for x in range(1, 11):
    print(f'{x:02} {x*x:3} {x*x*x:4}')
Py
该示例打印三列，每个列都有一个预定义的宽度。 第一列使用 0 填充较短的值。

$ python format_width.py
01   1    1
02   4    8
03   9   27
04  16   64
05  25  125
06  36  216
07  49  343
08  64  512
09  81  729
10 100 1000
Py
Python f 字符串对齐字符串
默认情况下，字符串在左边对齐。 我们可以使用&gt;字符来对齐右侧的字符串。 &gt;字符在冒号后面。

justify.py

#!/usr/bin/env python3

s1 = 'a'
s2 = 'ab'
s3 = 'abc'
s4 = 'abcd'

print(f'{s1:>10}')
print(f'{s2:>10}')
print(f'{s3:>10}')
print(f'{s4:>10}')
Py
我们有四个不同长度的弦。 我们将输出的宽度设置为十个字符。 值在右对齐。

$ python justify.py
         a
        ab
       abc
      abcd
Py
Python f 字符串数字符号
数字可以具有各种数字符号，例如十进制或十六进制。

format_notations.py

#!/usr/bin/env python3

a = 300

# hexadecimal
print(f"{a:x}")

# octal
print(f"{a:o}")

# scientific
print(f"{a:e}")
Py
该示例以三种不同的表示法打印值。

$ python format_notations.py
12c
454
3.000000e+02
Py
在本教程中，我们使用了 Python f 字符串。
"""


# https://geek-docs.com/python/python-tutorial/python-fstring.html
name = 'Peter'
age = 23

print('%s is %d years old' % (name, age))
print('{} is {} years old'.format(name, age))
print(f'{name} is {age} years old')


#该示例使用两个变量设置字符串格式。
print('%s is %d years old' % (name, age))

#这是最旧的选项。 它使用%运算符和经典字符串格式指定，例如%s和%d。
print('{} is {} years old'.format(name, age))

#从 Python 3.0 开始，format()函数被引入以提供高级格式化选项。
print(f'{name} is {age} years old')

#从 Python 3.6 开始，Python f 字符串可用。 该字符串具有f前缀，并使用{}评估变量。

#Python f 字符串表达式
#我们可以将表达式放在{}括号之间。



bags = 3
apples_in_bag = 12

print(f'There are total of {bags * apples_in_bag} apples')

#该示例对 f 字符串中的表达式求值。


#Python f 字符串字典
#我们可以使用 f 字符串中的字典。



user = {'name': 'John Doe', 'occupation': 'gardener'}

print(f"{user['name']} is a {user['occupation']}")
#该示例以 f 字符串形式评估字典。

#Python 多行 f 字符串
#我们可以使用多行字符串。



name = 'John Doe'
age = 32
occupation = 'gardener'

msg = (
    f'Name: {name}\n'
    f'Age: {age}\n'
    f'Occupation: {occupation}'
)

print(msg)

#该示例显示了多行 f 字符串。 F 弦放在方括号之间； 每个字符串前面都带有f字符。



#Python f 字符串调用函数
#我们还可以在 f 字符串中调用函数。


def mymax(x, y):

    return x if x > y else y

a = 3
b = 4

print(f'Max of {a} and {b} is {mymax(a, b)}')
#该示例在 f 字符串中调用自定义函数。


#Python f 字符串对象
#Python f 字符串也接受对象。 对象必须定义了__str__()或__repr__()魔术函数。

class User:
    def __init__(self, name, occupation):
        self.name = name
        self.occupation = occupation

    def __repr__(self):
        return f"{self.name} is a {self.occupation}"

u = User('John Doe', 'gardener')

print(f'{u}')
#该示例评估 f 字符串中的对象。



# Python F 字符串转义字符
# 下面的示例显示如何对 f 字符串中的某些字符进行转义。



print(f'Python uses {{}} to evaludate variables in f-strings')
print(f'This was a \'great\' film')

# 为了避免花括号，我们将字符加倍。 单引号以反斜杠字符转义。

# Python f 字符串格式化日期时间
# 以下示例格式化日期时间。


import datetime

now = datetime.datetime.now()

print(f'{now:%Y-%m-%d %H:%M}')
# 该示例显示格式化的当前日期时间。 日期时间格式说明符位于<colon>：</colon>字符之后。

# Python f 字符串格式化浮点数
# 浮点值的后缀为f。 我们还可以指定精度：小数位数。 精度是一个点字符后的值。


val = 12.38426492

print(f'{val:.2f}')
print(f'{val:.5f}')

#输出显示具有两位和五个小数位的数字。

# Python f 字符串格式化宽度
# 宽度说明符设置值的宽度。 如果该值短于指定的宽度，则该值可以用空格或其他字符填充。


for x in range(1, 11):
    print(f'{x:02} {x*x:3} {x*x*x:4}')
#该示例打印三列，每个列都有一个预定义的宽度。 第一列使用 0 填充较短的值。



# Python f 字符串对齐字符串
# 默认情况下，字符串在左边对齐。 我们可以使用&gt;字符来对齐右侧的字符串。 &gt;字符在冒号后面。



s1 = 'a'
s2 = 'ab'
s3 = 'abc'
s4 = 'abcd'

print(f'{s1:<10}')
print(f'{s2:>10}')
print(f'{s3:>10}')
print(f'{s4:>10}')
#我们有四个不同长度的弦。 我们将输出的宽度设置为十个字符。 值在右对齐。


#Python f 字符串数字符号
#数字可以具有各种数字符号，例如十进制或十六进制。

a = 300

# hexadecimal
print(f"{a:x}")

# octal
print(f"{a:o}")

# scientific
print(f"{a:e}")
#该示例以三种不同的表示法打印值。
