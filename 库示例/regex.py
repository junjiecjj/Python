#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:04:13 2022

@author: jack
"""

import re  
  
text = "JGood is a handsome boy, he is cool, clever, and so on..."  
print (re.sub(r'\s+', '-', text))  

text = "JGood is a handsome boy, he is cool, clever, and so on..."
print (re.sub(r'is\s+', '-', text))



print (re.sub(r'\s+\.', '.', text))

text="JGood is a handsome boy, , , he is cool, clever, and so on..."
print (re.sub(r'\s+,\s+', ', ', text))


text = "JGood is a handsome boy  , he is cool  , clever  , and so on..."
print (re.sub(r'\s+,\s+', ',', text))

text = "JGood is a handsome boy  , he is cool  , clever  .   and so on..."
print (re.sub(r'\s+\.', '.', text))



#========================================================================
# https://blog.csdn.net/HHG20171226/article/details/101646130
#========================================================================


def main():
    content = 'abc124hello46goodbye67shit'
    list1 = re.findall(r'\d+', content)
    print(list1)
    mylist = list(map(int, list1))
    print(mylist)
    print(sum(mylist))
    print(re.sub(r'\d+[hg]', 'foo1', content))
    print()
    print(re.sub(r'\d+', '456654', content))


if __name__ == '__main__':
    main()
# ['124', '46', '67']
# [124, 46, 67]
# 237
# abcfoo1ellofoo1oodbye67shit
# abc456654hello456654goodbye456654shit

text = "hello world ni hao"
ret = re.split('\W',text)
print(ret)


# 这里，[A-Z]+中的加号+表示，至少1次。[A-Z]+则表示，至少出现1个大写字母。
formula = 'YOU == ME**2'
ret = re.split('[A-Z]+', formula)
print(ret)


#========================================================================
#  https://www.runoob.com/python/python-reg-expressions.html
#========================================================================
"""

re.match函数
re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match() 就返回 none。

函数语法：

re.match(pattern, string, flags=0)
函数参数说明：

参数	描述
pattern	匹配的正则表达式
string	要匹配的字符串。
flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志
匹配成功 re.match 方法返回一个匹配的对象，否则返回 None。

我们可以使用 group(num) 或 groups() 匹配对象函数来获取匹配表达式。

匹配对象方法	描述
group(num=0)	匹配的整个表达式的字符串，group() 可以一次输入多个组号，在这种情况下它将返回一个包含那些组所对应值的元组。
groups()	返回一个包含所有小组字符串的元组，从 1 到 所含的小组号。

"""

print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配


line = "Cats are smarter than dogs"
 
matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)
 
if matchObj:
   print ("matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(1))
   print ("matchObj.group(2) : ", matchObj.group(2))
else:
   print ("No match!!")
"""
正则表达式：

r'(.*) are (.*?) .*'
解析:

首先，这是一个字符串，前面的一个 r 表示字符串为非转义的原始字符串，让编译器忽略反斜杠，也就是忽略转义字符。但是这个字符串里没有反斜杠，所以这个 r 可有可无。

 (.*) 第一个匹配分组，.* 代表匹配除换行符之外的所有字符。
 (.*?) 第二个匹配分组，.*? 后面多个问号，代表非贪婪模式，也就是说只匹配符合条件的最少字符
 后面的一个 .* 没有括号包围，所以不是分组，匹配效果和第一个一样，但是不计入匹配结果中。
matchObj.group() 等同于 matchObj.group(0)，表示匹配到的完整文本字符

matchObj.group(1) 得到第一组匹配结果，也就是(.*)匹配到的

matchObj.group(2) 得到第二组匹配结果，也就是(.*?)匹配到的
"""



"""
re.search方法
re.search 扫描整个字符串并返回第一个成功的匹配。

函数语法：

re.search(pattern, string, flags=0)
函数参数说明：

参数	描述
pattern	匹配的正则表达式
string	要匹配的字符串。
flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。
匹配成功re.search方法返回一个匹配的对象，否则返回None。

我们可以使用group(num) 或 groups() 匹配对象函数来获取匹配表达式。

匹配对象方法	描述
group(num=0)	匹配的整个表达式的字符串，group() 可以一次输入多个组号，在这种情况下它将返回一个包含那些组所对应值的元组。
groups()	返回一个包含所有小组字符串的元组，从 1 到 所含的小组号。
"""
print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.search('com', 'www.runoob.com').span())         # 不在起始位置匹配


line = "Cats are smarter than dogs";
 
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)
 
if searchObj:
   print ("searchObj.group() : ", searchObj.group())
   print ("searchObj.group(1) : ", searchObj.group(1))
   print ("searchObj.group(2) : ", searchObj.group(2))
else:
   print ("Nothing found!!")




#re.match与re.search的区别
#re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。


import re
 
line = "Cats are smarter than dogs";
 
matchObj = re.match( r'dogs', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
else:
   print ("No match!!")
 
matchObj = re.search( r'dogs', line, re.M|re.I)
if matchObj:
   print ("search --> searchObj.group() : ", matchObj.group())
else:
   print ("No match!!")





"""
检索和替换
Python 的 re 模块提供了re.sub用于替换字符串中的匹配项。

语法：

re.sub(pattern, repl, string, count=0, flags=0)
参数：

pattern : 正则中的模式字符串。
repl : 替换的字符串，也可为一个函数。
string : 要被查找替换的原始字符串。
count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

"""

phone = "2004-959-559 # 这是一个国外电话号码"
 
# 删除字符串中的 Python注释 
num = re.sub(r'#.*$', "", phone)
print( "电话号码是: ", num)
 
# 删除非数字(-)的字符串 
num = re.sub(r'\D', "", phone)
print ("电话号码是 : ", num)

#repl 参数是一个函数
#以下实例中将字符串中的匹配的数字乘以 2：

# 将匹配的数字乘以 2
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
 
s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))


"""
re.compile 函数
compile 函数用于编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用。

语法格式为：

re.compile(pattern[, flags])
参数：

pattern : 一个字符串形式的正则表达式

flags : 可选，表示匹配模式，比如忽略大小写，多行模式等，具体参数为：

re.I 忽略大小写
re.L 表示特殊字符集 \w, \W, \b, \B, \s, \S 依赖于当前环境
re.M 多行模式
re.S 即为 . 并且包括换行符在内的任意字符（. 不包括换行符）
re.U 表示特殊字符集 \w, \W, \b, \B, \d, \D, \s, \S 依赖于 Unicode 字符属性数据库
re.X 为了增加可读性，忽略空格和 # 后面的注释

正则表达式修饰符 - 可选标志
正则表达式可以包含一些可选标志修饰符来控制匹配的模式。修饰符被指定为一个可选的标志。多个标志可以通过按位 OR(|) 它们来指定。如 re.I | re.M 被设置成 I 和 M 标志：

修饰符	描述
re.I	使匹配对大小写不敏感
re.L	做本地化识别（locale-aware）匹配
re.M	多行匹配，影响 ^ 和 $
re.S	使 . 匹配包括换行在内的所有字符
re.U	根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.
re.X	该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。

语法： re.IGNORECASE 或简写为 re.I
作用： 进行忽略大小写匹配。

语法： re.ASCII 或简写为 re.A
作用： 顾名思义，ASCII表示ASCII码的意思，让 \w, \W, \b, \B, \d, \D, \s 和 \S 只匹配ASCII，而不是Unicode。

语法： re.DOTALL 或简写为 re.S
作用： DOT表示.，ALL表示所有，连起来就是.匹配所有，包括换行符\n。默认模式下.是不能匹配行符\n的。

语法： re.MULTILINE 或简写为 re.M
作用： 多行模式，当某字符串中有换行符\n，默认模式下是不支持换行符特性的，比如：行开头 和 行结尾，而多行模式下是支持匹配行开头的。


语法： re.VERBOSE 或简写为 re.X
作用： 详细模式，可以在正则表达式中加注解！


语法： re.LOCALE 或简写为 re.L
作用： 由当前语言区域决定 \w, \W, \b, \B 和大小写敏感匹配，这个标记只能对byte样式有效。这个标记官方已经不推荐使用，因为语言区域机制很不可靠，它一次只能处理一个 "习惯”，而且只对8位字节有效。
注意： 由于这个标记官方已经不推荐使用，而且猪哥也没使用过，所以就不给出实际的案例！

语法： re.UNICODE 或简写为 re.U
作用： 与 ASCII 模式类似，匹配unicode编码支持的字符，但是 Python 3 默认字符串已经是Unicode，所以有点冗余。

语法： re.DEBUG
作用： 显示编译时的debug信息。

"""

pattern = re.compile(r'\d+')                    # 用于匹配至少一个数字
m = pattern.match('one12twothree34four')        # 查找头部，没有匹配
print (m)

m = pattern.match('one12twothree34four', 2, 10) # 从'e'的位置开始匹配，没有匹配
print (m)

m = pattern.match('one12twothree34four', 3, 10) # 从'1'的位置开始匹配，正好匹配
print (m )                                        # 返回一个 Match 对象

m.group(0)   # 可省略 0
# '12'

m.start(0)   # 可省略 0
# 3

m.end(0)     # 可省略 0
# 5

m.span(0)    # 可省略 0
# (3,5)
"""
在上面，当匹配成功时返回一个 Match 对象，其中：

group([group1, …]) 方法用于获得一个或多个分组匹配的字符串，当要获得整个匹配的子串时，可直接使用 group() 或 group(0)；
start([group]) 方法用于获取分组匹配的子串在整个字符串中的起始位置（子串第一个字符的索引），参数默认值为 0；
end([group]) 方法用于获取分组匹配的子串在整个字符串中的结束位置（子串最后一个字符的索引+1），参数默认值为 0；
span([group]) 方法返回 (start(group), end(group))。
"""


pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)   # re.I 表示忽略大小写
m = pattern.match('Hello World Wide Web')
print(m)                               # 匹配成功，返回一个 Match 对象

m.group(0)                            # 返回匹配成功的整个子串
# 'Hello World'

m.span(0)                             # 返回匹配成功的整个子串的索引
# (0, 11)

m.group(1)                            # 返回第一个分组匹配成功的子串
# 'Hello'

m.span(1)                             # 返回第一个分组匹配成功的子串的索引
# (0, 5)

m.group(2)                            # 返回第二个分组匹配成功的子串
# 'World'

m.span(2)                             # 返回第二个分组匹配成功的子串
# (6, 11)

m.groups()                            # 等价于 (m.group(1), m.group(2), ...)
# ('Hello', 'World')

# m.group(3)                            # 不存在第三个分组,error




"""
findall
在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果有多个匹配模式，则返回元组列表，如果没有找到匹配的，则返回空列表。

注意： match 和 search 是匹配一次 findall 匹配所有。

语法格式为：

findall(string[, pos[, endpos]])
参数：

string : 待匹配的字符串。
pos : 可选参数，指定字符串的起始位置，默认为 0。
endpos : 可选参数，指定字符串的结束位置，默认为字符串的长度。
查找字符串中的所有数字：
"""
pattern = re.compile(r'\d+')   # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
 
print(result1)
print(result2)

#多个匹配模式，返回元组列表：
pattern = re.compile(r'\d+')   # 查找数字
result1 = pattern.findall('runoob 123 google 456')
result2 = pattern.findall('run88oob123google456', 0, 10)
 
print(result1)
print(result2)


"""
re.finditer
和 findall 类似，在字符串中找到正则表达式所匹配的所有子串，并把它们作为一个迭代器返回。

re.finditer(pattern, string, flags=0)
参数：

参数	描述
pattern	匹配的正则表达式
string	要匹配的字符串。
flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志
"""

 
it = re.finditer(r"\d+","12a32bc43jf3") 
for match in it: 
    print (match.group() )

"""
re.split
split 方法按照能够匹配的子串将字符串分割后返回列表，它的使用形式如下：

re.split(pattern, string[, maxsplit=0, flags=0])
参数：

参数	描述
pattern	匹配的正则表达式
string	要匹配的字符串。
maxsplit	分隔次数，maxsplit=1 分隔一次，默认为 0，不限制次数。
flags	标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等。参见：正则表达式修饰符 - 可选标志
"""

re.split('\W+', 'runoob, runoob, runoob.')
# ['runoob', 'runoob', 'runoob', '']

re.split('(\W+)', ' runoob, runoob, runoob.') 
# ['', ' ', 'runoob', ', ', 'runoob', ', ', 'runoob', '.', '']

re.split('\W+', ' runoob, runoob, runoob.', 1) 
# ['', 'runoob, runoob, runoob.']
 
re.split('a*', 'hello world')   # 对于一个找不到匹配的字符串而言，split 不会对其作出分割
# ['hello world']


"""
正则表达式模式
1 模式字符串使用特殊的语法来表示一个正则表达式：

2 字母和数字表示他们自身。一个正则表达式模式中的字母和数字匹配同样的字符串。

3 多数字母和数字前加一个反斜杠时会拥有不同的含义。

4 标点符号只有被转义时才匹配自身，否则它们表示特殊的含义。

5 反斜杠本身需要使用反斜杠转义。

6 由于正则表达式通常都包含反斜杠，所以你最好使用原始字符串来表示它们。模式元素(如 r'\t'，等价于 '\\t')匹配相应的特殊字符。

下表列出了正则表达式模式语法中的特殊元素。如果你使用模式的同时提供了可选的标志参数，某些模式元素的含义会改变。

模式	描述
^	          匹配字符串的开头
$	          匹配字符串的末尾。
.	          匹配任意字符，除了换行符，当re.DOTALL标记被指定时，则可以匹配包括换行符的任意字符。
[...]	      用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'
[^...]	      不在[]中的字符：[^abc] 匹配除了a,b,c之外的字符。
re*	          匹配0个或多个的表达式。
re+	           匹配1个或多个的表达式。
re?	          匹配0个或1个由前面的正则表达式定义的片段，非贪婪方式
re{ n}	      精确匹配 n 个前面表达式。例如， o{2} 不能匹配 "Bob" 中的 "o"，但是能匹配 "food" 中的两个 o。
re{ n,}	      匹配 n 个前面表达式。例如， o{2,} 不能匹配"Bob"中的"o"，但能匹配 "foooood"中的所有 o。"o{1,}" 等价于 "o+"。"o{0,}" 则等价于 "o*"。
re{ n, m}	   匹配 n 到 m 次由前面的正则表达式定义的片段，贪婪方式
a| b	      匹配a或b
(re)	      对正则表达式分组并记住匹配的文本
(?imx)	      正则表达式包含三种可选标志：i, m, 或 x 。只影响括号中的区域。
(?-imx)	      正则表达式关闭 i, m, 或 x 可选标志。只影响括号中的区域。
(?: re)	      类似 (...), 但是不表示一个组
(?imx: re)	  在括号中使用i, m, 或 x 可选标志
(?-imx: re)	  在括号中不使用i, m, 或 x 可选标志
(?#...)	      注释.
(?= re)	       前向肯定界定符。如果所含正则表达式，以 ... 表示，在当前位置成功匹配时成功，否则失败。但一旦所含表达式已经尝试，匹配引擎根本没有提高；模式的剩余部分还要尝试界定符的右边。
(?! re)	      前向否定界定符。与肯定界定符相反；当所含表达式不能在字符串当前位置匹配时成功
(?> re)	      匹配的独立模式，省去回溯。
\w	匹配字母数字及下划线
\W	匹配非字母数字及下划线
\s	匹配任意空白字符，等价于 [ \t\n\r\f]。
\S	匹配任意非空字符
\d	匹配任意数字，等价于 [0-9].
\D	匹配任意非数字
\A	匹配字符串开始
\Z	匹配字符串结束，如果是存在换行，只匹配到换行前的结束字符串。
\z	匹配字符串结束
\G	匹配最后匹配完成的位置。
\b	匹配一个单词边界，也就是指单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。
\B	匹配非单词边界。'er\B' 能匹配 "verb" 中的 'er'，但不能匹配 "never" 中的 'er'。
\n, \t, 等.	匹配一个换行符。匹配一个制表符。等
\1...\9	    匹配第n个分组的内容。
\10	        匹配第n个分组的内容，如果它经匹配。否则指的是八进制字符码的表达式。

正则表达式实例
字符匹配
实例	描述
python	匹配 "python".

字符类
实例	描述
[Pp]ython	匹配 "Python" 或 "python"
rub[ye]	匹配 "ruby" 或 "rube"
[aeiou]	匹配中括号内的任意一个字母
[0-9]	匹配任何数字。类似于 [0123456789]
[a-z]	匹配任何小写字母
[A-Z]	匹配任何大写字母
[a-zA-Z0-9]	匹配任何字母及数字
[^aeiou]	除了aeiou字母以外的所有字符
[^0-9]	匹配除了数字外的字符

特殊字符类
实例	描述
.	匹配除 "\n" 之外的任何单个字符。要匹配包括 '\n' 在内的任何字符，请使用象 '[.\n]' 的模式。
\d	匹配一个数字字符。等价于 [0-9]。
\D	匹配一个非数字字符。等价于 [^0-9]。
\s	匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
\S	匹配任何非空白字符。等价于 [^ \f\n\r\t\v]。
\w	匹配包括下划线的任何单词字符。等价于'[A-Za-z0-9_]'。
\W	匹配任何非单词字符。等价于 '[^A-Za-z0-9_]'。

"""





#========================================================================
#   https://www.liaoxuefeng.com/wiki/1016959663602400/1017639890281664
#========================================================================



"""
在正则表达式中，如果直接给出字符，就是精确匹配。用\d可以匹配一个数字，\w可以匹配一个字母或数字，所以：

'00\d'可以匹配'007'，但无法匹配'00A'；

'\d\d\d'可以匹配'010'；

'\w\w\d'可以匹配'py3'；

.可以匹配任意字符，所以：

'py.'可以匹配'pyc'、'pyo'、'py!'等等。
要匹配变长的字符，在正则表达式中，用*表示任意个字符（包括0个），用+表示至少一个字符，用?表示0个或1个字符，用{n}表示n个字符，用{n,m}表示n-m个字符：

来看一个复杂的例子：\d{3}\s+\d{3,8}。

我们来从左到右解读一下：

\d{3}表示匹配3个数字，例如'010'；

\s可以匹配一个空格（也包括Tab等空白符），所以\s+表示至少有一个空格，例如匹配' '，' '等；

\d{3,8}表示3-8个数字，例如'1234567'。

综合起来，上面的正则表达式可以匹配以任意个空格隔开的带区号的电话号码。

如果要匹配'010-12345'这样的号码呢？由于'-'是特殊字符，在正则表达式中，要用'\'转义，所以，上面的正则是\d{3}\-\d{3,8}。

但是，仍然无法匹配'010 - 12345'，因为带有空格。所以我们需要更复杂的匹配方式。

进阶
要做更精确地匹配，可以用[]表示范围，比如：

[0-9a-zA-Z\_]可以匹配一个数字、字母或者下划线；

[0-9a-zA-Z\_]+可以匹配至少由一个数字、字母或者下划线组成的字符串，比如'a100'，'0_Z'，'Py3000'等等；

[a-zA-Z\_][0-9a-zA-Z\_]*可以匹配由字母或下划线开头，后接任意个由一个数字、字母或者下划线组成的字符串，也就是Python合法的变量；

[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}更精确地限制了变量的长度是1-20个字符（前面1个字符+后面最多19个字符）。

A|B可以匹配A或B，所以(P|p)ython可以匹配'Python'或者'python'。

^表示行的开头，^\d表示必须以数字开头。

$表示行的结束，\d$表示必须以数字结束。

你可能注意到了，py也可以匹配'python'，但是加上^py$就变成了整行匹配，就只能匹配'py'了。


"""

#有了准备知识，我们就可以在Python中使用正则表达式了。Python提供re模块，包含所有正则表达式的功能。由于Python的字符串本身也用\转义，所以要特别注意：
s = 'ABC\\-001' # Python的字符串
# 对应的正则表达式字符串变成：
# 'ABC\-001'

# 因此我们强烈建议使用Python的r前缀，就不用考虑转义的问题了：

s = r'ABC\-001' # Python的字符串
# 对应的正则表达式字符串不变：
# 'ABC\-001'

re.match(r'^\d{3}\-\d{3,8}$', '010-12345')
re.match(r'^\d{3}\-\d{3,8}$', '010 12345')



print('a b   c'.split(' '))
# ['a', 'b', '', '', 'c']
#嗯，无法识别连续的空格，用正则表达式试试：

print(re.split(r'\s+', 'a b   c'))
# ['a', 'b', 'c']


#无论多少个空格都可以正常分割。加入,试试：
print(re.split(r'[\s\,]+', 'a,b, c  d'))


#再加入;试试：
print(re.split(r'[\s\,\;]+', 'a,b;; c  d'))




#除了简单地判断是否匹配之外，正则表达式还有提取子串的强大功能。用()表示的就是要提取的分组（Group）。比如：

# ^(\d{3})-(\d{3,8})$分别定义了两个组，可以直接从匹配的字符串中提取出区号和本地号码：

m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
# >>> m
# <_sre.SRE_Match object; span=(0, 9), match='010-12345'>
print(m.group(0))
# '010-12345'
print(m.group(1))
#'010'
print(m.group(2))
#'12345'

#如果正则表达式中定义了组，就可以在Match对象上用group()方法提取出子串来。
#注意到group(0)永远是与整个正则表达式相匹配的字符串，group(1)、group(2)……表示第1、2、……个子串。
#提取子串非常有用。来看一个更凶残的例子：
t = '19:05:30'
m = re.match(r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$', t)
print(m.groups())
# ('19', '05', '30')


#贪婪匹配
#最后需要特别指出的是，正则匹配默认是贪婪匹配，也就是匹配尽可能多的字符。举例如下，匹配出数字后面的0：

print(re.match(r'^(\d+)(0*)$', '102300').groups())
# ('102300', '')
#由于\d+采用贪婪匹配，直接把后面的0全部匹配了，结果0*只能匹配空字符串了。
#必须让\d+采用非贪婪匹配（也就是尽可能少匹配），才能把后面的0匹配出来，加个?就可以让\d+采用非贪婪匹配：
print(re.match(r'^(\d+?)(0*)$', '102300').groups())
# ('1023', '00')


#编译
#当我们在Python中使用正则表达式时，re模块内部会干两件事情：
#编译正则表达式，如果正则表达式的字符串本身不合法，会报错；
#用编译后的正则表达式去匹配字符串。
#如果一个正则表达式要重复使用几千次，出于效率的考虑，我们可以预编译该正则表达式，接下来重复使用时就不需要编译这个步骤了，直接匹配：


# 编译:
re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
# 使用：
print(re_telephone.match('010-12345').groups())
# ('010', '12345')
print(re_telephone.match('010-8086').groups())
# ('010', '8086')




"""

"""


s = "hello jack ! jka?sa jj??kk how are .. you ?? DrEaM come truE"
s1 = re.sub(r'([!.?])', r' \1', s)
s2 = re.sub(r'[^a-zA-Z.!?]+', r' ', s1)
s3 = re.sub(r'\s+', r' ', s2)
s4 = s3.lower()
print("s  = {}".format(s))
print("s1 = {}".format(s1))
print("s2 = {}".format(s2))
print("s3 = {}".format(s3))
print("s4 = {}".format(s4))
#========================================================================
#   Python核心编程
#========================================================================

"""
正则表达式模式       匹配的字符串

\w+-\d+                一个由字母数字组成的字符串和一串由一个连字符分隔的数字
[A-Za-z]\w*            第一个字符是字母;其余字符(如果存在)可以是字母或者数字(几乎等价于 Python 中的有 效标识符[参见练习])
\d{3}-\d{3}-\d{4}      美国电话号码的格式,前面是区号前缀,例如 800-555-1212
\w+@\w+\.com           以 XXX@YYY.com 格式表示的简单电子邮件地址

"""

#使用 match()方法匹配字符串

m = re.match('foo', 'foo')
 # 模式匹配字符串
if m is not None:
 # 如果匹配成功,就输出匹配内容
 m.group()


m = re.match('foo', 'food on the table') # 匹配成功
print(m.group())


print("# 匹配多个字符串")
bt = 'bat|bet|bit'   # 正则表达式模式: bat、bet、bit
m = re.match(bt, 'bat') # 'bat' 是一个匹配
m = re.match(bt, 'blt') # 对于 'blt' 没有匹配
m = re.match(bt, 'He bit me!') # 不能匹配字符串
m = re.search(bt, 'He bit me!') # 通过搜索查找 'bit'



print("# 匹配任何单个字符")
anyend = '.end'
m = re.match(anyend, 'bend')
if m is not None: print(m.group())
# 'bend'
m = re.match(anyend, 'end')
if m is not None: print(m.group())

m = re.match(anyend, '\nend')
if m is not None: print(m.group())
# 点号匹配 'b'


m = re.search('.end', 'The end.')# 在搜索中匹配 ' '
if m is not None: print(m.group())

patt314 = '3.14' # 表示正则表达式的点号
pi_patt = '3\.14' # 表示字面量的点号 (dec. point)

m = re.match(pi_patt, '3.14')  # 精确匹配
if m is not None: print(m.group())

m = re.match(patt314, '3014') # 点号匹配'0'
if m is not None: print(m.group())


m = re.match(patt314, '3.14') # 点号匹配 '.'
if m is not None: print(m.group())


print("#  创建字符集([ ])")
m = re.match('[cr][23][dp][o2]', 'c3po')# 匹配 'c3po'
if m is not None: print(m.group())
m = re.match('[cr][23][dp][o2]', 'c2do')# 匹配 'c2do'
if m is not None: print(m.group())
m = re.match('r2d2|c3po', 'c2do')# 不匹配 'c2do'
if m is not None: print(m.group())

m = re.match('r2d2|c3po', 'r2d2')# 匹配 'r2d2'
if m is not None: print(m.group())


print("# 重复、特殊字符以及分组")
patt = '\w+@(\w+\.)?\w+\.com'
print(re.match(patt, 'nobody@xxx.com').group())

print(re.match(patt, 'nobody@www.xxx.com').group())


patt = '\w+@(\w+\.)*\w+\.com'
print(re.match(patt, 'nobody@www.xxx.yyy.zzz.com').group())




m = re.match('\w\w\w-\d\d\d', 'abc-123')
if m is not None: print(m.group())
# abc-123

m = re.match('\w\w\w-\d\d\d', 'abc-xyz')
if m is not None: print(m.group())


m = re.match('(\w\w\w)-(\d\d\d)', 'abc-123')
if m is not None: print(m.group())# 完整匹配
# abc-123


print(m.group(1)) # 子组 1
# abc


print(m.group(2)) # 子组 2
# 123 

print(m.group()) # 全部子组
# abc-123

print(m.group(0)) # 全部子组
# abc-123

print(m.groups()) # 所有子组
# ('abc', '123')

# 如下为一个简单的示例,该示例展示了不同的分组排列,这将使整个事情变得更加清晰。
m = re.match('ab', 'ab')  # 没有子组

print(m.group()) # 完整匹配
# 'ab'

print(m.groups()) # 所有子组
# ()

m = re.match('(ab)', 'ab')# 一个子组
print(m.group())# 完整匹配
# 'ab'
print(m.group(1))# 子组 1
# 'ab'
print(m.groups())# 所有子组
# ('ab',)

m = re.match('(a)(b)', 'ab')# 两个子组
print(m.group())# 完整匹配
# 'ab'
print(m.group(1))# 子组 1
# 'a'
print(m.group(2))# 子组 2
# 'b'
print(m.groups())# 所有子组
#('a', 'b')

m = re.match('(a(b))', 'ab') # 两个子组
print(m.group())   # 完整匹配
# 'ab'
print(m.group(1))   # 子组 1
# 'ab'
print(m.group(2))  # 子组 2
# 'b'
print(m.groups()) # 所有子组
# ('ab', 'b')



print("# 匹配字符串的起始和结尾以及单词边界")


m = re.search('^The', 'The end.') # 匹配
if m is not None: print(m.group())
# 'The'

m = re.search('^The', 'end. The') # 不作为起始
if m is not None: print(m.group())
m = re.search(r'\bthe', 'bite the dog') # 在边界
if m is not None: print(m.group())
#'the'

m = re.search(r'\bthe', 'bitethe dog') # 有边界
if m is not None: print(m.group())

m = re.search(r'\Bthe', 'bitethe dog') # 没有边界
if m is not None: print(m.group())
#'the'


#使用 findall()和 finditer()查找每一次出现的位置
print(re.findall('car', 'car'))

print(re.findall('car', 'scary'))


print(re.findall('car', 'carry the barcardi to the car'))


s = 'This and that.'
S = 'This and that, those and these.'
# 对于一个成功的匹配,每个子组匹配是由 findall()返回的结果列表中的单一元素;对于
# 多个成功的匹配,每个子组匹配是返回的一个元组中的单一元素,而且每个元组(每个元组
# 都对应一个成功的匹配)是结果列表中的元素
print(re.findall(r'(th\w+) and (th\w+)', s, re.I))
# [('This', 'that')]

print(re.findall(r'(th\w+) and (th\w+)', S, re.I))
# [('This', 'that'), ('those', 'these')]

print(re.findall(r'(th\w+)', s, re.I))
# ['This', 'that']

print(re.findall(r'(th\w+)', S, re.I))
# ['This', 'that', 'those', 'these']


it = re.finditer(r'(th\w+)', s, re.I)


[g.groups() for g in re.finditer(r'(th\w+) and (th\w+)',s, re.I)]
# [('This', 'that')]



print([g.group() for g in re.finditer(r'(th\w+)', s, re.I)])
# ['This', 'that']

print([g.group(0) for g in re.finditer(r'(th\w+)', s, re.I)])
# ['This', 'that']

print([g.group(1) for g in re.finditer(r'(th\w+)', s, re.I)])
# ['This', 'that']

print([g.groups() for g in re.finditer(r'(th\w+)', s, re.I)])
# [('This',), ('that',)]

# 使用 sub()和 subn()搜索与替换
"""
有两个函数/方法用于实现搜索和替换功能:sub()和 subn()。两者几乎一样,都是将某字
符串中所有匹配正则表达式的部分进行某种形式的替换。用来替换的部分通常是一个字符串,
但它也可能是一个函数,该函数返回一个用来替换的字符串。subn()和 sub()一样,但 subn()
还返回一个表示替换的总数,替换后的字符串和表示替换总数的数字一起作为一个拥有两个
元素的元组返回。

"""
print(re.sub('X', 'Mr. Smith', 'attn: X\nDear X,\n'))
# attn: Mr. Smith
#Dear Mr. Smith,


print(re.subn('X', 'Mr. Smith', 'attn: X\nDear X,\n'))
# ('attn: Mr. Smith\nDear Mr. Smith,\n', 2)


print(re.sub('X', 'Mr. Smith', 'attn: X\nDear X,\n'))
# attn: Mr. Smith
# Dear Mr. Smith,

print(re.sub('[ae]', 'X', 'abcdef'))
# XbcdXf


print(re.subn('[ae]', 'X', 'abcdef'))
# ('XbcdXf', 2)


# 前面讲到,使用匹配对象的 group ()方法除了能够取出匹配分组编号外,还可以使用\N,其中 N 是在替换字符串中使用的分组编号。
print(re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})',r'\2/\1/\3', '2/20/91') )# Yes, Python is...'20/2/91'
# 20/2/91

print(re.sub(r'(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})',r'\2/\1/\3', '2/20/1991'))
# 20/2/1991

#在限定模式上使用 split()分隔字符串
print(re.split(':', 'str1:str2:str3'))
#['str1', 'str2', 'str3']


DATA = (
 'Mountain View, CA 94040',
 'Sunnyvale, CA',
 'Los Altos, 94023',
 'Cupertino 95014',
 'Palo Alto CA',)
for datum in DATA:
    print (re.split(', |(?= (?:\d{5}|[A-Z]{2})) ', datum))


#扩展符号

print(re.findall(r'(?i)yes', 'yes? Yes. YES!!'))


print(re.findall(r'(?i)th\w+', 'The quickest way is through this tunnel.'))



print(re.findall(r'(?im)(^th[\w ]+)', """ This line is the first, 
                 another line, 
                 that line,
                 it's the best """))


print(re.findall(r'th.+', '''
... The first line
... the second line
... the third line
... '''))


print(re.findall(r'(?s)th.+', '''
... The first line
... the second line
... the third line
... '''))

print(re.search(r'''(?x)  \((\d{3})\)   [ ]   (\d{3})  -   (\d{4})   ''', '(800) 555-1212').groups())
#('800', '555', '1212')




print(re.findall(r'http://(?:\w+\.)*(\w+\.com)', 'http://google.com http://www.google.com http://code.google.com'))
# ['google.com', 'google.com', 'google.com']


print(re.search(r'\((?P<areacode>\d{3})\) (?P<prefix>\d{3})-(?:\d{4})','(800) 555-1212').groupdict())
# {'areacode': '800', 'prefix': '555'}


print(re.sub(r'\((?P<areacode>\d{3})\) (?P<prefix>\d{3})-(?:\d{4})','(\g<areacode>) \g<prefix>-xxxx', '(800) 555-1212 (800) 555-xxxx'))


print(bool(re.match(r'\((?P<areacode>\d{3})\) (?P<prefix>\d{3})-(?P<number>\d{4}) (?P=areacode)-(?P=prefix)-(?P=number)1(?P=areacode)(?P=prefix)(?P=number)', '(800) 555-1212 800-555-1212 18005551212')))




print(bool(re.match(r'''(?x)
 \((?P<areacode>\d{3})\)[ ](?P<prefix>\d{3})-(?P<number>\d{4})
 [ ]
 (?P=areacode)-(?P=prefix)-(?P=number)
 [ ]
 1(?P=areacode)(?P=prefix)(?P=number)
 ''', '(800) 555-1212 800-555-1212 18005551212'))
)







print(re.findall(r'\w+(?= van Rossum)', '''
 Guido van Rossum
 Tim Peters
 Alex Martelli
 Just van Rossum
 Raymond Hettinger ''')

)


print(re.findall(r'(?m)^\s+(?!noreply|postmaster)(\w+)', '''
 sales@phptr.com
 postmaster@phptr.com
 eng@phptr.com
 noreply@phptr.com
 admin@phptr.com '''))



print( ['%s@aw.com' % e.group(1) for e in \
re.finditer(r'(?m)^\s+(?!noreply|postmaster)(\w+)', '''
 sales@phptr.com
 postmaster@phptr.com
 eng@phptr.com
 noreply@phptr.com
 admin@phptr.com
''')]
)

print(bool(re.search(r'(?:(x)|y)(?(1)y|x)', 'xy')))


print( bool(re.search(r'(?:(x)|y)(?(1)y|x)', 'xx')))







