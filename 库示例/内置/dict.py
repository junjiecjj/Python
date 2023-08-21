#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:18:52 2022

@author: jack

键必须不可变，所以可以用字符串,数字或元组充当，所以用列表就不行，如下实例：

"""

#========================================================================
# Python中创建字典的几种方法
#========================================================================
# 1 传统的文字表达式：
d={(20, 30):'good','name':'Allen', 'age':14, 'gender':'male',12:'kkk'}
print("dict d = {}".format(d))

#2.动态分配键值：
d={}
d['name']='Allen'
d['age']=21
d['gender']='male'
print("dict d = {}".format(d))
d[1]='abcd'
print("dict d = {}".format(d))


#3.字典键值表
d = dict(name='Allen', age=14, gender='male')
print("dict d = {}".format(d))
#这种形式所需的代码比常量少，但是键必须都是不带引号的字符串才行，其他的数字，元组都不行，所以下列代码会报错：
#c = dict(name='Allen', age=14, gender='male', 1='abcd')
#c = dict(name='Allen', age=14, gender='male', '1'='abcd')



# 4.字典键值元组表
d=dict([('name','Allen'),('age',21),('gender','male')])
print("dict d = {}".format(d))
# dict d = {'name': 'Allen', 'age': 21, 'gender': 'male'}

#5.所有键的值都相同或者赋予初始值：
d=dict.fromkeys(['height','weight'],'normal')
print("dict d = {}".format(d))
# dict d = {'height': 'normal', 'weight': 'normal'}


#  http://c.biancheng.net/view/2212.html
#  https://www.runoob.com/python/python-dictionary.html
#如下代码示范了使用花括号语法创建字典：
scores = {'语文': 89, '数学': 92, '英语': 93}
print(scores)
# {'语文': 89, '数学': 92, '英语': 93}

# 空的花括号代表空的dict
empty_dict = {}
print(empty_dict)
# 使用元组作为dict的key
dict2 = {(20, 30):'good', 30:'bad'}
print(dict2)
# {}
# {(20, 30): 'good', 30: 'bad'}

c = {(20, 30):'good','name':'Allen', 'age':14, 'gender':'male',12:'kkk'}
print(c)
print(c.get('name'))
# {(20, 30): 'good', 'name': 'Allen', 'age': 14, 'gender': 'male', 12: 'kkk'}
# Allen


#需要指出的是，元组可以作为 dict 的 key，但列表不能作为元组的 key。这是由于 dict 要求 key 必须是不可变类型，但列表是可变类型，因此列表不能作为元组的 key。
#在使用 dict() 函数创建字典时，可以传入多个列表或元组参数作为 key-value 对，每个列表或元组将被当成一个 key-value 对，因此这些列表或元组都只能包含两个元素。例如如下代码：
vegetables = [('celery', 1.58), ('brocoli', 1.29), ('lettuce', 2.19)]
# 创建包含3组key-value对的字典
dict3 = dict(vegetables)
print(dict3) # {'celery': 1.58, 'brocoli': 1.29, 'lettuce': 2.19}
# {'celery': 1.58, 'brocoli': 1.29, 'lettuce': 2.19}


cars = [['BMW', 8.5], ['BENS', 8.3], ['AUDI', 7.9]]
# 创建包含3组key-value对的字典
dict4 = dict(cars)
print(dict4) # {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}

#如果不为 dict() 函数传入任何参数，则代表创建一个空的字典。例如如下代码：
# 创建空的字典
dict5 = dict()
print(dict5) # {}

#还可通过为 dict 指定关键字参数创建字典，此时字典的 key 不允许使用表达式。例如如下代码：
# 使用关键字参数来创建字典
dict6 = dict(spinach = 1.39, cabbage = 2.59)
print(dict6) # {'spinach': 1.39, 'cabbage': 2.59}
# {'spinach': 1.39, 'cabbage': 2.59}

#如下代码示范了通过 key 访问 value：
scores = {'语文': 89}
# 通过key访问value
print(scores['语文'])
print(scores)
# 89
# {'语文': 89}


#如果要为 dict 添加 key-value 对，只需为不存在的 key 赋值即可：
# 对不存在的key赋值，就是增加key-value对
scores['数学'] = 93
scores[92] = 5.7
print(scores) # {'语文': 89, '数学': 93, 92: 5.7}

#如果要删除宇典中的 key-value 对，则可使用 del 语句。例如如下代码：

# 使用del语句删除key-value对
del scores['语文']
del scores['数学']
print(scores) # {92: 5.7}

#如果对 dict 中存在的 key-value 对赋值，新赋的 value 就会覆盖原有的 value，这样即可改变 dict 中的 key-value 对。例如如下代码：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# 对存在的key-value对赋值，改变key-value对
cars['BENS'] = 4.3
cars['AUDI'] = 3.8
print(cars)
# {'BMW': 8.5, 'BENS': 4.3, 'AUDI': 3.8}


#如果要判断字典是否包含指定的 key，则可以使用 in 或 not in 运算符。需要指出的是，对于 dict 而言，in 或 not in 运算符都是基于 key 来判断的。例如如下代码：
# 判断cars是否包含名为'AUDI'的key
print('AUDI' in cars) # True
# 判断cars是否包含名为'PORSCHE'的key
print('PORSCHE' in cars) # False
print('LAMBORGHINI' not in cars) # True


#字典由 dict 类代表，因此我们同样可使用 dir(dict) 来查看该类包含哪些方法。在交互式解释器中输入 dir(dict) 命令，将看到如下输出结果：
print("dir(dict) = {}".format(dir(dict)))



#clear()方法
#clear() 用于清空字典中所有的 key-value 对，对一个字典执行 clear() 方法之后，该字典就会变成一个空字典。例如如下代码：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
print(cars) # {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# 清空cars所有key-value对
cars.clear()
print(cars) # {}


#get()方法
#get() 方法其实就是根据 key 来获取 value，它相当于方括号语法的增强版，当使用方括号语法访问并不存在的 key 时，字典会引发 KeyError 错误；但如果使用 get() 方法访问不存在的 key，该方法会简单地返回 None，不会导致错误。例如如下代码：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# 获取'BMW'对应的value
print(cars.get('BMW')) # 8.5
print(cars.get('PORSCHE')) # None
# print(cars['PORSCHE']) # KeyError


#update()方法
#update() 方法可使用一个字典所包含的 key-value 对来更新己有的字典。在执行 update() 方法时，如果被更新的字典中己包含对应的 key-value 对，那么原 value 会被覆盖；如果被更新的字典中不包含对应的 key-value 对，则该 key-value 对被添加进去。例如如下代码：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
cars.update({'BMW':4.5, 'PORSCHE': 9.3})
print(cars)




#items()、keys()、values()
#items()、keys()、values() 分别用于获取字典中的所有 key-value 对、所有 key、所有 value。这三个方法依次返回 dict_items、dict_keys 和 dict_values 对象，Python 不希望用户直接操作这几个方法，但可通过 list() 函数把它们转换成列表。如下代码示范了这三个方法的用法：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# 获取字典所有的key-value对，返回一个dict_items对象
ims = cars.items()
print(type(ims)) # <class 'dict_items'>
# 将dict_items转换成列表
print(list(ims))
# [('BMW', 8.5), ('BENS', 8.3), ('AUDI', 7.9)]

# 访问第2个key-value对
print(list(ims)[1])
# ('BENS', 8.3)

# 获取字典所有的key，返回一个dict_keys对象
kys = cars.keys()
print(type(kys)) # <class 'dict_keys'>
# 将dict_keys转换成列表
print(list(kys))
# ['BMW', 'BENS', 'AUDI']


# 访问第2个key
print(list(kys)[1])
# 'BENS'

# 获取字典所有的value，返回一个dict_values对象
vals = cars.values()

print(type(vals))
# 将dict_values转换成列表
print(list(vals)) # [8.5, 8.3, 7.9]
# 访问第2个value
print(list(vals)[1]) # 8.3



#pop方法
# pop() 方法用于获取指定 key 对应的 value，并删除这个 key-value 对。如下方法示范了 pop() 方法的用法：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
print(cars.pop('AUDI')) # 7.9
print(cars) # {'BMW': 8.5, 'BENS': 8.3}


#popitem()方法
# popitem() 方法用于随机弹出字典中的一个 key-value 对。
#此处的随机其实是假的，正如列表的 pop() 方法总是弹出列表中最后一个元素，实际上字典的 popitem() 其实也是弹出字典中最后一个 key-value 对。由于字典存储 key-value 对的顺序是不可知的，因此开发者感觉字典的 popitem() 方法是“随机”弹出的，但实际上字典的 popitem() 方法总是弹出底层存储的最后一个 key-value 对。

#如下代码示范了 popitem() 方法的用法：
cars = {'AUDI': 7.9, 'BENS': 8.3, 'BMW': 8.5}
print(cars)
# {'AUDI': 7.9, 'BENS': 8.3, 'BMW': 8.5}
# 弹出字典底层存储的最后一个key-value对
print(cars.popitem())
# ('BMW', 8.5)
print(cars)
# {'BMW': 8.5, 'BENS': 8.3}

#由于实际上 popitem 弹出的就是一个元组，因此程序完全可以通过序列解包的方式用两个变量分别接收 key 和 value。例如如下代码：
# 将弹出项的key赋值给k、value赋值给v
k, v = cars.popitem()
print(k, v) # BENS 8.3


# setdefault()方法
# setdefault() 方法也用于根据 key 来获取对应 value 的值。但该方法有一个额外的功能，即当程序要获取的 key 在字典中不存在时，该方法会先为这个不存在的 key 设置一个默认的 value，然后再返回该 key 对应的 value。

#总之，setdefault() 方法总能返回指定 key 对应的 value；如果该 key-value 对存在，则直接返回该 key 对应的 value；如果该 key-value 对不存在，则先为该 key 设置默认的 value，然后再返回该 key 对应的 value。

#如下代码示范了 setdefault() 方法的用法：
cars = {'BMW': 8.5, 'BENS': 8.3, 'AUDI': 7.9}
# 设置默认值，该key在dict中不存在，新增key-value对
print(cars.setdefault('PORSCHE', 9.2)) # 9.2
print(cars)
# 设置默认值，该key在dict中存在，不会修改dict内容
print(cars.setdefault('BMW', 3.4)) # 8.5
print(cars)



#fromkeys()方法
#fromkeys() 方法使用给定的多个 key 创建字典，这些 key 对应的 value 默认都是 None；也可以额外传入一个参数作为默认的 value。该方法一般不会使用字典对象调用（没什么意义），通常会使用 dict 类直接调用。例如如下代码：
# 使用列表创建包含2个key的字典
a_dict = dict.fromkeys(['a', 'b'])
print(a_dict) # {'a': None, 'b': None}
# 使用元组创建包含2个key的字典
b_dict = dict.fromkeys((13, 17))
print(b_dict) # {13: None, 17: None}
# 使用元组创建包含2个key的字典，指定默认的value
c_dict = dict.fromkeys((13, 17), 'good')
print(c_dict) # {13: 'good', 17: 'good'}





#使用字典格式化字符串
#前面章节介绍过，在格式化字符串时，如果要格式化的字符串模板中包含多个变量，后面就需要按顺序给出多个变量，这种方式对于字符串模板中包含少量变量的情形是合适的，但如果字符串模板中包含大量变量，这种按顺序提供变量的方式则有些不合适。可改为在字符串模板中按 key 指定变量，然后通过字典为字符串模板中的 key 设置值。

#例如如下程序：
# 字符串模板中使用key
temp = '教程是:%(name)s, 价格是:%(price)010.2f, 出版社是:%(publish)s'
book = {'name':'Python基础教程', 'price': 99, 'publish': 'C语言中文网'}
# 使用字典为字符串模板中的key传入值
print(temp % book)
# 教程是:Python基础教程, 价格是:0000099.00, 出版社是:C语言中文网


book = {'name':'C语言小白变怪兽', 'price':159, 'publish': 'C语言中文网'}
# 使用字典为字符串模板中的key传入值
print(temp % book)
# 教程是:C语言小白变怪兽, 价格是:0000159.00, 出版社是:C语言中文网


# 看起来字典的keys()和values()方法返回的列表始终是一对一的映射（假设字典在调用这两个方法之间没有改变）。
d = {'one':1, 'two': 2, 'three': 3}
k, v = d.keys(), d.values()

for i in range(len(k)):
     print (d[k[i]] == v[i])


# 实例1：按键(key)排序
def dictionairy():

    # 声明字典
    key_value ={}

    # 初始化
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323

    print ("按键(key)排序:")

    # sorted(key_value) 返回重新排序的列表
    # 字典按键排序
    for i in sorted (key_value) :
        print ((i, key_value[i]), end =" ")
# 调用函数
dictionairy()
"""
按键(key)排序:
(1, 2) (2, 56) (3, 323) (4, 24) (5, 12) (6, 18)
"""

#实例2：按值(value)排序
def dictionairy():

    # 声明字典
    key_value ={}

    # 初始化
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323


    print ("按值(value)排序:")
    print(sorted(key_value.items(), key = lambda kv:(kv[1], kv[0])))
dictionairy()
"""
按值(value)排序:
[(1, 2), (5, 12), (6, 18), (4, 24), (2, 56), (3, 323)]
"""

#实例 3 : 字典列表排序
lis = [{ "name" : "Taobao", "age" : 100},
{ "name" : "Runoob", "age" : 7 },
{ "name" : "Google", "age" : 100 },
{ "name" : "Wiki" , "age" : 200 }]

# 通过 age 升序排序
print ("列表通过 age 升序排序: ")
print (sorted(lis, key = lambda i: i['age']) )

print ("\r")

# 先按 age 排序，再按 name 排序
print ("列表通过 age 和 name 排序: ")
print (sorted(lis, key = lambda i: (i['age'], i['name'])) )

print ("\r")

# 按 age 降序排序
print ("列表通过 age 降序排序: ")
print (sorted(lis, key = lambda i: i['age'],reverse=True) )
"""
列表通过 age 升序排序:
[{'name': 'Runoob', 'age': 7}, {'name': 'Taobao', 'age': 100}, {'name': 'Google', 'age': 100}, {'name': 'Wiki', 'age': 200}]

列表通过 age 和 name 排序:
[{'name': 'Runoob', 'age': 7}, {'name': 'Google', 'age': 100}, {'name': 'Taobao', 'age': 100}, {'name': 'Wiki', 'age': 200}]

列表通过 age 降序排序:
[{'name': 'Wiki', 'age': 200}, {'name': 'Taobao', 'age': 100}, {'name': 'Google', 'age': 100}, {'name': 'Runoob', 'age': 7}]
"""


#========================================================================
# https://www.1024sou.com/article/419794.html
# Python字典（dict ）的几种遍历方式
#========================================================================
#1.使用 for key in dict 遍历字典
x = {'a': 'A', 'b': 'B','chen':'liang','junjie':'qingxia'}
for key in x:
    print(f"key={key}, value={x[key]}")


for key in x:
    print(f"key={key}")

#2.使用 for key in dict.keys () 遍历字典的键
# keys
book = {
    'title': 'Python',
    'author': '-----',
    'press': '人生苦短，我用python'
}

for key in book.keys():
    print(key)

#3.使用 for values in dict.values () 遍历字典的值
# values
book = {
    'title': 'Python',
    'author': '-----',
    'press': '人生苦短，我用python'
}

for value in book.values():
    print(value)
# Python
# -----
# 人生苦短，我用python


#4.使用 for item in dict.items () 遍历字典的键值对
x = {'a': 'A', 'b': 'B'}
for item in x.items():
    key = item[0]
    value = item[1]
    print('%s   %s:%s' % (item, key, value))
# ('a', 'A')   a:A
# ('b', 'B')   b:B


item = (1, 2)
a, b = item
print(a, b)
# 1 2



x = {'a': 'A', 'b': 'B'}
for key, value in x.items():
    print('%s:%s' % (key, value))
# a:A
# b:B













