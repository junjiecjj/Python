#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:50:27 2022

@author: 
    
1. 语法

1.1 string.join()

'sep'.join(seq)
代码解析：

sep: 代表分隔符, 可以是单个字符如: , . - ; 等，也可以是字符串如: 'abc'。

seq: 代表要连接的元素序列，可以是字符串、元组、列表、字典等。

注：'sep'和seq都只能是string型，不能是int型和float型。
    
"""
import os


str = "-";
seq = ("a", "b", "c"); # 字符串序列
print(str.join( seq ));

str = " ";
seq = ("a", "b", "c"); # 字符串序列
print(str.join( seq ));

str = "";
seq = ("a", "b", "c"); # 字符串序列
print(str.join( seq ));

sep = " "            #分隔符(空格)
seq = '女神节日快乐'  #要连接的字符串 
str = sep.join(seq)  #用分隔符连接字符串中的元素
print("str = {}".format(str))

#分隔符是多个字符
sep = "  (*^__^*)  "  #分隔符(空格)
seq = '女神节日快乐'   #要连接的字符串 
str = sep.join(seq)   #用分隔符连接字符串中的元素
print("str = {}".format(str))

sep = " "                                 #分隔符(空格)
seq = {'W':1,'i':2,'n':3,'k':4, 'n':5}    #要连接的字典
str = sep.join(seq)                        #用分隔符连接字典的元素
print("str = {}".format(str))


sep = " (＾＿－) "                          #分隔符(空格)
seq = {'W':1,'i':2,'n':3,'k':4, 'n':5}      #要连接的字典  
str = sep.join(seq)                        #用分隔符连接字典的元素
print("str = {}".format(str))

#合并目录
path1 = 'D:'
path2 = '新建文件夹:'
path3 = '微信公众号:'
path4 = '17.python中的join函数'
Path_Final = os.path.join(path1, path2, path3, path4)
print("Path_Final = {}".format(Path_Final))
Path_Final1 = path1+path2+path3+path4
print("Path_Final = {}".format(Path_Final1))

#对列表进行操作（分别使用' '与':'作为分隔符）
seq1 = ['hello','good','boy','doiido']
print(' '.join(seq1))
print(':'.join(seq1))


#对字符串进行操作
seq2 = "hello good boy doiido"
print(':'.join(seq2))

#对字典进行操作
seq4 = {'hello':1,'good':2,'boy':3,'doiido':4}
print(':'.join(seq4))

#对元组进行操作
seq3 = ('hello','good','boy','doiido')
print(':'.join(seq3))


#用python代码实现分解素因数，并用join函数打印出来
num = int(input())                       #输入想要分解素因数的数
factor_list = []                         #设定保存素因数的列表                     
def factor_fun(n):
    for i in range(2, n+1):              #构建求素因数的循环           
        if n%i == 0:                     #如果n能整除i，则把i加入保存素因数的列表
            factor_list.append(i)
            if n!=i:                     #如果i不等于n，且i是n的因数，把n除以i得到的新数，调用factor_fun函数
                factor_fun(int(n/i))
            return factor_list
        
c = factor_fun(num)                     #调用函数
print(num, '=', end=' ',sep = ' ')
print('*'.join('%s' %id for id in factor_list))  #把factor_list列表中数值型的数据转换成字符，用*连接


#用join函数根据当前路径组成新路径
os.getcwd()   #获取当前路径
data_save = os.path.join(os.getcwd(), 'data_save')  #获取当前路径并组合新的路径










