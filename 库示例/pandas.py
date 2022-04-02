#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: pandas.py
# Author: 陈俊杰
# Created Time: 2021年07月19日 星期一 10时52分55秒

# mail: 2716705056@qq.com
#  此程序的功能是：

#########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time


"""
如果文本中的分隔符既有空格也有制表符('\t'),则sep='\s+'可以匹配任何空格。
"""
data1 = pd.read_table("data1.txt",sep='\s+')
data1a = pd.read_csv("data1.txt",sep='\s+')
data1_1 = pd.read_table("data1_1.txt",sep=',')
data1_1a = pd.read_csv("data1_1.txt",sep=',')

print(data1,'\n ********************* data1结束************************')



data2 = pd.read_table("data2.txt",sep=',', skiprows=[0,1,3,6])
data2a = pd.read_csv("data2.txt",sep=',',header=0, skiprows=[0,1,3,6])
data2a = pd.DataFrame(data2a)
# index_col=2,指定用作行标签的列
data2b = pd.read_csv("data2.txt",sep=',', index_col=2, skiprows=[0,1,3,6])
data2c = pd.read_csv("data2.txt",sep=',',  skiprows=[0,1,3,6])

# 指定header name
data2d = pd.read_csv("data2a.txt",sep=',', header=None, names=['col1','col2','col3','col4','col5'],)
print("data2:\n",data2,'\n')
print("data2a:\n",data2a,'\n')
print("data2b:\n",data2b,'\n')
print("data2c:\n",data2c,'\n')
print("data2d:\n",data2d,'\n')

print('***************索引列***********************\n\n')
print("data2d:\n",data2d,'\n')

print('data2d[\'col1\']:\n',data2d['col1'],'\n')
print('data2d.loc[:,[\'col1\']]:\n',data2d.loc[:,['col1']],'\n')
print('data2d[\'col3\']:\n',data2d['col3'],'\n')
print('data2d.loc[:,[\'col3\']] \n',data2d.loc[:,['col3']],'\n')
print('data2d[\'col1\',\'col3\']:\n',data2d[['col1','col3']],'\n')
print('data2d.loc[:,[\'col1\',\'col3\']] \n', data2d.loc[:,['col1','col3']],'\n')

print('data2c.loc[:,0]:\n',data2c.iloc[:,0],'\n')
print('data2c.loc[:,2 \n',data2c.iloc[:,2],'\n')
print('data2c.loc[:,[0,2]] \n', data2c.iloc[:,[0,2]],'\n')

print("data2a:\n",data2a,'\n')
print("data2a.loc[:,[ 'blue']]:\n",data2a.loc[:,'blue'],'\n')
print("data2a.loc[:,['red']] \n",data2a.loc[:,'red'],'\n')
print("data2a.loc[:,['blue','red']] \n", data2a.loc[:,['blue','red']],'\n')

print('data2a.loc[:,0]:\n',data2a.iloc[:,0],'\n')
print('data2a.loc[:,2] \n',data2a.iloc[:,2],'\n')
print('data2a.loc[:,[0,2]] \n', data2a.iloc[:,[0,2]],'\n')
print('***************索引行***********************\n\n')
print("data2c:\n",data2c,'\n')

print('data2c.iloc[0]:\n',data2c.iloc[0],'\n')
print('data2c.iloc[2]:\n',data2c.iloc[2],'\n')
print('data2c.iloc[0,2]:\n',data2c.iloc[0,2],'\n')

print('***************索引行与列***********************\n\n')
da = pd.DataFrame({'a':[1 ,2 , 3],'b':[4,5,6],'c':[7,8,9]})
print("da:\n",da,'\n')

print("da.loc[0:1,'b']:\n",da.loc[0:1,'b'],'\n')
print("da.loc[0:1,['a','b']]:\n",da.loc[0:1,['a','b']],'\n')

print("da[da['a']<=12]:\n",da[da['a']<=2],'\n')

print("da[da['a']<=2 & da['b']>=5]:\n",da[(da['a']<=2) & (da['b']>=5)],'\n')

print("da[(da['a']<=2) | (da['b']>=9)]:\n",da[(da['a']<=2) | (da['b']>=9)],'\n')

da.loc[da['a']!=1,'a'] = "0011010101011"
print("da:\n",da,'\n')

# header=1,index=1是否保留列名、行索引
da.to_csv("da.txt",encoding='utf-8',sep=' ',header=1,index=1,\
          columns=['a','b','c'], float_format='%.0f')
da1 = pd.read_csv("da.txt",sep='\s+',header=0, index_col=0, )





#======================================================================

df = pd.DataFrame({
                    'a': [1, 2] * 3,
                    'b': [True, False] * 3,
                    'c': [1.0, 2.0] * 3,
                    'd': ['1','2']*3
                  })

print(f"df.info() = {df.info()}\n")

#数值型特征: 包括int64,float64
df.select_dtypes(include = ['int64','float64'])

#仅int型的:
df.select_dtypes(include = 'int64')

#类别型特征(object):
df.select_dtypes(include = 'object')     

#布尔型特征(bool):
df.select_dtypes(include = 'bool')


#除了布尔型以外的所有特征:
df.select_dtypes(exclude = 'bool')


#这样得到的结果是DataFrame类型数据，下面将想要的特征名称取出来:
numerical_fea = list(df.select_dtypes(include = 'int64').columns)
numerical_fea


numerical_fea = list(df.select_dtypes(include =['int64','float64']).columns)
numerical_fea



df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)


pd.concat([pd.DataFrame([i], columns=['A']) for i in range(5)],ignore_index=True)


#如何在pandas中使用set_index( )与reset_index( )设置索引
"""
在数据分析过程中，有时出于增强数据可读性或其他原因，我们需要对数据表的索引值进行设定。在之前的文章中也有涉及过，在我们的pandas中，常用set_index( )与reset_index( )这两个函数进行索引设置，下面我们来了解一下这两个函数的用法。

DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)

参数解释：

keys：列标签或列标签/数组列表，需要设置为索引的列

drop：默认为True，删除用作新索引的列

append：是否将列附加到现有索引，默认为False。

inplace：输入布尔值，表示当前操作是否对原数据生效，默认为False。

verify_integrity：检查新索引的副本。否则，请将检查推迟到必要时进行。将其设置为false将提高该方法的性能，默认为false。



"""



import pandas as pd
import numpy as np
df = pd.DataFrame({'Country':['China','China', 'India', 'India', 'America', 'Japan', 'China', 'India'], 
 
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],
 
                    'Age':[50, 43, 34, 40, 25, 25, 45, 32]})




df_new = df.set_index('Country',drop=True, append=False, inplace=False, verify_integrity=False)
print(f"df_new = \n{df_new}\n")



# 可以看到，在上一步的代码中，是指定了drop=True，也就是删除用作新索引的列，下面师门尝试将drop=False.
df_new1 = df.set_index('Country',drop=False, append=False, inplace=False, verify_integrity=False)
print(f"df_new1 = \n{df_new1}\n")



df_new2 = df.set_index('Country',drop=False, append=True, inplace=False, verify_integrity=False)
print(f"df_new2 = \n{df_new2}\n")

df_new3 = df.set_index('Country',drop=True, append=True, inplace=False, verify_integrity=False)
print(f"df_new3 = \n{df_new3}\n")




"""
二、reset_index( )
1、函数体及主要参数解释：

DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
参数解释：

level：数值类型可以为：int、str、tuple或list，默认无，仅从索引中删除给定级别。默认情况下移除所有级别。控制了具体要还原的那个等级的索引 。

drop：当指定drop=False时，则索引列会被还原为普通列；否则，经设置后的新索引值被会丢弃。默认为False。

inplace：输入布尔值，表示当前操作是否对原数据生效，默认为False。

col_level：数值类型为int或str，默认值为0，如果列有多个级别，则确定将标签插入到哪个级别。默认情况下，它将插入到第一级。

col_fill：对象，默认‘’，如果列有多个级别，则确定其他级别的命名方式。如果没有，则重复索引名。

注意~~~reset_index（）还原可分为两种类型，第一种是对原来的数据表进行reset；第二种是对使用过set_index()函数的数据表进行reset。

"""





















































































































































































