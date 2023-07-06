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
data1 = pd.read_table("data1.txt", sep='\s+')
data1a = pd.read_csv("data1.txt", sep='\s+')
data1_1 = pd.read_table("data1_1.txt", sep=',')
data1_1a = pd.read_csv("data1_1.txt", sep=',')

print(data1,'\n ********************* data1结束************************')



data2 = pd.read_table("data2.txt", sep = ',', skiprows=[0,1,3,6])
data2a = pd.read_csv("data2.txt", sep = ',', header=0, skiprows=[0,1,3,6])
data2a = pd.DataFrame(data2a)
# index_col=2,指定用作行标签的列
data2b = pd.read_csv("data2.txt", sep = ',', index_col = 2, skiprows=[0,1,3,6])
data2c = pd.read_csv("data2.txt", sep = ',',  skiprows = [0,1,3,6])

# 指定header name
data2d = pd.read_csv("data2a.txt", sep=',', header=None, names = ['col1','col2','col3','col4','col5'],)
print("data2:\n",data2,'\n')
print("data2a:\n",data2a,'\n')
print("data2b:\n",data2b,'\n')
print("data2c:\n",data2c,'\n')
print("data2d:\n",data2d,'\n')

print('***************索引列***********************\n\n')
print("data2d:\n",data2d,'\n')

print('data2d[\'col1\']:\n', data2d['col1'], '\n')
print('data2d.loc[:,[\'col1\']]:\n', data2d.loc[:,['col1']], '\n')
print('data2d[\'col3\']:\n', data2d['col3'],'\n')
print('data2d.loc[:,[\'col3\']] \n', data2d.loc[:,['col3']], '\n')
print('data2d[\'col1\',\'col3\']:\n',data2d[['col1','col3']], '\n')
print('data2d.loc[:,[\'col1\',\'col3\']] \n', data2d.loc[:,['col1','col3']], '\n')

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





#=====================================================================
# 如何在pandas中使用set_index( )与reset_index( )设置索引
# https://zhuanlan.zhihu.com/p/110819220
#=====================================================================
     

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
#（1）我们先看一下第二种情况，即对使用过set_index()函数的数据表进行reset：

import pandas as pd
import numpy as np
df = pd.DataFrame({'Country':['China','China', 'India', 'India', 'America', 'Japan', 'China', 'India'], 
 
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],
 
                    'Age':[50, 43, 34, 40, 25, 25, 45, 32]})
print(f"df = \n{df}\n")

df_new = df.set_index('Country',drop=True, append=False, inplace=False, verify_integrity=False)
print(f"df_new = \n{df_new}\n")


df_new01 = df_new.reset_index(drop=False)
print(f"df_new01 = \n{df_new01}\n")


df_new02 = df_new.reset_index(drop=True)
print(f"df_new02 = \n{df_new02}\n")



#（2）再看下第一种情况，即对原来的数据表进行reset处理：

import pandas as pd
import numpy as np
df = pd.DataFrame({'Country':['China','China', 'India', 'India', 'America', 'Japan', 'China', 'India'], 
 
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000],
 
                    'Age':[50, 43, 34, 40, 25, 25, 45, 32]})

print(f"df = \n{df}\n")

print(f"df column name= \n{list(df)}")
print(f"df column name= \n{df.columns.values}")
print(f"df column name= \n{df.columns.tolist()}")



df_new03 = df.reset_index(drop=False)
print(f"df_new03 = \n{df_new03}\n")



df_new04 = df.reset_index(drop=True)
print(f"df_new04 = \n{df_new04}\n")


df1 = pd.DataFrame({'Country':['Franch','Gemm', 'England', 'Austra'], 
 
                   'Income':[778, 54, 3223, 476],
 
                    'Age':[12, 55,   98, 76]})
print(f"df1 = \n{df1}\n")


combined = df.append(df1)
print(f"combined = \n{combined}\n")

df2 = combined.reset_index()
print(f"df2 = \n{df2}\n")

# combined.reset_index(inplace=True)
# print(f"combined = \n{combined}\n")

#如果我们想直接使用重置后的索引，不保留原来的index，就可以加上(drop = True)，
df3 = combined.reset_index( drop = True)
print(f"df3 = \n{df3}\n")
# combined.reset_index(inplace=True, drop = True)
# print(f"combined = \n{combined}\n")


#=====================================================================
# 如何使用drop方法对数据进行删减处理
# https://zhuanlan.zhihu.com/p/109913870
#=====================================================================


"""
下面我们先简单说一下drop的用法及一些主要参数：
drop函数：drop(labels, axis=0, level=None, inplace=False, errors='raise')

关于参数axis：

axis为0时表示删除行，axis为1时表示删除列，还是一样~

关于参数errors：

errors='raise'会让程序在labels接收到没有的行名或者列名时抛出错误导致程序停止运行，errors='ignore'会忽略没有的行名或者列名，只对存在的行名或者列名进行操作，没有指定的话也是默认‘errors='raise'’。
label: 接受string或者array,代表删除的行或列的标签，无默认。
axis: 接受0或1，代表操作的轴向，默认为0，代表按行删除
levels: 接受Int或者索引名，代表标签所在的级别，默认为None。
inplaces: 接受boolean，代表操作是否对原数据生效，默认为false

"""
import pandas as pd
import numpy as np
cities = pd.DataFrame(np.random.randn(5, 5),
     index=['a', 'b', 'c', 'd', 'e'],
     columns=['shenzhen', 'guangzhou', 'beijing', 'nanjing', 'haerbin'])
print(f"cities = \n{cities}\n")



#（1）删除掉第a行：

df1=cities.drop(labels='a')
print(f"df1 = \n{df1}\n")


df2=cities.drop(index='a')
print(f"df2 = \n{df2}\n")

#可以看到，因为这里我们是删除行，所以我们用labels、index都是可以的。不过还是推荐使用labels。而已还是要注意~drop默认对原表不生效，如果要对原表生效，需要添加参数：inplace=True


#（2）删除非连续的多行：
df1=cities.drop(labels=['a','c','e'])
print(f"df1 = \n{df1}\n")

#这里我们要插播一下比较细节的东西，大家以后可能会遇到的一个问题：
#为了方便看，我们这次不设置索引名，下面重新创建一下数据表：
import pandas as pd
city = pd.DataFrame(np.random.randn(5, 5),
     columns=['shenzhen', 'guangzhou', 'beijing', 'nanjing', 'haerbin'])
print(f"city = \n{city}\n")

#我们还是删掉第1行（1实际上是第二行），而已这一次我们加上inplace=True：
city.drop(labels=1,axis=0,inplace=True)
print(f"city = \n{city}\n")


#如果这个时候，我们再输入一次：
city.drop(labels=1,axis=0,inplace=True)


"""
（报错了，很应该啊，好像也没什么，毕竟第1行本来就被我们删掉了）

但是！！！注意了，我们在这里想说明的是：
如果我们没标注索引，而已把数据一行一行删掉的话，该行对应的索引也是被我们删掉的！

比如说一个数据表里一共有5行，我们把第2、第3行给删掉了，就会顺道对应把索引2、索引3删掉，这时候数据的索引就会变成[1、4、5]，即使这个时候原来的第4行数据现在变成第2行了，现在也无法用索引第2行的方式来获取现在的第2行（原来的第4行），因为索引已经乱了。

所以说，大家在用drop的时候还是要注意这一点的。同时，我们该如何解决这个问题呢？

答案是要将索引重置，这样后面再次使用才不会因为索引不存在而报错。

重置索引的方法是：reset_index

reset_index，默认(drop = False)，当我们指定(drop = True)时，则不会保留原来的index，会直接使用重置后的索引。
"""


df1 = city.reset_index()
print(f"df1 = \n{df1}\n")

#如果我们想直接使用重置后的索引，不保留原来的index，就可以加上(drop = True)，如下所示：
df2 = city.reset_index(drop=True)
print(f"df2 = \n{df2}\n")



#这个时候我们再试试删除第1行，果然没有问题了。
df2.drop(labels=1,axis=0,inplace=True)
print(f"df2 = \n{df2}\n")



#（3）删除连续的多行：
#当我们想删除连续多行时，如果还是一个一个标签输入的话，显得不太智能，所以我们可以用更简便的方法：
#再建一个新的数据表:
import pandas as pd

people= pd.DataFrame(np.random.randn(6, 6),
     columns=['jack', 'rose', 'mike', 'chenqi', 'amy','tom'])
print(f"people  = \n{people}\n")    

#当我们想删去第2-4行时，可以用如下表示：
df2=people.drop(labels=range(2,5),axis=0)
print(f"df2 = \n{df2}\n")


#（4）删除列：
#在最新的表中删除‘jack’这一列：

df2=people.drop(labels='jack',axis=1)
print(f"df2 = \n{df2}\n")


#同时删除‘rose’、‘mike’这两列：
df2=people.drop(labels=['rose','mike'],axis=1)
print(f"df2 = \n{df2}\n")







#=====================================================================
# pandas.get_dummies 的用法
# https://zhuanlan.zhihu.com/p/109913870
#=====================================================================

import pandas as pd
df = pd.DataFrame([  
            ['green' , 'A'],   
            ['red'   , 'B'],   
            ['blue'  , 'A']])  

df.columns = ['color',  'class'] 
print(f"df = \n{df}\n")

df1 = pd.get_dummies(df) 
print(f"df1 = \n{df1}\n")

# 可以对指定列进行get_dummies
df3 = pd.get_dummies(df.color)
print(f"df3 = \n{df3}\n")


#将指定列进行get_dummies 后合并到元数据中
df4 = df.join(pd.get_dummies(df.color))
print(f"df4 = \n{df4}\n")
# or
df5 = pd.concat([df,df3], axis=1)
print(f"df5 = \n{df5}\n")


import pandas as pd
df = pd.DataFrame({
    'gender':['m','f','m','f','m','f','n']})
df_onehot = pd.get_dummies(df)

print(f"df_onehot = \n{df_onehot}\n")


import pandas as pd
data_df = pd.DataFrame({
    'id':[1,2,3,4,5,6,7],'gender':['m','f','m','f','m','f','n']})
df_onehot = pd.get_dummies(data_df)
print(f"df_onehot = \n{df_onehot}\n")


#prefix: string, list of strings, or dict of strings, default None
#用于附加DataFrame列名的字符串。在DataFrame上调用get_dummies时，传递长度等于列数的列表。或者，前缀可以是将列名映射到前缀的字典。
import pandas as pd
data_df = pd.DataFrame({
    'id':[1,2,3,4,5,6,7],'gender':['m','f','m','f','m','f','n']})
df_onehot = pd.get_dummies(data_df, prefix ='gen')
print(f"df_onehot = \n{df_onehot}\n")


# prefix_sep: string, default ‘_’
# 如果附加前缀，则使用分隔符/分隔符。或作为前缀传递列表或字典。

import pandas as pd
data_df = pd.DataFrame({
    'id':[1,2,3,4,5,6,7],'gender':['m','f','m','f','m','f','n']})
df_onehot = pd.get_dummies(data_df, prefix ='gen', prefix_sep = '/')
print(f"df_onehot = \n{df_onehot}\n")









































































