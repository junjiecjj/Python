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
