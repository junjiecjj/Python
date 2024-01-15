#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:46:27 2021

@author: jack
https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453844413&idx=1&sn=69acab3a82a2cb7a5cc2977d17e93c20&chksm=87eaa074b09d2962332ee0a54c3223454eec3b8f9084263b17e5f908371604427b720504537f&mpshare=1&scene=1&srcid=1228S33Bhqb6H69cYlVP4FCi&sharer_sharetime=1640694583741&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=ASjJvN04ZTVcy3G5%2BjHqfDE%3D&pass_ticket=yU1HCo%2BJnp5soZ5VMcomrNg3YTmO6ciODMnSgvX6uzdrFR2Id%2FdxjlFDXp8HTnAb&wx_header=0#rd

"""

str_value = "hello，python coders"  
print(f"{ str_value }")  

num_value = 123  
print(f"{num_value % 2 }")  

import datetime  
  
today = datetime.date.today()  
print(f"{today: %Y%m%d}")  
# 20211019  
print(f"{today : %Y%m%d}")  

a = 42  
print(f"{a:b}")   # 2进制  

print(f"{a:o}")   # 8进制  

print(f"{a:x}")   # 16进制，小写字母  
 
print(f"{a:X}")   # 16进制，大写字母  
  
print(f"{a:c}")   # ascii 码  


num_value = 123.456  
print(f'{num_value   :.2f}') #保留 2 位小数  

nested_format = ".2f" #可以作为变量  
print(f'{num_value:{nested_format}}')  



x = 'test'  
print(f'{x:>10}')   # 右对齐，左边补空格  
'      test'  
print(f'{x:*<10}')  # 左对齐，右边补*  
'test******'  
print(f'{x:=^10}')  # 居中，左右补=  
'===test==='  
x, n = 'test', 10  
print(f'{x:~^{n}}') # 可以传入变量 n  
'~~~test~~~'  
 

x = '中'  
print(f"{x!s}") # 相当于 str(x)  
'中'  
print(f"{x!r}") # 相当于 repr(x)  
"'中'"  

class MyClass:  
    def __format__(self, format_spec) -> str:  
        print(f'MyClass __format__ called with {format_spec !r}')  
        return "MyClass()"  
  
  
print(f'{MyClass():bala bala  %%MYFORMAT%%}')  





