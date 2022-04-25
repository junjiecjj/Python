#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:30:46 2022

@author: jack

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453847472&idx=1&sn=e236f3049a5d5d8a0dce7e9b93a6fcb7&chksm=87eaac79b09d256f76e0c4664ebfb01706ad032f0e4016c1ac8f519366489593d553d06a7161&mpshare=1&scene=24&srcid=0425ZWKDbAK4ACCJHN54HnVV&sharer_sharetime=1650850644620&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=AS4Li297gTxfzqBIdPBeJr4%3D&acctmode=0&pass_ticket=p716zXobtGbfw0swZspHFjoQO%2FNSfDcuD7NQpsXHDcKtjr3FZlocc12C39arlJpn&wx_header=0#rd

"""
# 一、关于列表
#1.问题描述
#在Python中，如果你试图在遍历一组数据的过程中，对其进行修改，这通常没什么问题。例如：
l = [3, 4, 56, 7, 10, 9, 6, 5]

for i in l:
    if not i % 2 == 0:
        continue
    l.remove(i)

print(l)

#上述这段代码遍历了一个包含数字的列表，为了去除掉所有偶数，直接修改了列表l。然而，运行后输出却是:
#[3, 56, 7, 9, 5]

#等一下！输出似乎不对。最终的结果仍然含有一个偶数56。为什么没有成功去除这个数呢？我们可以尝试打印出 for循环遍历的所有元素，运行如下代码：
l = [3, 4, 56, 7, 10, 9, 6, 5]

for i in l:
    print(i)
    if not i % 2 == 0:
        continue
    l.remove(i)

print(l)


# 方案一
l = [3, 4, 56, 7, 10, 9, 6, 5]

# 迭代翻转后的列表
for i in reversed(l):
    print(i)
    if not i % 2 == 0:
        continue
    l.remove(i)

print(l)


# 方案二
l = [3, 4, 56, 7, 10, 9, 6, 5]

# 在这里使用 'l.copy()' 来对列表 l 进行浅拷贝
for i in l.copy():  
    print(i)   
    if not i % 2 == 0:     
        continue  
    l.remove(i)
    
print(l)



# 二、关于字典
# 在对字典进行迭代时，不能修改字典。如下：
# {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
d = {k: k for k in range(10)}

for k, v in d.items():  
    if not v % 2 == 0:    
        continue  
    d.pop(k)




#2.解决方案
#我们可以先复制字典的所有 key ，随后在迭代 key 的过程中，移除不符合条件的元素。过程如下：
# {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
d = {k: k for k in range(10)}
# 这里复制了字典中的所有key值
# 没有复制整个字典
# 同时使用tuple()速度更快
for k in tuple(d.keys()):   
    if not d[k] % 2 == 0:    
        continue  
    d.pop(k)
    
print(d)


















































































































