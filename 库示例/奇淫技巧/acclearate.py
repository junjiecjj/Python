#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:30:45 2022
@author: jack
https://mp.weixin.qq.com/s?__biz=MzAxODI5ODMwOA==&mid=2666561612&idx=2&sn=502e4bd10cebaf4f5515bd07221aa156&chksm=80dc44e7b7abcdf166478f6315b800c06ebe88704e5efc88b18defe7915ae716c4cc50214e1f&mpshare=1&scene=24&srcid=0311geVY7fu2tyCixHSk2F0H&sharer_sharetime=1646957678144&sharer_shareid=0d5c82ce3c8b7c8f30cc9a686416d4a8&exportkey=AbQuqt%2BWwkRs373GJkPsdnM%3D&acctmode=0&pass_ticket=03qZsJ2pZlqB67Dci6UO5DtFdUL%2FyUiOT3Cv2PY3sldV3fcuNDSrhc4GyGCH9wF4&wx_header=0#rd
 让 Python 起飞的 24 个骚操作！
"""

import time


# 第 1 式：测算代码运行时间
# 平凡方法
tic = time.time()
much_job = [x**2 for x in range(1,1000000,3)]
toc = time.time()

print("used {:.5}s".format(toc-tic))


# 快捷方法（jupyter 环境）
# %%time
# much_job = [x**2 for x in range(1,1000000,3)]




# 第 2 式：测算代码多次运行平均时间
# 平凡方法
from timeit import timeit

g = lambda x:x**2+1
def main1():
    return (g(2)**120)

timeit('main1()',globals = {'main1':main1},number = 10)


#第 3 式：按调用函数分析代码运行时间
#平凡方法
def relu(x):
	return (x if x>0 else 0)

def main2():
	result = [relu(x) for x in range(-100000,100000,1)]
	return result
import profile
profile.run('main2()')

#第 4 式：按行分析代码运行时间
#平凡方法
from line_profiler import LineProfiler
lprofile = LineProfiler(main2,relu)
lprofile.run('main2()')
lprofile.print_stats()


#第 5 式：用 set 而非 list 进行查找
#低速方法
data = (i**2+1 for i in range(1000000))
list_data = list(data)
set_data = set(data)

#%%time
1098987 in list_data

#%%time
1098987 in set_data

#第 6 式：用 dict 而非两个 list 进行匹配查找
#低速方法
list_a = [2*i-1 for i in range(1000000)]
list_b = [i**2 for i in list_a]
dict_ab = dict(zip(list_a,list_b))


#%%time
print(list_b[list_a.index(876567)])


#%%time
print(dict_ab.get(876567,None) )



#第 7 式：优先使用 for 循环而不是 while 循环
#低速方法

#%%time
s,i = 0,0
while i<10000:
	i = i+1
	s = s+i
print(s)




#%%time
s = 0;
for i in range(1,10001):
	s = s+i
print(s)



#第 8 式：在循环体中避免重复计算
#低速方法
a = [i**2+1 for i in range(2000)]

#%%time
b = [i/sum(a) for i in a]


#高速方法
#%%time
sum_a = sum(a)
b = [i/sum_a for i in a]


#第 9 式：用循环机制代替递归函数
#低速方法
#%%time
def fib(n):
	return (1 if n in (1,2) else fib(n-1)+fib(n-2))
print(fib(30))



#高速方法
#%%time
def fib1(n):
	if n in (1,2):
		return (1)
	a,b = 1,1
	for i in range(2,n):
		a,b = b,a+b
	return(b)
print(fib1(30))



#第 10 式：用缓存机制加速递归函数
#低速方法
#%%time
def fib2(n):
	return (1 if n in (1,2) else fib2(n-1)+fib2(n-2))
print(fib2(30))


#高速方法
#%%time
from functools import lru_cache

@lru_cache(100)
def fib3(n):
	return (1 if n in (1,2) else fib2(n-1)+fib2(n-2))
print(fib3(30))



#第 11 式：用 numba 加速 Python 函数
#低速方法
def my_power(x):
	return (x**2)

def my_power_sum(n):
	s = 0
	for i in range(1,n+1):
		s = s+my_power(i)
	return(s)

print(my_power_sum(1000000))

#高速方法
from numba import jit

@jit
def my_power1(x):
	return (x**2)

@jit
def my_power_sum1(n):
	s = 0
	for i in range(1,n+1):
		s = s+my_power1(i)
	return(s)

print(my_power_sum1(1000000))


#第 12 式：使用 collections.Counter 加速计数
#低速方法
data = [x**2%1989 for x in range(2000000)]

#%%time
values_count = {}
for i in data:
	i_cnt = values_count.get(i,0)
	values_count[i] = i_cnt + 1
print(values_count.get(4,0))


#高速方法
#%%time
from collections import Counter
values_count = Counter(data)
print(values_count.get(4,0))






#第 13 式：使用 collections.ChainMap 加速字典合并
#低速方法
dic_a = {i:i+1 for i in range(1,1000000,2)}
dic_b = {i:2*i+1 for i in range(1,1000000,3)}
dic_c = {i:3*i+1 for i in range(1,1000000,5)}
dic_d = {i:4*i+1 for i in range(1,1000000,7)}

#%%time
result = dic_a.copy()
result.update(dic_b)
result.update(dic_c)
result.update(dic_d)
print(result.get(9999,0))


#高速方法
#%%time
from collections import ChainMap
chain = ChainMap(dic_a, dic_b, dic_c, dic_d)
print(chain.get(9999,0))



#第 14 式：使用 np.array 代替 list
#低速方法
#%%time
a = range(1,1000000,3)
b = range(1000000,1,-3)
c = [3*a[i] -2*b[i] for i in range(0,len(a))]



#高速方法

#%%time
import numpy as np
array_a  = np.arange(1,1000000,3)
array_b = np.arange(1000000,1,-3)
array_c = 3*array_a - 2*array_b




#第 15 式：使用 np.ufunc 代替 math.func
#低速方法
import math
a = range(1,1000000,3)
b = [math.log(x) for x in a]


#高速方法
#%%time
import numpy as np
array_a = np.arange(1,1000000,3)
array_b = np.log(array_a)


#第 16 式：使用 np.where 代替 if
#低速方法
import numpy as np
array_a = np.arange(-100000,1000000)
relu = np.vectorize(lambda x:x if x>0 else 0)
array_b = relu(array_a)



#高速方法

#%%time
relu = lambda x:np.where(x>0,x,0)
array_b = relu(array_a)



#第 17 式：使用 np.ufunc 函数代替 applymap
#低速方法
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(-10,11,size=(100000,26)),columns = list('abcdefghijklmnoqprstvuwxyz'))

#%%time
dfresult = df.applymap(lambda x:np.sin(x)+ np.cos(x))


#高速方法

#%%time
dfresult = np.sin(df) + np.cos(df)






#第 18 式：使用预分配存储代替动态扩容
#低速方法
#%%time
import pandas as pd
import numpy as np

df = pd.DataFrame(columns = list('abcdefghijklmnoqprstvuwxyz'))
for i in range(10000):
	df.loc[i,:] = range(i, i+26)
	


#高速方法
#%%time
import pandas as pd
import numpy as np
df = pd.DataFrame(np.zeros((10000,26)),columns = list('abcdefghijklmnoqprstvuwxyz'))
for i in range(10000):
	df.loc[i,:] = range(i, i+26)


#第 19 式：使用 csv 文件读写代替 excel 文件读写
#低速方法
#%%time
df.to_excel('data.xlsx')
#高速方法
df.to_csv('data.csv')



#第 20 式：使用 pandas 多进程工具 pandarallel
#低速方法
import pandas as pd
import numpy as np 
df = pd.DataFrame(np.random.randint(-10,11,size=(10000,26)),columns = list('abcdefghijklmnoqprstvuwxyz'))
#%%time
result = df.apply(np.sum,axis = 1)


#高速方法
from pandarallel import pandarallel
pandarallel.initialize(nb_workers = 4)
result = df.parallel_apply(np.sum,axis = 1)


#第 21 式：使用 dask 加速 dataframe
#低速方法
import pandas as pd
import numpy as np 
df = pd.DataFrame(np.random.randint(0,6,size=(100000000,5)),columns = list('abced'))

#%time 
df.groupby('a').mean()


#高速方法

import dask.dataframe as dd
df_task = dd.from_pandas(df, npartitions = 40)

#%time 
df_task.groupby('a').mean().compute()



#第 22 式：使用 dask.delayed 进行加速
#低速方法
import time
def muchjob(x):
	time.sleep(5)
	return (x**2)

#%%time
result = [muchjob(i) for i in range(5)]


#高速方法
#%%time
from dask import delayed, compute
from dask import threaded,multiprocessing
values = [delayed(muchjob)(i) for i in range(5)]
result = compute(*values,scjeduler='multiprocessing')




#第 23 式：应用多线程加速 IO 密集型任务
#低速方法
#%%time 
def writefile(i):
	with open('./testDictionary'+'.txt','w') as f:
		s = ('hello %d' % i)*10000000
		f.write(s)
		
#串行任务
for i in range(10):
	writefile(i)
	
#高速方法
#%%time

import threading

#多线程任务
thread_list = []
for i in range(10):
	t = threading.Thread(target=writefile,args=(i,))
	t.setDaemon(True)
	thread_list.append(t)

for i in thread_list:
	t.start()  # 启动线程
	
for i in thread_list:
	t.join() #等待子线程结束
	

	


#第 24 式：应用多进程加速 CPU 密集型任务
#低速方法
#%%time
import time

def muchjob(x):
	time.sleep(5)
	return (x**2)

#串行任务
ans = [muchjob(i) for i in range(8)]
print(ans)




#高速方法
#%%time
import time
import multiprocessing
data = range(8)

# 多进程任务
pool = multiprocessing.Pool(processes = 4)
result = []


for i in range(8):
	result.append(pool.apply_async(muchjob,(i,)))
pool.close()
pool.join()
ans = [res.get() for res in result]
print(ans)




