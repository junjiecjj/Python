#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:46:55 2022

@author: jack
"""

import numpy as np
import torch


#============================================================================
#  使用 numpy.tofile() 和 numpy.fromfile() 函数保存和加载 NumPy 数组
#============================================================================


a = np.arange(0,12)

a.shape = 3,4

a.tofile("/home/jack/公共的/MLData/TrashFile/a.bin")

b = np.fromfile("/home/jack/公共的/MLData/TrashFile/a.bin", dtype=np.float) # 按照float类型读入数据

print(f"b  = {b}") # 读入的数据是错误的

print(f"a.dtype = {a.dtype}") # 查看a的dtype
# a.dtype = int64

b = np.fromfile("/home/jack/公共的/MLData/TrashFile/a.bin", dtype=np.int64) # 按照int32类型读入数据
print(f"b  = {b}") 
b.shape = 3, 4 # 按照a的shape修改b的shape

print(f"b  = {b}") 

#============================================================================
#  1 np.savetxt()和np.loadtxt()
# 使用 np.savetxt 和 np.loadtxt 只能读写 1 维和 2 维的数组
#============================================================================

print("=="*30)
"""
loadtxt() 和 savetxt() 函数处理正常的文本文件(.txt 等)
np.loadtxt和np.savetxt可以读写1维和2维数组的文本文件：
同时可以指定各种分隔符、针对特定列的转换器函数、需要跳过的行数等。
——
注意：只能处理 1维和2维数组。可以用于CSV格式文本文件
——
np.savetxt(fname, X, fmt=’%.18e’, delimiter=’ ‘, newline=’\n’, header=’’, footer=’’, comments=’# ‘, encoding=None)

"""

# 保存一个二维数组为txt
path1 = '/home/jack/公共的/MLData/TrashFile/test1.txt'
ar1 =  np.arange(24).reshape(4,6)

np.savetxt(path1, ar1, fmt='%.2f', delimiter=',',)#使用默认分割符（空格），保留两位小数

path2 = '/home/jack/公共的/MLData/TrashFile/test2.txt'
np.savetxt(path2, ar1,   delimiter=',', fmt='%.18e')

ar1_load = np.loadtxt(path2, delimiter=',')#指定逗号分割符
print(f"ar1_load = \n{ar1_load}")
print(f"ar1_load.dtype = \n{ar1_load.dtype}")


# 保存一个二维数组为CSV
path3 = '/home/jack/公共的/MLData/TrashFile/test1.csv'
np.savetxt(path3, ar1, fmt='%.2f', delimiter=',',)#使用默认分割符（空格），保留两位小数

ar2_load = np.loadtxt(path3, delimiter=',')#指定逗号分割符
print(f"ar2_load = \n{ar2_load}")
print(f"ar2_load.dtype = \n{ar2_load.dtype}")


# 保存2个1维数组为txt,
path1 = '/home/jack/公共的/MLData/TrashFile/test1.txt'
ar1 =  np.arange(0.0,5.0,1.0)
ar2 = np.arange(0.0,  6.0,1.0)*2
np.savetxt(path1, (ar1,ar2), fmt='%.2f', delimiter=',',)#使用默认分割符（空格），保留两位小数

ar_load = np.loadtxt(path1, delimiter=',')

# ar_load
# Out[60]: 
# array([[0., 1., 2., 3., 4.],
#        [0., 2., 4., 6., 8.]])


# 保存2个二维数组为txt,会报错，
path1 = '/home/jack/公共的/MLData/TrashFile/test1.txt'
ar1 =  np.arange(0.0, 24.0, 1.0).reshape(4,6)
ar2 = np.arange(0.0, 6.0, 1.0).reshape(2,3)*2
np.savetxt(path1, (ar1,ar2), fmt='%.2f', delimiter=',',)#使用默认分割符（空格），保留两位小数

path2 = '/home/jack/公共的/MLData/TrashFile/test2.txt'
np.savetxt(path2, ar1,   delimiter=',', fmt='%.18e')

ar1_load = np.loadtxt(path2, delimiter=',')#指定逗号分割符
print(f"ar1_load = \n{ar1_load}")
print(f"ar1_load.dtype = \n{ar1_load.dtype}")


#保存字典为TXT,会报错，不能保存字典
path1 = '/home/jack/公共的/MLData/TrashFile/test3.txt'
ar1 = {"Hello":np.random.randint(1,5,size=(4,4)),"Jack":np.random.randint(1,3,size=(2,4))}
print(f"ar1 = {ar1}")

np.savetxt(path1, ar1, fmt='%.2f', delimiter=',', )#error, savetxt只能保存数组
ar2_load = np.loadtxt(path1, delimiter=',' )#指定逗号分割符




# 保存三维数组也会出错
path1 = '/home/jack/公共的/MLData/TrashFile/test3.txt'
ar1 = np.random.randint(1,5,size=(2,3,4))
print(f"ar1 = {ar1}")

np.savetxt(path1, ar1, fmt='%.2f', delimiter=',', )#error, savetxt只能保存1D/2D数组
ar2_load = np.loadtxt(path1, delimiter=',' )#指定逗号分割符



#============================================================================
#  2 np.save()和np.load()
#============================================================================
print("=="*30)
"""
#===================================== np.save() ================================================

np.load和np.save是读写磁盘数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为.npy的文件中。
他们会自动处理元素类型和形状等信息.

numpy.save()
numpy.save() 函数将数组保存到以 .npy 为扩展名的文件中。

numpy.save(file, arr, allow_pickle=True, fix_imports=True)
参数说明：

file：要保存的文件，扩展名为 .npy，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上。
arr: 要保存的数组
allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组，Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，
对对象进行序列化和反序列化。
fix_imports: 可选，为了方便 Pyhton2 中读取 Python3 保存的数据。

#===================================== np.load() ================================================
numpy.load()函数从具有npy扩展名(.npy)的磁盘文件返回输入数组。
用法：numpy.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True,encoding=’ASCII’)

参数：

file :：file-like对象，字符串或pathlib.Path。要读取的文件。 File-like对象必须支持seek()和read()方法。
mmap_mode :如果不为None，则使用给定模式memory-map文件(有关详细信息，请参见numpy.memmap
模式说明)。
allow_pickle :允许加载存储在npy文件中的腌制对象数组。
fix_imports :仅在在Python 3上加载Python 2生成的腌制文件时有用，该文件包括包含对象数组的npy /npz文件。
encoding :仅当在Python 3中加载Python 2生成的腌制文件时有用，该文件包含包含对象数组的npy /npz文件。
Returns :数据存储在文件中。对于.npz文件，必须关闭NpzFile类的返回实例，以避免泄漏文件描述符。

allow_pickle = True后才可打开，因为numpy版本过高
Alldata = np.load('populations.npz',allow_pickle = True)

查看此npz文件下的所有npy文件，此项目里包含“data”和“feature_names”两个文件
Alldata.files

"""

# 保存一个二维数组为npy
import numpy as np
path1 = '/home/jack/公共的/MLData/TrashFile/test1.npy'
ar1 =  np.arange(24).reshape(4,6)

print(f"ar1 = \n{ar1}")
np.save(path1, ar1)      #如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。

#读取数组数据, .npy文件
ar_load = np.load(path1)
print(f"ar_load = \n{ar_load}")


# ar1 = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# ar_load = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]





# 保存2个二维数组为npy
import numpy as np
path1 = '/home/jack/公共的/MLData/TrashFile/test1.npy'
ar1 =  np.arange(24).reshape(4,6)
ar2 = np.random.randint(1,10,size=(2,4))
print(f"ar1 = \n{ar1}")
print(f"ar2 = \n{ar2}")

np.save(path1, (ar1,ar2), allow_pickle=True )# 可以没有allow_pickle=True,因为默认为True

#读取数组数据, .npy文件
ar_load = np.load(path1, allow_pickle=True )# 必须有allow_pickle=True，否则出错
print(f"ar_load = \n{ar_load}")
print(f"ar_load[0] = \n{ar_load[0]}")
print(f"ar_load[1] = \n{ar_load[1]}")


# ar1 = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# ar2 = 
# [[8 7 7 3]
#  [5 7 5 7]]
# ar_load = 
# [array([[ 0,  1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10, 11],
#         [12, 13, 14, 15, 16, 17],
#         [18, 19, 20, 21, 22, 23]]) array([[8, 7, 7, 3],
#                                           [5, 7, 5, 7]])]
# ar_load[0] = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# ar_load[1] = 
# [[8 7 7 3]
#  [5 7 5 7]]



#保存1个字典为npy
path1 = '/home/jack/公共的/MLData/TrashFile/test2.npy'
ar1 = {"Hello":np.random.randint(1,5,size=(4,4)),"Jack":np.random.randint(1,3,size=(2,4))}
print(f"ar1 = {ar1}")

np.save(path1, ar1, allow_pickle=True )# 可以没有allow_pickle=True,因为默认为True
ar2_load = np.load(path1, allow_pickle=True)# 必须有allow_pickle=True，否则出错
print(f"ar2_load = \n{ar2_load}")
print(f"ar2_load[()] = \n{ar2_load[()]}")
print(f"ar2_load.item() = \n{ar2_load.item()}")

print(f"ar2_load[()][\"Hello\"] = \n{ar2_load[()]['Hello']}")
print(f"ar2_load[()]['Jack'] = \n{ar2_load[()]['Jack']}")

# ar1 = {'Hello': array([[3, 2, 4, 4],
#        [4, 2, 4, 3],
#        [3, 3, 4, 2],
#        [3, 2, 1, 4]]), 'Jack': array([[2, 1, 2, 1],
#        [2, 2, 1, 2]])}
# ar2_load[()] = 
# {'Hello': array([[3, 2, 4, 4],
#        [4, 2, 4, 3],
#        [3, 3, 4, 2],
#        [3, 2, 1, 4]]), 'Jack': array([[2, 1, 2, 1],
#        [2, 2, 1, 2]])}
# ar2_load.item() = 
# {'Hello': array([[3, 2, 4, 4],
#        [4, 2, 4, 3],
#        [3, 3, 4, 2],
#        [3, 2, 1, 4]]), 'Jack': array([[2, 1, 2, 1],
#        [2, 2, 1, 2]])}
# ar2_load[()]["Hello"] = 
# [[3 2 4 4]
#  [4 2 4 3]
#  [3 3 4 2]
#  [3 2 1 4]]
# ar2_load[()]['Jack'] = 
# [[2 1 2 1]
#  [2 2 1 2]]


import numpy as np
	# define
dict = {'a' : {1,2,3}, 'b': {4,5,6}}
	# save
np.save(path1,dict)
	# load
dict_load=np.load(path1, allow_pickle=True)
	
print("dict =",dict_load.item())
print("dict['a'] =",dict_load.item()['a'])


# dict = {'a': {1, 2, 3}, 'b': {4, 5, 6}}
# dict['a'] = {1, 2, 3}



#保存1个字典为npy
path1 = '/home/jack/公共的/MLData/TrashFile/test2.npy'
ar1 =  np.random.randint(1,5,size=(4,4))
ar2 = np.random.randint(1,3,size=(2,4))
Dic = {"first":[12,21,33], "name":"jack" }
print(f"ar1 = {ar1}")
print(f"ar2 = {ar2}")
print(f"Dic = \n{Dic}")

np.save(path1, Dic, allow_pickle=True )     # 可以没有allow_pickle=True
Dic_load = np.load(path1, allow_pickle=True)# 必须有allow_pickle=True，否则出错
print(f"Dic_load[()] = \n{Dic_load[()]}")
print(f"Dic_load.item() = \n{Dic_load.item()}")
print(f"Dic_load[()]['first'] = \n{Dic_load[()]['first']}")
print(f"Dic_load[()]['name'] = \n{Dic_load[()]['name']}")

# ar1 = [[3 3 1 4]
#  [1 3 1 1]
#  [3 2 1 1]
#  [3 3 2 2]]
# ar2 = [[2 1 2 1]
#  [1 1 1 2]]
# Dic = 
# {'first': [12, 21, 33], 'name': 'jack'}
# Dic_load[()] = 
# {'first': [12, 21, 33], 'name': 'jack'}
# Dic_load[()]['first'] = 
# [12, 21, 33]
# Dic_load[()]['name'] = 
# jack

#In [101]: Dic_load
# Out[101]: array({'first': [12, 21, 33], 'name': 'jack'}, dtype=object)

#同时保存多个数组和1个字典
path1 = '/home/jack/公共的/MLData/TrashFile/test2.npy'
ar1 =  np.random.randint(1,5,size=(4,4))
ar2 = np.random.randint(1,3,size=(2,4))
Dic = {"first":[12,21,33], "name":"jack" }
np.save(path1, (Dic,ar1,ar2), allow_pickle=True )
Data = np.load(path1, allow_pickle=True)# 必须有allow_pickle=True，否则出错
print(f"Data = \n{Data}")

print(f"Data[0]['first'] = \n{Data[0]['first']}")
print(f"Data[0]['name'] = \n{Data[0]['name']}")
print(f"Data[1]  = \n{Data[1] }")
print(f"Data[2]  = \n{Data[2] }")

# Data = 
# [{'first': [12, 21, 33], 'name': 'jack'} array([[4, 4, 1, 3],
#                                                 [4, 2, 4, 4],
#                                                 [4, 1, 3, 2],
#                                                 [1, 4, 1, 1]])
#  array([[2, 2, 1, 1],
#         [1, 2, 2, 1]])]
# Data[0]['first'] = 
# [12, 21, 33]
# Data[0]['name'] = 
# jack
# Data[1]  = 
# [[4 4 1 3]
#  [4 2 4 4]
#  [4 1 3 2]
#  [1 4 1 1]]
# Data[2]  = 
# [[2 2 1 1]
#  [1 2 2 1]]


#同时保存多个数组和多个字典
path1 = '/home/jack/公共的/MLData/TrashFile/test2.npy'
ar1 =  np.random.randint(1,5,size=(4,4))
ar2 = np.random.randint(1,3,size=(2,4))
Dic1 = {"first":[12,21,33], "name":"jack" }
Dic2 = {"first":np.arange(6).reshape(2,3), "name":"David" }
np.save(path1, (Dic1,ar1,ar2,Dic2), allow_pickle=True )
Data = np.load(path1, allow_pickle=True)# 必须有allow_pickle=True，否则出错
print(f"Data = \n{Data}")

print(f"Data[0]['first'] = \n{Data[0]['first']}")
print(f"Data[0]['name'] = \n{Data[0]['name']}")
print(f"Data[1]  = \n{Data[1] }")
print(f"Data[2]  = \n{Data[2] }")
print(f"Data[3]['first'] = \n{Data[3]['first']}")
print(f"Data[3]['name'] = \n{Data[3]['name']}")

# Data = 
# [{'first': [12, 21, 33], 'name': 'jack'} array([[1, 4, 3, 2],
#                                                 [1, 1, 2, 4],
#                                                 [2, 4, 4, 1],
#                                                 [2, 3, 2, 2]])
#  array([[2, 1, 1, 1],
#         [1, 1, 1, 2]]) {'first': array([[0, 1, 2],
#                               [3, 4, 5]]), 'name': 'David'}]
# Data[0]['first'] = 
# [12, 21, 33]
# Data[0]['name'] = 
# jack
# Data[1]  = 
# [[1 4 3 2]
#  [1 1 2 4]
#  [2 4 4 1]
#  [2 3 2 2]]
# Data[2]  = 
# [[2 1 1 1]
#  [1 1 1 2]]
# Data[3]['first'] = 
# [[0 1 2]
#  [3 4 5]]
# Data[3]['name'] = 
# David

#============================================================================
#  3   np.savez()和np.load()
#============================================================================

"""
savez() 函数用于将多个数组写入文件，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npz 的文件中。
savez函数的第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，
非关键字参数传递的数组会自动起名为arr_0, arr_1, …。
savez函数输出的是一个压缩文件(扩展名为npz)，其中每个文件都是一个save函数保存的npy文件，文件名对应于数组名。
load函数自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容。
——————


numpy.savez() 函数将多个数组保存到以 npz 为扩展名的文件中。
numpy.savez(file, *args, **kwds)
参数说明：

file：要保存的文件，扩展名为 .npz，如果文件路径末尾没有扩展名 .npz，该扩展名会被自动加上。
args: 要保存的数组，可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为 arr_0, arr_1, …　。
kwds: 要保存的数组使用关键字名称。

numpy.load()函数从具有npy扩展名(.npy)的磁盘文件返回输入数组。
用法：numpy.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True,encoding=’ASCII’)

参数：

file :：file-like对象，字符串或pathlib.Path。要读取的文件。 File-like对象必须支持seek()和read()方法。
mmap_mode :如果不为None，则使用给定模式memory-map文件(有关详细信息，请参见numpy.memmap
模式说明)。
allow_pickle :允许加载存储在npy文件中的腌制对象数组。
fix_imports :仅在在Python 3上加载Python 2生成的腌制文件时有用，该文件包括包含对象数组的npy /npz文件。
encoding :仅当在Python 3中加载Python 2生成的腌制文件时有用，该文件包含包含对象数组的npy /npz文件。
Returns :数据存储在文件中。对于.npz文件，必须关闭NpzFile类的返回实例，以避免泄漏文件描述符。
allow_pickle = True后才可打开，因为numpy版本过高
Alldata = np.load('populations.npz',allow_pickle = True)

查看此npz文件下的所有npy文件，此项目里包含“data”和“feature_names”两个文件
Alldata.files
"""

#保存多个数组
ar1 =  np.arange(24).reshape(4,6)
ar2 =  np.arange(6).reshape(2,3)*11
path = "/home/jack/公共的/MLData/TrashFile/test4.npz"

print("=="*30)
print(f"ar1 = \n{ar1}\n ar2 = \n{ar2}")
np.savez(path, A = ar1, B = ar2)


DATA = np.load(path)  # 可以没有allow_pickle=True
print(f"DATA = {DATA}")
print(f"DATA.files = {DATA.files}")

print(f"DATA['A'] = \n{DATA['A']}\nDATA['B'] = \n{DATA['B']}")

# ============================================================
# ar1 = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
#  ar2 = 
# [[ 0 11 22]
#  [33 44 55]]
# DATA = <numpy.lib.npyio.NpzFile object at 0x7f24c1853fd0>
# DATA.files = ['A', 'B']
# DATA['A'] = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# DATA['B'] = 
# [[ 0 11 22]
#  [33 44 55]]

#====================================================================
# 保存1个字典
dict_ = {'a' : 1, 'b': 2}
path = "/home/jack/公共的/MLData/TrashFile/test5.npz"
print("=="*30)
print(f"dict_ = \n{dict_}")
np.savez(path, A = dict_ ,)# 不能把allow_pickle=True作为参数，否则allow_pickle会被当做待存的数据


DATA = np.load(path,allow_pickle=True) # 必须有allow_pickle=True，否则出错

print(f"DATA.files = {DATA.files}")
print(f"DATA['A'] = \n{DATA['A']}")

# DATA.files = ['A']
# DATA['A'] = 
# {'a': 1, 'b': 2}


#同时保存多个数组和多个字典
dic1 = {"first":np.arange(12).reshape(3,4),"name":'jack'}
dic2 = {"fam":np.random.randint(1,10,size=(2,8)), "host":"129.20.1.12"}
ar1 = np.arange(32).reshape(4,8)
ar2 = np.arange(32,44).reshape(2,6)

np.savez(path, D1 = dic1, D2 = dic2, A1=ar1, A2=ar2 )
DATA = np.load(path,allow_pickle=True) # 必须有allow_pickle=True，否则出错
print(f"DATA.files = {DATA.files}")
print(f"DATA['D1'] = \n{DATA['D1']}")
print(f"DATA['D1'][()] = \n{DATA['D1'][()] }")
print(f"DATA['D2'] = \n{DATA['D2']}")
print(f"DATA['D2'][()] = \n{DATA['D2'][()] }")
print(f"DATA['A1'] = \n{DATA['A1']}")
print(f"DATA['A2'] = \n{DATA['A2']}")


# DATA.files = ['D1', 'D2', 'A1', 'A2']
# DATA['D1'] = 
# {'first': array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]]), 'name': 'jack'}
# DATA['D1'][()] = 
# {'first': array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]]), 'name': 'jack'}
# DATA['D2'] = 
# {'fam': array([[3, 9, 1, 9, 5, 8, 9, 5],
#        [8, 8, 2, 6, 1, 5, 1, 6]]), 'host': '129.20.1.12'}
# DATA['D2'][()] = 
# {'fam': array([[3, 9, 1, 9, 5, 8, 9, 5],
#        [8, 8, 2, 6, 1, 5, 1, 6]]), 'host': '129.20.1.12'}
# DATA['A1'] = 
# [[ 0  1  2  3  4  5  6  7]
#  [ 8  9 10 11 12 13 14 15]
#  [16 17 18 19 20 21 22 23]
#  [24 25 26 27 28 29 30 31]]
# DATA['A2'] = 
# [[32 33 34 35 36 37]
#  [38 39 40 41 42 43]]

#============================================================================
#  numpy.savez_compressed
#============================================================================
"""
这个就是在前面numpy.savez的基础上加了压缩,前面我介绍时尤其注明numpy.savez是得到的文件打包,不压缩的.这个文件就是对文件进行打包时使用了压缩,
可以理解为压缩前各npy的文件大小不变,使用该函数比前面的numpy.savez得到的npz文件更小.

注:函数所需参数和numpy.savez一致,用法完成一样.

用法:
numpy.savez_compressed(file, *args, **kwds)
以压缩的.npz 格式将多个数组保存到一个文件中。

提供数组作为关键字参数，以将它们存储在输出文件中的相应名称下：savez(fn, x=x, y=y)。

如果数组被指定为位置参数，即 savez(fn, x, y) ，它们的名称将是 arr_0、arr_1 等。

参数：
file： 字符串或文件
将保存数据的文件名(字符串)或打开的文件(file-like 对象)。如果文件是一个字符串或一个路径，.npz 扩展名将被附加到文件名(如果它不存在)。

args： 参数，可选
要保存到文件的数组。请使用关键字参数(参见下面的 kwds)为数组分配名称。指定为 args 的数组将命名为 “arr_0”、“arr_1” 等。

kwds： 关键字参数，可选
要保存到文件的数组。每个数组都将以其对应的关键字名称保存到输出文件中。

返回：
None

"""
#保存多个数组
ar1 =  np.arange(24).reshape(4,6)
ar2 =  np.arange(6).reshape(2,3)*11
path = "/home/jack/公共的/MLData/TrashFile/test5.npz"

print("=="*30)
print(f"ar1 = \n{ar1}\n ar2 = \n{ar2}")
np.savez_compressed(path, A = ar1, B = ar2)


DATA = np.load(path)
print(f"DATA = \n{DATA}")
print(f"DATA.files = {DATA.files}")
print(f"DATA['A'] = \n{DATA['A']}\nDATA['B'] = \n{DATA['B']}")


# ============================================================
# ar1 = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
#  ar2 = 
# [[ 0 11 22]
#  [33 44 55]]
# <numpy.lib.npyio.NpzFile object at 0x7f24c240afa0>
# DATA['A'] = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# DATA['B'] = 
# [[ 0 11 22]
#  [33 44 55]]



#===============================================================
#同时保存多个数组和多个字典
ar1 =  np.arange(24).reshape(4,6)
ar2 =  np.arange(6).reshape(2,3)*11
ar3 = np.random.randint(1,20,size=(4,4))


ar4 = np.random.randint(1,10,size=(2,4))

dic1 = {"Hello":ar3, "jack":ar4}
dic2 = {"fam":np.random.randint(1,10,size=(2,8)), "host":"129.20.1.12"}

path = "/home/jack/公共的/MLData/TrashFile/test7.npz"

print("=="*30)
print(f"ar1 = \n{ar1}\n ar2 = \n{ar2}\n ar3 = \n{ar3}\ndic1 = \n{dic1}\ndic2 = \n{dic2}")
np.savez_compressed(path, A1 = ar1, A2 = ar2, A3 = ar3,  D1= dic1, D2=dic2)


DATA = np.load(path ,allow_pickle=True) # 必须有allow_pickle=True，否则出错

print(f"DATA.files = {DATA.files}")
print(f"DATA['D1'] = \n{DATA['D1']}")
print(f"DATA['D1'][()] = \n{DATA['D1'][()] }")
print(f"DATA['D1'][()]['Hello'] = \n{DATA['D1'][()]['Hello'] }")
print(f"DATA['D1'][()]['jack'] = \n{DATA['D1'][()]['jack'] }")
print(f"DATA['D2'] = \n{DATA['D2']}")
print(f"DATA['D2'][()] = \n{DATA['D2'][()] }")
print(f"DATA['D2'][()]['fam'] = \n{DATA['D2'][()]['fam'] }")
print(f"DATA['D2'][()]['host'] = \n{DATA['D2'][()]['host'] }")
print(f"DATA['A1'] = \n{DATA['A1']}")
print(f"DATA['A2'] = \n{DATA['A2']}")
print(f"DATA['A3'] = \n{DATA['A3']}")


# ============================================================
# ar1 = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
#  ar2 = 
# [[ 0 11 22]
#  [33 44 55]]
#  ar3 = 
# [[18  2 17 15]
#  [13 12 18 16]
#  [11  8  1 13]
#  [16 17 14  4]]
# dic1 = 
# {'Hello': array([[18,  2, 17, 15],
#        [13, 12, 18, 16],
#        [11,  8,  1, 13],
#        [16, 17, 14,  4]]), 'jack': array([[5, 5, 4, 2],
#        [8, 8, 2, 4]])}
# dic2 = 
# {'fam': array([[5, 4, 8, 3, 4, 2, 9, 3],
#        [4, 1, 9, 6, 7, 9, 4, 8]]), 'host': '129.20.1.12'}
# DATA.files = ['A1', 'A2', 'A3', 'D1', 'D2']
# DATA['D1'] = 
# {'Hello': array([[18,  2, 17, 15],
#        [13, 12, 18, 16],
#        [11,  8,  1, 13],
#        [16, 17, 14,  4]]), 'jack': array([[5, 5, 4, 2],
#        [8, 8, 2, 4]])}
# DATA['D1'][()] = 
# {'Hello': array([[18,  2, 17, 15],
#        [13, 12, 18, 16],
#        [11,  8,  1, 13],
#        [16, 17, 14,  4]]), 'jack': array([[5, 5, 4, 2],
#        [8, 8, 2, 4]])}
# DATA['D1'][()]['Hello'] = 
# [[18  2 17 15]
#  [13 12 18 16]
#  [11  8  1 13]
#  [16 17 14  4]]
# DATA['D1'][()]['jack'] = 
# [[5 5 4 2]
#  [8 8 2 4]]
# DATA['D2'] = 
# {'fam': array([[5, 4, 8, 3, 4, 2, 9, 3],
#        [4, 1, 9, 6, 7, 9, 4, 8]]), 'host': '129.20.1.12'}
# DATA['D2'][()] = 
# {'fam': array([[5, 4, 8, 3, 4, 2, 9, 3],
#        [4, 1, 9, 6, 7, 9, 4, 8]]), 'host': '129.20.1.12'}
# DATA['D2'][()]['fam'] = 
# [[5 4 8 3 4 2 9 3]
#  [4 1 9 6 7 9 4 8]]
# DATA['D2'][()]['host'] = 
# 129.20.1.12
# DATA['A1'] = 
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]
# DATA['A2'] = 
# [[ 0 11 22]
#  [33 44 55]]
# DATA['A3'] = 
# [[18  2 17 15]
#  [13 12 18 16]
#  [11  8  1 13]
#  [16 17 14  4]]































































































































































































































































