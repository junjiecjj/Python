#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:46:50 2022

@author: jack
"""


import os




"""

os.walk()列出目录和子目录中的所有文件
os.walk()函数返回一个生成器，该生成器创建一个值元组（current_path、current_path 中的目录、current_path 中的文件）。

注意：使用该os.walk()函数，我们可以列出给定目录中的所有目录、子目录和文件。

它是一个递归函数，即每次调用生成器时，它都会递归地跟随每个目录以获取文件和目录的列表，直到初始目录中没有更多的子目录可用。

例如，调用os.walk('path')将为它访问的每个目录生成两个列表。第一个列表包含文件，第二个列表包含目录。


"""

pathnames = []
for (dirpath, dirnames, filenames) in os.walk('/home/jack/FL_semantic/results/Centralized_LeNet/test_results/raw_image'):
    for filename in filenames:
        pathnames += [os.path.join(dirpath, filename)]
print(pathnames)



print ('***获取当前目录***')
print( os.getcwd())
print (os.path.abspath(os.path.dirname(__file__)))

print( '***获取上级目录***')
print (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print (os.path.abspath(os.path.dirname(os.getcwd())))
print (os.path.abspath(os.path.join(os.getcwd(), "..")))

print ('***获取上上级目录***')
print( os.path.abspath(os.path.join(os.getcwd(), "../..")))


import sys, os

print(f"__file__ = {__file__}")    #当前.py文件的位置
print(f"os.path.abspath(__file__) = {os.path.abspath(__file__)}\n")  #返回当前.py文件的绝对路径
print(f"os.path.dirname(os.path.abspath(__file__)) = {os.path.dirname(os.path.abspath(__file__))}\n")   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
print(f"os.path.dirname(os.path.dirname(os.path.abspath(__file__))) = {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\n") #返回文件本身目录的上层目录
print(f"os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) = {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}\n")  #每多一层，即再上一层目录

print(f"os.path.realpath(__file__) = {os.path.realpath(__file__)}\n")   #当前文件的真实地址
print(f"os.path.dirname(os.path.realpath(__file__)) = {os.path.dirname(os.path.realpath(__file__))}\n") # 当前文件夹的路径

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)   #将目录或路径加入搜索路径

print(__name__)

# 打印当前的文件， 行数， 函数名，在调试时对定位很有用
print(f"[file:{os.path.realpath(__file__)}, line:{sys._getframe().f_lineno}, fun:{sys._getframe().f_code.co_name} ]")




"""
os.mkdir()
用法
mkdir(path, mode)
1
参数
path ：要创建的目录的路径（绝对路径或者相对路径）
mode ： Linux 目录权限数字表示，windows 请忽略该参数
权限种类分为三种，分别为 读，写，可执行。
身份为 owners，groups，others 三种。
3 × 3 共有 9种 权限
使用 3 个数字表示3个身份的权限
对于任何一个身份，可读 为 4，可写为 2，可执行为 1，用户具有的权限为相应的权重相加的结果，例如具有可读和可写权限，但是没有执行权限，则数字为 1 + 2=3。
默认是 777
注意： mkdir 只能创建一个目录，不能递归创建目录，例如创建 ./two/three 目录的时候，./two 目录必须存在，否则报错，另外需要注意的是，如果已经存在了目录，则创建目录也会失败！



os.makedirs()
用法
makedirs(path, mode=0o777, exist_ok=False):
1
参数
path : 要创建的目录，绝对路径或者相对路径
mode： 和上面一样，windows 用户请忽略
exist_ok：如果已经存在怎么处理，默认是 False ，即：已经存在程序报错。当为 True 时，创建目录的时候如果已经存在就不报错。

"""




# python 3
import os
path1="./one"
path2="./one/two"
path3="./two/three"

try:
    os.mkdir(path1)
    print("创建"+path1+"成功")
except:
    pass

try:
    os.mkdir(path2)
    print("创建" + path2 + "成功")
except:
    pass

try:
    os.mkdir(path3)
except:
    print("无法创建{0}目录".format(path3))

# 输出：
# 创建./one成功
# 创建./one/two成功
# 无法创建./two/three目录



import os
path1="./one"
path2="./one/two"
path3="./two/three"

try:
    os.makedirs(path1,exist_ok=True)
    print("创建"+path1+"成功，或者目录已经存在")
except:
    pass

try:
    os.makedirs(path2,exist_ok=True)
    print("创建" + path2 + "成功，或者目录已经存在")
except:
    pass

try:
    os.makedirs(path3,exist_ok=True)
    print("可以递归创建{0}目录".format(path3))
except:
    pass

# 输出结果
# 创建./one成功，或者目录已经存在
# 创建./one/two成功，或者目录已经存在
# 可以递归创建./two/three目录


"""
得到当前工作目录，即当前Python脚本工作的目录路径: os.getcwd()

返回指定目录下的所有文件和目录名:os.listdir()

函数用来删除一个文件:os.remove()

删除多个目录：os.removedirs（r“c：\python”）

检验给出的路径是否是一个文件：os.path.isfile()

检验给出的路径是否是一个目录：os.path.isdir()

判断是否是绝对路径：os.path.isabs()

检验给出的路径是否真地存:os.path.exists()
即使文件存在，你可能还需要判断文件是否可进行读写操作。
    判断文件是否可做读写操作
    使用os.access()方法判断文件是否可进行读写操作。
    语法：
    os.access(path, mode)
    path为文件路径，mode为操作模式，有这么几种:

    os.F_OK: 检查文件是否存在;

    os.R_OK: 检查文件是否可读;

    os.W_OK: 检查文件是否可以写入;

    os.X_OK: 检查文件是否可以执行


返回一个路径的目录名和文件名:os.path.split()

分离扩展名：os.path.splitext()

获取路径名：os.path.dirname()

获取文件名：os.path.basename()

运行shell命令: os.system()

读取和设置环境变量:os.getenv()与os.putenv()

给出当前平台使用的行终止符:os.linesep Windows使用’\r\n’，Linux使用’\n’而Mac使用’\r’

指示你正在使用的平台：os.name 对于Windows，它是’nt’，而对于Linux/Unix用户，它是’posix’

重命名：os.rename（old， new）

创建多级目录：os.makedirs（r“c：\python\test”）

创建单个目录：os.mkdir（“test”）

获取文件属性：os.stat（file）

修改文件权限与时间戳：os.chmod（file）

终止当前进程：os.exit（）

获取文件大小：os.path.getsize（filename）


创建目录
os.mkdir("file")
复制文件：
shutil.copyfile("oldfile","newfile") #oldfile和newfile都只能是文件
shutil.copy("oldfile","newfile") #oldfile只能是文件夹，newfile可以是文件，也可以是目标目录
复制文件夹：
shutil.copytree("olddir","newdir") #olddir和newdir都只能是目录，且newdir必须不存在
重命名文件（目录）
os.rename("oldname","newname") #文件或目录都是使用这条命令
移动文件（目录）
shutil.move("oldpos","newpos")
删除文件
os.remove("file")
删除目录
os.rmdir("dir") #只能删除空目录
shutil.rmtree("dir") #空目录、有内容的目录都可以删
转换目录
os.chdir("path") #换路径


os.path 模块提供了一些函数， 返回一个相对路径的绝对路径， 以及检查给定的路径是否为绝对路径。

调用 os.path.abspath(path)将返回参数的绝对路径的字符串。这是将相对路径转换为绝对路径的简便方法。
调用 os.path.isabs(path)，如果参数是一个绝对路径，就返回 True，如果参数是一个相对路径，就返回 False。
调用 os.path.relpath(path, start)将返回从 start 路径到 path 的相对路径的字符串。如果没有提供 start，就使用当前工作目录作为开始路径。
调用 os.listdir(path)将返回文件名字符串的列表，包含 path 参数中的每个文件（这个函数在 os 模块中，而不是 os.path）。

检查路径有效性
如果你提供的路径不存在， 许多 Python 函数就会崩溃并报错。 os.path 模块提供了一些函数，用于检测给定的路径是否存在，以及它是文件还是文件夹。

如果 path 参数所指的文件或文件夹存在， 调用 os.path.exists(path)将返回 True，否则返回 False。
如果 path 参数存在，并且是一个文件， 调用 os.path.isfile(path)将返回 True， 否则返回 False。
如果 path 参数存在， 并且是一个文件夹， 调用 os.path.isdir(path)将返回 True，否则返回 False。
在现代Python版本中，可以使用os.scandir()和 pathlib.Path 来替代 os.listdir()。
os.scandir()调用时返回一个迭代器而不是一个列表。

import os
with os.scandir('my_directory') as entries:
    for entry in entries:
        print(entry.name)


os.path.join()函数用于路径拼接文件路径，可以传入多个路径
(1) 如果不存在以 '/' 开始的参数，则函数会自动加上
(2) 存在以 '/' 开始的参数，从最后一个以 '/' 开头的参数开始拼接，之前的参数全部丢弃。
(3) 同时存在以 './' 与 '/' 开始的参数，以 '/' 为主，从最后一个以 '/' 开头的参数开始拼接，之前的参数全部丢弃。
(4) 只存在以 './' 开始的参数,会从 './' 开头的参数的上一个参数开始拼接。


"""

# change directory
import os, sys

path = "/home/jack/tmp"

retval = os.getcwd()
print ( f"当前工作目录为 {retval} ")


os.chdir(path)
retval = os.getcwd()
print ( f"当前工作目录为 {retval} ")


import glob
glob.glob(os.path.join('/home/jack/图片/Wallpapers', '*' + '.jpg'))

# ['/home/jack/图片/Wallpapers/DSC_4564.jpg',
#  '/home/jack/图片/Wallpapers/DSC_4352.jpg']


sorted(glob.glob(os.path.join('/home/jack/图片/Wallpapers', '*' + '.jpg')))
# ['/home/jack/图片/Wallpapers/DSC_4352.jpg',
#  '/home/jack/图片/Wallpapers/DSC_4564.jpg']



# os.scandir()获取目录的文件
# 该scandir()函数返回目录条目以及文件属性信息，为许多常见用例提供更好的性能。

# 它返回对象的迭代器os.DirEntry，其中包含文件名。

for path in os.scandir('/home/jack/FL_semantic/results/Centralized_LeNet/test_results/raw_image/'):
    if path.is_file():
        print(path.name)







