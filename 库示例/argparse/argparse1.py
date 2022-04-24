#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:36:14 2022

@author: jack
"""

import argparse

#https://zhuanlan.zhihu.com/p/56922793
#=======================================================
# 传入一个参数
#=======================================================
parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('integers', type=str, help='传入的数字')

args = parser.parse_args()

#获得传入的参数
print(args)


#==========================================================================
parser = argparse.ArgumentParser(description="miao shu")
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

args = parser.parse_args()

#获得传入的参数
print(args)

#=======================================================
# 操作args字典
#=======================================================
# 其实得到的这个结果Namespace(integers='5')是一种类似于python字典的数据类型。
# 我们可以使用 arg.参数名来提取这个参数

parser = argparse.ArgumentParser(description='命令行中传入一个数字')
#type是要传入的参数的数据类型  help是该参数的提示信息
parser.add_argument('integers', type=str, help='传入的数字')

args = parser.parse_args()

#获得integers参数
print(args.integers)




#=======================================================
# 传入多个参数
#=======================================================
parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=str, nargs='+',help='传入的数字')
#nargs是用来说明传入的参数个数，'+' 表示传入至少一个参数。这时候再重新在命令行中运行python demo.py 1 2 3 4得到
args = parser.parse_args()

print(args.integers)







#=======================================================
# 改变数据类型
#=======================================================
parser = argparse.ArgumentParser(description='命令行中传入一个数字')
parser.add_argument('integers', type=int, nargs='+',help='传入的数字')
args = parser.parse_args()

#对传入的数据进行加总
print(sum(args.integers))




#=======================================================
# 可选参数
#=======================================================
#为了在命令行中避免上述位置参数的bug（容易忘了顺序），可以使用可选参数，这个有点像关键词传参，但是需要在关键词前面加--，例如
parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str,help='姓')
parser.add_argument('--name', type=str,help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)


# python demo.py --family=张 --name=三


#=======================================================
# 默认值
#=======================================================

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str, default='陈',help='姓')
parser.add_argument('--name', type=str, default='俊杰', help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)

# 在命令行中分别输入 python demo.py 、 python demo.py --family=李
#=======================================================
# 必需参数
#=======================================================

parser = argparse.ArgumentParser(description='姓名')
parser.add_argument('--family', type=str, default='陈',help='姓')
parser.add_argument('--name', type=str, required=False, default='俊杰', help='名')
args = parser.parse_args()

#打印姓名
print(args.family+args.name)
